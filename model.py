import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class KokoroModel(nn.Module):
    """
    Simplified Kokoro-style model architecture.
    Optimized for MPS (Metal Performance Shaders) acceleration.
    """

    def __init__(self, vocab_size: int, mel_dim: int = 80, hidden_dim: int = 512):
        """
        Initialize the Kokoro TTS model
        
        Args:
            vocab_size: Size of the phoneme vocabulary
            mel_dim: Dimension of mel spectrogram features
            hidden_dim: Hidden dimension for internal layers
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim

        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        # Project bidirectional LSTM output back to hidden_dim
        self.text_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # Duration Predictor
        # Predicts a single log-duration value for each phoneme.
        # Common practice is to predict log durations and use MSE loss.
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Predicts 1 duration value per token
        )

        # Mel feature projection to match hidden dimension
        self.mel_projection_in = nn.Linear(mel_dim, hidden_dim)

        # Decoder
        # Input to decoder is mel_projected + attended_text_expanded
        self.decoder = nn.LSTM(
            hidden_dim + hidden_dim, hidden_dim, batch_first=True
        )

        # Attention mechanism
        # Query: decoder output, Key/Value: expanded text encoder output
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        # Output projection
        self.mel_projection_out = nn.Linear(hidden_dim, mel_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def encode_text(self, phoneme_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode phoneme indices to hidden representations.
        
        Args:
            phoneme_indices: Tensor of shape (batch_size, seq_len)
            
        Returns:
            Encoded text representations of shape (batch_size, seq_len, hidden_dim)
        """
        text_emb = self.text_embedding(phoneme_indices)
        text_emb = self.dropout(text_emb)
        text_encoded, _ = self.text_encoder(text_emb)
        
        # Project bidirectional output to hidden_dim
        text_encoded = self.text_projection(text_encoded)
        return text_encoded

    def _predict_durations(self, text_encoded: torch.Tensor) -> torch.Tensor:
        """
        Predicts log durations for each phoneme.
        Args:
            text_encoded: Output from text encoder (batch_size, text_seq_len, hidden_dim)
        Returns:
            Log durations of shape (batch_size, text_seq_len)
        """
        # Squeeze the last dimension which is 1
        log_durations = self.duration_predictor(text_encoded).squeeze(-1)
        return log_durations

    @staticmethod
    def _length_regulate(
        encoder_outputs: torch.Tensor,
        durations: torch.Tensor,
        max_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expands encoder outputs based on durations.

        Args:
            encoder_outputs: Tensor of shape (batch_size, text_seq_len, hidden_dim)
            durations: Tensor of shape (batch_size, text_seq_len), representing duration for each token.
                       Expected to be integer values or values that can be rounded.
            max_len: Optional. Maximum length for the expanded sequence. If not provided,
                     calculated from batch's max expanded length.

        Returns:
            - Expanded tensor of shape (batch_size, expanded_seq_len, hidden_dim)
            - A boolean mask for the expanded sequence (batch_size, expanded_seq_len)
              (True for padded elements, False for actual data).
        """
        # Ensure durations are integers for indexing
        durations_int = torch.round(durations).long()

        batch_size, text_seq_len, hidden_dim = encoder_outputs.shape

        # Calculate the total length for each item in the batch after expansion
        expanded_lengths = torch.sum(durations_int, dim=1)

        if max_len is None:
            max_expanded_len = expanded_lengths.max().item()
        else:
            max_expanded_len = max_len

        # Initialize output tensor with zeros and mask with True (indicating padding)
        expanded_outputs = torch.zeros(
            batch_size, max_expanded_len, hidden_dim, device=encoder_outputs.device
        )
        attention_mask = torch.ones(
            batch_size, max_expanded_len, dtype=torch.bool, device=encoder_outputs.device
        )

        for i in range(batch_size):
            current_idx = 0
            for j in range(text_seq_len):
                d = durations_int[i, j].item()
                if d > 0: # Only expand if duration is positive
                    # Repeat the encoder output 'd' times
                    segment = encoder_outputs[i, j].unsqueeze(0).repeat(d, 1)

                    # Place into the expanded_outputs tensor, respecting max_expanded_len
                    end_idx = min(current_idx + d, max_expanded_len)
                    # Handle cases where segment is too long due to truncation
                    segment_to_copy = segment[:end_idx - current_idx]

                    expanded_outputs[i, current_idx:end_idx] = segment_to_copy
                    attention_mask[i, current_idx:end_idx] = False # Mark as actual data

                    current_idx += d
                    if current_idx >= max_expanded_len: # Stop if max length is reached
                        break

        return expanded_outputs, attention_mask

    def forward_training(
        self,
        phoneme_indices: torch.Tensor,
        mel_specs: torch.Tensor,
        phoneme_durations: torch.Tensor # Ground truth durations (batch_size, text_seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass during training (teacher forcing).

        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            mel_specs: Target mel spectrograms of shape (batch_size, mel_seq_len, mel_dim)
            phoneme_durations: Ground truth durations for each phoneme (batch_size, text_seq_len)

        Returns:
            - Predicted mel spectrograms of shape (batch_size, mel_seq_len, mel_dim)
            - Predicted log durations of shape (batch_size, text_seq_len) (for duration loss)
        """
        batch_size = phoneme_indices.size(0)

        # Encode text
        text_encoded = self.encode_text(phoneme_indices)

        # Predict durations (these are used for a separate duration loss)
        predicted_log_durations = self._predict_durations(text_encoded)

        # Length regulate text_encoded using ground-truth durations for stable training.
        # The `max_len` is aligned with `mel_specs.size(1)` for attention consistency.
        text_encoded_expanded, attention_mask_text = self._length_regulate(
            text_encoded, phoneme_durations.float(), max_len=mel_specs.size(1)
        )
        
        outputs = []
        hidden = None
        
        for t in range(mel_specs.size(1)):
            # Current mel frame (teacher forcing)
            if t == 0:
                mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=mel_specs.device)
            else:
                mel_input = mel_specs[:, t-1:t, :]
            
            # Project mel to hidden dimension
            mel_projected = self.mel_projection_in(mel_input)
            mel_projected = self.dropout(mel_projected)
            
            # Attention over expanded text with padding mask
            attended, _ = self.attention(
                query=mel_projected,
                key=text_encoded_expanded,
                value=text_encoded_expanded,
                key_padding_mask=attention_mask_text # Mask padded elements
            )

            # Decoder step
            decoder_input = torch.cat([mel_projected, attended], dim=2)
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            decoder_out = self.dropout(decoder_out)
            
            # Project back to mel
            mel_pred = self.mel_projection_out(decoder_out)
            outputs.append(mel_pred)
        
        return torch.cat(outputs, dim=1), predicted_log_durations

    def forward_inference(self, phoneme_indices: torch.Tensor, max_len: int = 800) -> torch.Tensor:
        """
        Forward pass during inference (autoregressive).
        
        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            max_len: Maximum length for generated sequence
            
        Returns:
            Generated mel spectrograms of shape (batch_size, max_len, mel_dim)
        """
        batch_size = phoneme_indices.size(0)
        
        # Encode text
        text_encoded = self.encode_text(phoneme_indices)

        # Predict durations using the trained duration predictor
        predicted_log_durations = self._predict_durations(text_encoded)

        # Length regulate text_encoded using predicted durations.
        # max_len is not fixed here, it's derived from the summed predicted durations
        text_encoded_expanded, attention_mask_text = self._length_regulate(
            text_encoded, torch.exp(predicted_log_durations) # Convert log durations to actual durations
        )

        outputs = []
        hidden = None
        mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=phoneme_indices.device)

        for t in range(max_len):
            # Project mel to hidden dimension
            mel_projected = self.mel_projection_in(mel_input)

            # Attention over expanded text with padding mask
            attended, _ = self.attention(
                query=mel_projected,
                key=text_encoded_expanded,
                value=text_encoded_expanded,
                key_padding_mask=attention_mask_text
            )

            # Decoder step
            decoder_input = torch.cat([mel_projected, attended], dim=2)
            decoder_out, hidden = self.decoder(decoder_input, hidden)

            # Project back to mel
            mel_pred = self.mel_projection_out(decoder_out)
            outputs.append(mel_pred)

            # Use prediction as next input (autoregressive)
            mel_input = mel_pred

            # TODO: Add an "end of speech" prediction mechanism to stop generation
            # For now, it generates up to max_len

        return torch.cat(outputs, dim=1)

    def forward(self, phoneme_indices: torch.Tensor, mel_specs: Optional[torch.Tensor] = None, phoneme_durations: Optional[torch.Tensor] = None) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - automatically chooses training or inference mode.

        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            mel_specs: Target mel spectrograms for training mode (optional). If provided,
                       `phoneme_durations` must also be provided.
            phoneme_durations: Ground truth durations for training mode. Required if `mel_specs` is not None.

        Returns:
            - Mel spectrograms (predicted or generated) in inference mode.
            - Tuple (predicted mel, predicted log durations) in training mode.
        """
        if mel_specs is not None:
            if phoneme_durations is None:
                raise ValueError("phoneme_durations must be provided for training mode.")
            return self.forward_training(phoneme_indices, mel_specs, phoneme_durations)
        else:
            # Inference mode (autoregressive)
            return self.forward_inference(phoneme_indices)

    def get_model_info(self) -> dict:
        """
        Get model information and parameter count.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'vocab_size': self.vocab_size,
            'mel_dim': self.mel_dim,
            'hidden_dim': self.hidden_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
