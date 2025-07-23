import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch.utils.checkpoint import checkpoint # Import checkpoint

class KokoroModel(nn.Module):
    """
    Simplified Kokoro-style model architecture.
    Optimized for MPS (Metal Performance Shaders) acceleration
    """

    def __init__(self, vocab_size: int, mel_dim: int = 80, hidden_dim: int = 512):
        """
        Initialize the Kokoro model
        
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
        # We'll apply checkpointing to the LSTM directly in forward
        self.text_encoder = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.text_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # Duration Predictor
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
        # We'll apply checkpointing to the LSTM directly in forward
        self.decoder = nn.LSTM(
            hidden_dim + hidden_dim, hidden_dim, batch_first=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        # Output projection for Mel Spectrogram
        self.mel_projection_out = nn.Linear(hidden_dim, mel_dim)

        # End-of-Speech (Stop Token) Predictor
        self.stop_token_predictor = nn.Linear(hidden_dim, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def encode_text(self, phoneme_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode phoneme indices to hidden representations.
        This function will be checkpointed.

        Args:
            phoneme_indices: Tensor of shape (batch_size, seq_len)

        Returns:
            Encoded text representations of shape (batch_size, seq_len, hidden_dim)
        """
        text_emb = self.text_embedding(phoneme_indices)
        text_emb = self.dropout(text_emb)
        # Checkpoint the LSTM layer
        text_encoded, _ = checkpoint(self.text_encoder, text_emb, use_reentrant=False)
        
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
        # Duration predictor is small, might not benefit much from checkpointing itself,
        # but its input `text_encoded` might be from a checkpointed operation.
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
        This static method cannot be directly checkpointed using torch.utils.checkpoint.checkpoint
        unless wrapped in an nn.Module or a partial function that provides inputs.
        Given its stateless nature and pure tensor operations, it's generally not a target for checkpointing.

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
        durations_int = torch.round(durations).long()

        batch_size, text_seq_len, hidden_dim = encoder_outputs.shape

        expanded_lengths = torch.sum(durations_int, dim=1)

        if max_len is None:
            max_expanded_len = expanded_lengths.max().item()
        else:
            max_expanded_len = max_len

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
                if d > 0:
                    segment = encoder_outputs[i, j].unsqueeze(0).repeat(d, 1)

                    end_idx = min(current_idx + d, max_expanded_len)
                    segment_to_copy = segment[:end_idx - current_idx]

                    expanded_outputs[i, current_idx:end_idx] = segment_to_copy
                    attention_mask[i, current_idx:end_idx] = False

                    current_idx += d
                    if current_idx >= max_expanded_len:
                        break

        return expanded_outputs, attention_mask


    def forward_training(
        self,
        phoneme_indices: torch.Tensor,
        mel_specs: torch.Tensor,
        phoneme_durations: torch.Tensor, # Ground truth durations (batch_size, text_seq_len)
        stop_token_targets: torch.Tensor # Ground truth stop tokens (batch_size, mel_seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass during training (teacher forcing).

        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            mel_specs: Target mel spectrograms of shape (batch_size, mel_seq_len, mel_dim)
            phoneme_durations: Ground truth durations for each phoneme (batch_size, text_seq_len)
            stop_token_targets: Ground truth stop tokens (0 for all but last frame, which is 1)
                                of shape (batch_size, mel_seq_len)

        Returns:
            - Predicted mel spectrograms of shape (batch_size, mel_seq_len, mel_dim)
            - Predicted log durations of shape (batch_size, text_seq_len) (for duration loss)
            - Predicted stop token logits of shape (batch_size, mel_seq_len) (for stop token loss)
        """
        batch_size = phoneme_indices.size(0)

        # Encode text - checkpointing applied inside encode_text
        text_encoded = self.encode_text(phoneme_indices)

        # Predict durations (duration_predictor is small, no checkpointing here)
        predicted_log_durations = self._predict_durations(text_encoded)

        # Length regulate text_encoded using ground-truth durations
        text_encoded_expanded, attention_mask_text = self._length_regulate(
            text_encoded, phoneme_durations.float(), max_len=mel_specs.size(1)
        )

        outputs = []
        stop_token_outputs = [] # To store stop token predictions
        hidden = None

        # Iterating over time steps, each step's computation can be complex.
        # Checkpointing the decoder loop might be beneficial if the sequence is long.
        # However, checkpointing inside a loop where `hidden` state is passed
        # requires careful handling with `use_reentrant=False` and ensuring all inputs
        # requiring gradients are passed.
        # For simplicity and common practice with LSTMs, we often checkpoint the LSTM module directly.

        # To checkpoint the *entire* decoder loop would be more complex,
        # requiring wrapping the loop logic in a function to pass to checkpoint.
        # A more practical approach is to checkpoint the LSTM layer itself if it's recurrent.

        # Let's create a wrapper for the decoder step for checkpointing
        def decoder_step_fn(mel_proj, attended, hx):
            decoder_input = torch.cat([mel_proj, attended], dim=2)
            decoder_out, new_hx = self.decoder(decoder_input, hx)
            return decoder_out, new_hx

        for t in range(mel_specs.size(1)):
            # Current mel frame (teacher forcing)
            if t == 0:
                mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=mel_specs.device)
            else:
                mel_input = mel_specs[:, t-1:t, :]

            # Project mel to hidden dimension
            mel_projected = self.mel_projection_in(mel_input)
            mel_projected = self.dropout(mel_projected)

            # Attention mechanism - if this is a heavy part, checkpoint it.
            # MultiheadAttention can be memory intensive.
            # Query, Key, Value need to be passed for checkpointing.
            # If key_padding_mask requires_grad=False, it's fine.
            attended, _ = checkpoint(self.attention, mel_projected, text_encoded_expanded, text_encoded_expanded, key_padding_mask=attention_mask_text, use_reentrant=False)

            # Decoder step - checkpoint the LSTM if hidden state is passed
            # For recurrent models like LSTM, checkpointing them directly is often sufficient.
            # If `hidden` contains tensors that require gradients, they must be passed as inputs.
            # Initial `hidden` (None) does not require grad.
            if hidden is None:
                # Need dummy tensor if hidden is None, or handle first step outside checkpoint
                # Simpler: checkpoint the entire decoder (LSTM)
                decoder_out, hidden = self.decoder(torch.cat([mel_projected, attended], dim=2), hidden)
            else:
                # Pass hx and cx explicitly for checkpointing LSTM
                decoder_out, hidden = checkpoint(self.decoder, torch.cat([mel_projected, attended], dim=2), hidden, use_reentrant=False)

            decoder_out = self.dropout(decoder_out)

            # Project back to mel
            mel_pred = self.mel_projection_out(decoder_out)
            outputs.append(mel_pred)

            # Predict stop token
            stop_token_logit = self.stop_token_predictor(decoder_out)
            stop_token_outputs.append(stop_token_logit)

        # Concatenate outputs
        predicted_mel_frames = torch.cat(outputs, dim=1)
        predicted_stop_logits = torch.cat(stop_token_outputs, dim=1).squeeze(-1) # (batch_size, mel_seq_len)

        return predicted_mel_frames, predicted_log_durations, predicted_stop_logits

    def forward_inference(self, phoneme_indices: torch.Tensor, max_len: int = 800, stop_threshold: float = 0.5) -> torch.Tensor:
        """
        Forward pass during inference (autoregressive).
        Gradient checkpointing is generally not used during inference as it's a training-specific optimization.
        
        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            max_len: Maximum length for generated sequence
            stop_threshold: Probability threshold for stopping generation.
            
        Returns:
            Generated mel spectrograms of shape (batch_size, generated_mel_seq_len, mel_dim)
        """
        if phoneme_indices.size(0) > 1:
            print("Warning: Inference with stop token is most reliable with batch_size=1.")

        batch_size = phoneme_indices.size(0)
        
        # Encode text - no checkpointing for inference
        text_encoded = self.encode_text(phoneme_indices)

        # Predict durations
        predicted_log_durations = self._predict_durations(text_encoded)

        # Length regulate text_encoded using predicted durations
        text_encoded_expanded, attention_mask_text = self._length_regulate(
            text_encoded, torch.exp(predicted_log_durations)
        )

        outputs = []
        hidden = None
        mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=phoneme_indices.device)

        for t in range(max_len):
            mel_projected = self.mel_projection_in(mel_input)

            # Attention (no checkpointing for inference)
            attended, _ = self.attention(
                query=mel_projected,
                key=text_encoded_expanded,
                value=text_encoded_expanded,
                key_padding_mask=attention_mask_text
            )

            # Decoder step (no checkpointing for inference)
            decoder_input = torch.cat([mel_projected, attended], dim=2)
            decoder_out, hidden = self.decoder(decoder_input, hidden)

            mel_pred = self.mel_projection_out(decoder_out)
            outputs.append(mel_pred)

            stop_token_logit = self.stop_token_predictor(decoder_out)
            stop_probability = torch.sigmoid(stop_token_logit)

            mel_input = mel_pred

            if batch_size == 1 and stop_probability.item() > stop_threshold:
                print(f"Stopping generation at frame {t} (stop_prob: {stop_probability.item():.4f})")
                break

        return torch.cat(outputs, dim=1)

    def forward(
        self,
        phoneme_indices: torch.Tensor,
        mel_specs: Optional[torch.Tensor] = None,
        phoneme_durations: Optional[torch.Tensor] = None,
        stop_token_targets: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass - automatically chooses training or inference mode.

        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            mel_specs: Target mel spectrograms for training mode (optional). If provided,
                       `phoneme_durations` and `stop_token_targets` must also be provided.
            phoneme_durations: Ground truth durations for training mode. Required if `mel_specs` is not None.
            stop_token_targets: Ground truth stop tokens for training mode. Required if `mel_specs` is not None.

        Returns:
            - Mel spectrograms (predicted or generated) in inference mode.
            - Tuple (predicted mel, predicted log durations, predicted stop logits) in training mode.
        """
        if mel_specs is not None:
            if phoneme_durations is None or stop_token_targets is None:
                raise ValueError("phoneme_durations and stop_token_targets must be provided for training mode.")
            return self.forward_training(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets)
        else:
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
