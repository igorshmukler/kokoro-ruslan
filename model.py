import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch.utils.checkpoint import checkpoint # Keep for gradient checkpointing

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as a buffer, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape `(batch_size, seq_len, embedding_dim)`
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: the sequence to the encoder (batch_size, seq_len, d_model).
            src_mask: the mask for the src sequence (seq_len, seq_len).
            src_key_padding_mask: the mask for the src keys per batch (batch_size, seq_len).
        """
        # Self-attention part
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward part
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class KokoroModel(nn.Module):
    """
    Simplified Kokoro-style model architecture.
    Optimized for MPS (Metal Performance Shaders) acceleration
    """

    def __init__(self, vocab_size: int, mel_dim: int = 80, hidden_dim: int = 512,
                 n_encoder_layers: int = 6, n_heads: int = 8, encoder_ff_dim: int = 2048,
                 encoder_dropout: float = 0.1):
        """
        Initialize the Kokoro model with a Transformer encoder
        
        Args:
            vocab_size: Size of the phoneme vocabulary
            mel_dim: Dimension of mel spectrogram features
            hidden_dim: Hidden dimension for internal layers (d_model for Transformer)
            n_encoder_layers: Number of Transformer encoder layers
            n_heads: Number of attention heads in MultiheadAttention
            encoder_ff_dim: Dimension of the feed-forward network in Transformer blocks
            encoder_dropout: Dropout rate for the Transformer encoder
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim # This will now be d_model

        # Text encoder: Embedding + Positional Encoding + Stack of Transformer Blocks
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=encoder_dropout)

        # Stack of TransformerEncoderBlock layers
        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, n_heads, encoder_ff_dim, encoder_dropout)
            for _ in range(n_encoder_layers)
        ])

        # No longer need text_projection if hidden_dim is consistent
        # self.text_projection = nn.Linear(hidden_dim * 2, hidden_dim) # Removed as LSTM is replaced

        # Duration Predictor (remains the same)
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Predicts 1 duration value per token
        )

        # Mel feature projection to match hidden dimension
        self.mel_projection_in = nn.Linear(mel_dim, hidden_dim)

        # Decoder (can remain LSTM or be replaced with Transformer Decoder)
        # For simplicity, keeping LSTM for now as the request was only for encoder replacement.
        # In a full FastSpeech2/VITS, this would often be another Transformer stack.
        self.decoder = nn.LSTM(
            hidden_dim + hidden_dim, hidden_dim, batch_first=True
        )

        # Attention mechanism (remains the same MultiheadAttention)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=n_heads, batch_first=True # Use same n_heads as encoder for consistency
        )

        # Output projection for Mel Spectrogram
        self.mel_projection_out = nn.Linear(hidden_dim, mel_dim)

        # End-of-Speech (Stop Token) Predictor
        self.stop_token_predictor = nn.Linear(hidden_dim, 1)

        # Dropout for regularization (general model dropout)
        self.dropout = nn.Dropout(encoder_dropout) # Reusing encoder_dropout for general model dropout

    def encode_text(self, phoneme_indices: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode phoneme indices to hidden representations using Transformer encoder.

        Args:
            phoneme_indices: Tensor of shape (batch_size, seq_len)
            mask: Optional boolean mask (batch_size, seq_len) for padding.
                  True for padded positions.

        Returns:
            Encoded text representations of shape (batch_size, seq_len, hidden_dim)
        """
        text_emb = self.text_embedding(phoneme_indices) * self.hidden_dim ** 0.5 # Scale embeddings
        text_emb = self.positional_encoding(text_emb)

        # Apply Transformer Encoder Layers with gradient checkpointing
        x = text_emb
        for layer in self.transformer_encoder_layers:
            # Checkpoint each transformer block
            # If mask is None, then no padding mask is applied
            if mask is not None:
                x = checkpoint(layer, x, src_key_padding_mask=mask, use_reentrant=False)
            else:
                x = checkpoint(layer, x, use_reentrant=False)

        return x

    def _predict_durations(self, text_encoded: torch.Tensor) -> torch.Tensor:
        """
        Predicts log durations for each phoneme.
        Args:
            text_encoded: Output from text encoder (batch_size, text_seq_len, hidden_dim)
        Returns:
            Log durations of shape (batch_size, text_seq_len)
        """
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
        # Initialize mask as all True (padded) and set to False for actual content
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
        stop_token_targets: torch.Tensor, # Ground truth stop tokens (batch_size, mel_seq_len)
        text_padding_mask: Optional[torch.Tensor] = None # New: Mask for text encoder
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass during training (teacher forcing).

        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            mel_specs: Target mel spectrograms of shape (batch_size, mel_seq_len, mel_dim)
            phoneme_durations: Ground truth durations for each phoneme (batch_size, text_seq_len)
            stop_token_targets: Ground truth stop tokens (0 for all but last frame, which is 1)
                                of shape (batch_size, mel_seq_len)
            text_padding_mask: Optional boolean mask (batch_size, text_seq_len) for the
                                text encoder's padding. True for padded positions.

        Returns:
            - Predicted mel spectrograms of shape (batch_size, mel_seq_len, mel_dim)
            - Predicted log durations of shape (batch_size, text_seq_len) (for duration loss)
            - Predicted stop token logits of shape (batch_size, mel_seq_len) (for stop token loss)
        """
        batch_size = phoneme_indices.size(0)

        # Encode text using Transformer encoder
        # Pass text_padding_mask to the encoder
        text_encoded = self.encode_text(phoneme_indices, mask=text_padding_mask)

        # Predict durations
        predicted_log_durations = self._predict_durations(text_encoded)

        # Length regulate text_encoded using ground-truth durations
        text_encoded_expanded, attention_mask_text = self._length_regulate(
            text_encoded, phoneme_durations.float(), max_len=mel_specs.size(1)
        )

        outputs = []
        stop_token_outputs = [] # To store stop token predictions
        hidden = None # For LSTM decoder state

        for t in range(mel_specs.size(1)):
            # Current mel frame (teacher forcing)
            if t == 0:
                mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=mel_specs.device)
            else:
                mel_input = mel_specs[:, t-1:t, :]

            # Project mel to hidden dimension
            mel_projected = self.mel_projection_in(mel_input)
            mel_projected = self.dropout(mel_projected) # Using general dropout

            # Attention over expanded text with padding mask
            # Checkpoint the attention mechanism
            attended, _ = checkpoint(self.attention, mel_projected, text_encoded_expanded, text_encoded_expanded, 
                                     key_padding_mask=attention_mask_text, use_reentrant=False)

            # Decoder step (LSTM) - checkpoint the LSTM if hidden state is passed
            if hidden is None:
                decoder_input = torch.cat([mel_projected, attended], dim=2)
                decoder_out, hidden = self.decoder(decoder_input, hidden)
            else:
                # Pass hx and cx explicitly for checkpointing LSTM
                decoder_input = torch.cat([mel_projected, attended], dim=2)
                decoder_out, hidden = checkpoint(self.decoder, decoder_input, hidden, use_reentrant=False)

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

    def forward_inference(self, phoneme_indices: torch.Tensor, max_len: int = 800, stop_threshold: float = 0.5,
                          text_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass during inference (autoregressive).
        
        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            max_len: Maximum length for generated sequence
            stop_threshold: Probability threshold for stopping generation.
            text_padding_mask: Optional boolean mask (batch_size, text_seq_len) for the
                               text encoder's padding. True for padded positions.
            
        Returns:
            Generated mel spectrograms of shape (batch_size, generated_mel_seq_len, mel_dim)
        """
        if phoneme_indices.size(0) > 1:
            print("Warning: Inference with stop token is most reliable with batch_size=1.")

        batch_size = phoneme_indices.size(0)
        
        # Encode text (no checkpointing for inference)
        text_encoded = self.encode_text(phoneme_indices, mask=text_padding_mask)

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
        stop_token_targets: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None # New parameter
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass - automatically chooses training or inference mode.

        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            mel_specs: Target mel spectrograms for training mode (optional). If provided,
                       `phoneme_durations` and `stop_token_targets` must also be provided.
            phoneme_durations: Ground truth durations for training mode. Required if `mel_specs` is not None.
            stop_token_targets: Ground truth stop tokens for training mode. Required if `mel_specs` is not None.
            text_padding_mask: Optional boolean mask (batch_size, text_seq_len) for the
                               text encoder's padding. True for padded positions.

        Returns:
            - Mel spectrograms (predicted or generated) in inference mode.
            - Tuple (predicted mel, predicted log durations, predicted stop logits) in training mode.
        """
        if mel_specs is not None:
            if phoneme_durations is None or stop_token_targets is None:
                raise ValueError("phoneme_durations and stop_token_targets must be provided for training mode.")
            return self.forward_training(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets, text_padding_mask)
        else:
            return self.forward_inference(phoneme_indices, text_padding_mask=text_padding_mask)

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
