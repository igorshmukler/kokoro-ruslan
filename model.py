#!/usr/bin/env python3
"""
Kokoro Model Architecture
Simplified Kokoro-style Text-to-Speech model with attention mechanism
Optimized for MPS (Metal Performance Shaders) acceleration
"""

import torch
import torch.nn as nn
from typing import Optional


class KokoroModel(nn.Module):
    """
    Simplified Kokoro-style model architecture - optimized for MPS
    Text-to-Speech model with attention mechanism
    """

    def __init__(self, vocab_size: int, mel_dim: int = 80, hidden_dim: int = 512):
        """
        Initialize the Kokoro model.
        
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

        # Mel feature projection to match hidden dimension
        self.mel_projection_in = nn.Linear(mel_dim, hidden_dim)

        # Decoder
        self.decoder = nn.LSTM(
            hidden_dim + hidden_dim, hidden_dim, batch_first=True
        )

        # Attention mechanism
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

    def forward_training(self, phoneme_indices: torch.Tensor, mel_specs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training (teacher forcing).
        
        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            mel_specs: Target mel spectrograms of shape (batch_size, mel_seq_len, mel_dim)
            
        Returns:
            Predicted mel spectrograms of shape (batch_size, mel_seq_len, mel_dim)
        """
        batch_size = phoneme_indices.size(0)
        seq_len = mel_specs.size(1)
        
        # Encode text
        text_encoded = self.encode_text(phoneme_indices)
        
        outputs = []
        hidden = None
        
        for t in range(seq_len):
            # Current mel frame (teacher forcing)
            if t == 0:
                mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=mel_specs.device)
            else:
                mel_input = mel_specs[:, t-1:t, :]
            
            # Project mel to hidden dimension
            mel_projected = self.mel_projection_in(mel_input)
            mel_projected = self.dropout(mel_projected)
            
            # Attention over text
            attended, _ = self.attention(
                mel_projected,
                text_encoded,
                text_encoded
            )

            # Decoder step
            decoder_input = torch.cat([mel_projected, attended], dim=2)
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            decoder_out = self.dropout(decoder_out)
            
            # Project back to mel
            mel_pred = self.mel_projection_out(decoder_out)
            outputs.append(mel_pred)
        
        return torch.cat(outputs, dim=1)

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
        
        outputs = []
        hidden = None
        mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=phoneme_indices.device)
        
        for t in range(max_len):
            # Project mel to hidden dimension
            mel_projected = self.mel_projection_in(mel_input)
            
            # Attention over text
            attended, _ = self.attention(
                mel_projected,
                text_encoded,
                text_encoded
            )
            
            # Decoder step
            decoder_input = torch.cat([mel_projected, attended], dim=2)
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            
            # Project back to mel
            mel_pred = self.mel_projection_out(decoder_out)
            outputs.append(mel_pred)
            
            # Use prediction as next input (autoregressive)
            mel_input = mel_pred
        
        return torch.cat(outputs, dim=1)

    def forward(self, phoneme_indices: torch.Tensor, mel_specs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - automatically chooses training or inference mode.
        
        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            mel_specs: Target mel spectrograms for training mode (optional)
            
        Returns:
            Mel spectrograms (predicted or generated)
        """
        if mel_specs is not None:
            # Training mode with teacher forcing
            return self.forward_training(phoneme_indices, mel_specs)
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
