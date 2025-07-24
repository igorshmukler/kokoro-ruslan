import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the PositionalEncoding layer.

        Args:
            d_model: The embedding dimension.
            dropout: The dropout rate.
            max_len: The maximum length of sequences this positional encoding will support.
                     This value should be chosen to be greater than or equal to the
                     maximum expected sequence length during both training and inference.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model # Store d_model for potential future checks

        # Compute the positional encodings once in log space.
        position = torch.arange(max_len).unsqueeze(1)
        # Using 1e-4 instead of 10000.0 for compatibility with common Transformer PE implementations
        # (exp(-log(10000)/d_model) is equivalent to 1/((10000)^(1/d_model)))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(1e4)) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position.float() * div_term)
        pe[0, :, 1::2] = torch.cos(position.float() * div_term)

        # Register as a buffer: not a model parameter, but part of the state_dict
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, seq_offset: int = 0) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Input tensor (batch_size, seq_len, d_model).
            seq_offset: An integer offset to apply to the positional encoding.
                        Useful for autoregressive decoding where `x` is a single
                        token/frame, and `seq_offset` is its absolute position
                        in the sequence. For training (teacher forcing) it's usually 0.

        Returns:
            Tensor with positional encoding added (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)
        
        # Ensure PE is on the same device as the input tensor 'x'
        pe_on_device = self.pe.to(x.device)

        # Check if the requested slice goes beyond the pre-computed positional encoding length
        if seq_offset + seq_len > pe_on_device.size(1):
            logger.warning(
                f"Positional encoding `max_len` ({pe_on_device.size(1)}) is too small. "
                f"Requested slice (offset {seq_offset} + length {seq_len}) = {seq_offset + seq_len} "
                f"exceeds `max_len`. This may lead to incorrect positional embeddings or errors. "
                f"Consider increasing `max_len` during PositionalEncoding initialization."
            )
            # You might choose to clip `seq_len` or handle this more gracefully if you expect
            # sequences longer than `max_len` frequently in production, but for now, a warning is good.
            # If `max_len` is a hard limit for your model, this should ideally not be hit.

        # Slice `pe` based on actual sequence length and offset
        # Ensure slicing does not go out of bounds of pre-computed PE.
        # If seq_offset + seq_len exceeds pe_on_device.size(1), Python slicing will automatically
        # clip to the available length. However, the warning above is crucial for alerting the user.
        pe_slice = pe_on_device[:, seq_offset : seq_offset + seq_len]
        
        # Add positional encoding by broadcasting across the batch dimension
        # x: (B, L, D), pe_slice: (1, L, D) -> x + pe_slice becomes (B, L, D)
        x = x + pe_slice
        return self.dropout(x)
