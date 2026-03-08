import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4000):
        """
        Initializes the PositionalEncoding layer.

        Args:
            d_model: The embedding dimension.
            dropout: The dropout rate.
            max_len: The maximum length of sequences this positional encoding will support.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-torch.log(torch.tensor(10000.0)) / d_model))

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer
        # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, seq_offset: int = 0) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Input tensor (batch_size, seq_len, d_model).
            seq_offset: An integer offset to apply to the positional encoding.

        Returns:
            Tensor with positional encoding added (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, d_model = x.shape

        # Validate d_model matches
        if d_model != self.d_model:
            raise ValueError(f"Input d_model ({d_model}) doesn't match PE d_model ({self.d_model})")

        # Check bounds
        if seq_offset + seq_len > self.pe.size(1):
            logger.warning(
                f"Positional encoding max_len ({self.pe.size(1)}) is too small. "
                f"Requested slice (offset {seq_offset} + length {seq_len}) = {seq_offset + seq_len}. "
                f"Consider increasing max_len during initialization."
            )
            # Extend PE if needed
            self._extend_pe(seq_offset + seq_len)

        # Extract the positional encoding slice
        # pe shape: (1, max_len, d_model)
        # pe_slice shape: (1, seq_len, d_model)
        pe_slice = self.pe[:, seq_offset:seq_offset + seq_len, :]

        # Add positional encoding (broadcasting handles batch dimension)
        # x: (batch_size, seq_len, d_model)
        # pe_slice: (1, seq_len, d_model)
        # Result: (batch_size, seq_len, d_model)
        x = x + pe_slice

        return self.dropout(x)

    def _extend_pe(self, new_max_len: int):
        """Extend positional encoding if needed."""
        if new_max_len <= self.pe.size(1):
            return

        # Create extended PE
        old_max_len = self.pe.size(1)
        position = torch.arange(new_max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                            (-torch.log(torch.tensor(10000.0)) / self.d_model))

        pe_extended = torch.zeros(new_max_len, self.d_model, device=self.pe.device)
        pe_extended[:, 0::2] = torch.sin(position * div_term.to(self.pe.device))
        pe_extended[:, 1::2] = torch.cos(position * div_term.to(self.pe.device))

        # Update the buffer
        self.pe = pe_extended.unsqueeze(0)
        logger.info(f"Extended positional encoding from {old_max_len} to {new_max_len}")


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) — MPS-compatible relative positional encoding.

    RoPE encodes relative position information by rotating query and key vectors
    using a position-dependent rotation matrix.  Because the rotation is applied
    directly to Q and K in their native dtype, RoPE avoids all mixed-dtype
    arithmetic and is therefore safe on Apple MPS (where ALiBi's float16 matrix
    operations are buggy).

    The dot-product Q·Kᵀ automatically captures the *relative* displacement
    between positions, giving the model genuine relative-position awareness
    without adding any extra bias tensors to the attention logits.

    Reference:
        "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        Su et al. (2021) – https://arxiv.org/abs/2104.09864

    Args:
        head_dim:    Dimension of each attention head.  Must be even.
        max_seq_len: Pre-computed table length; extended automatically on demand.
        base:        Frequency base (10000 by default, as in the original paper).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE, got {head_dim}"
            )
        self.head_dim = head_dim
        self.base = base

        # θ_i = 1 / (base^(2i / head_dim)),  i ∈ [0, head_dim/2)
        theta = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )  # (head_dim // 2,)
        self.register_buffer('theta', theta, persistent=False)

        # Pre-build the cos/sin look-up tables
        self._build_cache(max_seq_len)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute and register cos/sin tables up to *seq_len* positions."""
        positions = torch.arange(
            seq_len, device=self.theta.device, dtype=self.theta.dtype
        )
        freqs = torch.outer(positions, self.theta)  # (seq_len, head_dim // 2)
        # Duplicate each frequency so the table covers the full head_dim
        emb = torch.cat([freqs, freqs], dim=-1)     # (seq_len, head_dim)
        self.register_buffer('cos_cache', emb.cos(), persistent=False)
        self.register_buffer('sin_cache', emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Return the 'rotate-half' variant: for x = [x1 | x2] → [-x2 | x1]."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_offset: int = 0,
        k_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q:        Query  ``(batch, heads, seq_len_q, head_dim)``
            k:        Key    ``(batch, heads, seq_len_k, head_dim)``
            q_offset: Starting position index for queries (non-zero during
                      autoregressive decoding when the KV-cache already holds
                      previous tokens).
            k_offset: Starting position index for keys (usually 0).

        Returns:
            ``(q_rot, k_rot)`` — rotated tensors with the same shapes and dtype
            as the inputs.  MPS-safe: all arithmetic stays in the input dtype.
        """
        seq_len_q = q.shape[2]
        seq_len_k = k.shape[2]
        max_pos = max(q_offset + seq_len_q, k_offset + seq_len_k)

        # Lazily extend the look-up table when a longer sequence arrives
        if max_pos > self.cos_cache.shape[0]:
            self._build_cache(max_pos * 2)
            logger.debug(
                "RoPE cache extended to %d positions", self.cos_cache.shape[0]
            )

        # Slice and cast to the input dtype — handles fp16/bf16/fp32 transparently
        cos_q = self.cos_cache[q_offset: q_offset + seq_len_q].to(q.dtype)  # (seq_q, D)
        sin_q = self.sin_cache[q_offset: q_offset + seq_len_q].to(q.dtype)
        cos_k = self.cos_cache[k_offset: k_offset + seq_len_k].to(k.dtype)  # (seq_k, D)
        sin_k = self.sin_cache[k_offset: k_offset + seq_len_k].to(k.dtype)

        # Broadcast over batch and head dims:  (1, 1, seq, head_dim)
        cos_q = cos_q.unsqueeze(0).unsqueeze(0)
        sin_q = sin_q.unsqueeze(0).unsqueeze(0)
        cos_k = cos_k.unsqueeze(0).unsqueeze(0)
        sin_k = sin_k.unsqueeze(0).unsqueeze(0)

        q_rot = q * cos_q + self._rotate_half(q) * sin_q
        k_rot = k * cos_k + self._rotate_half(k) * sin_k
        return q_rot, k_rot
