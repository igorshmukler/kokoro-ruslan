import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)

# Cache size for relative position matrices
REL_POS_CACHE_MAX_SIZE = 10


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic Depth (Drop Path) per sample.
    Randomly drops entire residual branches during training.

    Args:
        x: Input tensor
        drop_prob: Probability of dropping the path
        training: Whether in training mode

    Returns:
        Output tensor with stochastic depth applied
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    # Work with diff shapes: (batch_size,) for batch dimension
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarize: 0 or 1

    # Scale by keep_prob to maintain expected value
    output = x.div(keep_prob) * random_tensor
    return output

class MultiHeadAttentionImproved(nn.Module):
    """Improved multi-head attention with better initialization and optional relative positioning"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 use_relative_pos: bool = False, max_relative_distance: int = 32):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of each head's key/query/value
        self.use_relative_pos = use_relative_pos

        # Linear projections for Query, Key, Value
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output linear projection
        self.w_o = nn.Linear(d_model, d_model)

        # ALiBi: Attention with Linear Biases (replaces Shaw et al. relative pos)
        # Each head gets a learned slope for distance-based bias
        if use_relative_pos:
            # Initialize slopes following ALiBi paper: geometric sequence
            # For 8 heads: [1/2^1, 1/2^2, ..., 1/2^8] = [0.5, 0.25, 0.125, ...]
            slopes = torch.tensor([2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)])
            self.register_buffer('alibi_slopes', slopes)

        self.dropout_attn = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # Cache for distance matrices to avoid recomputation
        self._distance_cache = {}

        # Better initialization for linear layers
        self._init_weights()

    @staticmethod
    def _tensor_layout_metrics(tensor: torch.Tensor) -> str:
        shape = tuple(tensor.shape)
        stride = tensor.stride()
        numel = tensor.numel()
        elem_size = tensor.element_size()
        est_bytes = numel * elem_size
        max_offset = 0
        for size, step in zip(shape, stride):
            if size > 0:
                max_offset += (size - 1) * abs(step)
        return (
            f"shape={shape} stride={stride} contiguous={tensor.is_contiguous()} "
            f"numel={numel:,} elem_size={elem_size} est_bytes={est_bytes:,} "
            f"max_linear_offset={max_offset:,} dtype={tensor.dtype} device={tensor.device}"
        )

    def _log_phase(self, context_prefix: str, phase: str, tensor: Optional[torch.Tensor] = None):
        if tensor is None:
            logger.info(f"{context_prefix}  [AttentionPhase] {phase}")
            return
        logger.info(f"{context_prefix}  [AttentionPhase] {phase}: {self._tensor_layout_metrics(tensor)}")

    def _register_debug_grad_hook(self, tensor: torch.Tensor, hook_name: str, context_prefix: str):
        if not tensor.requires_grad:
            return

        def _hook(grad: Optional[torch.Tensor]):
            if grad is None:
                logger.info(f"{context_prefix}  [AttentionBW] {hook_name}: grad=None")
                return
            logger.info(f"{context_prefix}  [AttentionBW] {hook_name}: {self._tensor_layout_metrics(grad)}")

        tensor.register_hook(_hook)

    def _init_weights(self):
        # Glorot (Xavier) uniform for weight matrices
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        if self.w_o.bias is not None:
            nn.init.zeros_(self.w_o.bias)

    def _get_alibi_bias(self, seq_len_q: int, seq_len_k: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Generate ALiBi (Attention with Linear Biases) for attention.
        Returns bias of shape (num_heads, seq_len_q, seq_len_k)
        """
        cache_key = (seq_len_q, seq_len_k, device, dtype)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        # Create distance matrix: relative position from each query to each key
        # Shape: (seq_len_q, seq_len_k)
        q_pos = torch.arange(seq_len_q, device=device, dtype=dtype).unsqueeze(1)  # (seq_len_q, 1)
        k_pos = torch.arange(seq_len_k, device=device, dtype=dtype).unsqueeze(0)  # (1, seq_len_k)
        distance = k_pos - q_pos  # (seq_len_q, seq_len_k), negative for past, positive for future

        # Apply per-head slopes: (num_heads, 1, 1) * (1, seq_len_q, seq_len_k)
        # Result: (num_heads, seq_len_q, seq_len_k)
        # Convert slopes to correct dtype
        alibi_bias = self.alibi_slopes.to(dtype).view(-1, 1, 1) * distance.unsqueeze(0)

        # Cache for future use (limit cache size)
        if len(self._distance_cache) < REL_POS_CACHE_MAX_SIZE:
            self._distance_cache[cache_key] = alibi_bias

        return alibi_bias

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, # Causal mask for decoder self-attention (float('-inf'))
                key_padding_mask: Optional[torch.Tensor] = None # Padding mask (True for padded)
               ) -> Tuple[torch.Tensor, torch.Tensor]: # Return output and attention weights

        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        seq_len_v = value.size(1) # Should be same as seq_len_k
        is_batch_281 = hasattr(self, '_batch_281_log') and self._batch_281_log
        context_prefix = f"{getattr(self, '_crash_context', '')} " if hasattr(self, '_crash_context') else ""

        if is_batch_281:
            logger.info(
                f"{context_prefix}  [Attention] Entry: device={query.device}, dtype={query.dtype}, "
                f"query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)}"
            )
            logger.info(
                f"{context_prefix}    heads={self.num_heads}, d_k={self.d_k}, "
                f"attention_elements={batch_size * self.num_heads * seq_len_q * seq_len_k:,}"
            )
            self._log_phase(context_prefix, "entry_query", query)

        # 1. Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2) # (B, H, S_q, D_k)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)   # (B, H, S_k, D_k)
        V = self.w_v(value).view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S_v, D_k)

        if is_batch_281:
            logger.info(
                f"{context_prefix}    Q={tuple(Q.shape)} K={tuple(K.shape)} V={tuple(V.shape)} "
                f"(numel: Q={Q.numel():,}, K={K.numel():,}, V={V.numel():,})"
            )

        # 2. Prepare attention bias (ALiBi + masks) for Flash Attention
        attn_bias = None

        # Add ALiBi positional bias
        # CRITICAL: Disable ALiBi entirely on MPS due to fp16 dtype bugs in matrix multiplication
        # MPS backend has issues with mixed dtype operations that cause crashes
        use_alibi = self.use_relative_pos and query.device.type != 'mps'

        if use_alibi:
            # Get ALiBi bias: (num_heads, seq_len_q, seq_len_k)
            alibi_bias = self._get_alibi_bias(seq_len_q, seq_len_k, query.device, dtype=Q.dtype)
            # Expand to (B, H, S_q, S_k)
            attn_bias = alibi_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Apply causal mask
        if attn_mask is not None:
            # attn_mask: (S_q, S_k) with float('-inf') for masked positions
            mask_bias = attn_mask.unsqueeze(0).unsqueeze(0).to(dtype=Q.dtype, device=query.device)  # (1, 1, S_q, S_k)
            if attn_bias is None:
                attn_bias = mask_bias
            else:
                attn_bias = attn_bias + mask_bias

        # Apply padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, S_k) boolean (True for padded)
            key_padding_mask = key_padding_mask.to(torch.bool)
            padding_bias = torch.zeros(batch_size, 1, 1, seq_len_k, device=query.device, dtype=Q.dtype)
            padding_bias = padding_bias.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            if attn_bias is None:
                attn_bias = padding_bias
            else:
                attn_bias = attn_bias + padding_bias

        # 3. Compute attention
        # MPS has severe memory constraints - use ultra-aggressive chunking for long sequences
        # Chunking reduces peak memory during backward pass by processing in smaller pieces
        attention_size = batch_size * self.num_heads * seq_len_q * seq_len_k

        if query.device.type == 'mps' and (seq_len_q > 600 or seq_len_k > 600):
            if is_batch_281:
                self._log_phase(context_prefix, "pre_attention_chunked")
            # Ultra-aggressive chunking for MPS to avoid INT_MAX overflow during backward
            # Batch 281 crash: 398M attention elements crashed with chunk_size=32 (8M elements/chunk)
            # MPS backend creates internal NDArray during backward that overflows INT_MAX
            # Solution: Use even smaller chunks - 16 creates ~4M element chunks
            if attention_size > 300_000_000:
                chunk_size = 16  # Ultra-aggressive for 300M+ elements (4M elements/chunk)
            elif attention_size > 150_000_000:
                chunk_size = 32  # Extra aggressive for 150M+ elements (8M elements/chunk)
            elif attention_size > 50_000_000:
                chunk_size = 64  # Very aggressive for 50M+ elements
            else:
                chunk_size = 128  # Standard aggressive chunking

            num_chunks = (seq_len_q + chunk_size - 1) // chunk_size
            context_chunks = []

            # BATCH 281 LOGGING - Show fix is active
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                elements_per_chunk = batch_size * self.num_heads * chunk_size * seq_len_k
                logger.info(f"{context_prefix}  [Attention] MPS Chunking activated (fix for INT_MAX bug)")
                logger.info(f"{context_prefix}    attention_size: {attention_size:,} elements, chunk_size: {chunk_size}, chunks: {num_chunks}")

            # Process chunks without accumulating attention weights to save memory
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, seq_len_q)

                Q_chunk = Q[:, :, start_idx:end_idx, :]  # (B, H, chunk, D_k)

                # Compute attention for this chunk
                scores_chunk = torch.matmul(Q_chunk, K.transpose(-2, -1)) / self.scale  # (B, H, chunk, S_k)
                if attn_bias is not None:
                    # Only slice bias along query dim if it's actually expanded to full seq_len
                    # Padding masks have shape [B, 1, 1, S_k] and should broadcast as-is
                    if attn_bias.shape[2] == seq_len_q:
                        # Full bias tensor - slice it
                        bias_chunk = attn_bias[:, :, start_idx:end_idx, :]  # (B, H, chunk, S_k)
                        scores_chunk = scores_chunk + bias_chunk
                    elif attn_bias.shape[2] == 1:
                        # Broadcast bias (e.g., padding mask) - use as-is
                        scores_chunk = scores_chunk + attn_bias
                    else:
                        # Unexpected shape - log and skip
                        logger.error(f"Unexpected attn_bias shape: {attn_bias.shape}, expected dim 2 to be 1 or {seq_len_q}")

                attn_weights_chunk = F.softmax(scores_chunk, dim=-1)
                attn_weights_chunk = self.dropout_attn(attn_weights_chunk)
                context_chunk = torch.matmul(attn_weights_chunk, V)  # (B, H, chunk, D_k)

                context_chunks.append(context_chunk)

                # Clear intermediate tensors and free MPS cache after each chunk
                del scores_chunk, attn_weights_chunk, context_chunk
                if chunk_idx % 4 == 0:  # Periodic cache clearing
                    torch.mps.empty_cache()

            context = torch.cat(context_chunks, dim=2)  # (B, H, S_q, D_k)

            # BATCH 281 LOGGING - Confirm fix worked
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"{context_prefix}  [Attention] Chunked processing complete: {context.shape}")

            attn_weights = None  # Don't compute weights during chunked attention to save memory

        elif query.device.type == 'mps':
            if is_batch_281:
                self._log_phase(context_prefix, "pre_attention_dense")
            # Manual attention implementation for MPS (normal sequences)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            if attn_bias is not None:
                scores = scores + attn_bias

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout_attn(attn_weights)
            context = torch.matmul(attn_weights, V)
        else:
            # Use Flash Attention 2 on CUDA/CPU (2-4x faster)
            try:
                context = F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=attn_bias,
                    dropout_p=self.dropout_attn.p if self.training else 0.0,
                    is_causal=False  # We handle causality through attn_bias
                )
                # Flash Attention doesn't return weights, only compute during eval for visualization
                if not self.training:
                    with torch.no_grad():
                        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
                        if attn_bias is not None:
                            scores = scores + attn_bias
                        attn_weights = F.softmax(scores, dim=-1)
                else:
                    attn_weights = None

            except (RuntimeError, AttributeError) as e:
                # Fallback to manual implementation
                logger.warning(f"Flash Attention failed ({e}), using manual implementation")
                scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
                if attn_bias is not None:
                    scores = scores + attn_bias
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout_attn(attn_weights)
                context = torch.matmul(attn_weights, V)

        # 7. Concatenate heads and apply final linear layer
        # Transpose back (B, S_q, H, D_k) -> (B, S_q, D_model)
        context_t = context.transpose(1, 2)

        if is_batch_281:
            self._log_phase(context_prefix, "pre_contiguous_view_context_t", context_t)
            self._register_debug_grad_hook(context_t, "bw_context_t", context_prefix)
            logger.info(
                f"{context_prefix}  [Attention] Pre-projection: context={tuple(context.shape)}, "
                f"transposed={tuple(context_t.shape)}, transposed_stride={context_t.stride()}, "
                f"transposed_contiguous={context_t.is_contiguous()}, numel={context_t.numel():,}"
            )

        try:
            context = context_t.contiguous().view(
                batch_size, seq_len_q, self.d_model
            )
        except RuntimeError as e:
            logger.error(
                f"{context_prefix}  [Attention] Failed at contiguous/view: context_t_shape={tuple(context_t.shape)}, "
                f"stride={context_t.stride()}, contiguous={context_t.is_contiguous()}, error={e}"
            )
            raise

        if is_batch_281:
            self._log_phase(context_prefix, "post_contiguous_view_context", context)
            self._register_debug_grad_hook(context, "bw_context_post_view", context_prefix)
            logger.info(
                f"{context_prefix}  [Attention] Post-view: context={tuple(context.shape)}, "
                f"contiguous={context.is_contiguous()}, stride={context.stride()}"
            )

        if is_batch_281:
            self._log_phase(context_prefix, "pre_w_o", context)

        output = self.w_o(context)

        if is_batch_281:
            self._register_debug_grad_hook(output, "bw_output", context_prefix)
            self._log_phase(context_prefix, "post_w_o", output)
            logger.info(f"{context_prefix}  [Attention] Output projection complete: output={tuple(output.shape)}")

        # Return output and attention weights (None during training with Flash Attention)
        return output, attn_weights.mean(dim=1) if attn_weights is not None else None


class ImprovedTransformerEncoderBlock(nn.Module):
    """Enhanced Transformer encoder block with better normalization and activations"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float,
                 activation: str = 'gelu', use_prenorm: bool = True,
                 use_relative_pos: bool = False, drop_path_rate: float = 0.0):
        super().__init__()
        self.use_prenorm = use_prenorm
        self.drop_path_rate = drop_path_rate

        # Self-attention module
        self.self_attn = MultiHeadAttentionImproved(
            d_model, nhead, dropout, use_relative_pos
        )

        # Activation function for feed-forward network
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish': # SiLU is Swish
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # GLU-style feedforward (Gated Linear Unit)
        # Input to linear1 is d_model, output is dim_feedforward * 2 (for gate and linear paths)
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers (LayerNorm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout) # After self-attention residual
        self.dropout2 = nn.Dropout(dropout) # After feed-forward residual
        self.dropout_ff = nn.Dropout(dropout) # Inside feed-forward

        # Initialize weights for linear layers
        self._init_weights()

    def _init_weights(self):
        # Initialize only weight here, bias is usually initialized to zeros by default
        nn.init.xavier_uniform_(self.linear1.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """GLU-style feed-forward block"""
        x = self.linear1(x)
        # Split into two parts: one for gating, one for linear transformation
        gate, linear = x.chunk(2, dim=-1)
        # Apply activation to gate and multiply with linear path
        gated_output = self.activation(gate) * linear
        # Apply dropout and final linear transformation
        return self.linear2(self.dropout_ff(gated_output))

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, # Self-attention mask (e.g., causal)
                src_key_padding_mask: Optional[torch.Tensor] = None # Padding mask for src (True for padded)
               ) -> torch.Tensor:

        # Ensure src_key_padding_mask is boolean at this level
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(torch.bool)

        if self.use_prenorm:
            # Pre-normalization (e.g., Transformer-XL, GPT-2)
            # Self-attention sub-layer
            src_norm = self.norm1(src) # Apply LayerNorm BEFORE attention
            # MultiHeadAttentionImproved returns (output, attn_weights)
            attn_output, _ = self.self_attn(src_norm, src_norm, src_norm,
                                            attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)

            # BATCH 281 LOGGING
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"  [EncoderBlock] After self_attn: {attn_output.shape}")

            # Apply stochastic depth to attention output
            attn_output = drop_path(attn_output, self.drop_path_rate, self.training)
            src = src + self.dropout1(attn_output) # Residual connection + Dropout

            # BATCH 281 LOGGING
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"  [EncoderBlock] After residual: {src.shape}, starting FFN...")

            # Feed-forward sub-layer
            src_norm = self.norm2(src) # Apply LayerNorm BEFORE FFN
            ff_output = self._ff_block(src_norm)

            # BATCH 281 LOGGING
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"  [EncoderBlock] After FFN: {ff_output.shape}")

            # Apply stochastic depth to FFN output
            ff_output = drop_path(ff_output, self.drop_path_rate, self.training)
            src = src + self.dropout2(ff_output) # Residual connection + Dropout

            # BATCH 281 LOGGING
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"  [EncoderBlock] Block complete: {src.shape}")
        else:
            # Post-normalization (Original Transformer)
            # Self-attention sub-layer
            attn_output, _ = self.self_attn(src, src, src,
                                            attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
            # Apply stochastic depth to attention output
            attn_output = drop_path(attn_output, self.drop_path_rate, self.training)
            src = self.norm1(src + self.dropout1(attn_output)) # Residual + Dropout + LayerNorm

            # Feed-forward sub-layer
            ff_output = self._ff_block(src)
            # Apply stochastic depth to FFN output
            ff_output = drop_path(ff_output, self.drop_path_rate, self.training)
            src = self.norm2(src + self.dropout2(ff_output)) # Residual + Dropout + LayerNorm

        return src


class ImprovedTransformerDecoderBlock(nn.Module):
    """Enhanced Transformer decoder block with improved architecture"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float,
                 activation: str = 'gelu', use_prenorm: bool = True):
        super().__init__()
        self.use_prenorm = use_prenorm

        # Decoder self-attention: usually causal, can use relative positional encoding
        self.self_attn = MultiHeadAttentionImproved(d_model, nhead, dropout, use_relative_pos=True)

        # Cross-attention (encoder-decoder attention): no relative pos, queries from decoder, keys/values from encoder
        self.cross_attn = MultiHeadAttentionImproved(d_model, nhead, dropout, use_relative_pos=False)

        # Activation function for feed-forward network
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # GLU-style feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model) # For self-attention
        self.norm2 = nn.LayerNorm(d_model) # For cross-attention
        self.norm3 = nn.LayerNorm(d_model) # For feed-forward

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout) # After self-attention residual
        self.dropout2 = nn.Dropout(dropout) # After cross-attention residual
        self.dropout3 = nn.Dropout(dropout) # After feed-forward residual
        self.dropout_ff = nn.Dropout(dropout) # Inside feed-forward

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """GLU-style feed-forward block"""
        x = self.linear1(x)
        gate, linear = x.chunk(2, dim=-1)
        gated_output = self.activation(gate) * linear
        return self.linear2(self.dropout_ff(gated_output))

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, # memory is encoder output
                tgt_mask: Optional[torch.Tensor] = None, # Causal mask for decoder self-attention
                memory_mask: Optional[torch.Tensor] = None, # Not typically used for cross-attention
                tgt_key_padding_mask: Optional[torch.Tensor] = None, # Padding mask for decoder input
                memory_key_padding_mask: Optional[torch.Tensor] = None # Padding mask for encoder output
               ) -> torch.Tensor:

        # Ensure masks are boolean at this level
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(torch.bool)
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.to(torch.bool)

        context_prefix = f"{getattr(self, '_crash_context', '')} " if hasattr(self, '_crash_context') else ""

        # BATCH 281 LOGGING - Entry point
        if hasattr(self, '_batch_281_log') and self._batch_281_log:
            logger.info(f"{context_prefix}  [DecoderBlock] Entry: tgt={tgt.shape}, memory={memory.shape}")

        if self.use_prenorm:
            # Pre-normalization
            # Self-attention sub-layer
            tgt_norm = self.norm1(tgt)
            attn_output, _ = self.self_attn(tgt_norm, tgt_norm, tgt_norm,
                                            attn_mask=tgt_mask,
                                            key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(attn_output)

            # BATCH 281 LOGGING
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"{context_prefix}  [DecoderBlock] After self_attn: {tgt.shape}")

            # Cross-attention sub-layer
            tgt_norm = self.norm2(tgt)
            # Query is from decoder (tgt_norm), Key/Value from encoder (memory)
            cross_attn_output, _ = self.cross_attn(tgt_norm, memory, memory,
                                                   attn_mask=memory_mask, # Usually None
                                                   key_padding_mask=memory_key_padding_mask)

            # BATCH 281 LOGGING - Before residual
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"{context_prefix}  [DecoderBlock] Cross-attn returned: {cross_attn_output.shape}, applying dropout2...")

            dropped_output = self.dropout2(cross_attn_output)

            # BATCH 281 LOGGING - Before addition
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"{context_prefix}  [DecoderBlock] After dropout2, adding to tgt: {tgt.shape} + {dropped_output.shape}")

            tgt = tgt + dropped_output

            # BATCH 281 LOGGING
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"{context_prefix}  [DecoderBlock] After cross_attn residual: {tgt.shape}, starting FFN...")

            # Feed-forward sub-layer
            tgt_norm = self.norm3(tgt)
            ff_output = self._ff_block(tgt_norm)

            # BATCH 281 LOGGING
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"{context_prefix}  [DecoderBlock] After FFN: {ff_output.shape}")

            tgt = tgt + self.dropout3(ff_output)

            # BATCH 281 LOGGING
            if hasattr(self, '_batch_281_log') and self._batch_281_log:
                logger.info(f"{context_prefix}  [DecoderBlock] Block complete: {tgt.shape}")
        else:
            # Post-normalization
            # Self-attention sub-layer
            attn_output, _ = self.self_attn(tgt, tgt, tgt,
                                            attn_mask=tgt_mask,
                                            key_padding_mask=tgt_key_padding_mask)
            tgt = self.norm1(tgt + self.dropout1(attn_output))

            # Cross-attention sub-layer
            cross_attn_output, _ = self.cross_attn(tgt, memory, memory,
                                                   attn_mask=memory_mask, # Usually None
                                                   key_padding_mask=memory_key_padding_mask)
            tgt = self.norm2(tgt + self.dropout2(cross_attn_output))

            # Feed-forward sub-layer
            ff_output = self._ff_block(tgt)
            tgt = self.norm3(tgt + self.dropout3(ff_output))

        return tgt


class ImprovedTransformerDecoder(nn.Module):
    """Enhanced Transformer decoder with better layer organization"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, num_layers: int, use_prenorm: bool = True,
                 activation: str = 'gelu'):
        super().__init__()
        self.num_layers = num_layers
        self.use_prenorm = use_prenorm

        self.layers = nn.ModuleList([
            ImprovedTransformerDecoderBlock(
                d_model, nhead, dim_feedforward, dropout, activation, use_prenorm
            ) for _ in range(num_layers)
        ])

        # Final layer norm for pre-norm architecture (applied after all blocks)
        if use_prenorm:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = None # No final norm for post-norm architecture

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None, # Causal mask for decoder self-attention
                memory_key_padding_mask: Optional[torch.Tensor] = None, # Padding mask for encoder output
                tgt_key_padding_mask: Optional[torch.Tensor] = None # Padding mask for decoder input
               ) -> torch.Tensor:

        output = tgt

        for layer in self.layers:
            if self.training:
                # Use gradient checkpointing during training to save memory
                # Arguments to checkpoint must be positional and match the layer's forward signature
                # ImprovedTransformerDecoderBlock.forward takes:
                # tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask
                # Here, memory_mask is typically None for cross-attention.
                output = checkpoint(
                    layer, output, memory, tgt_mask, None,
                    tgt_key_padding_mask, memory_key_padding_mask,
                    use_reentrant=False
                )
            else:
                # Direct forward pass during inference
                output = layer(
                    output, memory, tgt_mask, None,
                    tgt_key_padding_mask, memory_key_padding_mask
                )

        # Apply final normalization if pre-norm architecture is used
        if self.norm is not None:
            output = self.norm(output)

        return output


# --- Compatibility Layer for Original Model (if needed) ---
# If your main model (KokoroModel) expects 'TransformerEncoderBlock' and 'TransformerDecoder'
# as directly defined classes (not the 'Improved' ones), these wrappers ensure compatibility.

class TransformerEncoderBlock(ImprovedTransformerEncoderBlock):
    """
    Backward compatibility wrapper for the original TransformerEncoderBlock.
    Defaults to original Transformer's post-norm, ReLU, no relative pos.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float,
                 drop_path_rate: float = 0.0):
        super().__init__(d_model, nhead, dim_feedforward, dropout,
                        activation='relu', use_prenorm=False, use_relative_pos=False,
                        drop_path_rate=drop_path_rate)


class TransformerDecoder(ImprovedTransformerDecoder):
    """
    Backward compatibility wrapper for the original TransformerDecoder.
    Defaults to original Transformer's post-norm, ReLU.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, num_layers: int):
        super().__init__(d_model, nhead, dim_feedforward, dropout, num_layers,
                        use_prenorm=False, activation='relu')


# --- Helper functions for optimized components (optional, for direct use) ---
# These can be used if you want to explicitly create the improved versions.

def create_optimized_encoder_layers(d_model: int, nhead: int, dim_feedforward: int,
                                   dropout: float, num_layers: int,
                                   use_prenorm: bool = True,
                                   activation: str = 'gelu',
                                   use_relative_pos: bool = False) -> nn.ModuleList:
    """Create a list of optimized encoder layers."""
    return nn.ModuleList([
        ImprovedTransformerEncoderBlock(
            d_model, nhead, dim_feedforward, dropout,
            activation, use_prenorm, use_relative_pos
        ) for _ in range(num_layers)
    ])


def create_optimized_decoder(d_model: int, nhead: int, dim_feedforward: int,
                           dropout: float, num_layers: int,
                           use_prenorm: bool = True,
                           activation: str = 'gelu') -> ImprovedTransformerDecoder:
    """Create an optimized transformer decoder."""
    return ImprovedTransformerDecoder(
        d_model, nhead, dim_feedforward, dropout, num_layers,
        use_prenorm, activation
    )
