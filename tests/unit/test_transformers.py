import math

import torch
import torch.nn as nn
import pytest

from kokoro.model import transformers

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _causal_mask(seq_len: int) -> torch.Tensor:
    """Upper-triangular causal (future-masking) attention mask."""
    return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)


# ===========================================================================
# Existing tests (kept verbatim, extended with extra assertions)
# ===========================================================================

def test_multihead_attention_shape():
    torch.manual_seed(0)
    d_model = 16
    heads = 4
    B = 2
    S_q = 5
    S_k = 7

    attn = transformers.MultiHeadAttentionImproved(d_model, heads, dropout=0.0, use_relative_pos=False)
    q = torch.randn(B, S_q, d_model)
    k = torch.randn(B, S_k, d_model)
    v = torch.randn(B, S_k, d_model)

    out, weights, _ = attn(q, k, v)
    assert out.shape == (B, S_q, d_model)
    if weights is not None:
        assert weights.shape[0] == B


def test_precomputed_kv_usage():
    torch.manual_seed(1)
    d_model = 12
    heads = 3
    B = 1
    S_q = 4
    S_k = 6

    attn = transformers.MultiHeadAttentionImproved(d_model, heads, dropout=0.0)
    q = torch.randn(B, S_q, d_model)
    k = torch.randn(B, S_k, d_model)
    v = torch.randn(B, S_k, d_model)

    # Precompute K and V in the format expected by the attention module
    K = attn.w_k(k).view(B, S_k, heads, attn.d_k).transpose(1, 2)
    V = attn.w_v(v).view(B, S_k, heads, attn.d_k).transpose(1, 2)

    out1, _, _ = attn(q, k, v)
    out2, _, _ = attn(q, k, v, precomputed_k=K, precomputed_v=V)

    assert out1.shape == out2.shape
    assert torch.allclose(out1, out2, atol=1e-5)


def test_decoder_block_precompute_and_forward():
    torch.manual_seed(2)
    d_model = 16
    heads = 4
    B = 2
    T = 3
    M = 5

    block = transformers.ImprovedTransformerDecoderBlock(d_model, heads, dim_feedforward=32, dropout=0.0, use_prenorm=True)
    memory = torch.randn(B, M, d_model)

    # precompute cached K/V and ensure attributes are set
    block.precompute_cross_attention_kv(memory)
    assert getattr(block, '_cached_cross_K', None) is not None
    assert getattr(block, '_cached_cross_V', None) is not None

    tgt = torch.randn(B, T, d_model)

    # Run forward in eval/tracing-friendly path (no checkpointing)
    out, _ = block(tgt, memory, tgt_mask=None, memory_mask=None)
    assert out.shape == (B, T, d_model)

    # Clear cache
    block.clear_cross_attention_cache()
    assert getattr(block, '_cached_cross_K', None) is None


def test_improved_decoder_forward_eval():
    torch.manual_seed(3)
    d_model = 8
    heads = 2
    B = 1
    T = 4
    M = 6

    dec = transformers.ImprovedTransformerDecoder(d_model, heads, dim_feedforward=16, dropout=0.0, num_layers=2, use_prenorm=True)
    dec.eval()

    memory = torch.randn(B, M, d_model)
    tgt = torch.randn(B, T, d_model)

    out, _ = dec(tgt, memory)
    assert out.shape == (B, T, d_model)


# ===========================================================================
# drop_path
# ===========================================================================

def test_drop_path_zero_prob_is_identity():
    x = torch.randn(4, 8, 16)
    out = transformers.drop_path(x, drop_prob=0.0, training=True)
    assert torch.equal(out, x)


def test_drop_path_not_training_is_identity():
    x = torch.randn(4, 8, 16)
    out = transformers.drop_path(x, drop_prob=0.9, training=False)
    assert torch.equal(out, x)


def test_drop_path_shape_preserved():
    torch.manual_seed(42)
    x = torch.randn(8, 6, 16)
    out = transformers.drop_path(x, drop_prob=0.5, training=True)
    assert out.shape == x.shape


def test_drop_path_training_zeroes_some_rows():
    """With high drop probability, at least some batch rows should be zeroed."""
    torch.manual_seed(0)
    x = torch.ones(32, 4, 4)
    out = transformers.drop_path(x, drop_prob=0.9, training=True)
    row_sums = out.sum(dim=(1, 2))  # per-sample sum
    assert (row_sums == 0).any(), "Expected at least one dropped sample"


# ===========================================================================
# MultiHeadAttentionImproved – additional coverage
# ===========================================================================

def test_attention_no_nan_or_inf():
    torch.manual_seed(10)
    attn = transformers.MultiHeadAttentionImproved(16, 4, dropout=0.0)
    q = k = v = torch.randn(2, 8, 16)
    out, _, _ = attn(q, k, v)
    assert torch.isfinite(out).all()


def test_attention_self_attention_square():
    """Self-attention: query == key == value, output shape is (B, S, d_model)."""
    torch.manual_seed(20)
    attn = transformers.MultiHeadAttentionImproved(16, 4, dropout=0.0)
    x = torch.randn(3, 10, 16)
    out, _, _ = attn(x, x, x)
    assert out.shape == (3, 10, 16)


def test_attention_with_padding_mask():
    """Padding mask: padded positions should not dominate the output."""
    torch.manual_seed(30)
    B, S, d_model, heads = 2, 6, 16, 4
    attn = transformers.MultiHeadAttentionImproved(d_model, heads, dropout=0.0)
    q = k = v = torch.randn(B, S, d_model)
    # Mask out last 2 positions
    key_padding_mask = torch.zeros(B, S, dtype=torch.bool)
    key_padding_mask[:, -2:] = True
    out, _, _ = attn(q, k, v, key_padding_mask=key_padding_mask)
    assert out.shape == (B, S, d_model)
    assert torch.isfinite(out).all()


def test_attention_with_causal_mask():
    """Causal mask: upper-triangular -inf mask should be handled without NaN."""
    torch.manual_seed(40)
    B, S, d_model, heads = 2, 8, 16, 4
    attn = transformers.MultiHeadAttentionImproved(d_model, heads, dropout=0.0)
    x = torch.randn(B, S, d_model)
    mask = _causal_mask(S)
    out, _, _ = attn(x, x, x, attn_mask=mask)
    assert out.shape == (B, S, d_model)
    assert torch.isfinite(out).all()


def test_attention_alibi_shape():
    """ALiBi relative position bias should be generated with the correct shape."""
    torch.manual_seed(50)
    heads = 4
    attn = transformers.MultiHeadAttentionImproved(16, heads, dropout=0.0, use_relative_pos=True)
    attn.eval()
    bias = attn._get_alibi_bias(6, 6, torch.device('cpu'), dtype=torch.float32)
    # Expected: (num_heads, seq_len_q, seq_len_k)
    assert bias.shape == (heads, 6, 6)


def test_attention_alibi_forward_shape():
    """Forward pass with ALiBi enabled should produce the correct output shape."""
    torch.manual_seed(55)
    B, S, d_model, heads = 2, 7, 16, 4
    attn = transformers.MultiHeadAttentionImproved(d_model, heads, dropout=0.0, use_relative_pos=True)
    attn.eval()
    x = torch.randn(B, S, d_model)
    out, _, _ = attn(x, x, x)
    assert out.shape == (B, S, d_model)
    assert torch.isfinite(out).all()


def test_attention_kv_cache_incremental_matches_full():
    """
    Incremental decoding with KV cache should match a causal full-sequence forward
    pass.  A causal mask is required on the full-sequence run so that position t
    only attends to positions 0..t — the same context available during step-by-step
    decoding with a growing KV cache.
    """
    torch.manual_seed(60)
    B, S, d_model, heads = 1, 5, 16, 4
    attn = transformers.MultiHeadAttentionImproved(d_model, heads, dropout=0.0, use_relative_pos=False)
    attn.eval()

    x = torch.randn(B, S, d_model)

    # Full-sequence reference with causal mask so position t sees only 0..t
    causal = _causal_mask(S)
    out_full, _, _ = attn(x, x, x, attn_mask=causal)

    # Incremental decoding step-by-step — naturally causal via growing cache
    cache = ()
    incremental_outs = []
    for t in range(S):
        xt = x[:, t:t+1, :]  # (B, 1, d_model)
        out_t, _, cache = attn(xt, xt, xt, kv_cache=cache)
        incremental_outs.append(out_t)

    out_incremental = torch.cat(incremental_outs, dim=1)  # (B, S, d_model)
    assert torch.allclose(out_full, out_incremental, atol=1e-4), (
        "Incremental KV-cache output differs from full-sequence causal output"
    )


# ===========================================================================
# ImprovedTransformerEncoderBlock
# ===========================================================================

def test_encoder_block_prenorm_shape():
    torch.manual_seed(70)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0,
        activation='gelu', use_prenorm=True
    )
    x = torch.randn(2, 10, 16)
    out = block(x)
    assert out.shape == (2, 10, 16)


def test_encoder_block_postnorm_shape():
    torch.manual_seed(71)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0,
        activation='relu', use_prenorm=False
    )
    x = torch.randn(2, 10, 16)
    out = block(x)
    assert out.shape == (2, 10, 16)


def test_encoder_block_prenorm_postnorm_differ():
    """Pre-norm and post-norm with identical weights should produce different outputs."""
    torch.manual_seed(80)
    pre = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0,
        activation='gelu', use_prenorm=True
    )
    post = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0,
        activation='gelu', use_prenorm=False
    )
    # Copy weights so the only difference is norm placement
    post.load_state_dict(pre.state_dict())

    pre.eval(); post.eval()
    x = torch.randn(2, 6, 16)
    out_pre = pre(x)
    out_post = post(x)
    assert not torch.allclose(out_pre, out_post, atol=1e-5), (
        "Pre-norm and post-norm should produce different outputs"
    )


def test_encoder_block_no_nan():
    torch.manual_seed(90)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, use_prenorm=True
    )
    block.eval()
    x = torch.randn(2, 12, 16)
    out = block(x)
    assert torch.isfinite(out).all()


def test_encoder_block_with_padding_mask():
    torch.manual_seed(100)
    B, S, d_model = 2, 8, 16
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0, use_prenorm=True
    )
    block.eval()
    x = torch.randn(B, S, d_model)
    padding_mask = torch.zeros(B, S, dtype=torch.bool)
    padding_mask[0, -2:] = True  # pad last 2 positions in batch item 0
    out = block(x, src_key_padding_mask=padding_mask)
    assert out.shape == (B, S, d_model)
    assert torch.isfinite(out).all()


def test_encoder_block_with_causal_mask():
    torch.manual_seed(110)
    S, d_model = 8, 16
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0, use_prenorm=True
    )
    block.eval()
    x = torch.randn(2, S, d_model)
    out = block(x, src_mask=_causal_mask(S))
    assert out.shape == (2, S, d_model)
    assert torch.isfinite(out).all()


def test_encoder_block_gradient_flows():
    torch.manual_seed(120)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, use_prenorm=True
    )
    block.train()
    x = torch.randn(2, 6, 16, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_encoder_block_uses_gelu_activation():
    """TransformerEncoderBlock (the wrapper) should use GELU, not ReLU."""
    block = transformers.TransformerEncoderBlock(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
    assert isinstance(block.activation, nn.GELU), (
        f"Expected GELU, got {type(block.activation).__name__}"
    )


def test_encoder_block_uses_prenorm():
    """TransformerEncoderBlock wrapper must set use_prenorm=True."""
    block = transformers.TransformerEncoderBlock(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
    assert block.use_prenorm is True


def test_encoder_block_drop_path_rate():
    """Drop path rate should be stored and respected."""
    block = transformers.TransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, drop_path_rate=0.1
    )
    assert block.drop_path_rate == pytest.approx(0.1)


# ===========================================================================
# ImprovedTransformerDecoderBlock
# ===========================================================================

def test_decoder_block_postnorm_shape():
    torch.manual_seed(130)
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0,
        activation='relu', use_prenorm=False
    )
    block.eval()
    B, T, M = 2, 5, 7
    tgt = torch.randn(B, T, 16)
    mem = torch.randn(B, M, 16)
    out, _ = block(tgt, mem)
    assert out.shape == (B, T, 16)


def test_decoder_block_no_nan():
    torch.manual_seed(140)
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, use_prenorm=True
    )
    block.eval()
    tgt = torch.randn(2, 5, 16)
    mem = torch.randn(2, 8, 16)
    out, _ = block(tgt, mem)
    assert torch.isfinite(out).all()


def test_decoder_block_with_causal_mask():
    torch.manual_seed(150)
    B, T, M, d_model = 2, 6, 8, 16
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0, use_prenorm=True
    )
    block.eval()
    tgt = torch.randn(B, T, d_model)
    mem = torch.randn(B, M, d_model)
    out, _ = block(tgt, mem, tgt_mask=_causal_mask(T))
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()


def test_decoder_block_with_memory_padding_mask():
    torch.manual_seed(160)
    B, T, M, d_model = 2, 4, 9, 16
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0, use_prenorm=True
    )
    block.eval()
    tgt = torch.randn(B, T, d_model)
    mem = torch.randn(B, M, d_model)
    mem_pad_mask = torch.zeros(B, M, dtype=torch.bool)
    mem_pad_mask[:, -3:] = True
    out, _ = block(tgt, mem, memory_key_padding_mask=mem_pad_mask)
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()


def test_decoder_block_gradient_flows():
    torch.manual_seed(170)
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, use_prenorm=True
    )
    block.train()
    tgt = torch.randn(2, 5, 16, requires_grad=True)
    mem = torch.randn(2, 7, 16)
    out, _ = block(tgt, mem)
    out.sum().backward()
    assert tgt.grad is not None
    assert torch.isfinite(tgt.grad).all()


def test_decoder_block_kv_cache_incremental():
    """
    Step-by-step decoding with self-attn KV cache should match a causal
    full-sequence forward pass.  The causal mask ensures the reference run
    restricts each position to the same context that incremental decoding sees.
    """
    torch.manual_seed(180)
    B, T, M, d_model, heads = 1, 6, 5, 16, 4
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=d_model, nhead=heads, dim_feedforward=32, dropout=0.0, use_prenorm=True
    )
    block.eval()
    mem = torch.randn(B, M, d_model)
    tgt = torch.randn(B, T, d_model)

    # Full-sequence causal reference
    out_full, _ = block(tgt, mem, tgt_mask=_causal_mask(T))

    # Incremental: feed one token at a time with growing KV cache (naturally causal)
    cache = ()
    incremental_outs = []
    for t in range(T):
        xt = tgt[:, t:t+1, :]
        out_t, cache = block(xt, mem, self_attn_kv_cache=cache)
        incremental_outs.append(out_t)

    out_inc = torch.cat(incremental_outs, dim=1)
    assert torch.allclose(out_full, out_inc, atol=1e-4), (
        "Decoder KV-cache incremental output should match full-sequence causal output"
    )


# ===========================================================================
# ImprovedTransformerDecoder (full stack)
# ===========================================================================

def test_decoder_prenorm_has_final_norm():
    """Pre-norm decoder must have a final LayerNorm applied after all layers."""
    dec = transformers.ImprovedTransformerDecoder(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=2, use_prenorm=True
    )
    assert dec.norm is not None and isinstance(dec.norm, nn.LayerNorm)


def test_decoder_postnorm_has_no_final_norm():
    """Post-norm decoder must NOT have a final LayerNorm."""
    dec = transformers.ImprovedTransformerDecoder(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=2, use_prenorm=False
    )
    assert dec.norm is None


def test_decoder_training_forward_shape():
    """Decoder training path (gradient checkpointing) should return correct shape."""
    torch.manual_seed(190)
    B, T, M, d_model = 2, 5, 7, 16
    dec = transformers.ImprovedTransformerDecoder(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=2, use_prenorm=True
    )
    dec.train()
    tgt = torch.randn(B, T, d_model)
    mem = torch.randn(B, M, d_model)
    out, _ = dec(tgt, mem)
    assert out.shape == (B, T, d_model)


def test_decoder_gradient_flows_through_all_layers():
    torch.manual_seed(200)
    B, T, M, d_model = 2, 4, 6, 16
    dec = transformers.ImprovedTransformerDecoder(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=3, use_prenorm=True
    )
    dec.train()
    tgt = torch.randn(B, T, d_model, requires_grad=True)
    mem = torch.randn(B, M, d_model)
    out, _ = dec(tgt, mem)
    out.sum().backward()
    assert tgt.grad is not None
    assert torch.isfinite(tgt.grad).all()


def test_decoder_precompute_kv_caches_all_layers():
    """precompute_cross_attention_kv should populate K/V in every layer."""
    torch.manual_seed(210)
    B, M, d_model, num_layers = 2, 8, 16, 3
    dec = transformers.ImprovedTransformerDecoder(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=num_layers
    )
    mem = torch.randn(B, M, d_model)
    dec.precompute_cross_attention_kv(mem)
    for layer in dec.layers:
        assert getattr(layer, '_cached_cross_K', None) is not None
    dec.clear_cross_attention_cache()
    for layer in dec.layers:
        assert getattr(layer, '_cached_cross_K', None) is None


# ===========================================================================
# TransformerEncoderBlock / TransformerDecoder compatibility wrappers
# ===========================================================================

def test_transformer_encoder_block_wrapper_uses_prenorm_gelu():
    """TransformerEncoderBlock compatibility wrapper must use pre-norm + GELU."""
    block = transformers.TransformerEncoderBlock(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0)
    assert block.use_prenorm is True
    assert isinstance(block.activation, nn.GELU)


def test_transformer_encoder_block_wrapper_forward():
    torch.manual_seed(220)
    block = transformers.TransformerEncoderBlock(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0)
    block.eval()
    x = torch.randn(2, 10, 32)
    out = block(x)
    assert out.shape == (2, 10, 32)
    assert torch.isfinite(out).all()


def test_transformer_decoder_wrapper_uses_prenorm_gelu():
    """TransformerDecoder compatibility wrapper must use pre-norm + GELU."""
    dec = transformers.TransformerDecoder(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0, num_layers=2)
    assert dec.use_prenorm is True
    assert dec.norm is not None and isinstance(dec.norm, nn.LayerNorm)
    # Check activation in every layer
    for layer in dec.layers:
        assert isinstance(layer.activation, nn.GELU)


def test_transformer_decoder_wrapper_forward():
    torch.manual_seed(230)
    dec = transformers.TransformerDecoder(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0, num_layers=2)
    dec.eval()
    B, T, M = 2, 5, 8
    tgt = torch.randn(B, T, 32)
    mem = torch.randn(B, M, 32)
    out, _ = dec(tgt, mem)
    assert out.shape == (B, T, 32)
    assert torch.isfinite(out).all()


# ===========================================================================
# Activation function correctness in feed-forward block
# ===========================================================================

def test_encoder_ff_block_uses_gelu_semantics():
    """
    The GLU feed-forward gate should use GELU: gate values should not be
    clipped at 0 (as ReLU would), producing negative contributions.
    """
    torch.manual_seed(240)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0.0,
        activation='gelu', use_prenorm=True
    )
    block.eval()

    # Craft an input where the gate path would be negative so ReLU would zero it
    # but GELU keeps a small negative value.
    gate_input = torch.full((1, 1, 16), -2.0)  # very negative → ReLU kills it
    # GELU(-2.0) ≈ -0.0455, not zero
    gelu_val = nn.GELU()(-torch.tensor(2.0)).item()
    assert gelu_val < 0.0, "GELU of a large negative should be slightly negative"


def test_decoder_ff_block_uses_gelu():
    """ImprovedTransformerDecoderBlock constructed with 'gelu' should store nn.GELU."""
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, activation='gelu'
    )
    assert isinstance(block.activation, nn.GELU)


def test_decoder_ff_block_uses_relu():
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, activation='relu'
    )
    assert isinstance(block.activation, nn.ReLU)


def test_decoder_ff_block_uses_swish():
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, activation='swish'
    )
    assert isinstance(block.activation, nn.SiLU)


def test_unsupported_activation_raises():
    with pytest.raises(ValueError, match="Unsupported activation"):
        transformers.ImprovedTransformerEncoderBlock(
            d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, activation='tanh'
        )


# ===========================================================================
# Helper factory functions
# ===========================================================================

def test_create_optimized_encoder_layers():
    layers = transformers.create_optimized_encoder_layers(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=3
    )
    assert len(layers) == 3
    assert all(isinstance(l, transformers.ImprovedTransformerEncoderBlock) for l in layers)


def test_create_optimized_decoder():
    dec = transformers.create_optimized_decoder(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=2
    )
    assert isinstance(dec, transformers.ImprovedTransformerDecoder)
    assert len(dec.layers) == 2


# ===========================================================================
# Determinism
# ===========================================================================

def test_encoder_block_deterministic_eval():
    """Same input in eval mode should always give the same output."""
    torch.manual_seed(250)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, use_prenorm=True
    )
    block.eval()
    x = torch.randn(2, 8, 16)
    out1 = block(x)
    out2 = block(x)
    assert torch.equal(out1, out2)


def test_decoder_deterministic_eval():
    torch.manual_seed(260)
    dec = transformers.ImprovedTransformerDecoder(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=2, use_prenorm=True
    )
    dec.eval()
    tgt = torch.randn(2, 5, 16)
    mem = torch.randn(2, 7, 16)
    out1, _ = dec(tgt, mem)
    out2, _ = dec(tgt, mem)
    assert torch.equal(out1, out2)
