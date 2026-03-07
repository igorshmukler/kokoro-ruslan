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

    block = transformers.ImprovedTransformerDecoderBlock(d_model, heads, dim_feedforward=32, dropout=0.0)
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

    dec = transformers.ImprovedTransformerDecoder(d_model, heads, dim_feedforward=16, dropout=0.0, num_layers=2)
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

def test_encoder_block_shape():
    torch.manual_seed(70)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0,
        activation='gelu'
    )
    x = torch.randn(2, 10, 16)
    out = block(x)
    assert out.shape == (2, 10, 16)


def test_encoder_block_has_glu_ff_submodule():
    """Encoder block must compose a GLUFeedForward submodule at self.ff."""
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0
    )
    assert isinstance(block.ff, transformers.GLUFeedForward), (
        f"Expected GLUFeedForward at block.ff, got {type(block.ff)}"
    )
    # Internal linear dimensions must match construction args
    assert block.ff.linear1.in_features == 16
    assert block.ff.linear1.out_features == 64   # dim_feedforward * 2
    assert block.ff.linear2.in_features == 32    # dim_feedforward
    assert block.ff.linear2.out_features == 16


def test_encoder_block_drop_path_changes_output_in_training():
    """With drop_path_rate > 0, stochastic depth must introduce run-to-run variation.

    We collect 20 forward passes and assert that at least two differ.  This is
    robust: it does not rely on a specific seed or all-ones input (which can
    produce identical all-pass results when every sample happens to be dropped).
    """
    d_model = 16
    torch.manual_seed(42)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0,
        drop_path_rate=0.5,  # 50 %: high enough for variation, low enough to pass residuals
    )
    block.train()
    x = torch.randn(8, 4, d_model)  # randn so residuals are non-trivial

    outputs = [block(x).detach() for _ in range(20)]
    any_different = any(not torch.equal(outputs[0], outputs[i]) for i in range(1, 20))
    assert any_different, (
        "drop_path_rate=0.5 in training mode should produce different outputs across forward passes"
    )


def test_encoder_block_no_nan():
    torch.manual_seed(90)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0
    )
    block.eval()
    x = torch.randn(2, 12, 16)
    out = block(x)
    assert torch.isfinite(out).all()


def test_encoder_block_with_padding_mask():
    torch.manual_seed(100)
    B, S, d_model = 2, 8, 16
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0
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
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0
    )
    block.eval()
    x = torch.randn(2, S, d_model)
    out = block(x, src_mask=_causal_mask(S))
    assert out.shape == (2, S, d_model)
    assert torch.isfinite(out).all()


def test_encoder_block_gradient_flows():
    torch.manual_seed(120)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0
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
    assert isinstance(block.ff.activation, nn.GELU), (
        f"Expected GELU, got {type(block.ff.activation).__name__}"
    )


def test_encoder_block_has_two_layer_norms():
    """Encoder block must have exactly norm1 and norm2 (pre-norm architecture)."""
    block = transformers.TransformerEncoderBlock(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)
    assert not hasattr(block, 'use_prenorm'), "use_prenorm attribute should have been removed"


def test_encoder_block_drop_path_rate():
    """Drop path rate should be stored and respected."""
    block = transformers.TransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, drop_path_rate=0.1
    )
    assert block.drop_path_rate == pytest.approx(0.1)


# ===========================================================================
# ImprovedTransformerDecoderBlock
# ===========================================================================

def test_decoder_block_has_glu_ff_submodule():
    """Decoder block must compose a GLUFeedForward submodule at self.ff."""
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0
    )
    assert isinstance(block.ff, transformers.GLUFeedForward), (
        f"Expected GLUFeedForward at block.ff, got {type(block.ff)}"
    )
    assert block.ff.linear1.in_features == 16
    assert block.ff.linear1.out_features == 64   # dim_feedforward * 2
    assert block.ff.linear2.in_features == 32    # dim_feedforward
    assert block.ff.linear2.out_features == 16
    assert not hasattr(block, 'use_prenorm'), "use_prenorm attribute should have been removed"


def test_decoder_block_no_nan():
    torch.manual_seed(140)
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0
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
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0
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
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0
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
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0
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
        d_model=d_model, nhead=heads, dim_feedforward=32, dropout=0.0
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

def test_decoder_always_has_final_norm():
    """Decoder must always carry a final LayerNorm (pre-norm architecture)."""
    dec = transformers.ImprovedTransformerDecoder(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=2
    )
    assert isinstance(dec.norm, nn.LayerNorm), (
        f"Expected nn.LayerNorm at dec.norm, got {type(dec.norm)}"
    )
    assert not hasattr(dec, 'use_prenorm'), "use_prenorm attribute should have been removed"


def test_decoder_final_norm_applied_in_forward():
    """The final LayerNorm must measurably affect the output (not be a no-op)."""
    torch.manual_seed(300)
    d_model = 16
    dec = transformers.ImprovedTransformerDecoder(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=1
    )
    dec.eval()
    tgt = torch.randn(1, 4, d_model)
    mem = torch.randn(1, 6, d_model)

    out_with_norm, _ = dec(tgt, mem)

    # Temporarily disable the final norm to confirm it has an effect
    original_norm = dec.norm
    dec.norm = nn.Identity()
    out_no_norm, _ = dec(tgt, mem)
    dec.norm = original_norm

    assert not torch.allclose(out_with_norm, out_no_norm, atol=1e-6), (
        "Final LayerNorm should change the output"
    )


def test_decoder_training_forward_shape():
    """Decoder training path (gradient checkpointing) should return correct shape."""
    torch.manual_seed(190)
    B, T, M, d_model = 2, 5, 7, 16
    dec = transformers.ImprovedTransformerDecoder(
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=2
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
        d_model=d_model, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=3
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

def test_transformer_encoder_block_wrapper_structure():
    """TransformerEncoderBlock compat wrapper: GELU via GLUFeedForward, two LayerNorms."""
    block = transformers.TransformerEncoderBlock(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0)
    assert isinstance(block.ff, transformers.GLUFeedForward)
    assert isinstance(block.ff.activation, nn.GELU)
    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)
    assert not hasattr(block, 'use_prenorm')


def test_transformer_encoder_block_wrapper_forward():
    torch.manual_seed(220)
    block = transformers.TransformerEncoderBlock(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0)
    block.eval()
    x = torch.randn(2, 10, 32)
    out = block(x)
    assert out.shape == (2, 10, 32)
    assert torch.isfinite(out).all()


def test_transformer_decoder_wrapper_structure():
    """TransformerDecoder compat wrapper: GELU via GLUFeedForward, final LayerNorm."""
    dec = transformers.TransformerDecoder(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0, num_layers=2)
    assert isinstance(dec.norm, nn.LayerNorm)
    assert not hasattr(dec, 'use_prenorm')
    for layer in dec.layers:
        assert isinstance(layer.ff, transformers.GLUFeedForward)
        assert isinstance(layer.ff.activation, nn.GELU)


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
        activation='gelu'
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
    assert isinstance(block.ff.activation, nn.GELU)


def test_decoder_ff_block_uses_relu():
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, activation='relu'
    )
    assert isinstance(block.ff.activation, nn.ReLU)


def test_decoder_ff_block_uses_swish():
    block = transformers.ImprovedTransformerDecoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, activation='swish'
    )
    assert isinstance(block.ff.activation, nn.SiLU)


def test_unsupported_activation_raises():
    with pytest.raises(ValueError, match="Unsupported activation"):
        transformers.ImprovedTransformerEncoderBlock(
            d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, activation='tanh'
        )


# ===========================================================================
# Determinism
# ===========================================================================

def test_encoder_block_deterministic_eval():
    """Same input in eval mode should always give the same output."""
    torch.manual_seed(250)
    block = transformers.ImprovedTransformerEncoderBlock(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0
    )
    block.eval()
    x = torch.randn(2, 8, 16)
    out1 = block(x)
    out2 = block(x)
    assert torch.equal(out1, out2)


def test_decoder_deterministic_eval():
    torch.manual_seed(260)
    dec = transformers.ImprovedTransformerDecoder(
        d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, num_layers=2
    )
    dec.eval()
    tgt = torch.randn(2, 5, 16)
    mem = torch.randn(2, 7, 16)
    out1, _ = dec(tgt, mem)
    out2, _ = dec(tgt, mem)
    assert torch.equal(out1, out2)


# ===========================================================================
# _build_activation
# ===========================================================================

class TestBuildActivation:
    def test_gelu_returns_gelu(self):
        act = transformers._build_activation('gelu')
        assert isinstance(act, nn.GELU)

    def test_gelu_case_insensitive(self):
        assert isinstance(transformers._build_activation('GELU'), nn.GELU)

    def test_relu_returns_relu(self):
        assert isinstance(transformers._build_activation('relu'), nn.ReLU)

    def test_swish_returns_silu(self):
        assert isinstance(transformers._build_activation('swish'), nn.SiLU)

    def test_silu_alias_returns_silu(self):
        assert isinstance(transformers._build_activation('silu'), nn.SiLU)

    def test_silu_case_insensitive(self):
        assert isinstance(transformers._build_activation('SiLU'), nn.SiLU)

    def test_unknown_activation_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            transformers._build_activation('tanh')

    def test_error_message_lists_valid_choices(self):
        """Error should name the valid options so the user knows what to use."""
        with pytest.raises(ValueError, match="gelu"):
            transformers._build_activation('bogus')

    def test_each_call_returns_fresh_module(self):
        """Factory should not share state — each call must return a new instance."""
        a = transformers._build_activation('gelu')
        b = transformers._build_activation('gelu')
        assert a is not b


# ===========================================================================
# GLUFeedForward
# ===========================================================================

class TestGLUFeedForward:
    def _make(self, d_model=16, dim_ff=32, dropout=0.0, activation='gelu'):
        return transformers.GLUFeedForward(d_model, dim_ff, dropout, activation)

    # --- structural ---

    def test_linear1_shape(self):
        ff = self._make(d_model=16, dim_ff=32)
        # linear1 must expand to 2× dim_ff so we can split gate and linear paths
        assert ff.linear1.in_features == 16
        assert ff.linear1.out_features == 64   # 32 * 2

    def test_linear2_shape(self):
        ff = self._make(d_model=16, dim_ff=32)
        assert ff.linear2.in_features == 32
        assert ff.linear2.out_features == 16

    def test_activation_set_correctly(self):
        for name, cls in [('gelu', nn.GELU), ('relu', nn.ReLU), ('swish', nn.SiLU)]:
            ff = self._make(activation=name)
            assert isinstance(ff.activation, cls), f"activation='{name}' should give {cls}"

    def test_xavier_init_linear1(self):
        """linear1 weights should be initialised with Xavier uniform (not zeros)."""
        torch.manual_seed(0)
        ff = self._make()
        # Xavier init produces values in roughly [-bound, +bound]; not all zero
        assert ff.linear1.weight.abs().max() > 0

    def test_bias_zeroed_linear1(self):
        ff = self._make()
        assert torch.all(ff.linear1.bias == 0)

    def test_bias_zeroed_linear2(self):
        ff = self._make()
        assert torch.all(ff.linear2.bias == 0)

    # --- forward ---

    def test_output_shape(self):
        torch.manual_seed(10)
        ff = self._make(d_model=16, dim_ff=32)
        ff.eval()
        x = torch.randn(2, 8, 16)
        out = ff(x)
        assert out.shape == (2, 8, 16)

    def test_output_finite(self):
        torch.manual_seed(20)
        ff = self._make()
        ff.eval()
        out = ff(torch.randn(3, 5, 16))
        assert torch.isfinite(out).all()

    def test_glu_gate_not_pure_relu(self):
        """GELU gate should pass negative values (unlike ReLU which clips at 0)."""
        ff = self._make(d_model=4, dim_ff=4, activation='gelu')
        ff.eval()
        # Zero out linear2 weights except one row so we can inspect the gate output
        with torch.no_grad():
            ff.linear1.weight.zero_()
            ff.linear1.bias.fill_(-2.0)   # large negative → gate ≈ GELU(-2) < 0
            ff.linear2.weight.fill_(1.0)
            ff.linear2.bias.zero_()
        x = torch.zeros(1, 1, 4)
        out = ff(x)
        # GELU(-2) ≈ -0.0455; ReLU(-2) = 0; so out should be non-zero
        assert out.abs().max() > 0, "GELU gate should produce non-zero output for negative inputs"

    def test_gradient_flows(self):
        torch.manual_seed(30)
        ff = self._make()
        ff.train()
        x = torch.randn(2, 4, 16, requires_grad=True)
        out = ff(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_dropout_applied_in_training(self):
        """With p=1.0 dropout the output should be all zeros in training mode."""
        ff = transformers.GLUFeedForward(d_model=8, dim_feedforward=16, dropout=1.0)
        ff.train()
        x = torch.randn(2, 4, 8)
        out = ff(x)
        assert torch.all(out == 0), "dropout=1.0 should produce all-zero output"

    def test_dropout_not_applied_in_eval(self):
        """In eval mode dropout is disabled; output must be non-zero for non-zero input."""
        ff = transformers.GLUFeedForward(d_model=8, dim_feedforward=16, dropout=1.0)
        ff.eval()
        x = torch.randn(2, 4, 8)
        out = ff(x)
        # With eval dropout=off the linear transformations on a random input are very
        # unlikely to produce an all-zero tensor.
        assert out.abs().max() > 0

    # --- composition: both blocks share the same class ---

    def test_encoder_and_decoder_use_same_glu_class(self):
        enc = transformers.ImprovedTransformerEncoderBlock(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
        dec = transformers.ImprovedTransformerDecoderBlock(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
        assert type(enc.ff) is type(dec.ff) is transformers.GLUFeedForward

    def test_no_legacy_activation_attribute_on_blocks(self):
        """After the refactor blocks must not expose a direct .activation attribute."""
        enc = transformers.ImprovedTransformerEncoderBlock(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
        dec = transformers.ImprovedTransformerDecoderBlock(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
        # activation lives inside self.ff, not directly on the block
        assert not hasattr(enc, 'activation'), "activation should be at enc.ff.activation, not enc.activation"
        assert not hasattr(dec, 'activation'), "activation should be at dec.ff.activation, not dec.activation"

    def test_no_legacy_linear_attributes_on_encoder_block(self):
        """After the refactor linear1/linear2/dropout_ff must not be on the encoder block."""
        block = transformers.ImprovedTransformerEncoderBlock(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
        for attr in ('linear1', 'linear2', 'dropout_ff'):
            assert not hasattr(block, attr), f"Encoder block should not have '{attr}' directly"

    def test_no_legacy_linear_attributes_on_decoder_block(self):
        """After the refactor linear1/linear2/dropout_ff must not be on the decoder block."""
        block = transformers.ImprovedTransformerDecoderBlock(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
        for attr in ('linear1', 'linear2', 'dropout_ff'):
            assert not hasattr(block, attr), f"Decoder block should not have '{attr}' directly"
