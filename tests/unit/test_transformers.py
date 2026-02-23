import torch
import pytest

from kokoro.model import transformers


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

    out, weights = attn(q, k, v)
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

    out1, _ = attn(q, k, v)
    out2, _ = attn(q, k, v, precomputed_k=K, precomputed_v=V)

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
    out = block(tgt, memory, tgt_mask=None, memory_mask=None)
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

    out = dec(tgt, memory)
    assert out.shape == (B, T, d_model)
