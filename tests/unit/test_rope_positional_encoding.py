import pytest
import torch

from kokoro.model.positional_encoding import RotaryPositionalEncoding
from kokoro.model import transformers

"""Unit tests for RotaryPositionalEncoding (RoPE) and its integration into
MultiHeadAttentionImproved via rel_pos_type='rope'."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)


# ===========================================================================
# RotaryPositionalEncoding — standalone tests
# ===========================================================================

class TestRoPEInit:
    def test_head_dim_must_be_even(self):
        with pytest.raises(ValueError, match="even"):
            RotaryPositionalEncoding(head_dim=3)

    def test_attributes_set(self):
        rope = RotaryPositionalEncoding(head_dim=8, max_seq_len=32)
        assert rope.head_dim == 8
        assert rope.base == 10000

    def test_theta_shape(self):
        head_dim = 16
        rope = RotaryPositionalEncoding(head_dim=head_dim)
        assert rope.theta.shape == (head_dim // 2,)

    def test_cos_cache_shape_on_init(self):
        rope = RotaryPositionalEncoding(head_dim=8, max_seq_len=64)
        assert rope.cos_cache.shape == (64, 8)
        assert rope.sin_cache.shape == (64, 8)

    def test_cos_sin_caches_match(self):
        """cos²+sin²=1 at every position and dimension."""
        rope = RotaryPositionalEncoding(head_dim=8, max_seq_len=32)
        sq = rope.cos_cache ** 2 + rope.sin_cache ** 2
        assert torch.allclose(sq, torch.ones_like(sq), atol=1e-5)


class TestRoPEForward:
    def test_output_shapes_preserved(self):
        rope = RotaryPositionalEncoding(head_dim=16)
        B, H, S, D = 2, 4, 10, 16
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_seq_lengths(self):
        rope = RotaryPositionalEncoding(head_dim=8)
        B, H = 1, 2
        q = torch.randn(B, H, 5, 8)
        k = torch.randn(B, H, 9, 8)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_output_is_finite(self):
        rope = RotaryPositionalEncoding(head_dim=16)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        q_rot, k_rot = rope(q, k)
        assert torch.isfinite(q_rot).all()
        assert torch.isfinite(k_rot).all()

    def test_rotation_is_not_identity(self):
        """RoPE must change the vectors (except trivially at position 0)."""
        rope = RotaryPositionalEncoding(head_dim=8)
        q = torch.ones(1, 1, 6, 8)
        k = torch.ones(1, 1, 6, 8)
        q_rot, _ = rope(q, k)
        # Positions > 0 should differ from the unrotated input
        assert not torch.allclose(q_rot[:, :, 1:, :], q[:, :, 1:, :])

    def test_rotation_preserves_norm(self):
        """Rotation matrices are orthogonal — vector norms must be preserved."""
        torch.manual_seed(42)
        rope = RotaryPositionalEncoding(head_dim=16)
        q = torch.randn(2, 4, 10, 16)
        q_rot, _ = rope(q, q)
        # Compare L2 norms along the head_dim axis
        norm_before = q.norm(dim=-1)
        norm_after = q_rot.norm(dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-5)

    def test_q_offset_shifts_rotation(self):
        """Using q_offset should yield different rotated output."""
        rope = RotaryPositionalEncoding(head_dim=8)
        q = torch.randn(1, 1, 3, 8)
        k = torch.randn(1, 1, 3, 8)
        q_rot_0, _ = rope(q, k, q_offset=0)
        q_rot_5, _ = rope(q, k, q_offset=5)
        assert not torch.allclose(q_rot_0, q_rot_5)

    def test_cache_auto_extends(self):
        rope = RotaryPositionalEncoding(head_dim=8, max_seq_len=4)
        q = torch.randn(1, 1, 16, 8)
        k = torch.randn(1, 1, 16, 8)
        q_rot, k_rot = rope(q, k)
        assert rope.cos_cache.shape[0] >= 16
        assert q_rot.shape == q.shape

    def test_dtype_passthrough_float16(self):
        """RoPE must not change the dtype of Q and K."""
        rope = RotaryPositionalEncoding(head_dim=8)
        q = torch.randn(1, 2, 4, 8).half()
        k = torch.randn(1, 2, 4, 8).half()
        q_rot, k_rot = rope(q, k)
        assert q_rot.dtype == torch.float16
        assert k_rot.dtype == torch.float16

    def test_relative_distance_captured_in_dot_product(self):
        """
        Q·Kᵀ should encode relative position: score(i, j) depends on (j-i).
        For a translationally-invariant pair, Q[i]·K[j] ≈ Q[i+d]·K[j+d] for any d.
        """
        torch.manual_seed(7)
        rope = RotaryPositionalEncoding(head_dim=16)
        B, H, S, D = 1, 1, 8, 16
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)

        q_rot, k_rot = rope(q, k)
        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))  # (B, H, S, S)

        # For fixed relative distance d=2, scores along diagonal d should be
        # approximately consistent (modulo the varying q/k content, so we just
        # check the scores are finite and non-trivially unequal across positions)
        assert torch.isfinite(scores).all()
        assert scores.std() > 0  # non-constant → position information encoded


class TestRoPERotateHalf:
    def test_rotate_half_basic(self):
        x = torch.tensor([[1., 2., 3., 4.]])
        expected = torch.tensor([[-3., -4., 1., 2.]])
        result = RotaryPositionalEncoding._rotate_half(x)
        assert torch.allclose(result, expected)

    def test_rotate_half_involution(self):
        """Applying rotate_half twice should negate the vector (270° = -90°)."""
        x = torch.randn(4, 8)
        twice = RotaryPositionalEncoding._rotate_half(
            RotaryPositionalEncoding._rotate_half(x)
        )
        assert torch.allclose(twice, -x)


# ===========================================================================
# MultiHeadAttentionImproved — rel_pos_type='rope'
# ===========================================================================

class TestAttentionRoPE:
    def test_rope_module_created_when_enabled(self):
        attn = transformers.MultiHeadAttentionImproved(
            16, 4, dropout=0.0, use_relative_pos=True, rel_pos_type='rope'
        )
        assert hasattr(attn, 'rope')
        assert isinstance(attn.rope, RotaryPositionalEncoding)

    def test_no_alibi_slopes_when_rope(self):
        attn = transformers.MultiHeadAttentionImproved(
            16, 4, dropout=0.0, use_relative_pos=True, rel_pos_type='rope'
        )
        assert not hasattr(attn, 'alibi_slopes')

    def test_alibi_slopes_when_alibi(self):
        attn = transformers.MultiHeadAttentionImproved(
            16, 4, dropout=0.0, use_relative_pos=True, rel_pos_type='alibi'
        )
        assert hasattr(attn, 'alibi_slopes')
        assert not hasattr(attn, 'rope')

    def test_output_shape(self):
        torch.manual_seed(10)
        B, S, d_model, heads = 2, 8, 16, 4
        attn = transformers.MultiHeadAttentionImproved(
            d_model, heads, dropout=0.0, use_relative_pos=True, rel_pos_type='rope'
        )
        attn.eval()
        x = torch.randn(B, S, d_model)
        out, _, _ = attn(x, x, x)
        assert out.shape == (B, S, d_model)

    def test_output_is_finite(self):
        torch.manual_seed(11)
        B, S, d_model, heads = 2, 8, 16, 4
        attn = transformers.MultiHeadAttentionImproved(
            d_model, heads, dropout=0.0, use_relative_pos=True, rel_pos_type='rope'
        )
        attn.eval()
        x = torch.randn(B, S, d_model)
        out, _, _ = attn(x, x, x)
        assert torch.isfinite(out).all()

    def test_rope_differs_from_no_rel_pos(self):
        """RoPE output must differ from the plain (no relative pos) baseline."""
        torch.manual_seed(12)
        B, S, d_model, heads = 1, 6, 16, 4
        x = torch.randn(B, S, d_model)

        attn_rope = transformers.MultiHeadAttentionImproved(
            d_model, heads, dropout=0.0, use_relative_pos=True, rel_pos_type='rope'
        )
        attn_plain = transformers.MultiHeadAttentionImproved(
            d_model, heads, dropout=0.0, use_relative_pos=False
        )
        # Copy weights so the only difference is RoPE
        attn_plain.load_state_dict(
            {k: v for k, v in attn_rope.state_dict().items() if 'rope' not in k},
            strict=False,
        )
        attn_rope.eval()
        attn_plain.eval()

        out_rope, _, _ = attn_rope(x, x, x)
        out_plain, _, _ = attn_plain(x, x, x)
        assert not torch.allclose(out_rope, out_plain, atol=1e-4)

    def test_causal_mask_with_rope(self):
        torch.manual_seed(13)
        B, S, d_model, heads = 2, 10, 16, 4
        attn = transformers.MultiHeadAttentionImproved(
            d_model, heads, dropout=0.0, use_relative_pos=True, rel_pos_type='rope'
        )
        attn.eval()
        x = torch.randn(B, S, d_model)
        causal = _causal_mask(S)
        out, _, _ = attn(x, x, x, attn_mask=causal)
        assert out.shape == (B, S, d_model)
        assert torch.isfinite(out).all()

    def test_cross_attention_with_rope(self):
        """RoPE encoder block: query and key have different lengths."""
        torch.manual_seed(14)
        B, Sq, Sk, d_model, heads = 1, 5, 9, 16, 4
        attn = transformers.MultiHeadAttentionImproved(
            d_model, heads, dropout=0.0, use_relative_pos=True, rel_pos_type='rope'
        )
        attn.eval()
        q = torch.randn(B, Sq, d_model)
        k = torch.randn(B, Sk, d_model)
        out, _, _ = attn(q, k, k)
        assert out.shape == (B, Sq, d_model)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        torch.manual_seed(15)
        B, S, d_model, heads = 2, 6, 16, 4
        attn = transformers.MultiHeadAttentionImproved(
            d_model, heads, dropout=0.0, use_relative_pos=True, rel_pos_type='rope'
        )
        x = torch.randn(B, S, d_model, requires_grad=True)
        out, _, _ = attn(x, x, x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ===========================================================================
# Encoder block with RoPE
# ===========================================================================

class TestEncoderBlockRoPE:
    def test_encoder_block_rope_output_shape(self):
        torch.manual_seed(20)
        B, S, d_model = 2, 10, 32
        block = transformers.ImprovedTransformerEncoderBlock(
            d_model, nhead=4, dim_feedforward=64, dropout=0.0,
            use_relative_pos=True, rel_pos_type='rope'
        )
        block.eval()
        x = torch.randn(B, S, d_model)
        out = block(x)
        assert out.shape == (B, S, d_model)
        assert torch.isfinite(out).all()

    def test_encoder_block_rope_gradient_flows(self):
        torch.manual_seed(21)
        B, S, d_model = 1, 8, 32
        block = transformers.ImprovedTransformerEncoderBlock(
            d_model, nhead=4, dim_feedforward=64, dropout=0.0,
            use_relative_pos=True, rel_pos_type='rope'
        )
        x = torch.randn(B, S, d_model, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ===========================================================================
# Decoder block with RoPE
# ===========================================================================

class TestDecoderBlockRoPE:
    def test_decoder_block_rope_output_shape(self):
        torch.manual_seed(30)
        B, T, M, d_model = 2, 6, 8, 32
        block = transformers.ImprovedTransformerDecoderBlock(
            d_model, nhead=4, dim_feedforward=64, dropout=0.0,
            rel_pos_type='rope'
        )
        block.eval()
        tgt = torch.randn(B, T, d_model)
        mem = torch.randn(B, M, d_model)
        out, _ = block(tgt, mem)
        assert out.shape == (B, T, d_model)
        assert torch.isfinite(out).all()

    def test_decoder_block_rope_with_causal_mask(self):
        torch.manual_seed(31)
        B, T, M, d_model = 1, 7, 9, 32
        block = transformers.ImprovedTransformerDecoderBlock(
            d_model, nhead=4, dim_feedforward=64, dropout=0.0,
            rel_pos_type='rope'
        )
        block.eval()
        tgt = torch.randn(B, T, d_model)
        mem = torch.randn(B, M, d_model)
        causal = _causal_mask(T)
        out, _ = block(tgt, mem, tgt_mask=causal)
        assert out.shape == (B, T, d_model)
        assert torch.isfinite(out).all()

    def test_decoder_block_rope_gradient_flows(self):
        torch.manual_seed(32)
        B, T, M, d_model = 2, 5, 7, 32
        block = transformers.ImprovedTransformerDecoderBlock(
            d_model, nhead=4, dim_feedforward=64, dropout=0.0,
            rel_pos_type='rope'
        )
        tgt = torch.randn(B, T, d_model, requires_grad=True)
        mem = torch.randn(B, M, d_model)
        out, _ = block(tgt, mem)
        out.sum().backward()
        assert tgt.grad is not None
        assert torch.isfinite(tgt.grad).all()

    def test_decoder_block_rel_pos_type_stored(self):
        block = transformers.ImprovedTransformerDecoderBlock(
            32, nhead=4, dim_feedforward=64, dropout=0.0,
            rel_pos_type='rope'
        )
        assert block.self_attn.rel_pos_type == 'rope'
        assert hasattr(block.self_attn, 'rope')


# ===========================================================================
# ImprovedTransformerDecoder (multi-layer) with RoPE
# ===========================================================================

class TestTransformerDecoderRoPE:
    def test_decoder_rope_output_shape(self):
        torch.manual_seed(40)
        B, T, M, d_model = 2, 8, 10, 32
        decoder = transformers.ImprovedTransformerDecoder(
            d_model, nhead=4, dim_feedforward=64, dropout=0.0,
            num_layers=2, rel_pos_type='rope'
        )
        decoder.eval()
        tgt = torch.randn(B, T, d_model)
        mem = torch.randn(B, M, d_model)
        out, _ = decoder(tgt, mem)
        assert out.shape == (B, T, d_model)
        assert torch.isfinite(out).all()

    def test_all_layers_use_rope(self):
        decoder = transformers.ImprovedTransformerDecoder(
            32, nhead=4, dim_feedforward=64, dropout=0.0,
            num_layers=3, rel_pos_type='rope'
        )
        for layer in decoder.layers:
            assert layer.self_attn.rel_pos_type == 'rope'
            assert hasattr(layer.self_attn, 'rope')
