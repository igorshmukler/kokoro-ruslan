import torch
import pytest

from src.kokoro.model.positional_encoding import (
    PositionalEncoding,
    RotaryPositionalEncoding,
)


def test_positional_encoding_shape_and_mismatch():
    d_model = 4
    pe = PositionalEncoding(d_model=d_model, max_len=8)
    pe.eval()

    x = torch.zeros(2, 5, d_model)
    out = pe(x)
    assert out.shape == x.shape

    # mismatched d_model should raise
    x_bad = torch.zeros(2, 5, d_model + 2)
    with pytest.raises(ValueError):
        pe(x_bad)


def test_positional_encoding_extend():
    d_model = 2
    pe = PositionalEncoding(d_model=d_model, max_len=4)
    pe.eval()
    x = torch.zeros(1, 6, d_model)
    # should extend without error
    out = pe(x, seq_offset=0)
    assert out.shape == x.shape


def test_rotary_positional_encoding_basic_and_rotate_half():
    head_dim = 4
    rope = RotaryPositionalEncoding(head_dim=head_dim, max_seq_len=8)

    # test _rotate_half
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = x.unsqueeze(0).unsqueeze(0)  # (1,1,4)
    rotated = rope._rotate_half(x)
    # for [1,2,3,4] -> [-3,-4,1,2]
    assert torch.allclose(rotated.squeeze(), torch.tensor([-3.0, -4.0, 1.0, 2.0]))

    # create q,k with shape (batch, heads, seq_len, head_dim)
    q = torch.randn(2, 3, 5, head_dim)
    k = torch.randn(2, 3, 6, head_dim)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert q_rot.dtype == q.dtype
    assert k_rot.dtype == k.dtype


def test_rotary_positional_encoding_odd_head_dim_raises():
    with pytest.raises(ValueError):
        RotaryPositionalEncoding(head_dim=3)

def test_positional_encoding_basic_and_extend():
    pe = PositionalEncoding(d_model=8, dropout=0.0, max_len=4)
    x = torch.zeros(2, 3, 8)
    out = pe(x, seq_offset=2)
    assert out.shape == x.shape

    # Trigger extension by requesting larger slice
    x2 = torch.zeros(1, 10, 8)
    out2 = pe(x2, seq_offset=0)
    assert out2.shape == x2.shape

    # Mismatched d_model should raise
    with pytest.raises(ValueError):
        pe(torch.zeros(1, 2, 4))


def test_rotary_positional_encoding_rotate_and_forward():
    rpe = RotaryPositionalEncoding(head_dim=4, max_seq_len=8)
    # Test rotate half
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    xr = rpe._rotate_half(x)
    assert xr.shape == x.shape

    # Forward pass shapes
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 5, 4)
    q_rot, k_rot = rpe(q, k, q_offset=0, k_offset=0)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
