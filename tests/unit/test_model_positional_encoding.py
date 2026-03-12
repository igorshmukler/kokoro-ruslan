import torch
import pytest
from src.kokoro.model.positional_encoding import PositionalEncoding, RotaryPositionalEncoding


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
