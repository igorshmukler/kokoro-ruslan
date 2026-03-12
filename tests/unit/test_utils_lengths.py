import torch
import pytest
from src.kokoro.utils.lengths import (
    vectorized_expand_tokens,
    LengthRegulator,
    length_regulate,
    average_by_duration,
)


def test_vectorized_expand_tokens_basic():
    tokens = torch.tensor([[1, 2], [3, 4]])
    durations = torch.tensor([[1, 2], [0, 1]])
    out = vectorized_expand_tokens(tokens, durations)
    # First batch: [1,2,2] -> length 3
    # Second batch: [3,4] -> durations [[0,1]] -> [4]
    assert out.shape[0] == 2
    assert out.shape[1] >= 1


def test_length_regulator_module():
    mod = LengthRegulator()
    tokens = torch.randn(2, 3, 4)
    durations = torch.tensor([[1, 0, 2], [0, 0, 0]])
    out = mod(tokens, durations)
    assert out.dim() == 3


def test_length_regulate_and_average():
    enc = torch.randn(2, 3, 4)
    durations = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    text_mask = torch.tensor([[False, False, True], [True, True, True]])
    expanded, mask = length_regulate(enc, durations, text_mask)
    assert expanded.shape[0] == 2
    assert mask.shape[0] == 2

    # average_by_duration expects frame-level values
    values = torch.arange(6.).view(2, 3)
    d = torch.tensor([[1, 2, 0], [1, 1, 1]])
    out_avg = average_by_duration(values, d)
    assert out_avg.shape[0] == 2
    assert out_avg.shape[1] == d.shape[1]
