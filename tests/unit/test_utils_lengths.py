import torch
from src.kokoro.utils.lengths import (
    vectorized_expand_tokens,
    LengthRegulator,
    length_regulate,
    average_by_duration,
)

def test_vectorized_expand_tokens_2d_and_maxlen():
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    durations = torch.tensor([[1, 2, 0], [0, 1, 2]])

    out = vectorized_expand_tokens(tokens, durations)
    assert out.shape == (2, 3)
    assert out[0].tolist() == [1, 2, 2]
    assert out[1].tolist() == [5, 6, 6]

    out2 = vectorized_expand_tokens(tokens, durations, max_len=2)
    assert out2.shape == (2, 2)
    assert out2[0].tolist() == [1, 2]
    assert out2[1].tolist() == [5, 6]


def test_vectorized_expand_tokens_3d_and_zero_durations():
    tokens = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])  # (2,2,1)
    durations = torch.tensor([[0, 0], [0, 0]])

    out = vectorized_expand_tokens(tokens, durations)
    # when all durations zero, returns at least length 1
    assert out.shape[0] == 2
    assert out.shape[1] >= 1


def test_vectorized_expand_tokens_pad_when_max_len_larger():
    tokens = torch.tensor([[1, 2]])
    durations = torch.tensor([[1, 1]])
    out = vectorized_expand_tokens(tokens, durations, max_len=4)
    assert out.shape == (1, 4)
    # padded values should be zero
    assert out[0, 2] == 0
    assert out[0, 3] == 0


def test_vectorized_expand_tokens_fallback_on_exception(monkeypatch):
    tokens = torch.tensor([[1, 2, 3]])
    durations = torch.tensor([[1, 1, 1]])

    # Force an exception inside the fast path to trigger fallback by monkeypatching
    # a function used only in the fast path (`torch.arange`). The fallback uses
    # `torch.repeat_interleave` and will still work.
    def bad_arange(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, 'arange', bad_arange)

    out = vectorized_expand_tokens(tokens, durations)
    assert out.shape[1] >= 3


def test_length_regulate_and_masking():
    enc = torch.tensor([[[1.0], [2.0], [0.0]]])  # (1,3,1)
    durations = torch.tensor([[1.0, 2.0, 0.0]])
    text_padding_mask = torch.tensor([[False, False, True]])

    out, mask = length_regulate(enc, durations, text_padding_mask)
    # expanded length = 3 (1 + 2)
    assert out.shape == (1, 3, 1)
    assert mask.shape == (1, 3)
    assert mask[0].tolist() == [False, False, False]
    assert out[0, 0, 0].item() == 1.0
    assert out[0, 1, 0].item() == 2.0


def test_average_by_duration_basic_and_mask():
    # values: 2 batches, 4 frames
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    durations = torch.tensor([[1, 3], [0, 4]])

    avg = average_by_duration(values, durations)
    # batch 0: phoneme0 frames [1] avg=1, phoneme1 frames [2,3,4] avg=3
    assert torch.allclose(avg[0], torch.tensor([1.0, 3.0]))
    # batch1: durations [0,4] -> first phoneme masked to 0, second avg=(10+20+30+40)/4=25
    assert torch.allclose(avg[1], torch.tensor([0.0, 25.0]))

    # test mask param: mask phoneme 1 in batch 0
    mask = torch.tensor([[False, True], [False, False]])
    avg2 = average_by_duration(values, durations, mask=mask)
    assert torch.allclose(avg2[0], torch.tensor([1.0, 0.0]))


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
