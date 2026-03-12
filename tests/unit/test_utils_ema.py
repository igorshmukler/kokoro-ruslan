import torch
from src.kokoro.utils.ema import recommended_ema_decay


def test_recommended_ema_decay_invalid_inputs():
    assert recommended_ema_decay(0, 1, 5) == 0.9999
    assert recommended_ema_decay(10, 0, 5) == 0.9999


def test_recommended_ema_decay_range():
    val = recommended_ema_decay(1000, 10, 5)
    assert 0.9 <= val <= 0.9999
    assert isinstance(val, float)
