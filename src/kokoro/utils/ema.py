"""Exponential Moving Average (EMA) utilities for training."""

import math


def recommended_ema_decay(n_train: int, batch_size: int, k: float) -> float:
    """
    Calculate EMA decay so that the half-life equals k epochs.

    Half-life definition: after k epochs of optimizer steps, the
    contribution of the original weights has decayed to 50%.

    Formula: decay = exp(-ln(2) / (steps_per_epoch * k))
    """
    if n_train <= 0 or batch_size <= 0:
        return 0.9999

    steps_per_epoch = n_train / batch_size
    half_life_steps = steps_per_epoch * k

    if half_life_steps <= 0:
        return 0.9999

    decay = math.exp(-math.log(2) / half_life_steps)

    # Clip to a sensible range for TTS
    return max(0.9, min(decay, 0.9999))
