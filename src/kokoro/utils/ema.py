"""Exponential Moving Average (EMA) utilities for training."""


def recommended_ema_decay(n_train: int, batch_size: int, k: float) -> float:
    """
    Calculate EMA decay so that the 'center of mass' or smoothing
    window aligns with k epochs.
    """
    if n_train <= 0 or batch_size <= 0:
        return 0.9999

    # 1. Calculate total steps per epoch
    steps_per_epoch = n_train / batch_size

    # 2. Total steps in the desired window (k epochs)
    total_steps = steps_per_epoch * k

    # 3. Standard EMA decay formula:
    # Often expressed as alpha = (window - 1) / (window + 1)
    # or alpha = exp(-1 / total_steps) for a precise half-life
    decay = (total_steps - 1) / (total_steps + 1)

    # Clip to a sensible range for TTS (usually between 0.99 and 0.9999)
    return max(0.9, min(decay, 0.9999))
