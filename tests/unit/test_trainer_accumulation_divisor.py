from kokoro.training.trainer import KokoroTrainer


def test_effective_accumulation_divisor_full_window_returns_configured_steps():
    divisor = KokoroTrainer._effective_accumulation_divisor(
        gradient_accumulation_steps=4,
        accumulated_step=1,
        batch_idx=5,
        num_batches=100,
    )
    assert divisor == 4


def test_effective_accumulation_divisor_tail_window_uses_remaining_microbatches():
    # Epoch with 10 batches, grad_accum=4.
    # At batch_idx=8 with accumulated_step=0, only two micro-batches remain (8 and 9).
    divisor = KokoroTrainer._effective_accumulation_divisor(
        gradient_accumulation_steps=4,
        accumulated_step=0,
        batch_idx=8,
        num_batches=10,
    )
    assert divisor == 2

    # Next micro-batch in same tail cycle should keep the same effective divisor.
    divisor_next = KokoroTrainer._effective_accumulation_divisor(
        gradient_accumulation_steps=4,
        accumulated_step=1,
        batch_idx=9,
        num_batches=10,
    )
    assert divisor_next == 2


def test_effective_accumulation_divisor_is_always_at_least_one():
    divisor = KokoroTrainer._effective_accumulation_divisor(
        gradient_accumulation_steps=0,
        accumulated_step=0,
        batch_idx=0,
        num_batches=0,
    )
    assert divisor == 1
