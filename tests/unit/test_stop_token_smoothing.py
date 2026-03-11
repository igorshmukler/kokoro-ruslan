"""
Tests for stop-token temporal smoothing (build_stop_token_targets).

The smoothing spreads the positive gradient over a short tail of frames
leading up to the true stop boundary, preventing a single-frame spike when
pos_weight is large.

Covered behaviours:
  1.  tail=0 → original hard [0…0, 1] target (backward-compat)
  2.  Last frame is always 1.0 for any T ≥ 1
  3.  Frame at stop-1 equals decay^1 for default decay
  4.  Frame at stop-k equals decay^k  (exponential law)
  5.  Frames before the tail are exactly 0.0
  6.  All values are in [0, 1]
  7.  tail clamped when T ≤ tail+1 (short sequences)
  8.  T=1 → target is [1.0] regardless of tail/decay
  9.  T=0 → empty tensor; no crash
  10. decay=1.0 → tail frames all equal 1.0 (constant plateau)
  11. Config defaults: stop_token_smooth_tail=4, stop_token_smooth_decay=0.5
  12. Custom config values are respected
"""

import pytest
import torch

from kokoro.data.dataset import build_stop_token_targets
from kokoro.training.config import TrainingConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build(T: int, tail: int = 4, decay: float = 0.5) -> torch.Tensor:
    return build_stop_token_targets(T, tail=tail, decay=decay)


# ---------------------------------------------------------------------------
# 1. Backward-compatibility: tail=0 reproduces the original hard target
# ---------------------------------------------------------------------------
class TestHardTargetCompat:

    def test_tail_zero_last_frame_is_one(self):
        t = _build(10, tail=0)
        assert t[-1].item() == pytest.approx(1.0)

    def test_tail_zero_all_others_are_zero(self):
        t = _build(10, tail=0)
        assert t[:-1].sum().item() == pytest.approx(0.0)

    def test_tail_zero_equals_original_hard_target(self):
        T = 50
        expected = torch.zeros(T)
        expected[-1] = 1.0
        assert torch.allclose(_build(T, tail=0), expected)


# ---------------------------------------------------------------------------
# 2–4. Exponential decay law
# ---------------------------------------------------------------------------
class TestExponentialDecay:

    def test_last_frame_always_one(self):
        for T in [1, 5, 20, 200]:
            t = _build(T, tail=4, decay=0.5)
            assert t[-1].item() == pytest.approx(1.0), f"T={T}: last frame should be 1.0"

    def test_second_to_last_equals_decay(self):
        decay = 0.5
        t = _build(20, tail=4, decay=decay)
        assert t[-2].item() == pytest.approx(decay, rel=1e-5)

    def test_frame_k_before_stop_equals_decay_pow_k(self):
        decay = 0.5
        tail  = 4
        T     = 30
        t = _build(T, tail=tail, decay=decay)
        for k in range(1, tail + 1):
            expected = decay ** k
            actual   = t[T - 1 - k].item()
            assert actual == pytest.approx(expected, rel=1e-5), (
                f"frame T-1-{k}: expected decay^{k}={expected:.4f}, got {actual:.4f}"
            )

    def test_decay_0_3_law(self):
        decay = 0.3
        tail  = 3
        T     = 20
        t = _build(T, tail=tail, decay=decay)
        for k in range(tail + 1):
            expected = decay ** k
            actual   = t[T - 1 - k].item()
            assert actual == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 5. Frames before the tail are zero
# ---------------------------------------------------------------------------
class TestZeroOutsideTail:

    def test_frames_before_tail_are_zero(self):
        tail = 3
        T    = 20
        t = _build(T, tail=tail)
        # frames 0 … T-1-tail-1 should all be 0
        prefix = t[:T - tail - 1]
        assert prefix.sum().item() == pytest.approx(0.0), (
            f"Expected zero before the tail; got {prefix.tolist()}"
        )

    def test_only_tail_plus_one_frames_nonzero(self):
        tail = 4
        T    = 50
        t = _build(T, tail=tail)
        n_nonzero = (t > 0).sum().item()
        assert n_nonzero == tail + 1


# ---------------------------------------------------------------------------
# 6. All values in [0, 1]
# ---------------------------------------------------------------------------
class TestValueRange:

    def test_all_values_in_unit_interval(self):
        for T in [1, 10, 200]:
            t = _build(T, tail=4, decay=0.5)
            assert t.min().item() >= 0.0, f"T={T}: min value negative"
            assert t.max().item() <= 1.0, f"T={T}: max value exceed 1.0"


# ---------------------------------------------------------------------------
# 7. Tail clamped for short sequences
# ---------------------------------------------------------------------------
class TestShortSequenceClamping:

    def test_tail_larger_than_T_minus_1_does_not_error(self):
        # T=3, tail=10 → only 3 frames available
        t = _build(3, tail=10, decay=0.5)
        assert t.shape == (3,)
        assert t[-1].item() == pytest.approx(1.0)

    def test_short_sequence_values_are_finite(self):
        for T in [1, 2, 3]:
            t = _build(T, tail=10, decay=0.5)
            assert torch.isfinite(t).all(), f"Non-finite values at T={T}"

    def test_short_sequence_last_frame_is_one(self):
        for T in [1, 2, 3, 4, 5]:
            t = _build(T, tail=10, decay=0.5)
            assert t[-1].item() == pytest.approx(1.0), f"T={T}: last frame not 1.0"


# ---------------------------------------------------------------------------
# 8–9. Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:

    def test_T_equals_1_gives_single_one(self):
        t = _build(1, tail=4, decay=0.5)
        assert t.shape == (1,)
        assert t[0].item() == pytest.approx(1.0)

    def test_T_equals_0_gives_empty_tensor(self):
        t = _build(0, tail=4, decay=0.5)
        assert t.shape == (0,)

    def test_output_dtype_is_float32(self):
        t = _build(10)
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# 10. decay=1.0 → all tail frames equal 1.0
# ---------------------------------------------------------------------------
class TestDecayOneConstantPlateau:

    def test_decay_1_tail_frames_all_equal_one(self):
        tail = 4
        T    = 20
        t = _build(T, tail=tail, decay=1.0)
        # The tail+1 frames at the end should all be 1.0
        assert torch.allclose(t[T - tail - 1 :], torch.ones(tail + 1))

    def test_decay_1_prefix_still_zero(self):
        tail = 4
        T    = 20
        t = _build(T, tail=tail, decay=1.0)
        assert t[:T - tail - 1].sum().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11–12. Config defaults and custom values
# ---------------------------------------------------------------------------
class TestConfigDefaults:

    def test_smooth_tail_default(self):
        assert TrainingConfig.__dataclass_fields__['stop_token_smooth_tail'].default == 6

    def test_smooth_decay_default(self):
        assert TrainingConfig.__dataclass_fields__['stop_token_smooth_decay'].default == pytest.approx(0.5)

    def test_custom_tail_survives_post_init(self):
        cfg = TrainingConfig(stop_token_smooth_tail=2)
        assert cfg.stop_token_smooth_tail == 2

    def test_custom_decay_survives_post_init(self):
        cfg = TrainingConfig(stop_token_smooth_decay=0.3)
        assert cfg.stop_token_smooth_decay == pytest.approx(0.3)

    def test_tail_zero_disables_smoothing(self):
        cfg = TrainingConfig(stop_token_smooth_tail=0)
        t = build_stop_token_targets(10, tail=cfg.stop_token_smooth_tail,
                                     decay=cfg.stop_token_smooth_decay)
        expected = torch.zeros(10)
        expected[-1] = 1.0
        assert torch.allclose(t, expected)
