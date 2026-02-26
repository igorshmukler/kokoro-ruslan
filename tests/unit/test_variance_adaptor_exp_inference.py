"""
Tests covering the fix: VarianceAdaptor.forward uses exp() on duration_pred
before rounding when duration_target is None (inference mode).

Bug: old code did `torch.round(duration_pred)` directly on log1p-domain values
     (e.g. log1p(5) ≈ 1.79 → rounds to 2 frames, not the correct 6).
Fix: new code does `torch.round(torch.exp(duration_pred))` → exp(1.79) ≈ 6.
"""
import math
import torch
import torch.nn as nn
import pytest

from kokoro.model.variance_predictor import VarianceAdaptor


# ---------------------------------------------------------------------------
# Helper: a fixed-output duration predictor (bypasses learning)
# ---------------------------------------------------------------------------
class FixedDurationPredictor(nn.Module):
    """Returns a constant log-domain value for every phoneme position."""

    def __init__(self, log_val: float, hidden_dim: int = 1):
        super().__init__()
        self.log_val = log_val
        # A dummy parameter so the module registers correctly
        self._dummy = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        return torch.full((x.size(0), x.size(1)), self.log_val)


def _adaptor_with_fixed_predictor(log_val: float) -> VarianceAdaptor:
    adaptor = VarianceAdaptor(n_bins=8, pitch_min=0.0, pitch_max=1.0,
                              energy_min=0.0, energy_max=1.0)
    adaptor.duration_predictor = FixedDurationPredictor(log_val)
    return adaptor


# ---------------------------------------------------------------------------
# 1. Frame count aligns with exp() of the log-domain prediction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_frames,n_phonemes", [
    (5, 4),
    (7, 3),
    (1, 6),   # edge: exp(log1p(1)) ≈ round(2.0) = 2 frames each
    (10, 2),
])
def test_inference_frame_count_matches_exp_of_log_prediction(n_frames, n_phonemes):
    """
    With a fixed predictor outputting log1p(n_frames), the expanded sequence
    should have n_phonemes * round(exp(log1p(n_frames))) frames.
    """
    log_val = math.log1p(n_frames)
    adaptor = _adaptor_with_fixed_predictor(log_val)

    hidden = adaptor.hidden_dim
    encoder_output = torch.randn(1, n_phonemes, hidden)

    adapted, _, _, _, _ = adaptor(
        encoder_output, mask=None,
        pitch_target=None, energy_target=None, duration_target=None
    )

    expected_frames_per_phoneme = round(math.exp(log_val))
    expected_total = n_phonemes * expected_frames_per_phoneme
    assert adapted.shape[1] == expected_total, (
        f"Expected {expected_total} frames ({n_phonemes} phonemes × "
        f"{expected_frames_per_phoneme} frames each), got {adapted.shape[1]}"
    )


# ---------------------------------------------------------------------------
# 2. Inference gives more frames than naïve rounding of log-domain prediction
# ---------------------------------------------------------------------------
def test_inference_exp_gives_more_frames_than_raw_round():
    """
    Regression: without exp(), log1p(5) ≈ 1.79 rounds to 2 frames.
    With exp(), it correctly produces ~6 frames.
    """
    log_val = math.log1p(5)   # ≈ 1.7918
    adaptor = _adaptor_with_fixed_predictor(log_val)

    hidden = adaptor.hidden_dim
    n_phonemes = 4
    encoder_output = torch.randn(1, n_phonemes, hidden)

    # Inference path — should use exp()
    adapted_infer, _, _, _, _ = adaptor(
        encoder_output, duration_target=None,
        pitch_target=None, energy_target=None
    )

    # Training path forced to old (wrong) raw-round behavior: 2 frames/phoneme
    wrong_durations = torch.full((1, n_phonemes), 2.0)
    adapted_wrong, _, _, _, _ = adaptor(
        encoder_output, duration_target=wrong_durations,
        pitch_target=None, energy_target=None
    )

    assert adapted_infer.shape[1] > adapted_wrong.shape[1], (
        "Inference with exp() should yield more frames than raw-round of log1p value "
        f"(got {adapted_infer.shape[1]} vs {adapted_wrong.shape[1]})"
    )


# ---------------------------------------------------------------------------
# 3. Negative prediction clamped to 0 (not negative frames)
# ---------------------------------------------------------------------------
def test_inference_negative_prediction_clamped_to_zero_frames():
    """exp of a large negative number → near 0 after clamp(min=0)."""
    adaptor = _adaptor_with_fixed_predictor(-100.0)  # exp(-100) ≈ 0 → clamp → 0

    hidden = adaptor.hidden_dim
    n_phonemes = 3
    encoder_output = torch.randn(1, n_phonemes, hidden)

    adapted, _, _, _, _ = adaptor(
        encoder_output, duration_target=None,
        pitch_target=None, energy_target=None
    )
    # Should not crash; all durations = 0 so adapted has minimal frames
    # (VarianceAdaptor pads to min_frames=3 to avoid kernel size errors)
    assert adapted.shape[0] == 1
    assert adapted.shape[2] == hidden


# ---------------------------------------------------------------------------
# 4. exp(0) = 1 → each phoneme gets at least one frame
# ---------------------------------------------------------------------------
def test_inference_zero_log_prediction_gives_one_frame_per_phoneme():
    """log_pred=0 → exp(0)=1.0 → round(1.)=1 → 1 frame per phoneme."""
    adaptor = _adaptor_with_fixed_predictor(0.0)

    hidden = adaptor.hidden_dim
    n_phonemes = 5
    encoder_output = torch.randn(1, n_phonemes, hidden)

    adapted, _, _, _, _ = adaptor(
        encoder_output, duration_target=None,
        pitch_target=None, energy_target=None
    )

    assert adapted.shape[1] == n_phonemes, (
        "exp(0)=1 frame per phoneme → total should equal n_phonemes"
    )


# ---------------------------------------------------------------------------
# 5. Training path (duration_target provided) is unchanged
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frames_per_phoneme", [3, 5, 8])
def test_training_path_uses_provided_duration_target(frames_per_phoneme: int):
    """When duration_target is supplied, it must be used directly (no exp)."""
    adaptor = VarianceAdaptor(n_bins=8, pitch_min=0.0, pitch_max=1.0,
                              energy_min=0.0, energy_max=1.0)

    batch, n_phonemes, hidden = 2, 6, adaptor.hidden_dim
    encoder_output = torch.randn(batch, n_phonemes, hidden)
    duration_target = torch.full((batch, n_phonemes), float(frames_per_phoneme))

    adapted, _, _, _, _ = adaptor(
        encoder_output, duration_target=duration_target,
        pitch_target=None, energy_target=None
    )

    expected_frames = n_phonemes * frames_per_phoneme
    assert adapted.shape == (batch, expected_frames, hidden), (
        f"Training path: expected ({batch}, {expected_frames}, {hidden}), "
        f"got {adapted.shape}"
    )


# ---------------------------------------------------------------------------
# 6. Output tensor attributes remain valid after change
# ---------------------------------------------------------------------------
def test_inference_output_is_finite_and_correct_shape():
    """Adapted output must be finite and have the right dimensions."""
    log_val = math.log1p(4)
    adaptor = _adaptor_with_fixed_predictor(log_val)

    batch, n_phonemes = 2, 5
    hidden = adaptor.hidden_dim
    encoder_output = torch.randn(batch, n_phonemes, hidden)

    adapted, dur_pred, pitch_pred, energy_pred, frame_mask = adaptor(
        encoder_output, duration_target=None,
        pitch_target=None, energy_target=None
    )

    expected_frames_per_phoneme = round(math.exp(log_val))  # round(5) = 5
    expected_frames = n_phonemes * expected_frames_per_phoneme

    assert adapted.shape == (batch, expected_frames, hidden)
    assert dur_pred.shape == (batch, n_phonemes)
    assert pitch_pred.shape == (batch, expected_frames)
    assert energy_pred.shape == (batch, expected_frames)
    assert frame_mask.shape == (batch, expected_frames)
    assert torch.isfinite(adapted).all(), "Adapted output contains non-finite values"
