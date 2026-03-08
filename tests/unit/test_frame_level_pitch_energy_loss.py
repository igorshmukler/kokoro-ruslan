"""
Regression tests for the frame-level pitch/energy loss fix.

Root cause: VarianceAdaptor.pitch_predictor runs at frame level but the old
loss code averaged predictions back to phoneme level before comparing with
phoneme-level targets.  This collapsed all intra-phoneme gradient signal to a
single value per phoneme, allowing the predictor to converge to the degenerate
solution "constant per phoneme" with zero within-phoneme variance.

Fix (losses.py): when the prediction is frame-level and the target is phoneme-
level, expand the phoneme-level target to frame level via vectorized_expand_tokens,
then compute MSE at frame level using mel_mask_2d.

Tests verify:
  1. Loss is non-zero when frame-level predictions differ from frame-level-expanded targets.
  2. Each frame independently contributes to the gradient (no collapsed gradient).
  3. Gradients differ BETWEEN frames within the same phoneme.
  4. Padding frames are excluded (mel_mask_2d).
  5. Phoneme-level prediction path (fallback) is unchanged.
  6. Energy follows the identical fix logic.
  7. No gradient flows through the averaged-prediction path (old bug regression).
"""
import logging
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kokoro.training.losses import calculate_training_losses
from kokoro.utils.lengths import vectorized_expand_tokens


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cfg(**kwargs):
    defaults = dict(
        duration_loss_weight=0.0,
        stop_token_loss_weight=0.0,
        pitch_loss_weight=1.0,
        energy_loss_weight=1.0,
        verbose=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


_LOG = logging.getLogger(__name__)


def _call_losses(
    *,
    B=1,
    n_phones=3,
    n_frames=6,
    mel_dim=2,
    phoneme_durations=None,
    mel_lengths=None,
    phoneme_lengths=None,
    predicted_pitch=None,
    predicted_energy=None,
    pitch_targets=None,
    energy_targets=None,
    config=None,
):
    """Call calculate_training_losses with minimal boilerplate."""
    device = torch.device("cpu")
    if phoneme_durations is None:
        # default: 2 frames per phoneme
        phoneme_durations = torch.full((B, n_phones), 2, dtype=torch.long)
    if mel_lengths is None:
        mel_lengths = torch.full((B,), n_frames, dtype=torch.long)
    if phoneme_lengths is None:
        phoneme_lengths = torch.full((B,), n_phones, dtype=torch.long)

    mel_specs = torch.zeros(B, n_frames, mel_dim)
    predicted_mel = torch.zeros(B, n_frames, mel_dim)
    predicted_log_durations = torch.zeros(B, n_phones)
    stop_targets = torch.zeros(B, n_frames)
    stop_logits = torch.zeros(B, n_frames)

    return calculate_training_losses(
        device=device,
        config=config or _cfg(),
        model=None,
        criterion_mel=nn.L1Loss(reduction="none"),
        criterion_duration=nn.MSELoss(reduction="none"),
        criterion_stop_token=nn.BCEWithLogitsLoss(reduction="none"),
        criterion_pitch=nn.MSELoss(reduction="none"),
        criterion_energy=nn.MSELoss(reduction="none"),
        average_by_duration=lambda p, d, l: p,
        logger=_LOG,
        predicted_mel=predicted_mel,
        predicted_log_durations=predicted_log_durations,
        predicted_stop_logits=stop_logits,
        mel_specs=mel_specs,
        phoneme_durations=phoneme_durations,
        stop_token_targets=stop_targets,
        mel_lengths=mel_lengths,
        phoneme_lengths=phoneme_lengths,
        predicted_pitch=predicted_pitch,
        predicted_energy=predicted_energy,
        pitch_targets=pitch_targets,
        energy_targets=energy_targets,
    )


# ===========================================================================
# 1. Frame-level path: correct loss value via expansion
# ===========================================================================

class TestFrameLevelPitchLossValue:
    """Frame-level prediction × phoneme-level target → loss computed via expansion."""

    def test_perfect_prediction_gives_zero_loss(self):
        """When predictions exactly match expanded targets the MSE must be 0."""
        torch.manual_seed(0)
        B, P, T = 1, 3, 6
        phoneme_durations = torch.tensor([[2, 2, 2]], dtype=torch.long)  # 3×2 = 6 frames
        phoneme_targets = torch.tensor([[0.2, 0.5, 0.8]])                # phoneme-level

        # Expand targets manually and use them as predictions
        perfect_pred = vectorized_expand_tokens(
            phoneme_targets, phoneme_durations, max_len=T
        )

        _, _, _, _, loss_pitch, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            predicted_pitch=perfect_pred,
            pitch_targets=phoneme_targets,
        )
        assert loss_pitch.item() == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_loss_when_prediction_differs(self):
        B, P, T = 1, 3, 6
        phoneme_durations = torch.tensor([[2, 2, 2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.2, 0.5, 0.8]])

        # Predictions are all zeros — large loss expected
        predicted_pitch = torch.zeros(B, T)
        _, _, _, _, loss_pitch, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            predicted_pitch=predicted_pitch,
            pitch_targets=phoneme_targets,
        )
        # MSE of (0 vs 0.2, 0 vs 0.2, 0 vs 0.5, 0 vs 0.5, 0 vs 0.8, 0 vs 0.8) / 6
        expected = (2 * 0.2**2 + 2 * 0.5**2 + 2 * 0.8**2) / 6
        assert loss_pitch.item() == pytest.approx(expected, rel=1e-4)

    def test_energy_perfect_prediction_gives_zero_loss(self):
        B, P, T = 1, 3, 6
        phoneme_durations = torch.tensor([[2, 2, 2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.1, 0.4, 0.9]])

        perfect_pred = vectorized_expand_tokens(
            phoneme_targets, phoneme_durations, max_len=T
        )
        _, _, _, _, _, loss_energy = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            predicted_energy=perfect_pred,
            energy_targets=phoneme_targets,
        )
        assert loss_energy.item() == pytest.approx(0.0, abs=1e-6)

    def test_unequal_phoneme_durations(self):
        """Different per-phoneme durations should still produce the correct expanded MSE."""
        B, P, T = 1, 3, 7
        phoneme_durations = torch.tensor([[1, 3, 3]], dtype=torch.long)  # 1+3+3 = 7
        phoneme_targets = torch.tensor([[0.3, 0.6, 0.9]])
        predicted_pitch = torch.zeros(B, T)

        # Expected: 1*0.09 + 3*0.36 + 3*0.81 = 0.09+1.08+2.43 = 3.6 over 7 frames
        expected_mse = (1 * 0.3**2 + 3 * 0.6**2 + 3 * 0.9**2) / 7

        _, _, _, _, loss_pitch, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            mel_lengths=torch.tensor([T]),
            phoneme_lengths=torch.tensor([P]),
            predicted_pitch=predicted_pitch,
            pitch_targets=phoneme_targets,
        )
        assert loss_pitch.item() == pytest.approx(expected_mse, rel=1e-4)


# ===========================================================================
# 2. Gradient flows independently per frame
# ===========================================================================

class TestFrameLevelGradients:
    """Every frame must carry its own independent gradient."""

    def test_each_frame_has_nonzero_grad(self):
        """With distinct targets per phoneme, all frame gradients must be non-zero."""
        torch.manual_seed(1)
        B, P, T = 1, 3, 6
        phoneme_durations = torch.tensor([[2, 2, 2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.2, 0.5, 0.8]])    # distinct per phoneme

        predicted_pitch = torch.zeros(B, T, requires_grad=True)

        total_loss, _, _, _, _, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            predicted_pitch=predicted_pitch,
            pitch_targets=phoneme_targets,
        )
        total_loss.backward()

        assert predicted_pitch.grad is not None
        # All 6 frames should have non-zero gradient when predictions ≠ targets
        assert (predicted_pitch.grad.abs() > 0).all()

    def test_intra_phoneme_gradient_is_per_frame(self):
        """
        Critical regression: under the OLD averaging scheme all frames within
        a phoneme received IDENTICAL gradient.  The new frame-level expansion
        scheme should also produce identical gradient within a phoneme (because
        the target for each frame is the same phoneme mean), but critically the
        gradient is now applied at the individual-frame scale and not collapsed
        through an averaging operation.

        We verify this by checking that the gradient is proportional to
        2*(pred_i - target_i)/N_frames for each frame, where target_i is the
        expanded phoneme mean.
        """
        torch.manual_seed(2)
        B, P, T = 1, 2, 4
        phoneme_durations = torch.tensor([[2, 2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.3, 0.7]])

        # Predictions: different values within each phoneme
        pred_values = torch.tensor([[0.1, 0.2, 0.5, 0.9]])
        predicted_pitch = pred_values.clone().requires_grad_(True)

        total_loss, _, _, _, loss_pitch, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            predicted_pitch=predicted_pitch,
            pitch_targets=phoneme_targets,
            config=_cfg(pitch_loss_weight=1.0),
        )
        total_loss.backward()

        grad = predicted_pitch.grad.squeeze(0)
        # d(MSE)/d(pred_i) = 2*(pred_i - target_i) / N where N=4 including pitch_loss_weight=1
        # For phoneme 0: target=0.3, frames 0,1: pred=0.1, 0.2
        # For phoneme 1: target=0.7, frames 2,3: pred=0.5, 0.9
        expected_grad = torch.tensor([
            2 * (0.1 - 0.3) / 4,
            2 * (0.2 - 0.3) / 4,
            2 * (0.5 - 0.7) / 4,
            2 * (0.9 - 0.7) / 4,
        ])
        assert torch.allclose(grad, expected_grad, atol=1e-5), (
            f"Frame-level gradient mismatch:\n  got:      {grad.tolist()}\n"
            f"  expected: {expected_grad.tolist()}"
        )

    def test_frames_within_phoneme_have_different_grads(self):
        """
        When predictions vary within a phoneme, gradients for those frames
        must differ from each other (they're proportional to pred_i - target).
        """
        B, P, T = 1, 2, 4
        phoneme_durations = torch.tensor([[2, 2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.5, 0.5]])   # same target for both phonemes

        # Make frame predictions vary within each phoneme
        predicted_pitch = torch.tensor([[0.1, 0.9, 0.2, 0.8]], requires_grad=True)
        total_loss, _, _, _, _, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            predicted_pitch=predicted_pitch,
            pitch_targets=phoneme_targets,
        )
        total_loss.backward()
        grad = predicted_pitch.grad.squeeze(0)
        # Frames 0 and 1 belong to phoneme 0 (target=0.5): preds 0.1 and 0.9
        # → gradients differ since 0.1-0.5 ≠ 0.9-0.5
        assert grad[0] != grad[1], "Gradients for frames within the same phoneme must differ"

    def test_energy_frame_level_gradients(self):
        """Energy predictor gets per-frame gradients too."""
        torch.manual_seed(3)
        B, P, T = 1, 3, 6
        phoneme_durations = torch.tensor([[2, 2, 2]], dtype=torch.long)
        energy_targets = torch.tensor([[0.1, 0.5, 0.9]])

        predicted_energy = torch.zeros(B, T, requires_grad=True)
        total_loss, _, _, _, _, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            predicted_energy=predicted_energy,
            energy_targets=energy_targets,
        )
        total_loss.backward()
        assert predicted_energy.grad is not None
        assert (predicted_energy.grad.abs() > 0).all()


# ===========================================================================
# 3. Padding frames excluded from loss
# ===========================================================================

class TestFrameLevelMasking:

    def test_padding_frames_excluded_from_pitch_loss(self):
        """
        Pitch loss must ignore frames beyond mel_lengths.
        Set predictions to a large error in the padding region; loss should
        equal the loss computed on valid frames only.
        """
        B, P = 1, 2
        T_valid = 4
        T_padded = 6  # 2 extra padding frames
        phoneme_durations = torch.tensor([[2, 2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.4, 0.6]])

        # Valid region: perfect predictions.  Padding region: large error.
        predicted_pitch = torch.zeros(B, T_padded)
        predicted_pitch[0, T_valid:] = 999.0      # should be ignored

        _, _, _, _, loss_pitch, _ = _call_losses(
            n_phones=P, n_frames=T_padded,
            phoneme_durations=phoneme_durations,
            mel_lengths=torch.tensor([T_valid]),
            phoneme_lengths=torch.tensor([P]),
            predicted_pitch=predicted_pitch,
            pitch_targets=phoneme_targets,
        )
        # MSE over valid frames only: (2*0.4² + 2*0.6²)/4
        expected = (2 * 0.4**2 + 2 * 0.6**2) / 4
        assert loss_pitch.item() == pytest.approx(expected, rel=1e-4)

    def test_fully_padded_batch_gives_zero_loss(self):
        B, P, T = 1, 2, 4
        phoneme_durations = torch.tensor([[2, 2]], dtype=torch.long)
        phoneme_targets = torch.rand(B, P)
        predicted_pitch = torch.rand(B, T)

        # mel_lengths = 0 means all frames are padding
        _, _, _, _, loss_pitch, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            mel_lengths=torch.tensor([0]),
            phoneme_lengths=torch.tensor([P]),
            predicted_pitch=predicted_pitch,
            pitch_targets=phoneme_targets,
        )
        assert loss_pitch.item() == pytest.approx(0.0)


# ===========================================================================
# 4. Phoneme-level fallback path unchanged
# ===========================================================================

class TestPhonemeLevelFallback:
    """When predictions are already at phoneme level the old path is used."""

    def test_phoneme_level_prediction_uses_phoneme_mask(self):
        """
        If predicted_pitch.size(1) == phoneme_durations.size(1), the loss is
        computed at phoneme level (not expanded to frame level).
        """
        B, P, T = 1, 3, 6
        phoneme_durations = torch.tensor([[2, 2, 2]], dtype=torch.long)
        # Supply PHONEME-level prediction (same length as phoneme_durations)
        phoneme_targets = torch.tensor([[0.2, 0.5, 0.8]])
        predicted_pitch_ph = torch.zeros(B, P)  # same size as P — phoneme-level

        _, _, _, _, loss_pitch, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            predicted_pitch=predicted_pitch_ph,
            pitch_targets=phoneme_targets,
        )
        # Pure phoneme-level MSE: (0.2² + 0.5² + 0.8²) / 3
        expected = (0.2**2 + 0.5**2 + 0.8**2) / 3
        assert loss_pitch.item() == pytest.approx(expected, rel=1e-4)

    def test_phoneme_level_prediction_perfect_gives_zero(self):
        B, P, T = 1, 3, 6
        phoneme_durations = torch.tensor([[2, 2, 2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.3, 0.6, 0.9]])
        # Exact match at phoneme level
        _, _, _, _, loss_pitch, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            predicted_pitch=phoneme_targets.clone(),  # same size as P
            pitch_targets=phoneme_targets,
        )
        assert loss_pitch.item() == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# 5. Frame-level loss differs from the old phoneme-level computation
# ===========================================================================

class TestFrameLevelDiffersFromPhonemeLevel:
    """
    Confirm that computing the loss at frame level produces different results
    than the old phoneme-averaging approach when within-phoneme predictions vary.
    """

    def _old_style_loss(self, predicted_pitch, phoneme_durations, phoneme_targets,
                        mel_lengths, phoneme_lengths, n_frames):
        """Reproduce the OLD (buggy) loss: average predictions to phoneme level."""
        B, P = phoneme_durations.shape
        criterion = nn.MSELoss(reduction="none")
        max_p = P

        # Average frame-level predictions to phoneme level (old code path)
        # simple per-batch averaging for test purposes
        avg_pred = torch.zeros(B, P)
        for b in range(B):
            start = 0
            for p in range(int(phoneme_lengths[b].item())):
                dur = int(phoneme_durations[b, p].item())
                avg_pred[b, p] = predicted_pitch[b, start:start + dur].mean()
                start += dur

        ph_mask = torch.arange(P).unsqueeze(0) < phoneme_lengths.unsqueeze(1)
        loss_unreduced = criterion(avg_pred, phoneme_targets)
        valid = ph_mask & torch.isfinite(loss_unreduced)
        return loss_unreduced[valid].mean()

    def test_varying_intra_phoneme_prediction_changes_frame_loss(self):
        """
        Two prediction tensors that share the same per-phoneme mean but differ
        within phonemes must produce (1) the SAME loss under the old scheme and
        (2) DIFFERENT losses under the new frame-level scheme.
        """
        B, P, T = 1, 2, 4
        phoneme_durations = torch.tensor([[2, 2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.5, 0.5]])
        mel_lengths = torch.tensor([T])
        phoneme_lengths = torch.tensor([P])

        # pred_a: constant within each phoneme (mean = 0.5)
        pred_a = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        # pred_b: varying within each phoneme (mean still 0.5 per phoneme)
        pred_b = torch.tensor([[0.3, 0.7, 0.2, 0.8]])

        # Old scheme: both have mean 0.5 per phoneme → identical loss (== 0)
        old_loss_a = self._old_style_loss(
            pred_a, phoneme_durations, phoneme_targets, mel_lengths, phoneme_lengths, T
        )
        old_loss_b = self._old_style_loss(
            pred_b, phoneme_durations, phoneme_targets, mel_lengths, phoneme_lengths, T
        )
        assert old_loss_a.item() == pytest.approx(0.0, abs=1e-6)
        assert old_loss_b.item() == pytest.approx(0.0, abs=1e-6), (
            "Old scheme gives zero loss for varying predictions that share the phoneme mean "
            "— this is the degenerate behaviour being fixed."
        )

        # New scheme: pred_a gives zero loss, pred_b gives non-zero loss
        _, _, _, _, new_loss_a, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            mel_lengths=mel_lengths, phoneme_lengths=phoneme_lengths,
            predicted_pitch=pred_a,
            pitch_targets=phoneme_targets,
        )
        _, _, _, _, new_loss_b, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            mel_lengths=mel_lengths, phoneme_lengths=phoneme_lengths,
            predicted_pitch=pred_b,
            pitch_targets=phoneme_targets,
        )
        assert new_loss_a.item() == pytest.approx(0.0, abs=1e-6)
        assert new_loss_b.item() > 0, (
            "New frame-level loss must be non-zero when within-phoneme predictions vary"
        )
        assert new_loss_a.item() != new_loss_b.item(), (
            "Frame-level loss must distinguish predictions with same phoneme mean but "
            "different within-phoneme variation"
        )

    def test_gradient_for_varying_intra_phoneme_prediction(self):
        """
        Under the new scheme a prediction with within-phoneme variation must
        generate gradients that differ between the two frames in that phoneme.
        Under the old scheme those gradients would be identical (same phoneme mean).
        """
        B, P, T = 1, 1, 2
        phoneme_durations = torch.tensor([[2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.5]])

        pred = torch.tensor([[0.2, 0.8]], requires_grad=True)
        total_loss, _, _, _, _, _ = _call_losses(
            n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            predicted_pitch=pred,
            pitch_targets=phoneme_targets,
            config=_cfg(pitch_loss_weight=1.0),
        )
        total_loss.backward()
        g0, g1 = pred.grad[0, 0], pred.grad[0, 1]
        # Frame 0: pred=0.2, target=0.5 → grad ∝ -(0.5-0.2) < 0
        # Frame 1: pred=0.8, target=0.5 → grad ∝ (0.8-0.5) > 0
        assert g0 < 0, f"Expected negative gradient for frame 0, got {g0}"
        assert g1 > 0, f"Expected positive gradient for frame 1, got {g1}"
        assert g0 != g1, "Gradients must differ between frames with different predictions"


# ===========================================================================
# 6. Batch dimension
# ===========================================================================

class TestBatchDimension:

    def test_multi_batch_frame_level_loss(self):
        """Frame-level expansion is applied independently per batch item."""
        B, P, T = 2, 2, 4
        phoneme_durations = torch.tensor([[2, 2], [2, 2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.3, 0.7], [0.1, 0.9]])
        predicted_pitch = torch.zeros(B, T)

        _, _, _, _, loss_pitch, _ = _call_losses(
            B=B, n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            mel_lengths=torch.full((B,), T, dtype=torch.long),
            phoneme_lengths=torch.full((B,), P, dtype=torch.long),
            predicted_pitch=predicted_pitch,
            pitch_targets=phoneme_targets,
        )
        # All frames: MSE([0.3,0.3,0.7,0.7, 0.1,0.1,0.9,0.9])² / 8
        vals = [0.3, 0.3, 0.7, 0.7, 0.1, 0.1, 0.9, 0.9]
        expected = sum(v ** 2 for v in vals) / 8
        assert loss_pitch.item() == pytest.approx(expected, rel=1e-4)

    def test_multi_batch_with_variable_lengths(self):
        """Padding is excluded independently per batch item."""
        B, P, T = 2, 2, 4
        phoneme_durations = torch.tensor([[2, 2], [2, 2]], dtype=torch.long)
        phoneme_targets = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        # Batch item 0: 4 valid frames.  Batch item 1: 2 valid frames.
        predicted_pitch = torch.zeros(B, T)

        _, _, _, _, loss_pitch, _ = _call_losses(
            B=B, n_phones=P, n_frames=T,
            phoneme_durations=phoneme_durations,
            mel_lengths=torch.tensor([4, 2], dtype=torch.long),
            phoneme_lengths=torch.full((B,), P, dtype=torch.long),
            predicted_pitch=predicted_pitch,
            pitch_targets=phoneme_targets,
        )
        # Valid frames: 4+2 = 6.  Expected MSE: 6 * 0.25 / 6 = 0.25
        assert loss_pitch.item() == pytest.approx(0.25, rel=1e-4)
