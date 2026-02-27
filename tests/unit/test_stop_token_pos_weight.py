"""
Tests for the stop-token class-imbalance fix (0.0.21).

BUG (fixed):
    BCEWithLogitsLoss was constructed without pos_weight.  Stop token targets
    are extremely imbalanced: only the final frame of each sequence is positive
    (1.0) and all other ~T frames are negative (0.0).  For a T=400 sequence
    the ratio is 399:1.  Without pos_weight the loss-minimising solution is to
    always predict logit≈0 (sigmoid≈0.5 → 0.0 after threshold), achieving
    near-zero BCE loss while never firing the stop signal during inference.

FIX:
    BCEWithLogitsLoss(reduction='none', pos_weight=tensor([pos_weight_val]))
    where pos_weight_val ≈ average sequence length (default 150.0).  This
    makes each positive frame contribute pos_weight times more to the gradient
    than a negative frame, cancelling the class imbalance.

Tests:
  1.  Config default: stop_token_pos_weight == 150.0
  2.  Config custom value is respected
  3.  Criterion pos_weight stored and accessible
  4.  Criterion pos_weight matches config value
  5.  Without pos_weight, always-predict-zero achieves near-zero stop loss
      (documents the original bug)
  6.  With pos_weight, always-predict-zero incurs significant stop loss on
      the positive frame (confirms the fix)
  7.  pos_weight scales loss on positive frames linearly
  8.  pos_weight does NOT change loss on negative frames
  9.  Gradient on positive frame is amplified by pos_weight vs negative frame
  10. _calculate_losses stop loss matches manual BCE with pos_weight
  11. _calculate_losses stop loss is zero when logits perfectly predict targets
  12. pos_weight tensor device matches the device passed during construction
  13. Total loss in _calculate_losses increases with higher pos_weight when
      the stop signal is wrong (positive frame predicted as negative)
"""

import math
import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace

from kokoro.training.config import TrainingConfig
from kokoro.training.trainer import KokoroTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stop_criterion(pos_weight: float = 150.0,
                         device: str = 'cpu') -> nn.BCEWithLogitsLoss:
    """Replicate the exact criterion construction in KokoroTrainer.__init__."""
    _pos_w = torch.tensor([pos_weight], device=torch.device(device))
    return nn.BCEWithLogitsLoss(reduction='none', pos_weight=_pos_w)


def _build_trainer_with_pos_weight(pos_weight: float = 150.0) -> KokoroTrainer:
    """Minimal trainer with the real pos_weight-corrected stop criterion."""
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.device = torch.device('cpu')
    trainer.criterion_mel = nn.L1Loss(reduction='none')
    trainer.criterion_duration = nn.MSELoss(reduction='none')
    trainer.criterion_stop_token = _make_stop_criterion(pos_weight)
    trainer.criterion_pitch = None
    trainer.criterion_energy = None
    trainer.config = SimpleNamespace(
        duration_loss_weight=0.1,
        stop_token_loss_weight=1.0,
        pitch_loss_weight=0.1,
        energy_loss_weight=0.1,
    )
    return trainer


def _imbalanced_batch(T: int = 200):
    """
    Single-sample batch with T mel frames, 1 phoneme, 1 positive stop frame.
    Used to test stop-token loss behaviour under realistic imbalance.
    """
    mel_specs          = torch.zeros(1, T, 80)
    predicted_mel      = torch.zeros(1, T, 80)
    phoneme_durations  = torch.tensor([[T]], dtype=torch.long)
    # duration predictor predicts log(T+1) so MSE loss = 0
    predicted_log_dur  = torch.log(phoneme_durations.float() + 1.0)
    stop_targets       = torch.zeros(1, T)
    stop_targets[0, -1] = 1.0           # only last frame is positive
    mel_lengths        = torch.tensor([T], dtype=torch.long)
    phoneme_lengths    = torch.tensor([1], dtype=torch.long)
    return (mel_specs, predicted_mel, phoneme_durations, predicted_log_dur,
            stop_targets, mel_lengths, phoneme_lengths)


# ---------------------------------------------------------------------------
# 1–2. TrainingConfig default and custom value
# ---------------------------------------------------------------------------
class TestTrainingConfigDefaults:

    def test_stop_token_pos_weight_default_is_150(self):
        cfg = TrainingConfig.__new__(TrainingConfig)
        # Access the raw dataclass default without calling __post_init__
        assert TrainingConfig.__dataclass_fields__['stop_token_pos_weight'].default == 150.0

    def test_stop_token_pos_weight_custom_value_survives_post_init(self):
        cfg = TrainingConfig(stop_token_pos_weight=300.0)
        assert cfg.stop_token_pos_weight == 300.0

    def test_stop_token_pos_weight_present_in_config(self):
        assert hasattr(TrainingConfig(), 'stop_token_pos_weight')


# ---------------------------------------------------------------------------
# 3–4. Criterion construction
# ---------------------------------------------------------------------------
class TestCriterionConstruction:

    def test_stop_criterion_has_pos_weight_attribute(self):
        criterion = _make_stop_criterion(150.0)
        assert criterion.pos_weight is not None, (
            "criterion_stop_token.pos_weight is None — pos_weight not set"
        )

    def test_stop_criterion_pos_weight_value(self):
        criterion = _make_stop_criterion(200.0)
        assert criterion.pos_weight.item() == pytest.approx(200.0), (
            f"pos_weight is {criterion.pos_weight.item()}, expected 200.0"
        )

    def test_stop_criterion_without_pos_weight_has_none(self):
        """Documents baseline: plain BCEWithLogitsLoss has pos_weight=None."""
        plain = nn.BCEWithLogitsLoss(reduction='none')
        assert plain.pos_weight is None

    def test_stop_criterion_pos_weight_shape(self):
        criterion = _make_stop_criterion(150.0)
        assert criterion.pos_weight.shape == (1,)


# ---------------------------------------------------------------------------
# 5. BUG DOCUMENTATION: without pos_weight, always-zero prediction ≈ free
# ---------------------------------------------------------------------------
class TestOriginalBugDocumented:

    def test_without_pos_weight_always_zero_gets_near_zero_stop_loss(self):
        """
        Demonstrates the original bug: on a 200-frame sequence with 1 positive
        frame and 199 negatives, predicting logit=0 everywhere achieves a very
        small average stop loss because the single positive frame is overwhelmed.

        Average BCE (logit=0, target=0):  log(2) ≈ 0.6931  (per negative frame)
        Average BCE (logit=0, target=1):  log(2) ≈ 0.6931  (per positive frame)
        Mean over 200 frames ≈ 0.6931 — BUT the model could reduce this to
        near-0 by predicting large negative logits for all frames since 199/200
        frames benefit, and the 1/200 positive barely matters.

        With a large negative logit (e.g. -10) everywhere:
        - target=0: BCE ≈ exp(-10) ≈ 0.0  (near zero, reward for 199 frames)
        - target=1: BCE ≈ 10.0             (penalty for 1 frame)
        Mean ≈ (199 * 0.0 + 1 * 10) / 200 ≈ 0.05  — small enough to ignore

        Without pos_weight the gradient from the 1 positive is negligible.
        """
        T = 200
        plain_criterion = nn.BCEWithLogitsLoss(reduction='none')  # no pos_weight
        logits = torch.full((1, T), -10.0)   # strong "always negative" prediction
        targets = torch.zeros(1, T)
        targets[0, -1] = 1.0

        losses = plain_criterion(logits, targets)
        avg_loss = losses.mean().item()

        # With -10 logits: 199 frames give ≈0 loss, 1 frame gives ≈10 loss
        # Average ≈ 10/200 = 0.05 — small enough to converge to "always zero"
        assert avg_loss < 0.1, (
            f"Without pos_weight, always-negative prediction has avg stop loss "
            f"{avg_loss:.4f} — this is the exploitable imbalance the fix addresses."
        )

    def test_without_pos_weight_gradient_is_dominated_by_negative_frames(self):
        """Sum of gradients from negative frames >> gradient from positive frame."""
        T = 200
        plain_criterion = nn.BCEWithLogitsLoss(reduction='none')
        logits = torch.zeros(1, T, requires_grad=True)
        targets = torch.zeros(1, T)
        targets[0, -1] = 1.0

        loss = plain_criterion(logits, targets).mean()
        loss.backward()

        grad = logits.grad[0]
        grad_positives = grad[-1:].abs().sum().item()
        grad_negatives = grad[:-1].abs().sum().item()

        # Without pos_weight: one positive frame vs 199 negative frames
        # Positive gradient should be tiny relative to summed negative gradients
        assert grad_positives / (grad_negatives + 1e-8) < 0.1, (
            f"Without pos_weight: positive frame gradient ({grad_positives:.4f}) "
            f"is NOT overwhelmed by negative frame gradients ({grad_negatives:.4f}). "
            "Expected ratio < 0.1."
        )


# ---------------------------------------------------------------------------
# 6–9. Fix verification: with pos_weight the imbalance is corrected
# ---------------------------------------------------------------------------
class TestPosWeightFixVerification:

    def test_with_pos_weight_always_zero_incurs_meaningful_stop_loss(self):
        """
        With pos_weight=150, the 1 positive frame contributes 150× more to the
        loss than each negative frame, making the average loss much larger and
        forcing the model to learn to fire the stop signal.
        """
        T = 200
        pos_weight = 150.0
        criterion = _make_stop_criterion(pos_weight)
        logits  = torch.full((1, T), -10.0)
        targets = torch.zeros(1, T)
        targets[0, -1] = 1.0

        losses  = criterion(logits, targets)
        avg_loss = losses.mean().item()

        # Positive frame: loss ≈ 150 * 10 = 1500; negatives ≈ 0
        # Average ≈ 1500 / 200 = 7.5 — not exploitable
        assert avg_loss > 1.0, (
            f"With pos_weight={pos_weight}, always-negative prediction should "
            f"incur avg stop loss > 1.0; got {avg_loss:.4f}"
        )

    def test_pos_weight_scales_positive_frame_loss_linearly(self):
        """Loss on the positive frame scales proportionally with pos_weight."""
        logit  = torch.tensor([[0.0]])   # single frame, target=1
        target = torch.tensor([[1.0]])

        losses = []
        for pw in [1.0, 10.0, 100.0]:
            c = _make_stop_criterion(pw)
            losses.append(c(logit, target).item())

        base = losses[0]
        assert losses[1] == pytest.approx(10.0 * base, rel=1e-4), (
            "pos_weight=10 should give 10× the positive-frame loss of pos_weight=1"
        )
        assert losses[2] == pytest.approx(100.0 * base, rel=1e-4), (
            "pos_weight=100 should give 100× the positive-frame loss of pos_weight=1"
        )

    def test_pos_weight_does_not_change_negative_frame_loss(self):
        """pos_weight must NOT alter loss on negative-class frames (target=0)."""
        logit  = torch.tensor([[0.0]])
        target = torch.tensor([[0.0]])

        loss_no_pw  = nn.BCEWithLogitsLoss(reduction='none')(logit, target).item()
        loss_with_pw = _make_stop_criterion(150.0)(logit, target).item()

        assert loss_no_pw == pytest.approx(loss_with_pw, rel=1e-5), (
            "pos_weight should not affect loss for target=0 frames"
        )

    def test_pos_weight_amplifies_positive_frame_gradient(self):
        """
        Gradient on the positive frame with pos_weight=W should be W times
        larger than without pos_weight.
        """
        pw = 150.0
        logit_pw    = torch.tensor([[0.0]], requires_grad=True)
        logit_plain = torch.tensor([[0.0]], requires_grad=True)
        target = torch.tensor([[1.0]])

        _make_stop_criterion(pw)(logit_pw, target).backward()
        nn.BCEWithLogitsLoss(reduction='none')(logit_plain, target).backward()

        grad_pw    = logit_pw.grad.abs().item()
        grad_plain = logit_plain.grad.abs().item()

        assert grad_pw == pytest.approx(pw * grad_plain, rel=1e-4), (
            f"Gradient with pos_weight={pw}: {grad_pw:.4f}; "
            f"expected {pw} × plain gradient ({grad_plain:.4f}) = {pw * grad_plain:.4f}"
        )

    def test_gradient_ratio_positive_to_negative_equals_pos_weight(self):
        """
        After the fix, |grad on positive frame| / |grad on negative frame|
        should equal pos_weight (both at logit=0 where sigmoid=0.5).
        """
        pw = 150.0
        T  = 2
        logits  = torch.zeros(1, T, requires_grad=True)
        targets = torch.zeros(1, T)
        targets[0, 0] = 1.0   # frame 0 positive, frame 1 negative

        criterion = _make_stop_criterion(pw)
        criterion(logits, targets).mean().backward()

        grad_pos = logits.grad[0, 0].abs().item()
        grad_neg = logits.grad[0, 1].abs().item()

        assert grad_pos / grad_neg == pytest.approx(pw, rel=1e-3), (
            f"Expected gradient ratio (positive/negative) ≈ {pw}; "
            f"got {grad_pos / grad_neg:.2f}"
        )


# ---------------------------------------------------------------------------
# 10–11. _calculate_losses integration
# ---------------------------------------------------------------------------
class TestCalculateLossesStopToken:

    def test_stop_loss_matches_manual_bce_with_pos_weight(self):
        """
        _calculate_losses stop loss should equal manually computed weighted BCE
        over the valid (non-padded) frames.
        """
        T   = 10
        pw  = 150.0
        trainer = _build_trainer_with_pos_weight(pw)

        (mel_specs, predicted_mel, phoneme_durations, predicted_log_dur,
         stop_targets, mel_lengths, phoneme_lengths) = _imbalanced_batch(T)

        # Predict logit=0 (no information) for all frames
        predicted_stop_logits = torch.zeros(1, T)

        _, _, _, loss_stop, _, _ = trainer._calculate_losses(
            predicted_mel, predicted_log_dur, predicted_stop_logits,
            mel_specs, phoneme_durations, stop_targets,
            mel_lengths, phoneme_lengths,
        )

        # Manual BCE with pos_weight
        criterion = _make_stop_criterion(pw)
        manual_losses = criterion(predicted_stop_logits, stop_targets)  # (1, T)
        expected = manual_losses[0, :T].mean().item()   # all T frames are valid

        assert loss_stop.item() == pytest.approx(expected, rel=1e-5)

    def test_stop_loss_is_zero_for_perfect_prediction(self):
        """
        If stop logits perfectly predict targets (large positive for the stop
        frame, large negative for all others), stop loss should be near zero.
        """
        T   = 20
        pw  = 150.0
        trainer = _build_trainer_with_pos_weight(pw)

        (mel_specs, predicted_mel, phoneme_durations, predicted_log_dur,
         stop_targets, mel_lengths, phoneme_lengths) = _imbalanced_batch(T)

        # Perfect predictor: very negative everywhere except last frame
        predicted_stop_logits = torch.full((1, T), -20.0)
        predicted_stop_logits[0, -1] = 20.0   # very positive for the stop frame

        _, _, _, loss_stop, _, _ = trainer._calculate_losses(
            predicted_mel, predicted_log_dur, predicted_stop_logits,
            mel_specs, phoneme_durations, stop_targets,
            mel_lengths, phoneme_lengths,
        )

        assert loss_stop.item() < 0.01, (
            f"Perfect stop predictor should have near-zero loss; got {loss_stop.item():.4f}"
        )

    def test_higher_pos_weight_gives_higher_total_loss_when_stop_wrong(self):
        """
        With a wrong stop prediction (logit=0 everywhere, but last frame should
        fire), higher pos_weight should give larger total loss via larger stop loss.
        """
        T = 50

        (mel_specs, predicted_mel, phoneme_durations, predicted_log_dur,
         stop_targets, mel_lengths, phoneme_lengths) = _imbalanced_batch(T)
        predicted_stop_logits = torch.zeros(1, T)

        losses = {}
        for pw in [1.0, 50.0, 150.0]:
            trainer = _build_trainer_with_pos_weight(pw)
            total, *_ = trainer._calculate_losses(
                predicted_mel, predicted_log_dur, predicted_stop_logits,
                mel_specs, phoneme_durations, stop_targets,
                mel_lengths, phoneme_lengths,
            )
            losses[pw] = total.item()

        assert losses[50.0] > losses[1.0], (
            "pos_weight=50 should give higher total loss than pos_weight=1 "
            "when the stop frame is predicted incorrectly"
        )
        assert losses[150.0] > losses[50.0], (
            "pos_weight=150 should give higher total loss than pos_weight=50"
        )


# ---------------------------------------------------------------------------
# 12. Device consistency
# ---------------------------------------------------------------------------
class TestPosWeightDeviceConsistency:

    def test_pos_weight_tensor_matches_requested_device(self):
        """pos_weight tensor must be on the same device as the criterion is used on."""
        # CPU only test (MPS not available in all CI environments)
        criterion = _make_stop_criterion(150.0, device='cpu')
        assert criterion.pos_weight.device.type == 'cpu'

    def test_criterion_can_compute_loss_with_cpu_tensors(self):
        criterion = _make_stop_criterion(150.0, device='cpu')
        logits  = torch.zeros(2, 10)
        targets = torch.zeros(2, 10)
        targets[:, -1] = 1.0
        loss = criterion(logits, targets)
        assert loss.shape == (2, 10)
        assert torch.isfinite(loss).all()
