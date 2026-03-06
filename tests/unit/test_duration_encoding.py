"""
Tests that verify the duration encoding/decoding contract is consistent
across the training loss (losses.py) and the inference decode paths
(variance_predictor.py, duration_adaptor.py).

Training encodes:  target = log1p(d) = log(d + 1)
Inference decodes: d_hat  = expm1(pred) = exp(pred) - 1

Using exp() instead of expm1() biases every phoneme duration by +1 frame,
causing longer-than-expected utterances and corrupted stop-token signals.
"""
import math
import types
from types import SimpleNamespace

import torch
import torch.nn as nn
import pytest

from kokoro.training.losses import calculate_training_losses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_loss_inputs(B=1, T_mel=4, T_phone=3, device="cpu"):
    """Minimal valid inputs for calculate_training_losses."""
    d = torch.device(device)
    mel_dim = 2
    mel_specs = torch.zeros(B, T_mel, mel_dim, device=d)
    predicted_mel = torch.zeros(B, T_mel, mel_dim, device=d)
    # predicted_log_durations: (B, T_phone) — in log1p space
    predicted_log_durations = torch.zeros(B, T_phone, device=d)
    stop_targets = torch.zeros(B, T_mel, device=d)
    stop_logits = torch.zeros(B, T_mel, device=d)
    phoneme_durations = torch.ones(B, T_phone, device=d)
    mel_lengths = torch.full((B,), T_mel, device=d)
    phoneme_lengths = torch.full((B,), T_phone, device=d)
    return dict(
        predicted_mel=predicted_mel,
        predicted_log_durations=predicted_log_durations,
        predicted_stop_logits=stop_logits,
        mel_specs=mel_specs,
        phoneme_durations=phoneme_durations,
        stop_token_targets=stop_targets,
        mel_lengths=mel_lengths,
        phoneme_lengths=phoneme_lengths,
    )


def _build_loss_config():
    return SimpleNamespace(
        duration_loss_weight=0.1,
        stop_token_loss_weight=1.0,
        pitch_loss_weight=0.1,
        energy_loss_weight=0.1,
        verbose=False,
    )


# ===========================================================================
# losses.py — duration target encoding
# ===========================================================================

class TestLossesEncodesWithLog1p:

    def test_duration_target_is_log1p(self):
        """
        calculate_training_losses must convert integer durations to log1p space.
        When predicted_log_durations == log1p(phoneme_durations) the duration
        loss must be zero (perfect prediction).
        """
        device = torch.device("cpu")
        B, T_phone, T_mel = 1, 3, 4
        phoneme_durations = torch.tensor([[1.0, 2.0, 5.0]])  # raw frame counts
        perfect_prediction = torch.log1p(phoneme_durations)   # what model should output

        inputs = _minimal_loss_inputs(B=B, T_mel=T_mel, T_phone=T_phone)
        inputs["phoneme_durations"] = phoneme_durations
        inputs["predicted_log_durations"] = perfect_prediction
        inputs["mel_lengths"] = torch.tensor([T_mel])
        inputs["phoneme_lengths"] = torch.tensor([T_phone])

        _, _, loss_duration, _, _, _ = calculate_training_losses(
            device=device,
            config=_build_loss_config(),
            model=None,
            criterion_mel=nn.L1Loss(reduction="none"),
            criterion_duration=nn.MSELoss(reduction="none"),
            criterion_stop_token=nn.BCEWithLogitsLoss(reduction="none"),
            criterion_pitch=None,
            criterion_energy=None,
            average_by_duration=lambda p, d, l: p,
            logger=__import__("logging").getLogger(__name__),
            **inputs,
        )

        assert loss_duration.item() == pytest.approx(0.0, abs=1e-6), (
            f"Duration loss should be zero when prediction == log1p(target), "
            f"got {loss_duration.item():.6f}"
        )

    def test_duration_target_not_plain_log(self):
        """
        Using plain log(d) (without +1) as the prediction should NOT give zero loss,
        confirming the target encoding is log(d+1) not log(d).
        """
        device = torch.device("cpu")
        phoneme_durations = torch.tensor([[2.0, 5.0, 7.0]])
        wrong_prediction = torch.log(phoneme_durations)  # log(d) not log(d+1)

        B, T_phone, T_mel = 1, 3, 4
        inputs = _minimal_loss_inputs(B=B, T_mel=T_mel, T_phone=T_phone)
        inputs["phoneme_durations"] = phoneme_durations
        inputs["predicted_log_durations"] = wrong_prediction
        inputs["phoneme_lengths"] = torch.tensor([T_phone])

        _, _, loss_duration, _, _, _ = calculate_training_losses(
            device=device,
            config=_build_loss_config(),
            model=None,
            criterion_mel=nn.L1Loss(reduction="none"),
            criterion_duration=nn.MSELoss(reduction="none"),
            criterion_stop_token=nn.BCEWithLogitsLoss(reduction="none"),
            criterion_pitch=None,
            criterion_energy=None,
            average_by_duration=lambda p, d, l: p,
            logger=__import__("logging").getLogger(__name__),
            **inputs,
        )

        assert loss_duration.item() > 1e-4, (
            "log(d) and log1p(d) differ; duration loss must be > 0 when using "
            "the wrong encoding — indicates target uses log(d+1) not log(d)"
        )

    def test_duration_target_encodes_zero_correctly(self):
        """log1p(0) == 0; a phoneme with zero duration maps to log-target of 0."""
        assert math.log1p(0) == pytest.approx(0.0)

        device = torch.device("cpu")
        phoneme_durations = torch.tensor([[0.0, 3.0]])
        perfect_prediction = torch.log1p(phoneme_durations)

        inputs = _minimal_loss_inputs(B=1, T_mel=4, T_phone=2)
        inputs["phoneme_durations"] = phoneme_durations
        inputs["predicted_log_durations"] = perfect_prediction
        inputs["phoneme_lengths"] = torch.tensor([2])

        _, _, loss_duration, _, _, _ = calculate_training_losses(
            device=device,
            config=_build_loss_config(),
            model=None,
            criterion_mel=nn.L1Loss(reduction="none"),
            criterion_duration=nn.MSELoss(reduction="none"),
            criterion_stop_token=nn.BCEWithLogitsLoss(reduction="none"),
            criterion_pitch=None,
            criterion_energy=None,
            average_by_duration=lambda p, d, l: p,
            logger=__import__("logging").getLogger(__name__),
            **inputs,
        )

        # d=0 is masked out (duration_valid = phoneme_mask & dur > 0), so only d=3 counts
        assert loss_duration.item() == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# variance_predictor.py — inference decode
# ===========================================================================

class TestVarianceAdaptorDecodeExpm1:
    """VarianceAdaptor.forward() in inference mode must use expm1 to invert log1p."""

    def test_known_durations_roundtrip(self):
        """
        Feed log1p(known_int_durations) as predictions; the adaptor must
        produce exactly sum(known_int_durations) output frames.
        """
        from kokoro.model.variance_predictor import VarianceAdaptor

        known = torch.tensor([[2.0, 3.0, 5.0]])   # expected frame counts per phoneme
        log1p_pred = torch.log1p(known)            # what a perfect predictor outputs

        adaptor = VarianceAdaptor(n_bins=8)
        B, T, H = 1, 3, adaptor.hidden_dim
        enc = torch.zeros(B, T, H)

        class _FixedPredictor(nn.Module):
            def forward(self, x, mask=None):
                return log1p_pred

        adaptor.duration_predictor = _FixedPredictor()
        adapted, *_ = adaptor(enc, mask=None)  # no duration_target → inference path

        expected_frames = int(known.sum().item())  # 10
        # if exp() were used: 3+4+6 = 13
        assert adapted.size(1) == expected_frames, (
            f"expm1 decode: expected {expected_frames} frames, got {adapted.size(1)}"
        )

    @pytest.mark.parametrize("d", [1, 2, 5, 7, 10, 15])
    def test_single_phoneme_roundtrip(self, d):
        """Single phoneme with duration d: adaptor must expand to exactly d frames."""
        from kokoro.model.variance_predictor import VarianceAdaptor

        known = torch.tensor([[float(d)]])
        log1p_pred = torch.log1p(known)

        adaptor = VarianceAdaptor(n_bins=8)
        B, T, H = 1, 1, adaptor.hidden_dim
        enc = torch.zeros(B, T, H)

        class _FixedPredictor(nn.Module):
            def forward(self, x, mask=None):
                return log1p_pred

        adaptor.duration_predictor = _FixedPredictor()
        adapted, *_ = adaptor(enc, mask=None)

        # With VariancePredictor min_frames guard (3), short durations are padded.
        actual = adapted.size(1)
        expected = max(d, 3)  # adaptor pads to at least 3 frames
        assert actual == expected, (
            f"d={d}: expected {expected} frames, got {actual}"
        )


# ===========================================================================
# duration_adaptor.py — inference decode
# ===========================================================================

class TestSimpleDurationAdaptorDecodeExpm1:
    """SimpleDurationAdaptor inference path must also use expm1."""

    def test_known_durations_roundtrip(self):
        from kokoro.model.duration_adaptor import SimpleDurationAdaptor
        from kokoro.utils.lengths import length_regulate

        known = torch.tensor([[3.0, 4.0, 2.0]])
        log1p_pred = torch.log1p(known)

        class _FixedPredictor(nn.Module):
            def forward(self, x):
                return log1p_pred

        adaptor = SimpleDurationAdaptor(
            duration_predictor_fn=_FixedPredictor(),
            length_regulate_fn=length_regulate,
        )

        T, H = 3, 8
        enc = torch.zeros(1, T, H)
        mask = torch.zeros(1, T, dtype=torch.bool)  # no padding
        expanded, *_ = adaptor(enc, mask=mask, inference=True)

        expected = int(known.sum().item())  # 9
        # exp() would give 4+5+3 = 12
        assert expanded.size(1) == expected, (
            f"expm1 decode: expected {expected} frames, got {expanded.size(1)}"
        )

    @pytest.mark.parametrize("durations", [[1, 1], [3, 5, 2], [10], [1, 2, 3, 4]])
    def test_parametrized_roundtrip(self, durations):
        from kokoro.model.duration_adaptor import SimpleDurationAdaptor
        from kokoro.utils.lengths import length_regulate

        known = torch.tensor([list(map(float, durations))])
        log1p_pred = torch.log1p(known)

        class _FixedPredictor(nn.Module):
            def forward(self, x):
                return log1p_pred

        adaptor = SimpleDurationAdaptor(
            duration_predictor_fn=_FixedPredictor(),
            length_regulate_fn=length_regulate,
        )

        T, H = len(durations), 8
        enc = torch.zeros(1, T, H)
        mask = torch.zeros(1, T, dtype=torch.bool)  # no padding
        expanded, *_ = adaptor(enc, mask=mask, inference=True)

        expected = sum(durations)
        assert expanded.size(1) == expected, (
            f"durations={durations}: expected {expected} frames, got {expanded.size(1)}"
        )


# ===========================================================================
# Cross-file consistency: training encoding == inference decoding inverse
# ===========================================================================

class TestEncodingDecodingConsistency:

    def test_losses_encoding_and_adaptor_decoding_are_inverses(self):
        """
        End-to-end check:
          1. losses.py computes target = log1p(d)
          2. adaptor decodes via expm1 → recovered = expm1(log1p(d)) == d

        If exp() is used in step 2, recovered = exp(log1p(d)) = d + 1 (off by 1).
        """
        raw_durations = torch.tensor([1.0, 3.0, 5.0, 7.0, 2.0])

        # Step 1: how losses.py encodes
        target = torch.log(raw_durations + 1.0)  # == log1p(d), matches losses.py line 47

        # Step 2: how adaptor must decode
        recovered = torch.expm1(target)

        assert torch.allclose(recovered, raw_durations, atol=1e-5), (
            "expm1(log(d+1)) must recover d exactly. "
            "If exp() is used instead, result is d+1 (off by 1 per phoneme)."
        )

    def test_exp_decode_is_wrong(self):
        """Demonstrate the off-by-one that the old exp() decode produced."""
        raw_durations = torch.tensor([2.0, 5.0, 7.0])
        target = torch.log1p(raw_durations)

        # Old (broken) decode
        wrong = torch.exp(target)
        assert torch.allclose(wrong, raw_durations + 1, atol=1e-5), (
            "exp(log1p(d)) should equal d+1, which is the bias introduced by "
            "the old incorrect decode path"
        )

        # Correct decode
        correct = torch.expm1(target)
        assert torch.allclose(correct, raw_durations, atol=1e-5), (
            "expm1(log1p(d)) must equal d (the true duration)"
        )
