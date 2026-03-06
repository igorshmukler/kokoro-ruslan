"""
Tests for training/eval mode management in KokoroModel.forward() and
forward_inference().

The core invariant: forward() and forward_inference() must NEVER call
self.train() or self.eval(). Mode is owned by the caller (trainer,
validation loop, inference script).  Violating this means that:
  - validate_epoch()'s .eval() is overridden before every batch, so
    dropout / stochastic-depth contaminate val metrics.
  - Early stopping and checkpoint-saving decisions become unreliable.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, call

from kokoro.model.model import KokoroModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_model(**kwargs) -> KokoroModel:
    """Minimal KokoroModel for mode tests (small dims, no variance predictor)."""
    defaults = dict(
        vocab_size=20,
        mel_dim=16,
        hidden_dim=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_heads=2,
        encoder_ff_dim=64,
        decoder_ff_dim=64,
        encoder_dropout=0.1,
        decoder_input_dropout=0.1,
        use_variance_predictor=False,
        gradient_checkpointing=False,
        use_stress_embedding=False,
        use_stochastic_depth=False,
    )
    defaults.update(kwargs)
    return KokoroModel(**defaults)


def _training_batch(model: KokoroModel, B: int = 1, T_phone: int = 4, T_mel: int = 4):
    """Minimal teacher-forcing batch.

    Durations are all-ones so they sum exactly to T_mel (== T_phone), avoiding
    expand/pad branches that could throw shape errors and obscure mode bugs.
    """
    assert T_phone == T_mel, "Keep T_phone == T_mel so durations sum == T_mel"
    phonemes = torch.randint(1, model.vocab_size, (B, T_phone))
    durations = torch.ones(B, T_phone, dtype=torch.long)  # sum = T_phone = T_mel
    mel = torch.randn(B, T_mel, model.mel_dim)
    stop = torch.zeros(B, T_mel)
    return phonemes, mel, durations, stop


# ===========================================================================
# forward() must not mutate mode
# ===========================================================================

class TestForwardDoesNotChangeModeInTraining:
    """When forward() is called with mel_specs (training path), it must leave
    the model in whatever mode it was in before the call."""

    def test_stays_eval_when_called_in_eval_mode(self):
        """Simulates validate_epoch: model is in eval, forward() must not flip it."""
        model = _make_model()
        model.eval()
        assert not model.training, "pre-condition: model must be in eval"

        phonemes, mel, dur, stop = _training_batch(model)
        with torch.no_grad():
            try:
                model(phonemes, mel, dur, stop)
            except Exception:
                pass  # shape mismatches are acceptable; mode flip is not

        assert not model.training, (
            "forward() with mel_specs must NOT call self.train() — "
            "doing so overrides eval() set by validate_epoch()"
        )

    def test_stays_train_when_called_in_train_mode(self):
        """Normal training path: model stays in train mode."""
        model = _make_model()
        model.train()
        assert model.training, "pre-condition: model must be in train"

        phonemes, mel, dur, stop = _training_batch(model)
        try:
            model(phonemes, mel, dur, stop)
        except Exception:
            pass

        assert model.training, "forward() must not flip train → eval"


class TestForwardDoesNotChangeModeInference:
    """When forward() is called without mel_specs (inference dispatch), it must
    not change mode — the caller owns the lifecycle."""

    def test_stays_eval_when_inference_called_in_eval_mode(self):
        model = _make_model()
        model.eval()

        phonemes = torch.randint(1, model.vocab_size, (1, 4))
        with torch.no_grad():
            try:
                model(phonemes)  # no mel_specs → inference path
            except Exception:
                pass

        assert not model.training, (
            "forward() without mel_specs must NOT call self.eval() inside — "
            "it should already be eval if caller set it"
        )

    def test_stays_train_when_inference_called_in_train_mode(self):
        """Edge case: if caller somehow calls inference path in train mode,
        the model must not silently switch to eval."""
        model = _make_model()
        model.train()

        phonemes = torch.randint(1, model.vocab_size, (1, 4))
        with torch.no_grad():
            try:
                model(phonemes)
            except Exception:
                pass

        assert model.training, (
            "forward() (inference path) must not switch train → eval without caller consent"
        )


# ===========================================================================
# forward_inference() must not mutate mode
# ===========================================================================

class TestForwardInferenceDoesNotChangeMode:

    def test_stays_eval_mode(self):
        model = _make_model()
        model.eval()

        phonemes = torch.randint(1, model.vocab_size, (1, 4))
        with torch.no_grad():
            try:
                model.forward_inference(phonemes)
            except Exception:
                pass

        assert not model.training, "forward_inference() must not call self.eval() (redundant + side-effects)"

    def test_stays_train_mode(self):
        """If a caller (e.g. a custom loop) calls forward_inference in train mode,
        the model must not silently switch."""
        model = _make_model()
        model.train()

        phonemes = torch.randint(1, model.vocab_size, (1, 4))
        with torch.no_grad():
            try:
                model.forward_inference(phonemes)
            except Exception:
                pass

        assert model.training, "forward_inference() must not override the caller's train mode"


# ===========================================================================
# No self.train() / self.eval() calls inside forward or forward_inference
# ===========================================================================

class TestNoModeCallsInsideForward:
    """Monkey-patch .train() / .eval() to detect any calls made during a
    forward pass and assert none occur."""

    def _spy_mode_calls(self, model):
        original_train = model.train
        original_eval = model.eval
        calls = []

        def spy_train(mode=True):
            calls.append(('train', mode))
            return original_train(mode)

        def spy_eval():
            calls.append(('eval',))
            return original_eval()

        model.train = spy_train
        model.eval = spy_eval
        return calls

    def test_no_mode_call_during_training_forward(self):
        model = _make_model()
        model.train()
        logged = self._spy_mode_calls(model)

        phonemes, mel, dur, stop = _training_batch(model)
        with torch.no_grad():
            try:
                model(phonemes, mel, dur, stop)
            except Exception:
                pass

        mode_mutations = [c for c in logged if c != ('train', True)]
        assert not mode_mutations, (
            f"forward() (training path) called mode-changing methods: {mode_mutations}"
        )

    def test_no_mode_call_during_inference_forward(self):
        model = _make_model()
        model.eval()
        logged = self._spy_mode_calls(model)

        phonemes = torch.randint(1, model.vocab_size, (1, 4))
        with torch.no_grad():
            try:
                model(phonemes)
            except Exception:
                pass

        assert not logged, (
            f"forward() (inference path) unexpectedly called: {logged}"
        )

    def test_no_mode_call_during_forward_inference_direct(self):
        model = _make_model()
        model.eval()
        logged = self._spy_mode_calls(model)

        phonemes = torch.randint(1, model.vocab_size, (1, 4))
        with torch.no_grad():
            try:
                model.forward_inference(phonemes)
            except Exception:
                pass

        assert not logged, (
            f"forward_inference() unexpectedly called: {logged}"
        )


# ===========================================================================
# Dropout is disabled in eval mode (forward_training called explicitly)
# ===========================================================================

class TestDropoutDisabledInEvalMode:
    """
    With the mode bug fixed, calling forward_training in eval mode must
    produce deterministic (dropout-free) outputs.
    Two identical forward passes in eval mode must return the same result.
    In train mode the outputs should differ across calls (stochastic dropout).
    """

    def test_eval_mode_forward_training_is_deterministic(self):
        torch.manual_seed(0)
        model = _make_model(encoder_dropout=0.5, decoder_input_dropout=0.5)
        model.eval()

        phonemes, mel, dur, stop = _training_batch(model, T_phone=5, T_mel=5)
        with torch.no_grad():
            out1 = model.forward_training(phonemes, mel, dur, stop)
            out2 = model.forward_training(phonemes, mel, dur, stop)

        mel1 = out1[0]
        mel2 = out2[0]
        assert torch.allclose(mel1, mel2, atol=1e-6), (
            "forward_training() in eval mode must be deterministic — "
            "dropout should be inactive"
        )

    def test_train_mode_forward_training_can_differ(self):
        """Sanity check: with high dropout in train mode, two passes differ."""
        torch.manual_seed(1)
        model = _make_model(encoder_dropout=0.5, decoder_input_dropout=0.5)
        model.train()

        phonemes, mel, dur, stop = _training_batch(model, T_phone=5, T_mel=5)
        with torch.no_grad():
            out1 = model.forward_training(phonemes, mel, dur, stop)
            out2 = model.forward_training(phonemes, mel, dur, stop)

        mel1 = out1[0]
        mel2 = out2[0]
        # Not guaranteed to differ every time, but with p=0.5 dropout over
        # multiple layers it would be extraordinarily unlikely to coincide.
        assert not torch.allclose(mel1, mel2, atol=1e-6), (
            "Sanity: with dropout=0.5 in train mode, two forward passes should differ"
        )


# ===========================================================================
# Validate-epoch simulation: mode must be eval for every batch
# ===========================================================================

class TestValidateEpochModePersistence:
    """
    Simulate the validate_epoch loop: set eval once before the loop and
    verify the model remains in eval mode throughout multiple forward passes
    with mel_specs (teacher-forcing).
    """

    def test_model_stays_eval_across_multiple_val_batches(self):
        model = _make_model()
        model.eval()

        for _ in range(5):
            phonemes, mel, dur, stop = _training_batch(model)
            assert not model.training, "Model must remain in eval mode before each batch"
            with torch.no_grad():
                try:
                    model(phonemes, mel, dur, stop)
                except Exception:
                    pass
            assert not model.training, (
                "forward() must not flip model back to train mode between val batches"
            )

    def test_eval_mode_is_not_leaked_into_submodules(self):
        """
        All submodules must stay in eval mode too — .eval() on the top-level
        module propagates recursively, so any subsequent .train() call would
        flip them all back.
        """
        model = _make_model()
        model.eval()

        phonemes, mel, dur, stop = _training_batch(model)
        with torch.no_grad():
            try:
                model(phonemes, mel, dur, stop)
            except Exception:
                pass

        for name, module in model.named_modules():
            assert not module.training, (
                f"Submodule '{name}' was switched to train mode by forward()"
            )
