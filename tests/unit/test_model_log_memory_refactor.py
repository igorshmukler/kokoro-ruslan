"""
Tests for the _log_memory() helper refactor in KokoroModel.

Refactor summary:
  - Before: each profiling site did `if self.enable_profiling: self.profiler.log_memory_stats(stage)` inline
  - After: all 18 sites call `self._log_memory(stage)`, which encapsulates the guard

Tests cover:
  1. _log_memory helper contract  — correct delegation and guard behaviour
  2. Profiling ON: all expected stage names are emitted by a full forward_inference pass
  3. Profiling OFF: profiler.log_memory_stats is NEVER called regardless of path
  4. Stage name pass-through — the helper forwards the stage string unchanged
"""

import torch
import pytest
from unittest.mock import MagicMock, call, patch

from kokoro.model.model import KokoroModel


# ---------------------------------------------------------------------------
# Shared tiny model factories
# ---------------------------------------------------------------------------

def _tiny_model(profiling: bool = False) -> KokoroModel:
    """Return a minimal KokoroModel suitable for fast unit tests."""
    return KokoroModel(
        vocab_size=20,
        mel_dim=4,
        hidden_dim=16,
        n_encoder_layers=2,          # 2 layers so encoder checkpointing segment logic fires
        n_heads=2,
        encoder_ff_dim=32,
        encoder_dropout=0.0,
        n_decoder_layers=1,
        decoder_ff_dim=32,
        max_decoder_seq_len=100,
        use_variance_predictor=True,
        variance_filter_size=16,
        variance_kernel_size=3,
        n_variance_bins=8,
        pitch_min=0.0,
        pitch_max=1.0,
        energy_min=0.0,
        energy_max=1.0,
        gradient_checkpointing=False,
        use_stochastic_depth=False,
        enable_profiling=profiling,
    )


_PHONEMES = torch.tensor([[1, 2, 3, 4]])


# ---------------------------------------------------------------------------
# 1. _log_memory helper contract
# ---------------------------------------------------------------------------

class TestLogMemoryHelper:
    """Direct unit tests of the _log_memory() helper method."""

    def test_calls_profiler_when_profiling_enabled(self):
        """When enable_profiling=True, _log_memory must call profiler.log_memory_stats."""
        model = _tiny_model(profiling=True)
        model.profiler = MagicMock()

        model._log_memory("my_stage")

        model.profiler.log_memory_stats.assert_called_once_with("my_stage")

    def test_does_not_call_profiler_when_profiling_disabled(self):
        """When enable_profiling=False, _log_memory must NOT touch the profiler."""
        model = _tiny_model(profiling=False)
        model.profiler = MagicMock()

        model._log_memory("my_stage")

        model.profiler.log_memory_stats.assert_not_called()

    def test_stage_name_forwarded_unchanged(self):
        """The stage string is passed through to log_memory_stats verbatim."""
        model = _tiny_model(profiling=True)
        model.profiler = MagicMock()

        stage = "encoder_layer_4_start"
        model._log_memory(stage)

        args, _ = model.profiler.log_memory_stats.call_args
        assert args[0] == stage

    def test_multiple_calls_accumulate(self):
        """Each _log_memory call produces a separate profiler call."""
        model = _tiny_model(profiling=True)
        model.profiler = MagicMock()

        stages = ["stage_a", "stage_b", "stage_c"]
        for s in stages:
            model._log_memory(s)

        assert model.profiler.log_memory_stats.call_count == 3
        model.profiler.log_memory_stats.assert_has_calls(
            [call(s) for s in stages], any_order=False
        )


# ---------------------------------------------------------------------------
# 2. forward_inference emits all top-level stage names when profiling is ON
# ---------------------------------------------------------------------------

class TestForwardInferenceStageNames:
    """
    Verify that running forward_inference with profiling ON produces the
    expected _log_memory stage names.  We patch _log_memory with a wrapping
    mock so the real implementation still runs, while we capture call args.
    """

    def _run_and_capture_stages(self) -> list:
        model = _tiny_model(profiling=True)
        model.profiler = MagicMock()   # prevent real GPU profiler work

        captured = []

        original = model._log_memory

        def _capturing(stage):
            captured.append(stage)
            return original(stage)

        with patch.object(model, '_log_memory', side_effect=_capturing):
            model.forward_inference(_PHONEMES, max_len=5, stop_threshold=2.0)

        return captured

    def test_inference_start_emitted(self):
        stages = self._run_and_capture_stages()
        assert "inference_start" in stages

    def test_inference_pre_generation_emitted(self):
        stages = self._run_and_capture_stages()
        assert "inference_pre_generation" in stages

    def test_inference_end_emitted(self):
        stages = self._run_and_capture_stages()
        assert "inference_end" in stages

    def test_text_embedding_stages_emitted(self):
        stages = self._run_and_capture_stages()
        assert "text_embedding_start" in stages
        assert "text_embedding_end" in stages

    def test_duration_prediction_stages_emitted_when_no_variance_predictor(self):
        """
        _predict_durations is only called in the fallback path
        (use_variance_predictor=False).  Confirm its stages are emitted there.
        """
        # Build a model WITHOUT the variance predictor so _predict_durations is called
        model = KokoroModel(
            vocab_size=20, mel_dim=4, hidden_dim=16,
            n_encoder_layers=2, n_heads=2, encoder_ff_dim=32, encoder_dropout=0.0,
            n_decoder_layers=1, decoder_ff_dim=32, max_decoder_seq_len=100,
            use_variance_predictor=False,
            variance_filter_size=16, variance_kernel_size=3, n_variance_bins=8,
            pitch_min=0.0, pitch_max=1.0, energy_min=0.0, energy_max=1.0,
            gradient_checkpointing=False, use_stochastic_depth=False,
            enable_profiling=True,
        )
        model.profiler = MagicMock()

        captured = []
        original = model._log_memory

        def _capturing(stage):
            captured.append(stage)
            return original(stage)

        with patch.object(model, '_log_memory', side_effect=_capturing):
            model.forward_inference(_PHONEMES, max_len=5, stop_threshold=2.0)

        assert "duration_prediction_start" in captured, (
            f"Expected duration_prediction_start; got: {captured}"
        )
        assert "duration_prediction_end" in captured

    def test_decoder_stages_emitted_during_training_forward(self):
        """
        decoder_start / decoder_end come from _checkpoint_decoder_forward which
        is called via the training forward() path, not forward_inference().
        """
        model = _tiny_model(profiling=True)
        model.profiler = MagicMock()
        model.train()

        captured = []
        original = model._log_memory

        def _capturing(stage):
            captured.append(stage)
            return original(stage)

        mel_len = 6
        mel_specs = torch.zeros(1, mel_len, 4)
        durations = torch.ones(1, _PHONEMES.shape[1], dtype=torch.long)
        stop_targets = torch.zeros(1, mel_len)

        with patch.object(model, '_log_memory', side_effect=_capturing):
            model(_PHONEMES, mel_specs, durations, stop_targets)

        assert "decoder_start" in captured, (
            f"Expected decoder_start in training forward; got: {captured}"
        )
        assert "decoder_end" in captured

    def test_no_duplicate_inference_start(self):
        """inference_start must appear exactly once per forward call."""
        stages = self._run_and_capture_stages()
        assert stages.count("inference_start") == 1

    def test_stages_ordering(self):
        """Core stages must appear in the correct relative order."""
        stages = self._run_and_capture_stages()
        ordered = ["inference_start", "text_embedding_start", "text_embedding_end",
                   "inference_end"]
        positions = [stages.index(s) for s in ordered]
        assert positions == sorted(positions), (
            f"Expected stages to appear in order {ordered}, got positions {positions}"
        )


# ---------------------------------------------------------------------------
# 3. No profiling — profiler.log_memory_stats must NEVER be called
# ---------------------------------------------------------------------------

class TestNoProfiling:
    """Verify that when enable_profiling=False no profiler calls escape."""

    def _run_with_mock_profiler(self) -> MagicMock:
        model = _tiny_model(profiling=False)
        mock_profiler = MagicMock()
        model.profiler = mock_profiler
        model.forward_inference(_PHONEMES, max_len=5, stop_threshold=2.0)
        return mock_profiler

    def test_profiler_log_memory_never_called_in_forward_inference(self):
        mock_profiler = self._run_with_mock_profiler()
        mock_profiler.log_memory_stats.assert_not_called()

    def test_direct_log_memory_call_silent_when_disabled(self):
        model = _tiny_model(profiling=False)
        mock_profiler = MagicMock()
        model.profiler = mock_profiler
        # No exception, no profiler call
        model._log_memory("should_be_silent")
        mock_profiler.log_memory_stats.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Gradient-checkpointing path emits checkpoint-specific stage names
# ---------------------------------------------------------------------------

class TestGradientCheckpointingStages:
    """
    With gradient_checkpointing=True, _checkpoint_encoder_layers and
    _checkpoint_decoder_forward take different code paths that emit
    different stage names.  We verify those stages appear.
    """

    def _run_checkpointing_and_capture(self) -> list:
        model = _tiny_model(profiling=True)
        model.profiler = MagicMock()
        model.gradient_checkpointing = True
        model.train()   # checkpointing only applies during training

        captured = []
        original = model._log_memory

        def _capturing(stage):
            captured.append(stage)
            return original(stage)

        phonemes = _PHONEMES
        mel_len = 6
        mel_specs = torch.zeros(1, mel_len, 4)
        durations = torch.ones(1, phonemes.shape[1], dtype=torch.long)
        stop_targets = torch.zeros(1, mel_len)

        with patch.object(model, '_log_memory', side_effect=_capturing):
            model(phonemes, mel_specs, durations, stop_targets)

        return captured

    def test_encoder_segment_stages_emitted(self):
        """Checkpointing path for encoder produces encoder_segment_*_start/end stages."""
        stages = self._run_checkpointing_and_capture()
        segment_starts = [s for s in stages if s.endswith("_start") and "segment" in s]
        assert len(segment_starts) >= 1, (
            f"Expected at least one encoder_segment_*_start stage, got stages: {stages}"
        )

    def test_decoder_checkpoint_stages_emitted(self):
        """Checkpointing path for decoder produces decoder_checkpoint_start/end stages."""
        stages = self._run_checkpointing_and_capture()
        assert "decoder_checkpoint_start" in stages, (
            f"Expected decoder_checkpoint_start in stages, got: {stages}"
        )
        assert "decoder_checkpoint_end" in stages, (
            f"Expected decoder_checkpoint_end in stages, got: {stages}"
        )
