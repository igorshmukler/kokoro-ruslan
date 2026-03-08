"""Unit tests for _purge_tb_events_after_step in checkpoint_manager.

Tests verify that:
  - Events at step <= keep_up_to_step are preserved.
  - Events at step > keep_up_to_step are discarded.
  - Original event files are deleted (consolidated into one new file).
  - The function returns a usable SummaryWriter.
  - Events from multiple source files are handled correctly.
  - Edge cases (empty dir, keep_up_to=0, keep_up_to above all steps) work.
  - Fallback behaviour when EventFileLoader is unavailable is graceful.
  - resume_from_checkpoint invokes the purge with current_optimizer_step.
"""

import glob
import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "tensorboard",
    reason="tensorboard must be installed for TB-purge tests",
)

# Must come after the importorskip so we don't fail on missing tensorboard.
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

from kokoro.training.checkpoint_manager import _purge_tb_events_after_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_scalars(log_dir: Path, steps: list[int], tag: str = "loss/total") -> Path:
    """Write one scalar per step to a new SummaryWriter and flush/close it.
    Returns the path to the event file created."""
    sw = SummaryWriter(log_dir=str(log_dir))
    for s in steps:
        sw.add_scalar(tag, float(s) * 0.1, global_step=s)
    sw.flush()
    sw.close()
    # Small sleep so file timestamps differ when we create multiple files.
    time.sleep(0.05)
    files = sorted(log_dir.glob("events.out.tfevents.*"))
    return files[-1]


def _read_scalar_steps(log_dir: Path, tag: str = "loss/total") -> list[int]:
    """Return sorted list of global_step values for *tag* across all event files."""
    steps = []
    for ef in sorted(log_dir.glob("events.out.tfevents.*")):
        loader = EventFileLoader(str(ef))
        for event in loader.Load():
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                if value.tag == tag:
                    steps.append(event.step)
    return sorted(steps)


def _event_file_count(log_dir: Path) -> int:
    return len(list(log_dir.glob("events.out.tfevents.*")))


# ---------------------------------------------------------------------------
# Core filtering behaviour
# ---------------------------------------------------------------------------

class TestFilterByStep:
    """Events before/at the threshold are kept; events after are purged."""

    def test_events_at_threshold_kept(self, tmp_path):
        _write_scalars(tmp_path, [100, 200, 300])
        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=300)
        writer.close()
        assert _read_scalar_steps(tmp_path) == [100, 200, 300]

    def test_events_after_threshold_purged(self, tmp_path):
        _write_scalars(tmp_path, [100, 200, 300, 400, 500])
        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=300)
        writer.close()
        assert _read_scalar_steps(tmp_path) == [100, 200, 300]

    def test_no_events_survive_when_keep_up_to_is_zero(self, tmp_path):
        """keep_up_to_step=0 removes all step>0 scalars."""
        _write_scalars(tmp_path, [1, 2, 3])
        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=0)
        writer.close()
        assert _read_scalar_steps(tmp_path) == []

    def test_all_events_kept_when_threshold_above_all(self, tmp_path):
        """keep_up_to_step larger than any written step preserves everything."""
        _write_scalars(tmp_path, [10, 20, 30])
        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=10_000)
        writer.close()
        assert _read_scalar_steps(tmp_path) == [10, 20, 30]

    def test_exact_boundary_step_included(self, tmp_path):
        _write_scalars(tmp_path, [199, 200, 201])
        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=200)
        writer.close()
        assert _read_scalar_steps(tmp_path) == [199, 200]

    def test_multiple_tags_filtered_consistently(self, tmp_path):
        sw = SummaryWriter(log_dir=str(tmp_path))
        for s in [50, 100, 150, 200]:
            sw.add_scalar("loss/mel", s * 0.01, global_step=s)
            sw.add_scalar("loss/dur", s * 0.02, global_step=s)
        sw.flush()
        sw.close()

        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=100)
        writer.close()

        mel_steps = _read_scalar_steps(tmp_path, "loss/mel")
        dur_steps = _read_scalar_steps(tmp_path, "loss/dur")
        assert mel_steps == [50, 100]
        assert dur_steps == [50, 100]


# ---------------------------------------------------------------------------
# File management
# ---------------------------------------------------------------------------

class TestFileManagement:
    """Old event files are deleted; exactly one new file remains after the call."""

    def test_original_file_deleted(self, tmp_path):
        orig = _write_scalars(tmp_path, [1, 2, 3])
        assert orig.exists()

        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=3)
        writer.close()

        assert not orig.exists(), "Original event file should have been deleted"

    def test_exactly_one_event_file_after_purge(self, tmp_path):
        _write_scalars(tmp_path, [1, 2, 3])
        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=3)
        writer.close()
        assert _event_file_count(tmp_path) == 1

    def test_multiple_source_files_consolidated_into_one(self, tmp_path):
        """Writing two separate event files then purging should leave one file."""
        sw1 = SummaryWriter(log_dir=str(tmp_path))
        sw1.add_scalar("loss/total", 1.0, global_step=100)
        sw1.flush()
        sw1.close()
        time.sleep(0.05)

        sw2 = SummaryWriter(log_dir=str(tmp_path))
        sw2.add_scalar("loss/total", 0.5, global_step=200)
        sw2.flush()
        sw2.close()

        assert _event_file_count(tmp_path) == 2

        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=200)
        writer.close()

        assert _event_file_count(tmp_path) == 1
        assert _read_scalar_steps(tmp_path) == [100, 200]

    def test_multiple_source_files_filtered_correctly(self, tmp_path):
        """Events from both source files are filtered by the threshold."""
        sw1 = SummaryWriter(log_dir=str(tmp_path))
        for s in [100, 200, 300]:
            sw1.add_scalar("loss/total", s * 0.01, global_step=s)
        sw1.flush()
        sw1.close()
        time.sleep(0.05)

        sw2 = SummaryWriter(log_dir=str(tmp_path))
        for s in [400, 500, 600]:
            sw2.add_scalar("loss/total", s * 0.01, global_step=s)
        sw2.flush()
        sw2.close()

        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=300)
        writer.close()

        assert _read_scalar_steps(tmp_path) == [100, 200, 300]


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

class TestReturnValue:
    """The function must return a usable SummaryWriter."""

    def test_returns_summary_writer(self, tmp_path):
        _write_scalars(tmp_path, [1, 2])
        result = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=2)
        assert isinstance(result, SummaryWriter)
        result.close()

    def test_returned_writer_is_writable(self, tmp_path):
        """The returned writer must accept new scalars without error."""
        _write_scalars(tmp_path, [100])
        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=100)
        writer.add_scalar("loss/total", 0.42, global_step=101)
        writer.flush()
        writer.close()
        assert 101 in _read_scalar_steps(tmp_path)

    def test_custom_summarywriter_class_used(self, tmp_path):
        """_SummaryWriter override is forwarded to the fresh writer construction."""
        mock_sw_instance = MagicMock(spec=SummaryWriter)
        mock_sw_instance.file_writer = MagicMock()
        mock_sw_class = MagicMock(return_value=mock_sw_instance)

        _write_scalars(tmp_path, [1])
        result = _purge_tb_events_after_step(
            str(tmp_path), keep_up_to_step=1, _SummaryWriter=mock_sw_class
        )

        mock_sw_class.assert_called_once_with(log_dir=str(tmp_path))
        assert result is mock_sw_instance


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Behaviour with an empty log directory or no event files."""

    def test_empty_log_dir_returns_writer(self, tmp_path):
        result = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=500)
        assert isinstance(result, SummaryWriter)
        result.close()

    def test_empty_log_dir_creates_event_file(self, tmp_path):
        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=500)
        writer.close()
        assert _event_file_count(tmp_path) == 1

    def test_non_existent_log_dir_created(self, tmp_path):
        new_dir = tmp_path / "sub" / "logs"
        writer = _purge_tb_events_after_step(str(new_dir), keep_up_to_step=0)
        assert isinstance(writer, SummaryWriter)
        writer.close()

    def test_all_events_above_threshold_leaves_empty_history(self, tmp_path):
        _write_scalars(tmp_path, [200, 300, 400])
        writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=100)
        writer.close()
        assert _read_scalar_steps(tmp_path) == []


# ---------------------------------------------------------------------------
# Fallback when tensorboard event reader is unavailable
# ---------------------------------------------------------------------------

class TestFallbackBehaviour:
    """When EventFileLoader cannot be imported, fall back gracefully (delete & reopen)."""

    def test_fallback_returns_writer_when_eventfileloader_missing(self, tmp_path):
        _write_scalars(tmp_path, [100, 200, 300])

        with patch.dict(
            sys.modules,
            {
                "tensorboard.backend.event_processing.event_file_loader": None,
                "tensorboard.compat.proto.event_pb2": None,
            },
        ):
            result = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=200)

        assert isinstance(result, SummaryWriter)
        result.close()

    def test_fallback_leaves_single_event_file(self, tmp_path):
        _write_scalars(tmp_path, [100, 200, 300])

        with patch.dict(
            sys.modules,
            {
                "tensorboard.backend.event_processing.event_file_loader": None,
                "tensorboard.compat.proto.event_pb2": None,
            },
        ):
            writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=200)

        writer.close()
        assert _event_file_count(tmp_path) == 1

    def test_fallback_old_files_deleted(self, tmp_path):
        orig = _write_scalars(tmp_path, [100])

        with patch.dict(
            sys.modules,
            {"tensorboard.backend.event_processing.event_file_loader": None},
        ):
            writer = _purge_tb_events_after_step(str(tmp_path), keep_up_to_step=100)

        writer.close()
        assert not orig.exists()


# ---------------------------------------------------------------------------
# Integration: resume_from_checkpoint invokes purge with correct step
# ---------------------------------------------------------------------------

class TestResumeCheckpointTBPurge:
    """resume_from_checkpoint must call _purge_tb_events_after_step with
    trainer.current_optimizer_step as the keep_up_to_step argument."""

    def _make_minimal_trainer(self, log_dir: Path, optimizer_step: int = 300):
        """Minimal trainer namespace sufficient for the purge path in resume_from_checkpoint."""
        from kokoro.training.checkpoint_manager import _purge_tb_events_after_step as real_fn

        trainer = SimpleNamespace(
            config=SimpleNamespace(
                resume_checkpoint="auto",
                output_dir=str(log_dir.parent),
            ),
            device=MagicMock(),
            device_type="cpu",
            model=MagicMock(),
            optimizer=MagicMock(
                state_dict=lambda: {"param_groups": [{"lr": 1e-4}]},
                param_groups=[{"lr": 1e-4}],
            ),
            scheduler=MagicMock(spec=[]),   # not OneCycleLR → skips scheduler block
            scheduler_per_batch=False,
            use_mixed_precision=False,
            scaler=None,
            use_ema=False,
            ema_model=None,
            start_epoch=0,
            best_loss=float("inf"),
            best_val_loss=float("inf"),
            current_optimizer_step=optimizer_step,
            optimizer_steps_completed=optimizer_step,
            log_dir=str(log_dir),
            writer=MagicMock(),
            dataset=SimpleNamespace(phoneme_processor=MagicMock()),
            _encoder_lr_multiplier=1.0,
            _onecycle_steps=None,
            _onecycle_max_lr=None,
        )
        return trainer

    def test_purge_called_with_current_optimizer_step(self, tmp_path):
        """_purge_tb_events_after_step must receive keep_up_to_step == current_optimizer_step."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        trainer = self._make_minimal_trainer(log_dir, optimizer_step=750)

        captured_calls = {}

        def fake_purge(log_dir, keep_up_to_step, _SummaryWriter=None):
            captured_calls["log_dir"] = log_dir
            captured_calls["keep_up_to_step"] = keep_up_to_step
            return MagicMock(spec=SummaryWriter)

        fake_load_checkpoint = MagicMock(
            return_value=(4, 0.25, MagicMock())
        )
        fake_checkpoint = {
            "current_optimizer_step": 750,
            "optimizer_steps_completed": 750,
        }

        with (
            patch(
                "kokoro.training.checkpoint_manager._purge_tb_events_after_step",
                side_effect=fake_purge,
            ),
            patch(
                "kokoro.training.checkpoint_manager.find_latest_checkpoint",
                return_value="/fake/checkpoint.pth",
            ),
            patch(
                "kokoro.training.checkpoint_manager.torch.load",
                return_value=fake_checkpoint,
            ),
        ):
            from kokoro.training.checkpoint_manager import resume_from_checkpoint
            resume_from_checkpoint(
                trainer,
                _load_checkpoint_fn=fake_load_checkpoint,
                _SummaryWriter=MagicMock(return_value=MagicMock()),
            )

        assert "keep_up_to_step" in captured_calls, (
            "_purge_tb_events_after_step was not called during resume"
        )
        assert captured_calls["keep_up_to_step"] == 750, (
            f"Expected keep_up_to_step=750, got {captured_calls['keep_up_to_step']}"
        )

    def test_purge_log_dir_matches_trainer_log_dir(self, tmp_path):
        """The log_dir argument to purge must match trainer.log_dir."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        trainer = self._make_minimal_trainer(log_dir, optimizer_step=500)
        captured_calls = {}

        def fake_purge(log_dir, keep_up_to_step, _SummaryWriter=None):
            captured_calls["log_dir"] = log_dir
            captured_calls["keep_up_to_step"] = keep_up_to_step
            return MagicMock(spec=SummaryWriter)

        fake_checkpoint = {"current_optimizer_step": 500, "optimizer_steps_completed": 500}

        with (
            patch(
                "kokoro.training.checkpoint_manager._purge_tb_events_after_step",
                side_effect=fake_purge,
            ),
            patch(
                "kokoro.training.checkpoint_manager.find_latest_checkpoint",
                return_value="/fake/checkpoint.pth",
            ),
            patch(
                "kokoro.training.checkpoint_manager.torch.load",
                return_value=fake_checkpoint,
            ),
        ):
            from kokoro.training.checkpoint_manager import resume_from_checkpoint
            resume_from_checkpoint(
                trainer,
                _load_checkpoint_fn=MagicMock(return_value=(2, 0.5, MagicMock())),
                _SummaryWriter=MagicMock(return_value=MagicMock()),
            )

        assert captured_calls.get("log_dir") == str(log_dir)
