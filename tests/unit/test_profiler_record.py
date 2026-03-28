import contextlib
from unittest.mock import patch, MagicMock

import pytest
import torch

from kokoro.training.trainer import KokoroTrainer

"""
Unit tests for KokoroTrainer._profiler_record

The helper must:
  - Return a torch.profiler.record_function context when active=True
  - Return a contextlib.nullcontext (no-op) when active=False
  - Be usable as a context manager in both modes without raising
  - Not execute torch.profiler.record_function when active=False
  - Execute torch.profiler.record_function with the correct name when active=True
"""

def _make_trainer() -> KokoroTrainer:
    """Minimal trainer sufficient to call _profiler_record."""
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    return trainer


# ---------------------------------------------------------------------------
# active=False: must be a no-op context manager
# ---------------------------------------------------------------------------

class TestInactive:
    def test_returns_context_manager_when_inactive(self):
        trainer = _make_trainer()
        ctx = trainer._profiler_record("SomeName", active=False)
        # Must support the context manager protocol
        assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")

    def test_inactive_context_does_not_raise(self):
        trainer = _make_trainer()
        with trainer._profiler_record("SomeName", active=False):
            pass  # must not raise

    def test_inactive_body_executes_normally(self):
        trainer = _make_trainer()
        executed = []
        with trainer._profiler_record("SomeName", active=False):
            executed.append(True)
        assert executed == [True], "Body inside inactive context should execute"

    def test_inactive_does_not_call_record_function(self):
        trainer = _make_trainer()
        with patch("torch.profiler.record_function") as mock_rf:
            with trainer._profiler_record("SomeName", active=False):
                pass
            mock_rf.assert_not_called()

    def test_inactive_returns_nullcontext_type(self):
        """When inactive the returned object should be a nullcontext instance."""
        trainer = _make_trainer()
        ctx = trainer._profiler_record("SomeName", active=False)
        # nullcontext is the expected type for the no-op path.
        assert isinstance(ctx, contextlib.nullcontext)


# ---------------------------------------------------------------------------
# active=True: must wrap with torch.profiler.record_function
# ---------------------------------------------------------------------------

class TestActive:
    def test_returns_context_manager_when_active(self):
        trainer = _make_trainer()
        ctx = trainer._profiler_record("SomeName", active=True)
        assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")

    def test_active_context_does_not_raise(self):
        trainer = _make_trainer()
        with trainer._profiler_record("SomeName", active=True):
            pass  # must not raise

    def test_active_body_executes_normally(self):
        trainer = _make_trainer()
        executed = []
        with trainer._profiler_record("SomeName", active=True):
            executed.append(True)
        assert executed == [True], "Body inside active context should execute"

    def test_active_calls_record_function_with_correct_name(self):
        trainer = _make_trainer()
        # torch.profiler.record_function is itself a context manager factory;
        # we verify it is called with the right name.
        with patch("torch.profiler.record_function") as mock_rf:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_rf.return_value = mock_ctx

            with trainer._profiler_record("ForwardPass", active=True):
                pass

            mock_rf.assert_called_once_with("ForwardPass")

    def test_active_returns_record_function_instance(self):
        """When active the returned object must be a record_function, not nullcontext."""
        trainer = _make_trainer()
        ctx = trainer._profiler_record("SomeName", active=True)
        assert not isinstance(ctx, contextlib.nullcontext), (
            "Active _profiler_record must NOT return a nullcontext"
        )
        assert isinstance(ctx, torch.profiler.record_function)


# ---------------------------------------------------------------------------
# name forwarding
# ---------------------------------------------------------------------------

class TestNameForwarding:
    @pytest.mark.parametrize("stage_name", [
        "Data_Loading",
        "Model_Forward",
        "Loss_Calculation",
        "Backward_Pass",
    ])
    def test_known_stage_names_passed_correctly(self, stage_name):
        trainer = _make_trainer()
        with patch("torch.profiler.record_function") as mock_rf:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_rf.return_value = mock_ctx

            with trainer._profiler_record(stage_name, active=True):
                pass

            mock_rf.assert_called_once_with(stage_name)

    def test_inactive_name_is_irrelevant(self):
        """active=False must be a no-op regardless of name."""
        trainer = _make_trainer()
        with patch("torch.profiler.record_function") as mock_rf:
            with trainer._profiler_record("anything", active=False):
                pass
            mock_rf.assert_not_called()


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------

class TestExceptionPropagation:
    def test_inactive_propagates_exception(self):
        trainer = _make_trainer()
        with pytest.raises(ValueError, match="test error"):
            with trainer._profiler_record("SomeName", active=False):
                raise ValueError("test error")

    def test_active_propagates_exception(self):
        trainer = _make_trainer()
        with pytest.raises(ValueError, match="test error"):
            with trainer._profiler_record("SomeName", active=True):
                raise ValueError("test error")
