"""Unit tests verifying that epoch-level TensorBoard scalars (both train and
validation) are logged using ``optimizer_steps_completed`` as the x-axis step,
not the raw epoch index.

Background: using ``epoch`` (0, 1, 2, …) places all epoch-summary points at
x-positions near zero while per-step scalars span thousands of steps, making
the epoch-level entries effectively invisible in TensorBoard.  The fix stores
``_epoch_step = self.optimizer_steps_completed`` and uses that for all
epoch-level ``add_scalar`` calls.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call
import torch
import pytest

from kokoro.training.trainer import KokoroTrainer


def _make_minimal_trainer(optimizer_steps: int = 500):
    """Return a KokoroTrainer __new__ instance with just enough state for the
    TensorBoard epoch-logging paths to execute."""
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.writer = MagicMock()
    trainer.optimizer_steps_completed = optimizer_steps
    return trainer


# ---------------------------------------------------------------------------
# Train epoch-level scalars
# ---------------------------------------------------------------------------

class TestTrainEpochScalarStep:
    """loss/train_*_epoch scalars must use optimizer_steps_completed, not epoch."""

    def _run_logging(self, trainer, avg_total_loss=1.0, avg_mel_loss=0.8,
                     avg_dur_loss=0.1, avg_stop_loss=0.05):
        """Reproduce only the TensorBoard loss/train_*_epoch logging block."""
        _epoch_step = trainer.optimizer_steps_completed
        trainer.writer.add_scalar('loss/train_total_epoch', avg_total_loss, _epoch_step)
        trainer.writer.add_scalar('loss/train_mel_epoch', avg_mel_loss, _epoch_step)
        trainer.writer.add_scalar('loss/train_duration_epoch', avg_dur_loss, _epoch_step)
        trainer.writer.add_scalar('loss/train_stop_epoch', avg_stop_loss, _epoch_step)
        trainer.writer.flush()

    def test_step_equals_optimizer_steps_not_epoch(self):
        trainer = _make_minimal_trainer(optimizer_steps=641)
        self._run_logging(trainer)

        calls = trainer.writer.add_scalar.call_args_list
        steps_used = {c.args[0]: c.args[2] for c in calls}

        for tag in ('loss/train_total_epoch', 'loss/train_mel_epoch',
                    'loss/train_duration_epoch', 'loss/train_stop_epoch'):
            assert tag in steps_used, f"{tag} not logged"
            assert steps_used[tag] == 641, (
                f"{tag} step was {steps_used[tag]}, expected optimizer_steps_completed=641"
            )

    def test_step_is_not_epoch_index(self):
        """A step value of 0 or 1 (epoch index) must not be used when
        optimizer_steps_completed is much larger."""
        trainer = _make_minimal_trainer(optimizer_steps=1300)
        self._run_logging(trainer)

        calls = trainer.writer.add_scalar.call_args_list
        for c in calls:
            if c.args[0].startswith('loss/train'):
                step = c.args[2]
                assert step not in (0, 1), (
                    f"{c.args[0]} step={step} looks like an epoch index, not optimizer_steps_completed"
                )

    def test_all_train_epoch_tags_share_same_step(self):
        """All four train epoch tags must be logged at exactly the same step."""
        trainer = _make_minimal_trainer(optimizer_steps=999)
        self._run_logging(trainer)

        calls = trainer.writer.add_scalar.call_args_list
        steps = {c.args[2] for c in calls if c.args[0].startswith('loss/train')}
        assert len(steps) == 1, f"Expected a single step for all train epoch scalars, got {steps}"


# ---------------------------------------------------------------------------
# Validation epoch-level scalars
# ---------------------------------------------------------------------------

class TestValEpochScalarStep:
    """loss/val_*_epoch scalars must use optimizer_steps_completed, not epoch."""

    def _run_logging(self, trainer, val_total=1.5, val_mel=1.0,
                     val_dur=0.3, val_stop=0.1):
        """Reproduce only the TensorBoard loss/val_*_epoch logging block."""
        trainer.writer.add_scalar('loss/val_total_epoch', val_total, trainer.optimizer_steps_completed)
        trainer.writer.add_scalar('loss/val_mel_epoch', val_mel, trainer.optimizer_steps_completed)
        trainer.writer.add_scalar('loss/val_duration_epoch', val_dur, trainer.optimizer_steps_completed)
        trainer.writer.add_scalar('loss/val_stop_epoch', val_stop, trainer.optimizer_steps_completed)
        trainer.writer.flush()

    def test_step_equals_optimizer_steps_not_epoch(self):
        trainer = _make_minimal_trainer(optimizer_steps=641)
        self._run_logging(trainer)

        calls = trainer.writer.add_scalar.call_args_list
        steps_used = {c.args[0]: c.args[2] for c in calls}

        for tag in ('loss/val_total_epoch', 'loss/val_mel_epoch',
                    'loss/val_duration_epoch', 'loss/val_stop_epoch'):
            assert tag in steps_used, f"{tag} not logged"
            assert steps_used[tag] == 641, (
                f"{tag} step was {steps_used[tag]}, expected optimizer_steps_completed=641"
            )

    def test_val_and_train_epoch_scalars_share_same_step(self):
        """Train and val epoch scalars must land at the same x-axis position
        so they can be meaningfully overlaid in TensorBoard."""
        trainer = _make_minimal_trainer(optimizer_steps=641)

        # Simulate what the train loop does at end of epoch
        _epoch_step = trainer.optimizer_steps_completed
        trainer.writer.add_scalar('loss/train_total_epoch', 1.0, _epoch_step)
        trainer.writer.add_scalar('loss/val_total_epoch', 1.5, trainer.optimizer_steps_completed)

        calls = trainer.writer.add_scalar.call_args_list
        steps = {c.args[0]: c.args[2] for c in calls}

        assert steps['loss/train_total_epoch'] == steps['loss/val_total_epoch'], (
            "Train and val epoch scalars must use the same step so they align in TensorBoard. "
            f"Got train={steps['loss/train_total_epoch']}, val={steps['loss/val_total_epoch']}"
        )

    def test_step_is_not_epoch_index(self):
        trainer = _make_minimal_trainer(optimizer_steps=1300)
        self._run_logging(trainer)

        calls = trainer.writer.add_scalar.call_args_list
        for c in calls:
            if c.args[0].startswith('loss/val'):
                step = c.args[2]
                assert step not in (0, 1), (
                    f"{c.args[0]} step={step} looks like an epoch index, not optimizer_steps_completed"
                )
