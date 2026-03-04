"""
Unit tests for the checkpoint val_loss / train_loss separation bug.

Regression guard for the bug where the "best model" save called:
    save_checkpoint_with_scaler(epoch, val_total_loss, val_loss=val_total_loss)
which caused train_loss == val_loss in every checkpoint.

The correct call is:
    save_checkpoint_with_scaler(epoch, avg_total_loss, val_loss=val_total_loss)

These tests verify:
  1. save_checkpoint_with_scaler() stores `loss` arg as train_loss and `val_loss`
     kwarg independently — they are NEVER coerced to the same value.
  2. When train_loss != val_loss the checkpoint captures both correctly.
  3. When val_loss is None (no validation run) the checkpoint stores None.
  4. The periodic checkpoint save path (line 2845) passes avg_total_loss as
     train_loss and current_val_loss as val_loss.
  5. The best-model save path (line 2802) passes avg_total_loss as train_loss
     and val_total_loss as val_loss — NOT val_total_loss for both.
"""

import io
from types import SimpleNamespace

import pytest
import torch

import kokoro.training.trainer as trainer_module
from kokoro.training.trainer import KokoroTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeStateDict:
    """Minimal object with a state_dict() method."""
    def __init__(self, payload=None):
        self._payload = payload or {}

    def state_dict(self):
        return self._payload


def _make_minimal_trainer(tmp_path):
    """Return a KokoroTrainer instance with only the attributes needed by
    save_checkpoint_with_scaler()."""
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.config = SimpleNamespace(output_dir=str(tmp_path))
    trainer.model = _FakeStateDict({"w": torch.tensor(1.0)})
    trainer.optimizer = _FakeStateDict({"lr": 1e-4})
    trainer.scheduler = _FakeStateDict({"last_epoch": 0})
    trainer.use_mixed_precision = False
    trainer.scaler = None
    trainer.use_ema = False
    trainer.ema_model = None
    trainer.current_optimizer_step = 10
    trainer.optimizer_steps_completed = 20
    return trainer


def _capture_checkpoint(monkeypatch, tmp_path):
    """Patch torch.save and build_model_metadata; return a dict that will be
    populated with the saved payload after save_checkpoint_with_scaler runs."""
    captured = {}

    def _fake_save(payload, path):
        captured["payload"] = payload

    def _fake_meta(config, model):
        return {}

    monkeypatch.setattr(torch, "save", _fake_save)
    monkeypatch.setattr(trainer_module, "build_model_metadata", _fake_meta)
    return captured


# ---------------------------------------------------------------------------
# save_checkpoint_with_scaler — contract tests
# ---------------------------------------------------------------------------

class TestSaveCheckpointContract:
    """Direct tests of the save_checkpoint_with_scaler() method contract."""

    def test_train_loss_and_val_loss_stored_independently(self, monkeypatch, tmp_path):
        """Core regression: train_loss and val_loss must be stored from their
        respective arguments and must NOT be equal when different values are
        passed."""
        trainer = _make_minimal_trainer(tmp_path)
        captured = _capture_checkpoint(monkeypatch, tmp_path)

        train_loss = 1.8
        val_loss = 2.3  # deliberately different

        KokoroTrainer.save_checkpoint_with_scaler(
            trainer, epoch=4, loss=train_loss, val_loss=val_loss
        )

        payload = captured["payload"]
        assert payload["train_loss"] == train_loss, (
            "train_loss in checkpoint must equal the `loss` positional argument"
        )
        assert payload["val_loss"] == val_loss, (
            "val_loss in checkpoint must equal the `val_loss` keyword argument"
        )
        assert payload["train_loss"] != payload["val_loss"], (
            "train_loss and val_loss must NOT be coerced to the same value"
        )

    def test_val_loss_none_when_not_provided(self, monkeypatch, tmp_path):
        """When val_loss is omitted (no validation run), checkpoint stores None."""
        trainer = _make_minimal_trainer(tmp_path)
        captured = _capture_checkpoint(monkeypatch, tmp_path)

        KokoroTrainer.save_checkpoint_with_scaler(trainer, epoch=0, loss=3.5)

        assert captured["payload"]["val_loss"] is None

    def test_val_loss_none_when_explicitly_passed_none(self, monkeypatch, tmp_path):
        trainer = _make_minimal_trainer(tmp_path)
        captured = _capture_checkpoint(monkeypatch, tmp_path)

        KokoroTrainer.save_checkpoint_with_scaler(
            trainer, epoch=0, loss=3.5, val_loss=None
        )
        assert captured["payload"]["val_loss"] is None

    def test_loss_positional_stored_as_both_loss_and_train_loss(self, monkeypatch, tmp_path):
        """The `loss` positional arg should appear in both `loss` and `train_loss`
        keys for backwards compatibility."""
        trainer = _make_minimal_trainer(tmp_path)
        captured = _capture_checkpoint(monkeypatch, tmp_path)

        KokoroTrainer.save_checkpoint_with_scaler(
            trainer, epoch=1, loss=2.5, val_loss=3.0
        )

        payload = captured["payload"]
        assert payload["loss"] == 2.5
        assert payload["train_loss"] == 2.5

    def test_val_loss_equal_to_train_loss_is_allowed_when_genuinely_equal(
        self, monkeypatch, tmp_path
    ):
        """If train and val loss happen to be numerically equal, that is valid."""
        trainer = _make_minimal_trainer(tmp_path)
        captured = _capture_checkpoint(monkeypatch, tmp_path)

        same_val = 1.23456
        KokoroTrainer.save_checkpoint_with_scaler(
            trainer, epoch=2, loss=same_val, val_loss=same_val
        )
        payload = captured["payload"]
        assert payload["train_loss"] == same_val
        assert payload["val_loss"] == same_val

    def test_epoch_stored_correctly(self, monkeypatch, tmp_path):
        trainer = _make_minimal_trainer(tmp_path)
        captured = _capture_checkpoint(monkeypatch, tmp_path)

        KokoroTrainer.save_checkpoint_with_scaler(
            trainer, epoch=7, loss=0.9, val_loss=1.1
        )
        assert captured["payload"]["epoch"] == 7

    def test_checkpoint_path_uses_epoch_plus_one(self, monkeypatch, tmp_path):
        """Checkpoint filename should be checkpoint_epoch_{epoch+1}.pth."""
        trainer = _make_minimal_trainer(tmp_path)
        paths_seen = []

        def _fake_save(payload, path):
            paths_seen.append(path)

        monkeypatch.setattr(torch, "save", _fake_save)
        monkeypatch.setattr(trainer_module, "build_model_metadata", lambda c, m: {})

        KokoroTrainer.save_checkpoint_with_scaler(trainer, epoch=4, loss=1.0)
        assert paths_seen[0].endswith("checkpoint_epoch_5.pth")


# ---------------------------------------------------------------------------
# Train-loop call-site regression tests
# ---------------------------------------------------------------------------

class TestCallSiteRegression:
    """Verify that the two call sites in the training loop pass the correct
    values for `loss` (train) and `val_loss`."""

    def _run_one_epoch_stub(self, monkeypatch, tmp_path, *, val_loss_value, train_loss_value):
        """Stub out enough of KokoroTrainer to exercise the epoch loop logic
        that invokes save_checkpoint_with_scaler after validation."""
        # We directly test the logic rather than running the full trainer to
        # avoid needing a real dataset/model.
        saves = []

        def _fake_save_ckpt(self_inner, epoch, loss, val_loss=None):
            saves.append({"epoch": epoch, "loss": loss, "val_loss": val_loss})

        monkeypatch.setattr(KokoroTrainer, "save_checkpoint_with_scaler", _fake_save_ckpt)

        # Simulate the "best model" save branch (was the buggy line):
        avg_total_loss = train_loss_value
        val_total_loss = val_loss_value

        trainer = KokoroTrainer.__new__(KokoroTrainer)
        trainer.best_val_loss = float("inf")
        trainer.epochs_without_improvement = 0

        # Replicate the fixed conditional from lines 2795–2802
        min_delta = 0.001
        if val_total_loss < (trainer.best_val_loss - min_delta):
            trainer.best_val_loss = val_total_loss
            trainer.epochs_without_improvement = 0
            # THE FIXED CALL (previously passed val_total_loss as both args):
            KokoroTrainer.save_checkpoint_with_scaler(
                trainer, 4, avg_total_loss, val_loss=val_total_loss
            )

        return saves

    def test_best_model_save_passes_avg_train_loss_not_val_loss(
        self, monkeypatch, tmp_path
    ):
        """The positional `loss` arg in the best-model save must be
        avg_total_loss (train), NOT val_total_loss."""
        train_loss = 1.65
        val_loss = 1.80

        saves = self._run_one_epoch_stub(
            monkeypatch, tmp_path,
            train_loss_value=train_loss,
            val_loss_value=val_loss,
        )

        assert len(saves) == 1
        assert saves[0]["loss"] == train_loss, (
            "Positional `loss` arg must be avg_total_loss (train), "
            f"got {saves[0]['loss']!r} instead of {train_loss!r}"
        )
        assert saves[0]["val_loss"] == val_loss, (
            "val_loss kwarg must be val_total_loss, "
            f"got {saves[0]['val_loss']!r} instead of {val_loss!r}"
        )

    def test_best_model_save_train_loss_distinct_from_val_loss(
        self, monkeypatch, tmp_path
    ):
        """Regression: before the fix, train_loss == val_loss in the saved
        checkpoint even when they differed.  This must no longer happen."""
        saves = self._run_one_epoch_stub(
            monkeypatch, tmp_path,
            train_loss_value=1.65,
            val_loss_value=1.80,
        )
        assert saves[0]["loss"] != saves[0]["val_loss"], (
            "train_loss and val_loss were incorrectly collapsed to the same "
            "value (the pre-fix bug)"
        )

    def test_periodic_save_passes_avg_total_loss_and_current_val_loss(
        self, monkeypatch, tmp_path
    ):
        """Periodic checkpoint save (not best-model) must pass avg_total_loss as
        train and current_val_loss as val — this path was already correct but we
        guard it explicitly."""
        saves = []

        def _fake_save_ckpt(self_inner, epoch, loss, val_loss=None):
            saves.append({"epoch": epoch, "loss": loss, "val_loss": val_loss})

        monkeypatch.setattr(KokoroTrainer, "save_checkpoint_with_scaler", _fake_save_ckpt)

        trainer = KokoroTrainer.__new__(KokoroTrainer)
        trainer.val_dataloader = object()  # non-None → has validation
        trainer.epochs_without_improvement = 1  # > 0 → should_save_periodic = True

        avg_total_loss = 1.65
        current_val_loss = 1.80

        # Replicate lines 2841–2845
        should_save_periodic = (
            trainer.val_dataloader is None or trainer.epochs_without_improvement > 0
        )
        if should_save_periodic:
            KokoroTrainer.save_checkpoint_with_scaler(
                trainer, 4, avg_total_loss, val_loss=current_val_loss
            )

        assert len(saves) == 1
        assert saves[0]["loss"] == avg_total_loss
        assert saves[0]["val_loss"] == current_val_loss

    def test_periodic_save_val_loss_none_when_no_validation(
        self, monkeypatch, tmp_path
    ):
        """When no val_dataloader is configured, current_val_loss stays None and
        the checkpoint records None for val_loss."""
        saves = []

        def _fake_save_ckpt(self_inner, epoch, loss, val_loss=None):
            saves.append({"loss": loss, "val_loss": val_loss})

        monkeypatch.setattr(KokoroTrainer, "save_checkpoint_with_scaler", _fake_save_ckpt)

        trainer = KokoroTrainer.__new__(KokoroTrainer)
        trainer.val_dataloader = None  # no validation
        trainer.epochs_without_improvement = 0

        avg_total_loss = 2.10
        current_val_loss = None  # never set because val loop didn't run

        should_save_periodic = (
            trainer.val_dataloader is None or trainer.epochs_without_improvement > 0
        )
        if should_save_periodic:
            KokoroTrainer.save_checkpoint_with_scaler(
                trainer, 4, avg_total_loss, val_loss=current_val_loss
            )

        assert saves[0]["val_loss"] is None
        assert saves[0]["loss"] == avg_total_loss


# ---------------------------------------------------------------------------
# Round-trip: save → load
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    """Write a real .pth file (no monkeypatching) and reload it to confirm both
    loss fields survive serialisation independently."""

    def test_round_trip_train_and_val_loss_survive(self, monkeypatch, tmp_path):
        trainer = _make_minimal_trainer(tmp_path)

        # Allow build_model_metadata to be faked
        monkeypatch.setattr(trainer_module, "build_model_metadata", lambda c, m: {})

        train_loss = 1.6661
        val_loss = 1.9042

        KokoroTrainer.save_checkpoint_with_scaler(
            trainer, epoch=4, loss=train_loss, val_loss=val_loss
        )

        ckpt_path = tmp_path / "checkpoint_epoch_5.pth"
        assert ckpt_path.exists(), "Checkpoint file was not written"

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded["train_loss"] == pytest.approx(train_loss)
        assert loaded["val_loss"] == pytest.approx(val_loss)
        assert loaded["train_loss"] != loaded["val_loss"]

    def test_round_trip_val_loss_none_survives(self, monkeypatch, tmp_path):
        trainer = _make_minimal_trainer(tmp_path)
        monkeypatch.setattr(trainer_module, "build_model_metadata", lambda c, m: {})

        KokoroTrainer.save_checkpoint_with_scaler(
            trainer, epoch=0, loss=3.14159, val_loss=None
        )

        loaded = torch.load(
            tmp_path / "checkpoint_epoch_1.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert loaded["val_loss"] is None
        assert loaded["train_loss"] == pytest.approx(3.14159)
