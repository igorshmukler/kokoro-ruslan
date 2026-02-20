"""Unit tests for trainer checkpoint step counter persistence and resume restoration."""

from types import SimpleNamespace

import torch

from kokoro.training.trainer import KokoroTrainer
import kokoro.training.trainer as trainer_module


class _StateDictHolder:
    def __init__(self, payload=None):
        self.payload = payload or {"ok": True}

    def state_dict(self):
        return self.payload


def test_save_checkpoint_with_scaler_persists_step_counters(monkeypatch, tmp_path):
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.config = SimpleNamespace(output_dir=str(tmp_path))
    trainer.model = _StateDictHolder({"model": 1})
    trainer.optimizer = _StateDictHolder({"opt": 1})
    trainer.scheduler = _StateDictHolder({"sched": 1})
    trainer.use_mixed_precision = False
    trainer.scaler = None
    trainer.use_ema = False
    trainer.ema_model = None
    trainer.current_optimizer_step = 123
    trainer.optimizer_steps_completed = 456

    captured = {}

    def _fake_save(payload, path):
        captured["payload"] = payload
        captured["path"] = path

    monkeypatch.setattr(torch, "save", _fake_save)

    KokoroTrainer.save_checkpoint_with_scaler(trainer, epoch=2, loss=1.23)

    assert "payload" in captured
    assert captured["payload"]["current_optimizer_step"] == 123
    assert captured["payload"]["optimizer_steps_completed"] == 456
    assert captured["payload"]["epoch"] == 2


def test_setup_checkpoint_resumption_restores_step_counters(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint_epoch_2.pth"
    torch.save(
        {
            "current_optimizer_step": 777,
            "optimizer_steps_completed": 888,
        },
        checkpoint_path,
    )

    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.config = SimpleNamespace(
        resume_checkpoint=str(checkpoint_path),
        output_dir=str(tmp_path),
    )
    trainer.model = object()
    trainer.optimizer = object()
    trainer.scheduler = object()
    trainer.dataset = SimpleNamespace(phoneme_processor=None)
    trainer.device = torch.device("cpu")
    trainer.device_type = "cpu"

    trainer.use_mixed_precision = False
    trainer.scaler = None
    trainer.use_ema = False
    trainer.ema_model = None

    trainer.current_optimizer_step = 0
    trainer.optimizer_steps_completed = 0
    trainer.reset_called = False

    def _fake_reset():
        trainer.reset_called = True

    def _fake_load_checkpoint(path, model, optimizer, scheduler, output_dir):
        assert path == str(checkpoint_path)
        assert output_dir == str(tmp_path)
        return 2, 5.0973, "dummy_processor"

    monkeypatch.setattr(trainer_module, "load_checkpoint", _fake_load_checkpoint)

    KokoroTrainer.setup_checkpoint_resumption(trainer)

    assert trainer.start_epoch == 2
    assert trainer.best_loss == 5.0973
    assert trainer.dataset.phoneme_processor == "dummy_processor"
    assert trainer.current_optimizer_step == 777
    assert trainer.optimizer_steps_completed == 888
    assert trainer.reset_called is True


def test_setup_checkpoint_resumption_keeps_defaults_when_counter_keys_missing(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint_epoch_2.pth"
    torch.save({"some_other_key": 1}, checkpoint_path)

    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.config = SimpleNamespace(
        resume_checkpoint=str(checkpoint_path),
        output_dir=str(tmp_path),
    )
    trainer.model = object()
    trainer.optimizer = object()
    trainer.scheduler = object()
    trainer.dataset = SimpleNamespace(phoneme_processor=None)
    trainer.device = torch.device("cpu")
    trainer.device_type = "cpu"

    trainer.use_mixed_precision = False
    trainer.scaler = None
    trainer.use_ema = False
    trainer.ema_model = None

    trainer.current_optimizer_step = 11
    trainer.optimizer_steps_completed = 22

    def _fake_load_checkpoint(path, model, optimizer, scheduler, output_dir):
        assert path == str(checkpoint_path)
        assert output_dir == str(tmp_path)
        return 2, 5.0973, "dummy_processor"

    monkeypatch.setattr(trainer_module, "load_checkpoint", _fake_load_checkpoint)

    KokoroTrainer.setup_checkpoint_resumption(trainer)

    assert trainer.current_optimizer_step == 11
    assert trainer.optimizer_steps_completed == 22
