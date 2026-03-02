import torch
import torch.nn as nn
from types import SimpleNamespace

from kokoro.training.trainer import KokoroTrainer


class _CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.forward_calls = 0

    def forward(
        self,
        phoneme_indices,
        mel_specs,
        phoneme_durations,
        stop_token_targets,
        pitch_targets=None,
        energy_targets=None,
        stress_indices=None,
    ):
        self.forward_calls += 1
        batch_size = mel_specs.size(0)
        mel_len = mel_specs.size(1)
        phoneme_len = phoneme_indices.size(1)
        predicted_mel = mel_specs + self.w
        predicted_log_durations = torch.zeros(batch_size, phoneme_len, device=mel_specs.device) + self.w
        predicted_stop_logits = torch.zeros(batch_size, mel_len, device=mel_specs.device) + self.w
        return predicted_mel, predicted_log_durations, predicted_stop_logits, None, None


def _build_trainer_for_stabilization_test(loss_multiplier: float = 1.0) -> KokoroTrainer:
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.device = torch.device("cpu")
    trainer.device_type = "cpu"
    trainer.use_mixed_precision = False
    trainer.scaler = None
    trainer.model = _CountingModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.scheduler_per_batch = False
    trainer.use_ema = False
    trainer.enable_adaptive_memory = False
    trainer.memory_report_interval = 1000
    trainer.current_optimizer_step = 0
    trainer.profiler = None
    trainer.grad_explosion_norm_ema = None
    trainer.grad_explosion_ema_alpha = 0.95
    trainer.grad_explosion_abs_floor = 1000.0
    trainer.grad_explosion_multiplier = 3.0
    trainer.grad_explosion_warmup_steps = 0
    trainer.grad_explosion_warmup_floor = 1000.0
    trainer.grad_explosion_min_ema_steps = 0
    trainer.grad_explosion_ema_steps = 0
    trainer.grad_explosion_streak = 0
    trainer.projection_spike_clip_norm = 30.0
    trainer.attention_spike_clip_norm = 20.0
    trainer.optimizer_steps_completed = 0
    trainer.mixed_precision_stats = {
        "scale_updates": 0,
        "scale_decreases": 0,
        "overflow_count": 0,
        "successful_steps": 0,
        "skipped_steps": 0,
    }

    noop = lambda *args, **kwargs: None
    trainer.writer = SimpleNamespace(
        add_scalar=noop,
        add_histogram=noop,
        add_image=noop,
        flush=noop,
    )
    trainer.log_memory_stats = noop
    trainer.clear_device_cache = noop
    trainer._update_ema = noop
    trainer._step_scheduler_with_warmup = noop
    trainer.adaptive_memory_cleanup = lambda *args, **kwargs: {"pressure_level": "low", "cleaned": False}
    trainer.interbatch_profiler = SimpleNamespace(
        reset=noop,
        start_batch=noop,
        end_batch=noop,
        start_data_loading=noop,
        end_data_loading=noop,
        start_forward_pass=noop,
        end_forward_pass=noop,
        start_backward_pass=noop,
        end_backward_pass=noop,
        get_statistics=lambda: {},
        print_report=noop,
    )

    trainer.config = SimpleNamespace(
        profile_epoch_start=999,
        enable_profiling=False,
        enable_interbatch_profiling=False,
        gradient_accumulation_steps=1,
        num_epochs=1,
        duration_loss_weight=0.1,
        stop_token_loss_weight=1.0,
    )

    def _simple_loss(*args, **kwargs):
        total = (trainer.model.w * loss_multiplier) + 1.0
        zero = torch.tensor(0.0)
        return total, zero, zero, zero, zero, zero

    trainer._calculate_losses = _simple_loss
    return trainer


def _make_batch(mel_len: int, max_duration: int):
    phoneme_len = 4
    durations = torch.ones(1, phoneme_len, dtype=torch.long)
    durations[0, 0] = max_duration
    return {
        "mel_specs": torch.zeros(1, mel_len, 2),
        "phoneme_indices": torch.zeros(1, phoneme_len, dtype=torch.long),
        "phoneme_durations": durations,
        "stop_token_targets": torch.zeros(1, mel_len),
        "mel_lengths": torch.tensor([mel_len], dtype=torch.long),
        "phoneme_lengths": torch.tensor([phoneme_len], dtype=torch.long),
    }


def test_train_epoch_processes_high_risk_batch_with_adaptive_stabilization(monkeypatch):
    trainer = _build_trainer_for_stabilization_test()
    risky_batch = _make_batch(mel_len=1785, max_duration=240)
    trainer.dataloader = [risky_batch]

    captured = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def _capture_clip(parameters, max_norm, *args, **kwargs):
        captured["max_norm"] = float(max_norm)
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _capture_clip)

    # Prevent tests from calling torch.mps.empty_cache (some torch builds raise when MPS not available)
    monkeypatch.setattr(getattr(torch, 'mps', object()), "empty_cache", lambda: None, raising=False)

    avg_total_loss, _, _, _ = trainer.train_epoch(0)

    risk_ratio = max(1785 / 1400, 240 / 150)
    expected_clip = max(0.05, 0.5 / (risk_ratio ** 0.5))

    assert trainer.model.forward_calls == 1
    assert avg_total_loss > 0.0
    assert "max_norm" in captured
    assert abs(captured["max_norm"] - expected_clip) < 1e-6


def test_train_epoch_keeps_default_clip_for_normal_batch(monkeypatch):
    trainer = _build_trainer_for_stabilization_test()
    normal_batch = _make_batch(mel_len=800, max_duration=60)
    trainer.dataloader = [normal_batch]

    captured = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def _capture_clip(parameters, max_norm, *args, **kwargs):
        captured["max_norm"] = float(max_norm)
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _capture_clip)

    # Prevent tests from calling torch.mps.empty_cache
    monkeypatch.setattr(getattr(torch, 'mps', object()), "empty_cache", lambda: None, raising=False)

    avg_total_loss, _, _, _ = trainer.train_epoch(0)

    assert trainer.model.forward_calls == 1
    assert avg_total_loss > 0.0
    assert captured["max_norm"] == 0.5


def test_train_epoch_uses_emergency_clip_norm_on_gradient_explosion(monkeypatch):
    trainer = _build_trainer_for_stabilization_test(loss_multiplier=2000.0)
    normal_batch = _make_batch(mel_len=800, max_duration=60)
    trainer.dataloader = [normal_batch]

    captured = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def _capture_clip(parameters, max_norm, *args, **kwargs):
        captured["max_norm"] = float(max_norm)
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _capture_clip)

    # Prevent tests from calling torch.mps.empty_cache
    monkeypatch.setattr(getattr(torch, 'mps', object()), "empty_cache", lambda: None, raising=False)

    avg_total_loss, _, _, _ = trainer.train_epoch(0)

    assert trainer.model.forward_calls == 1
    assert avg_total_loss > 0.0
    assert captured["max_norm"] == 0.05
