import torch
from types import SimpleNamespace
import torch.nn as nn

from kokoro.training.trainer import KokoroTrainer


def _build_minimal_trainer() -> KokoroTrainer:
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.device = torch.device("cpu")
    trainer.criterion_mel = torch.nn.L1Loss(reduction="none")
    trainer.criterion_duration = torch.nn.MSELoss(reduction="none")
    trainer.criterion_stop_token = torch.nn.BCEWithLogitsLoss(reduction="none")
    trainer.criterion_pitch = None
    trainer.criterion_energy = None
    trainer.config = SimpleNamespace(
        duration_loss_weight=0.1,
        stop_token_loss_weight=1.0,
        pitch_loss_weight=0.1,
        energy_loss_weight=0.1,
    )
    return trainer


def test_calculate_losses_ignores_nonfinite_values_in_padded_frames():
    trainer = _build_minimal_trainer()

    mel_specs = torch.zeros(1, 4, 2)
    predicted_mel = mel_specs.clone()
    predicted_mel[0, 2, 0] = float("nan")
    predicted_mel[0, 3, 1] = float("nan")

    phoneme_durations = torch.tensor([[1, 1, 0]], dtype=torch.long)
    # Match trainer's target computation (uses +1.0) so duration loss is zero
    predicted_log_durations = torch.log(phoneme_durations.float() + 1.0)

    stop_token_targets = torch.zeros(1, 4)
    predicted_stop_logits = torch.zeros(1, 4)

    mel_lengths = torch.tensor([2], dtype=torch.long)
    phoneme_lengths = torch.tensor([2], dtype=torch.long)

    total_loss, loss_mel, loss_duration, loss_stop_token, loss_pitch, loss_energy = trainer._calculate_losses(
        predicted_mel,
        predicted_log_durations,
        predicted_stop_logits,
        mel_specs,
        phoneme_durations,
        stop_token_targets,
        mel_lengths,
        phoneme_lengths,
    )

    assert torch.isfinite(total_loss)
    assert torch.isfinite(loss_mel)
    assert torch.isfinite(loss_duration)
    assert torch.isfinite(loss_stop_token)
    assert torch.isfinite(loss_pitch)
    assert torch.isfinite(loss_energy)

    assert torch.isclose(loss_mel, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(total_loss, torch.tensor(0.6931472), atol=1e-6)


class _SingleParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))

    def forward(self, phoneme_indices, mel_specs, phoneme_durations, stop_token_targets,
                pitch_targets=None, energy_targets=None):
        batch_size = mel_specs.size(0)
        mel_len = mel_specs.size(1)
        phoneme_len = phoneme_indices.size(1)
        predicted_mel = mel_specs + self.w
        predicted_log_durations = torch.zeros(batch_size, phoneme_len, device=mel_specs.device) + self.w
        predicted_stop_logits = torch.zeros(batch_size, mel_len, device=mel_specs.device) + self.w
        return predicted_mel, predicted_log_durations, predicted_stop_logits, None, None


def _make_minimal_training_batch():
    return {
        "mel_specs": torch.zeros(1, 2, 2),
        "phoneme_indices": torch.zeros(1, 2, dtype=torch.long),
        "phoneme_durations": torch.ones(1, 2, dtype=torch.long),
        "stop_token_targets": torch.zeros(1, 2),
        "mel_lengths": torch.tensor([2], dtype=torch.long),
        "phoneme_lengths": torch.tensor([2], dtype=torch.long),
    }


def test_train_epoch_skips_optimizer_step_on_nonfinite_gradients():
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.device = torch.device("cpu")
    trainer.device_type = "cpu"
    trainer.use_mixed_precision = False
    trainer.scaler = None
    trainer.model = _SingleParamModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scheduler_per_batch = False
    trainer.use_ema = False
    trainer.enable_adaptive_memory = False
    trainer.memory_report_interval = 1000
    trainer.mixed_precision_stats = {
        "scale_updates": 0,
        "scale_decreases": 0,
        "overflow_count": 0,
        "successful_steps": 0,
        "skipped_steps": 0,
    }
    trainer.current_optimizer_step = 0
    trainer.profiler = None

    noop = lambda *args, **kwargs: None
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
    trainer.dataloader = [_make_minimal_training_batch()]

    def _finite_loss_nonfinite_grad(*args, **kwargs):
        total = torch.sqrt(trainer.model.w)
        zero = torch.tensor(0.0)
        return total, zero, zero, zero, zero, zero

    trainer._calculate_losses = _finite_loss_nonfinite_grad

    trainer.train_epoch(0)

    assert torch.isfinite(trainer.model.w).all()
    assert torch.isclose(trainer.model.w.detach(), torch.tensor(0.0), atol=1e-8)
