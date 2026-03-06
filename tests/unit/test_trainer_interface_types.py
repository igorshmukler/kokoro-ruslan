import torch
import torch.nn as nn
from types import SimpleNamespace

from kokoro.training.trainer import (
    KokoroTrainer,
    BatchOnDevice,
    StepResult,
    EpochMetrics,
)


class _MinimalStepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))

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
        batch_size = mel_specs.size(0)
        mel_len = mel_specs.size(1)
        phoneme_len = phoneme_indices.size(1)
        predicted_mel = mel_specs + self.w
        predicted_log_durations = torch.zeros(batch_size, phoneme_len, device=mel_specs.device) + self.w
        predicted_stop_logits = torch.zeros(batch_size, mel_len, device=mel_specs.device) + self.w
        return predicted_mel, predicted_log_durations, predicted_stop_logits, None, None


def _make_trainer() -> KokoroTrainer:
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.device = torch.device("cpu")
    trainer.device_type = "cpu"
    trainer.use_mixed_precision = False
    trainer.scaler = None
    trainer.model = _MinimalStepModel()
    trainer.config = SimpleNamespace(
        duration_loss_weight=0.1,
        stop_token_loss_weight=1.0,
        pitch_loss_weight=0.1,
        energy_loss_weight=0.1,
    )
    return trainer


def _make_batch() -> dict:
    return {
        "mel_specs": torch.zeros(1, 2, 2),
        "phoneme_indices": torch.zeros(1, 2, dtype=torch.long),
        "phoneme_durations": torch.ones(1, 2, dtype=torch.long),
        "stop_token_targets": torch.zeros(1, 2),
        "mel_lengths": torch.tensor([2], dtype=torch.long),
        "phoneme_lengths": torch.tensor([2], dtype=torch.long),
    }


def test_execute_training_step_returns_stepresult_dataclass():
    trainer = _make_trainer()

    def _finite_losses(*args, **kwargs):
        total = trainer.model.w + 1.0
        zero = torch.tensor(0.0)
        return total, zero, zero, zero, zero, zero

    trainer._calculate_losses = _finite_losses

    transferred = trainer._transfer_batch_to_device(_make_batch())
    assert isinstance(transferred, BatchOnDevice)

    step_result = trainer._execute_training_step(transferred, transferred.mel_specs)

    assert isinstance(step_result, StepResult)
    assert torch.isfinite(step_result.total_loss)
    assert step_result.predicted_mel.shape == transferred.mel_specs.shape


def test_epochmetrics_iterable_and_tuple_shape():
    metrics = EpochMetrics(total_loss=1.0, mel_loss=0.2, dur_loss=0.3, stop_loss=0.4)

    assert metrics.as_tuple() == (1.0, 0.2, 0.3, 0.4)
    assert tuple(metrics) == (1.0, 0.2, 0.3, 0.4)
