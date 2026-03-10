import torch
import torch.nn as nn
from types import SimpleNamespace

from kokoro.training.trainer import KokoroTrainer, BatchOnDevice, StepResult


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))


class _SamplerStub:
    def __init__(self):
        self.batches = []


class _RebuildingLoader:
    def __init__(self, items, sampler, stale_len):
        self._items = list(items)
        self._sampler = sampler
        self._stale_len = int(stale_len)

    def __len__(self):
        # Simulates stale cached length from previous epoch build.
        return self._stale_len

    def __iter__(self):
        # Simulates DynamicFrameBatchSampler rebuilding batches at iteration start.
        self._sampler.batches = list(self._items)
        yield from self._items


def _make_batch_on_device() -> BatchOnDevice:
    mel_len = 4
    ph_len = 3
    return BatchOnDevice(
        mel_specs=torch.zeros(1, mel_len, 2),
        phoneme_indices=torch.zeros(1, ph_len, dtype=torch.long),
        phoneme_durations=torch.ones(1, ph_len, dtype=torch.long),
        stop_token_targets=torch.zeros(1, mel_len),
        mel_lengths=torch.tensor([mel_len], dtype=torch.long),
        phoneme_lengths=torch.tensor([ph_len], dtype=torch.long),
        pitches=None,
        energies=None,
        stress_indices=None,
    )


def _make_step_result(loss_value: float) -> StepResult:
    batch_size = 1
    mel_len = 4
    ph_len = 3
    loss = torch.tensor(loss_value, dtype=torch.float32)
    return StepResult(
        total_loss=loss,
        loss_mel=loss,
        loss_duration=torch.tensor(0.0),
        loss_stop_token=torch.tensor(0.0),
        loss_pitch=None,
        loss_energy=None,
        predicted_mel=torch.zeros(batch_size, mel_len, 2),
        predicted_log_durations=torch.zeros(batch_size, ph_len),
        predicted_pitch=None,
        predicted_energy=None,
    )


def _build_minimal_trainer(gradient_accumulation_steps: int) -> KokoroTrainer:
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.device = torch.device("cpu")
    trainer.device_type = "cpu"
    trainer.use_mixed_precision = False
    trainer.scaler = None
    trainer.model = _TinyModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.scheduler_per_batch = False
    trainer.use_ema = False
    trainer.enable_adaptive_memory = False
    trainer.memory_report_interval = 1000
    trainer.current_optimizer_step = 0
    trainer.profiler = None
    trainer.optimizer_steps_completed = 0
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
    trainer.ffn_spike_clip_norm = 15.0
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=1,
        duration_loss_weight=0.1,
        stop_token_loss_weight=1.0,
        max_grad_norm=5.0,
        verbose=False,
        use_spec_augment=False,
        spec_augment_start_epoch=999,
        profile_wait_steps=1,
        profile_warmup_steps=1,
        profile_steps=1,
        interbatch_report_interval=100,
    )

    # We don't test transfer in this file; keep train_epoch focused.
    trainer._transfer_batch_to_device = lambda batch: _make_batch_on_device()

    return trainer


def test_train_epoch_uses_rebuilt_batch_count_for_last_batch_step():
    trainer = _build_minimal_trainer(gradient_accumulation_steps=3)
    sampler = _SamplerStub()

    # Stale length says 1 batch, but rebuilt iterator has 2 batches.
    items = [object(), object()]
    trainer.batch_sampler = sampler
    trainer.dataloader = _RebuildingLoader(items=items, sampler=sampler, stale_len=1)

    step_calls = {"count": 0}

    def _fake_optimizer_step_with_clipping(*args, **kwargs):
        step_calls["count"] += 1
        return True, 0.0

    trainer._optimizer_step_with_clipping = _fake_optimizer_step_with_clipping

    results = iter([
        _make_step_result(1.0),
        _make_step_result(1.0),
    ])
    trainer._execute_training_step = lambda *args, **kwargs: next(results)

    metrics = trainer.train_epoch(0)

    assert metrics.total_loss > 0.0
    # With correct rebuilt num_batches=2, second batch is last and triggers step.
    assert step_calls["count"] == 1
    assert trainer.optimizer_steps_completed == 1


def test_train_epoch_averages_only_successfully_processed_batches():
    trainer = _build_minimal_trainer(gradient_accumulation_steps=1)
    sampler = _SamplerStub()
    items = [object(), object(), object()]
    trainer.batch_sampler = sampler
    trainer.dataloader = _RebuildingLoader(items=items, sampler=sampler, stale_len=3)

    trainer._optimizer_step_with_clipping = lambda *args, **kwargs: (True, 0.0)

    # Middle batch is skipped (None), so only 2 batches contribute losses: (1 + 3) / 2 = 2.
    results = iter([
        _make_step_result(1.0),
        None,
        _make_step_result(3.0),
    ])
    trainer._execute_training_step = lambda *args, **kwargs: next(results)

    metrics = trainer.train_epoch(0)

    assert metrics.total_loss == 2.0
    assert metrics.mel_loss == 2.0
