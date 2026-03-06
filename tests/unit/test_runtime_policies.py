from types import SimpleNamespace

import torch

from kokoro.training.runtime_policies import RuntimeStepPolicy, MemoryOOMPolicy


class _SingleParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))


class _FakeCudaScaler:
    def __init__(self, scales):
        self._scales = list(scales)
        self._idx = 0
        self.unscaled = False

    def unscale_(self, optimizer):
        self.unscaled = True

    def get_scale(self):
        return self._scales[self._idx]

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        if self._idx < len(self._scales) - 1:
            self._idx += 1


class _FakeMpsScaler:
    def __init__(self, scales, step_successful=True):
        self._scales = list(scales)
        self._idx = 0
        self._step_successful = step_successful

    def get_scale(self):
        return self._scales[self._idx]

    def step(self, optimizer):
        optimizer.step()
        return self._step_successful

    def update(self):
        if self._idx < len(self._scales) - 1:
            self._idx += 1


class _FakeMemoryManager:
    def __init__(self, adaptive_result=None, emergency_result=None):
        self.adaptive_result = adaptive_result or {"cleaned": True, "pressure_level": "low"}
        self.emergency_result = emergency_result or {"success": True, "memory_freed_mb": 123.0}
        self.adaptive_calls = []
        self.emergency_calls = 0

    def adaptive_cleanup(self, batch_idx, force):
        self.adaptive_calls.append((batch_idx, force))
        return self.adaptive_result

    def emergency_cleanup(self):
        self.emergency_calls += 1
        return self.emergency_result


def _mp_stats():
    return {
        "scale_updates": 0,
        "scale_decreases": 0,
        "overflow_count": 0,
        "successful_steps": 0,
        "skipped_steps": 0,
    }


def test_runtime_step_policy_non_mixed_precision_preserves_scheduler_and_ema_cadence():
    policy = RuntimeStepPolicy(logger=SimpleNamespace(info=lambda *a, **k: None))
    model = _SingleParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler_calls = {"count": 0}
    ema_calls = {"count": 0}

    step_successful = policy.optimizer_step_with_clipping(
        model=model,
        optimizer=optimizer,
        use_mixed_precision=False,
        device_type="cpu",
        scaler=None,
        mixed_precision_stats=_mp_stats(),
        clip_norm=0.5,
        step_scheduler=True,
        scheduler_per_batch=True,
        step_scheduler_fn=lambda: scheduler_calls.__setitem__("count", scheduler_calls["count"] + 1),
        update_ema=True,
        update_ema_fn=lambda: ema_calls.__setitem__("count", ema_calls["count"] + 1),
    )

    assert step_successful is True
    assert scheduler_calls["count"] == 1
    assert ema_calls["count"] == 1


def test_runtime_step_policy_scheduler_not_stepped_when_disabled():
    policy = RuntimeStepPolicy(logger=SimpleNamespace(info=lambda *a, **k: None))
    model = _SingleParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler_calls = {"count": 0}

    step_successful = policy.optimizer_step_with_clipping(
        model=model,
        optimizer=optimizer,
        use_mixed_precision=False,
        device_type="cpu",
        scaler=None,
        mixed_precision_stats=_mp_stats(),
        clip_norm=0.5,
        step_scheduler=False,
        scheduler_per_batch=True,
        step_scheduler_fn=lambda: scheduler_calls.__setitem__("count", scheduler_calls["count"] + 1),
        update_ema=False,
        update_ema_fn=lambda: None,
    )

    assert step_successful is True
    assert scheduler_calls["count"] == 0


def test_runtime_step_policy_cuda_updates_mixed_precision_stats_on_scale_drop():
    policy = RuntimeStepPolicy(logger=SimpleNamespace(info=lambda *a, **k: None))
    model = _SingleParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _FakeCudaScaler(scales=[1024.0, 512.0])
    stats = _mp_stats()

    step_successful = policy.optimizer_step_with_clipping(
        model=model,
        optimizer=optimizer,
        use_mixed_precision=True,
        device_type="cuda",
        scaler=scaler,
        mixed_precision_stats=stats,
        clip_norm=0.5,
        step_scheduler=False,
        scheduler_per_batch=False,
        step_scheduler_fn=lambda: None,
        update_ema=False,
        update_ema_fn=lambda: None,
    )

    assert step_successful is False
    assert scaler.unscaled is True
    assert stats["scale_updates"] == 1
    assert stats["scale_decreases"] == 1
    assert stats["overflow_count"] == 1


def test_runtime_step_policy_mps_failed_step_updates_skip_counters():
    policy = RuntimeStepPolicy(logger=SimpleNamespace(info=lambda *a, **k: None))
    model = _SingleParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _FakeMpsScaler(scales=[1024.0, 512.0], step_successful=False)
    stats = _mp_stats()

    step_successful = policy.optimizer_step_with_clipping(
        model=model,
        optimizer=optimizer,
        use_mixed_precision=True,
        device_type="mps",
        scaler=scaler,
        mixed_precision_stats=stats,
        clip_norm=0.5,
        step_scheduler=False,
        scheduler_per_batch=False,
        step_scheduler_fn=lambda: None,
        update_ema=False,
        update_ema_fn=lambda: None,
    )

    assert step_successful is False
    assert stats["skipped_steps"] == 1
    assert stats["overflow_count"] == 1
    assert stats["scale_updates"] == 1
    assert stats["scale_decreases"] == 1


def test_runtime_step_policy_skipped_step_does_not_advance_scheduler_or_ema():
    policy = RuntimeStepPolicy(logger=SimpleNamespace(info=lambda *a, **k: None))
    model = _SingleParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _FakeMpsScaler(scales=[1024.0, 512.0], step_successful=False)

    scheduler_calls = {"count": 0}
    ema_calls = {"count": 0}

    step_successful = policy.optimizer_step_with_clipping(
        model=model,
        optimizer=optimizer,
        use_mixed_precision=True,
        device_type="mps",
        scaler=scaler,
        mixed_precision_stats=_mp_stats(),
        clip_norm=0.5,
        step_scheduler=True,
        scheduler_per_batch=True,
        step_scheduler_fn=lambda: scheduler_calls.__setitem__("count", scheduler_calls["count"] + 1),
        update_ema=True,
        update_ema_fn=lambda: ema_calls.__setitem__("count", ema_calls["count"] + 1),
    )

    assert step_successful is False
    assert scheduler_calls["count"] == 0
    assert ema_calls["count"] == 0


def test_memory_policy_adaptive_cleanup_path_uses_manager():
    policy = MemoryOOMPolicy(logger=SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None))
    manager = _FakeMemoryManager(adaptive_result={"cleaned": True, "pressure_level": "high"})

    result = policy.adaptive_cleanup(
        enable_adaptive_memory=True,
        memory_manager=manager,
        batch_idx=7,
        force=False,
        clear_device_cache_fn=lambda: None,
    )

    assert result == {"cleaned": True, "pressure_level": "high"}
    assert manager.adaptive_calls == [(7, False)]


def test_memory_policy_non_adaptive_cleanup_keeps_decision_contract():
    policy = MemoryOOMPolicy(logger=SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None))
    manager = _FakeMemoryManager()
    clear_calls = {"count": 0}

    result = policy.adaptive_cleanup(
        enable_adaptive_memory=False,
        memory_manager=manager,
        batch_idx=200,
        force=False,
        clear_device_cache_fn=lambda: clear_calls.__setitem__("count", clear_calls["count"] + 1),
    )

    assert result == {"cleaned": False, "pressure_level": "disabled"}
    assert clear_calls["count"] == 1


def test_memory_policy_oom_recovery_decision_matches_emergency_cleanup_success():
    policy = MemoryOOMPolicy(logger=SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None))
    manager_ok = _FakeMemoryManager(emergency_result={"success": True, "memory_freed_mb": 64.0})
    manager_fail = _FakeMemoryManager(emergency_result={"success": False, "memory_freed_mb": 0.0})

    can_continue_ok = policy.handle_oom(
        enable_adaptive_memory=True,
        memory_manager=manager_ok,
        batch_idx=3,
        error=RuntimeError("out of memory"),
        device_type="cuda",
        clear_device_cache_fn=lambda: None,
    )
    can_continue_fail = policy.handle_oom(
        enable_adaptive_memory=True,
        memory_manager=manager_fail,
        batch_idx=4,
        error=RuntimeError("out of memory"),
        device_type="cuda",
        clear_device_cache_fn=lambda: None,
    )

    assert can_continue_ok is True
    assert can_continue_fail is False


def test_memory_policy_oom_non_adaptive_fallback_always_continues_after_cleanup():
    policy = MemoryOOMPolicy(logger=SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None))
    manager = _FakeMemoryManager()
    clear_calls = {"count": 0}
    gc_calls = {"count": 0}

    can_continue = policy.handle_oom(
        enable_adaptive_memory=False,
        memory_manager=manager,
        batch_idx=5,
        error=RuntimeError("out of memory"),
        device_type="mps",
        clear_device_cache_fn=lambda: clear_calls.__setitem__("count", clear_calls["count"] + 1),
        gc_collect_fn=lambda: gc_calls.__setitem__("count", gc_calls["count"] + 1) or 0,
    )

    assert can_continue is True
    assert clear_calls["count"] == 1
    assert gc_calls["count"] == 1
