import types
import time
import builtins
import torch


def test_get_current_memory_stats_cpu_with_psutil(monkeypatch):
    import kokoro.utils.adaptive_memory_manager as amm

    # Fake psutil.virtual_memory() result
    class VM:
        total = 16 * 1024 ** 3  # 16 GB in bytes
        available = 8 * 1024 ** 3  # 8 GB available

    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: VM())

    monkeypatch.setattr(amm, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(amm, "psutil", fake_psutil)

    mgr = amm.AdaptiveMemoryManager(device=torch.device("cpu"))
    stats = mgr.get_current_memory_stats()

    # Expect 8GB used out of 16GB -> 50%
    assert abs(stats["current_mb"] - ((VM.total - VM.available) / 1024 ** 2)) < 1e-3
    assert abs(stats["usage_percent"] - 50.0) < 1e-6


def test_get_current_memory_stats_mps_with_psutil_and_torch(monkeypatch):
    import kokoro.utils.adaptive_memory_manager as amm

    # Fake psutil total
    class VM:
        total = 32 * 1024 ** 3  # 32 GB
        # make `available` present so module calculations that reference it do not fail
        available = total - (1 * 1024 ** 3)

    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: VM())

    # Fake torch.mps current allocated memory (in bytes)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 1 * 1024 ** 3)

    monkeypatch.setattr(amm, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(amm, "psutil", fake_psutil)

    mgr = amm.AdaptiveMemoryManager(device=torch.device("mps"))
    stats = mgr.get_current_memory_stats()

    expected_current_mb = (1 * 1024 ** 3) / 1024 ** 2
    assert abs(stats["current_mb"] - expected_current_mb) < 1e-6
    # usage_percent = current / total
    expected_percent = (expected_current_mb / (VM.total / 1024 ** 2)) * 100
    assert abs(stats["usage_percent"] - expected_percent) < 1e-6


def test_get_current_memory_stats_cuda_with_monkeypatched_torch(monkeypatch):
    import kokoro.utils.adaptive_memory_manager as amm
    from types import SimpleNamespace

    # Simulate CUDA device properties and memory APIs
    total_bytes = 8 * 1024 ** 3
    allocated_bytes = 2 * 1024 ** 3
    reserved_bytes = 3 * 1024 ** 3
    peak_bytes = 4 * 1024 ** 3

    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: allocated_bytes)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: reserved_bytes)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: peak_bytes)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=total_bytes))

    # Ensure psutil fallback is disabled for CUDA
    monkeypatch.setattr(amm, "PSUTIL_AVAILABLE", False)

    mgr = amm.AdaptiveMemoryManager(device=torch.device("cuda:0"))
    stats = mgr.get_current_memory_stats()

    assert abs(stats["current_mb"] - allocated_bytes / 1024 ** 2) < 1e-6
    assert abs(stats["reserved_mb"] - reserved_bytes / 1024 ** 2) < 1e-6
    assert abs(stats["peak_mb"] - peak_bytes / 1024 ** 2) < 1e-6
    assert abs(stats["total_mb"] - (total_bytes / 1024 ** 2)) < 1e-6
"""Unit tests for adaptive memory manager behavior."""

import torch
import pytest

from kokoro.utils.adaptive_memory_manager import AdaptiveMemoryManager, MemoryPressureLevel


def test_memory_trend_uses_recent_rolling_window():
    manager = AdaptiveMemoryManager(torch.device("cpu"))

    usages = [4.0, 4.1, 4.2, 4.0, 4.3, 4.5, 4.4, 4.6, 4.7, 4.8]
    for batch_idx, usage in enumerate(usages):
        manager.batch_count = batch_idx
        manager._update_memory_trend(usage)

    old_avg = sum(usages[:5]) / 5
    new_avg = sum(usages[5:]) / 5
    assert manager.memory_trend == pytest.approx(new_avg - old_avg)


def test_should_cleanup_low_pressure_only_on_extended_interval():
    manager = AdaptiveMemoryManager(torch.device("cpu"))
    manager.current_pressure = MemoryPressureLevel.LOW

    interval = manager.strategies[MemoryPressureLevel.LOW].check_interval

    manager.batch_count = interval
    assert manager.should_cleanup() is False

    manager.batch_count = interval * 2
    assert manager.should_cleanup() is True


def test_get_memory_report_uses_recent_memory_history_stats():
    manager = AdaptiveMemoryManager(torch.device("cpu"))
    manager.batch_count = 100
    manager.cleanup_count = 5
    manager.total_cleanup_time = 0.25
    manager.current_pressure = MemoryPressureLevel.MODERATE
    manager.memory_trend = -0.21
    manager.memory_history = [
        {"batch": i, "usage_percent": float(i), "timestamp": float(i)}
        for i in range(1, 11)
    ]

    report = manager.get_memory_report()

    assert report["current_pressure"] == "moderate"
    assert report["current_memory_usage_percent"] == pytest.approx(10.0)
    assert report["avg_memory_usage_percent"] == pytest.approx(5.5)
    assert report["max_memory_usage_percent"] == pytest.approx(10.0)
    assert report["min_memory_usage_percent"] == pytest.approx(1.0)
    assert report["memory_trend"] == pytest.approx(-0.21)
