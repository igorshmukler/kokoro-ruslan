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
