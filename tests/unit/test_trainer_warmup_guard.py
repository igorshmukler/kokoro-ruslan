"""
Tests covering KokoroTrainer._apply_warmup_guard — the staticmethod extracted
from KokoroTrainer.__init__ to guard OneCycleLR against receiving a
non-positive total_steps when warmup_steps >= total_steps.

Before fix: onecycle_steps = total_steps - self.warmup_steps
            → negative when warmup_steps > total_steps → OneCycleLR crash
After fix:  Guard clamps warmup_steps to total_steps - 1 first, then
            onecycle_steps = max(1, total_steps - warmup_steps) >= 1
"""
import pytest
from kokoro.training.trainer import KokoroTrainer


# ---------------------------------------------------------------------------
# Normal operation: warmup_steps < total_steps — no clamping, no change
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("warmup_steps,total_steps", [
    (0,  100),
    (10, 100),
    (99, 100),
    (1,  2),
    (0,  1),
])
def test_guard_normal_case_no_clamping(warmup_steps: int, total_steps: int):
    """When warmup_steps < total_steps, neither value is modified."""
    clamped_warmup, onecycle_steps = KokoroTrainer._apply_warmup_guard(
        warmup_steps, total_steps
    )
    assert clamped_warmup == warmup_steps, (
        "warmup_steps should not be modified when it is strictly less than total_steps"
    )
    assert onecycle_steps == total_steps - warmup_steps


# ---------------------------------------------------------------------------
# Edge: warmup_steps == total_steps — must clamp
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 5, 100])
def test_guard_clamps_when_warmup_equals_total(n: int):
    clamped_warmup, onecycle_steps = KokoroTrainer._apply_warmup_guard(n, n)
    assert clamped_warmup == n - 1, (
        f"warmup_steps ({n}) == total_steps ({n}) → should be clamped to {n - 1}"
    )
    assert onecycle_steps == 1


# ---------------------------------------------------------------------------
# warmup_steps > total_steps — must clamp
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("warmup_steps,total_steps", [
    (101, 100),
    (200, 50),
    (1000, 10),
])
def test_guard_clamps_when_warmup_exceeds_total(warmup_steps: int, total_steps: int):
    clamped_warmup, onecycle_steps = KokoroTrainer._apply_warmup_guard(
        warmup_steps, total_steps
    )
    assert clamped_warmup < total_steps, (
        "Clamped warmup_steps must be strictly less than total_steps"
    )
    assert onecycle_steps >= 1, "onecycle_steps must be >= 1 after guard"


# ---------------------------------------------------------------------------
# onecycle_steps is always >= 1 regardless of inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("warmup_steps,total_steps", [
    (0, 1),
    (1, 1),
    (5, 1),
    (0, 100),
    (100, 100),
    (150, 100),
])
def test_guard_onecycle_steps_always_positive(warmup_steps: int, total_steps: int):
    _, onecycle_steps = KokoroTrainer._apply_warmup_guard(warmup_steps, total_steps)
    assert onecycle_steps >= 1, (
        f"onecycle_steps must always be >= 1 (got {onecycle_steps} for "
        f"warmup_steps={warmup_steps}, total_steps={total_steps})"
    )


# ---------------------------------------------------------------------------
# Return types are int
# ---------------------------------------------------------------------------
def test_guard_returns_integers():
    clamped_warmup, onecycle_steps = KokoroTrainer._apply_warmup_guard(10, 100)
    assert isinstance(clamped_warmup, int)
    assert isinstance(onecycle_steps, int)


# ---------------------------------------------------------------------------
# total_steps=1, warmup_steps=0 — minimal valid configuration
# ---------------------------------------------------------------------------
def test_guard_minimal_valid_configuration():
    clamped_warmup, onecycle_steps = KokoroTrainer._apply_warmup_guard(0, 1)
    assert clamped_warmup == 0
    assert onecycle_steps == 1


# ---------------------------------------------------------------------------
# Clamped warmup_steps result is consistent: clamped = max(0, total - 1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("warmup_steps,total_steps,expected_warmup,expected_onecycle", [
    (100, 100, 99, 1),
    (200, 100, 99, 1),
    (1,   1,   0,  1),
    (2,   1,   0,  1),
])
def test_guard_exact_clamped_values(
    warmup_steps: int, total_steps: int,
    expected_warmup: int, expected_onecycle: int
):
    clamped_warmup, onecycle_steps = KokoroTrainer._apply_warmup_guard(
        warmup_steps, total_steps
    )
    assert clamped_warmup == expected_warmup, (
        f"Expected clamped_warmup={expected_warmup}, got {clamped_warmup}"
    )
    assert onecycle_steps == expected_onecycle, (
        f"Expected onecycle_steps={expected_onecycle}, got {onecycle_steps}"
    )
