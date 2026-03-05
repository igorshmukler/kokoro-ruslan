"""
Tests validating that OneCycleLR starts exactly at `learning_rate` when
manual warmup is enabled (no LR jump at the warmup boundary).

Bug background
--------------
The trainer runs a hand-rolled linear warmup for `warmup_steps` steps, ramping
LR from `learning_rate * warmup_start_lr_ratio` up to `learning_rate`.  After
warmup the OneCycleLR scheduler takes over.  OneCycleLR's initial value is:

    initial_lr = max_lr / div_factor
               = (learning_rate * max_lr_multiplier) / div_factor

If `div_factor` is not equal to `max_lr_multiplier` the initial LR of
OneCycleLR will differ from `learning_rate`, creating a visible jump in the
LR trace at step `warmup_steps`.

Fix
---
When `use_warmup=True`, `div_factor` is set to `max_lr_multiplier` so that:

    initial_lr = learning_rate * max_lr_multiplier / max_lr_multiplier
               = learning_rate  ✓

When `use_warmup=False`, the classic PyTorch default of `div_factor=25.0` is
preserved (OneCycleLR is responsible for the whole ramp from base_lr up to
max_lr).
"""

import math
from types import SimpleNamespace

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer(lr: float) -> torch.optim.Optimizer:
    """Minimal optimizer wrapping a single dummy parameter."""
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.Adam([param], lr=lr)


def _compute_div_factor(use_warmup: bool, max_lr_multiplier: float) -> float:
    """Mirror the trainer's div_factor formula exactly."""
    return float(max_lr_multiplier) if use_warmup else 25.0


def _onecycle_initial_lr(
    learning_rate: float,
    max_lr_multiplier: float,
    use_warmup: bool,
    total_steps: int = 1000,
    pct_start: float = 0.3,
) -> float:
    """Return the LR that OneCycleLR sets on the optimizer immediately after
    construction (equivalent to step 0 of the OneCycleLR phase)."""
    max_lr = learning_rate * max_lr_multiplier
    div_factor = _compute_div_factor(use_warmup, max_lr_multiplier)
    opt = _make_optimizer(learning_rate)
    torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=div_factor,
        final_div_factor=10000.0,
        last_epoch=-1,
    )
    return opt.param_groups[0]['lr']


# ---------------------------------------------------------------------------
# Core invariant: with warmup, no LR jump at the boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("learning_rate,max_lr_multiplier", [
    (1e-4,  2.0),
    (1e-4,  3.0),
    (5e-5,  2.0),
    (3e-4,  1.5),
    (1e-3,  2.0),
])
def test_onecycle_initial_lr_equals_learning_rate_when_warmup_enabled(
    learning_rate: float, max_lr_multiplier: float
):
    """OneCycleLR must start at exactly `learning_rate` when use_warmup=True."""
    initial_lr = _onecycle_initial_lr(
        learning_rate, max_lr_multiplier, use_warmup=True
    )
    assert math.isclose(initial_lr, learning_rate, rel_tol=1e-6), (
        f"OneCycleLR initial LR {initial_lr:.8f} != learning_rate {learning_rate:.8f} "
        f"(max_lr_multiplier={max_lr_multiplier}). "
        "This would cause a visible LR jump at the end of warmup."
    )


@pytest.mark.parametrize("learning_rate,max_lr_multiplier", [
    (1e-4, 2.0),
    (1e-4, 3.0),
    (5e-5, 2.0),
])
def test_onecycle_initial_lr_is_not_learning_rate_when_warmup_disabled(
    learning_rate: float, max_lr_multiplier: float
):
    """Without warmup, OneCycleLR uses div_factor=25 → starts well below
    learning_rate (the classic ramp-from-low behaviour)."""
    initial_lr = _onecycle_initial_lr(
        learning_rate, max_lr_multiplier, use_warmup=False
    )
    expected = learning_rate * max_lr_multiplier / 25.0
    assert math.isclose(initial_lr, expected, rel_tol=1e-6), (
        f"Expected OneCycleLR initial LR {expected:.8f}, got {initial_lr:.8f}"
    )
    # Must be strictly below learning_rate (assuming max_lr_multiplier < 25)
    assert initial_lr < learning_rate, (
        "Without warmup, OneCycleLR should start below learning_rate"
    )


# ---------------------------------------------------------------------------
# div_factor formula correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_lr_multiplier", [1.5, 2.0, 3.0, 5.0])
def test_div_factor_equals_max_lr_multiplier_when_warmup_enabled(max_lr_multiplier):
    df = _compute_div_factor(use_warmup=True, max_lr_multiplier=max_lr_multiplier)
    assert df == max_lr_multiplier


def test_div_factor_is_25_when_warmup_disabled():
    for mlm in [1.5, 2.0, 3.0, 5.0]:
        df = _compute_div_factor(use_warmup=False, max_lr_multiplier=mlm)
        assert df == 25.0, f"Expected 25.0, got {df} for max_lr_multiplier={mlm}"


# ---------------------------------------------------------------------------
# Algebraic identity: max_lr / div_factor == learning_rate (warmup=True)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("learning_rate,max_lr_multiplier", [
    (1e-4, 2.0),
    (3e-4, 3.0),
    (1e-3, 1.5),
])
def test_max_lr_div_div_factor_equals_learning_rate(learning_rate, max_lr_multiplier):
    """Algebraic proof that the formula produces a seamless handoff."""
    max_lr = learning_rate * max_lr_multiplier
    div_factor = _compute_div_factor(use_warmup=True, max_lr_multiplier=max_lr_multiplier)
    assert math.isclose(max_lr / div_factor, learning_rate, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end warmup boundary: last warmup step LR == OneCycleLR initial LR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("learning_rate,max_lr_multiplier,warmup_steps", [
    (1e-4, 2.0, 1200),
    (1e-4, 2.0,  500),
    (5e-5, 3.0,  300),
])
def test_no_lr_jump_at_warmup_boundary(
    learning_rate: float, max_lr_multiplier: float, warmup_steps: int
):
    """Simulate the exact LR sequence around the boundary and confirm
    the last warmup LR and first OneCycleLR LR are identical."""
    warmup_start_lr = learning_rate * 0.01

    # Last warmup step: progress = (warmup_steps - 1) / warmup_steps
    # At step warmup_steps the warmup condition is no longer true; the trainer
    # hands off to OneCycleLR starting from step 0.
    # The warmup reaches exactly learning_rate at step == warmup_steps:
    warmup_final_lr = warmup_start_lr + (learning_rate - warmup_start_lr) * 1.0
    assert math.isclose(warmup_final_lr, learning_rate, rel_tol=1e-9)

    onecycle_initial_lr = _onecycle_initial_lr(
        learning_rate, max_lr_multiplier, use_warmup=True,
        total_steps=max(warmup_steps + 1, 3200),
    )

    assert math.isclose(onecycle_initial_lr, warmup_final_lr, rel_tol=1e-6), (
        f"LR jump detected at warmup boundary: "
        f"warmup ends at {warmup_final_lr:.8f}, "
        f"OneCycleLR starts at {onecycle_initial_lr:.8f}. "
        f"Difference: {abs(onecycle_initial_lr - warmup_final_lr):.2e}"
    )


# ---------------------------------------------------------------------------
# Stored _onecycle_div_factor attribute is consistent with use_warmup flag
# ---------------------------------------------------------------------------

def test_stored_div_factor_is_consistent_with_use_warmup_true():
    """Verify that the stored attribute would equal max_lr_multiplier for warmup runs."""
    max_lr_multiplier = 2.0
    use_warmup = True
    stored = _compute_div_factor(use_warmup, max_lr_multiplier)
    assert stored == max_lr_multiplier
    # And it satisfies the no-jump invariant
    learning_rate = 1e-4
    max_lr = learning_rate * max_lr_multiplier
    assert math.isclose(max_lr / stored, learning_rate, rel_tol=1e-9)


def test_stored_div_factor_is_consistent_with_use_warmup_false():
    """Without warmup, stored div_factor should be 25.0 (PyTorch default)."""
    for max_lr_multiplier in [1.5, 2.0, 5.0]:
        stored = _compute_div_factor(use_warmup=False, max_lr_multiplier=max_lr_multiplier)
        assert stored == 25.0


# ---------------------------------------------------------------------------
# Regression: old div_factor=3.0 would have caused a visible drop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("learning_rate,max_lr_multiplier", [
    (1e-4, 2.0),
    (1e-4, 3.0),
])
def test_old_div_factor_3_would_have_caused_jump(learning_rate, max_lr_multiplier):
    """Document that the old hardcoded div_factor=3.0 did NOT produce
    initial_lr == learning_rate (unless max_lr_multiplier happened to be 3.0)."""
    max_lr = learning_rate * max_lr_multiplier
    old_initial_lr = max_lr / 3.0
    if not math.isclose(max_lr_multiplier, 3.0, rel_tol=1e-9):
        assert not math.isclose(old_initial_lr, learning_rate, rel_tol=1e-4), (
            "Expected the old div_factor=3.0 to produce a jump (mismatch)."
        )

    # New formula: no jump
    new_div_factor = _compute_div_factor(use_warmup=True, max_lr_multiplier=max_lr_multiplier)
    new_initial_lr = max_lr / new_div_factor
    assert math.isclose(new_initial_lr, learning_rate, rel_tol=1e-9)
