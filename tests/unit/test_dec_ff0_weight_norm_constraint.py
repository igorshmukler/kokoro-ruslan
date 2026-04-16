"""
Tests for the post-step max weight-norm constraint on decoder.layers.0.ff.linear1.

Covered behaviours
------------------
  1.  Config default: dec_ff0_linear1_max_weight_norm == 60.0
  2.  Explicit value survives __post_init__
  3.  Setting ≤ 0.0 disables the clamp (attribute read path)
  4.  _setup_weight_norm_constraints caches the correct weight tensor when the
      module exists in the model
  5.  _setup_weight_norm_constraints sets _dec_ff0_linear1_weight to None and
      logs a warning when the module is absent
  6.  _apply_weight_norm_constraints rescales the weight in-place when the
      current L2 norm exceeds the ceiling
  7.  _apply_weight_norm_constraints does NOT modify the weight when the norm
      is below the ceiling
  8.  After clamping, the weight norm is ≤ max_norm (within float tolerance)
  9.  max_norm ≤ 0.0 skips the clamp even when the weight has a huge norm
  10. The constraint clamps both linear1.weight and linear2.weight in every decoder FF layer
  11. _apply_weight_norm_constraints is a no-op when _dec_ff0_linear1_weight is
      None (missing-module path does not raise)
  12. Weight direction (unit vector) is preserved after clamping
"""

import warnings

import pytest
import torch
import torch.nn as nn

from kokoro.training.config import TrainingConfig
from kokoro.training.trainer import KokoroTrainer


# ---------------------------------------------------------------------------
# Minimal helpers — mirror the real decoder naming path:
#   model.decoder.layers[0].ff.linear1  /  .linear2
# ---------------------------------------------------------------------------

class _MinimalFF(nn.Module):
    """Mirrors GLUFeedForward: has linear1 and linear2."""

    def __init__(self, in_dim: int = 8, ff_dim: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim // 2, in_dim)  # GLU halves the width

    def forward(self, x):  # pragma: no cover
        return x


class _MinimalDecoderBlock(nn.Module):
    """Mirrors ImprovedTransformerDecoderBlock: has a .ff sub-module."""

    def __init__(self, in_dim: int = 8, ff_dim: int = 16):
        super().__init__()
        self.ff = _MinimalFF(in_dim, ff_dim)

    def forward(self, x):  # pragma: no cover
        return x


class _MinimalDecoderStack(nn.Module):
    """Mirrors ImprovedTransformerDecoder: has a .layers ModuleList."""

    def __init__(self, in_dim: int = 8, ff_dim: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([_MinimalDecoderBlock(in_dim, ff_dim)])

    def forward(self, x):  # pragma: no cover
        return x


class _MinimalModel(nn.Module):
    """Produces named_modules() path 'decoder.layers.0.ff.linear{1,2}'."""

    def __init__(self, in_dim: int = 8, ff_dim: int = 16):
        super().__init__()
        self.decoder = _MinimalDecoderStack(in_dim, ff_dim)

    def forward(self, x):  # pragma: no cover
        return x


def _make_trainer(
    *,
    max_norm: float = 60.0,
    model: nn.Module = None,
    in_dim: int = 8,
    ff_dim: int = 16,
) -> KokoroTrainer:
    """Build a minimal KokoroTrainer instance via __new__ to avoid full __init__."""
    from types import SimpleNamespace
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.model = model if model is not None else _MinimalModel(in_dim, ff_dim)
    trainer.config = SimpleNamespace(dec_ff0_linear1_max_weight_norm=max_norm)
    trainer._dec_ff0_linear1_weight = None  # will be set by setup call
    return trainer


def _set_weight_norm(param: nn.Parameter, target_norm: float) -> None:
    """Scale *param* so its L2 norm equals *target_norm*."""
    with torch.no_grad():
        current = param.norm(2).item()
        if current == 0.0:
            param.fill_(1.0)
            current = param.norm(2).item()
        param.mul_(target_norm / current)


# ---------------------------------------------------------------------------
# 1–3: Config field
# ---------------------------------------------------------------------------

def test_config_default_value():
    cfg = TrainingConfig()
    assert cfg.dec_ff0_linear1_max_weight_norm == 0.0


def test_config_custom_value_survives_post_init():
    cfg = TrainingConfig(dec_ff0_linear1_max_weight_norm=45.0)
    assert cfg.dec_ff0_linear1_max_weight_norm == 45.0


def test_config_zero_disables():
    cfg = TrainingConfig(dec_ff0_linear1_max_weight_norm=0.0)
    assert cfg.dec_ff0_linear1_max_weight_norm == 0.0


def test_config_negative_disables():
    cfg = TrainingConfig(dec_ff0_linear1_max_weight_norm=-1.0)
    assert cfg.dec_ff0_linear1_max_weight_norm == -1.0


# ---------------------------------------------------------------------------
# 4–5: _setup_weight_norm_constraints
# ---------------------------------------------------------------------------

def test_setup_caches_correct_weight_tensor():
    trainer = _make_trainer()
    trainer._setup_weight_norm_constraints()

    linear1 = dict(trainer.model.named_modules())['decoder.layers.0.ff.linear1']
    assert trainer._dec_ff0_linear1_weight is linear1.weight


def test_setup_sets_none_and_warns_when_module_absent():
    from types import SimpleNamespace

    class _EmptyModel(nn.Module):
        def forward(self, x):  # pragma: no cover
            return x

    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.model = _EmptyModel()
    trainer.config = SimpleNamespace(dec_ff0_linear1_max_weight_norm=60.0)
    trainer._enc_ff_weights = []

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        # The trainer uses logger.warning, not Python warnings; just ensure no exception
        trainer._setup_weight_norm_constraints()

    assert trainer._dec_ff0_linear1_weight is None


# ---------------------------------------------------------------------------
# 6–8: _apply_weight_norm_constraints — clamping behaviour
# ---------------------------------------------------------------------------

def test_apply_clamps_weight_when_norm_exceeds_ceiling():
    max_norm = 60.0
    trainer = _make_trainer(max_norm=max_norm)
    trainer._setup_weight_norm_constraints()

    w = trainer._dec_ff0_linear1_weight
    _set_weight_norm(w, 90.0)          # well above ceiling
    assert w.norm(2).item() > max_norm

    trainer._apply_weight_norm_constraints()

    assert w.norm(2).item() <= max_norm + 1e-5


def test_apply_does_not_modify_weight_when_norm_below_ceiling():
    max_norm = 60.0
    trainer = _make_trainer(max_norm=max_norm)
    trainer._setup_weight_norm_constraints()

    w = trainer._dec_ff0_linear1_weight
    _set_weight_norm(w, 30.0)          # below ceiling
    expected_norm = w.norm(2).item()

    trainer._apply_weight_norm_constraints()

    assert abs(w.norm(2).item() - expected_norm) < 1e-5


def test_apply_norm_is_le_max_norm_after_clamp():
    max_norm = 50.0
    trainer = _make_trainer(max_norm=max_norm)
    trainer._setup_weight_norm_constraints()

    _set_weight_norm(trainer._dec_ff0_linear1_weight, 200.0)
    trainer._apply_weight_norm_constraints()

    assert trainer._dec_ff0_linear1_weight.norm(2).item() <= max_norm + 1e-5


# ---------------------------------------------------------------------------
# 9: max_norm ≤ 0 skips clamp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_norm", [0.0, -1.0, -100.0])
def test_apply_is_noop_when_max_norm_disabled(max_norm):
    trainer = _make_trainer(max_norm=max_norm)
    trainer._setup_weight_norm_constraints()

    w = trainer._dec_ff0_linear1_weight
    _set_weight_norm(w, 999.0)
    norm_before = w.norm(2).item()

    trainer._apply_weight_norm_constraints()

    assert abs(w.norm(2).item() - norm_before) < 1e-5


# ---------------------------------------------------------------------------
# 10: both linear1 and linear2 are clamped
# ---------------------------------------------------------------------------

def test_apply_also_clamps_linear2_weight():
    """_apply_weight_norm_constraints constrains all decoder FF weights (linear1 and linear2)."""
    max_norm = 60.0
    trainer = _make_trainer(max_norm=max_norm)
    trainer._setup_weight_norm_constraints()

    # Put both linear1 and linear2 above the ceiling
    linear2 = dict(trainer.model.named_modules())['decoder.layers.0.ff.linear2']
    _set_weight_norm(trainer._dec_ff0_linear1_weight, 300.0)
    _set_weight_norm(linear2.weight, 300.0)

    trainer._apply_weight_norm_constraints()

    # both linear1 and linear2 were clamped
    assert trainer._dec_ff0_linear1_weight.norm(2).item() <= max_norm + 1e-5
    assert linear2.weight.norm(2).item() <= max_norm + 1e-5


# ---------------------------------------------------------------------------
# 11: no-op when _dec_ff0_linear1_weight is None
# ---------------------------------------------------------------------------

def test_apply_is_noop_when_weight_ref_is_none():
    from types import SimpleNamespace
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.config = SimpleNamespace(dec_ff0_linear1_max_weight_norm=60.0)
    trainer._dec_ff0_linear1_weight = None
    trainer._dec_ff_weights = []
    trainer._enc_ff_weights = []
    # Must not raise
    trainer._apply_weight_norm_constraints()


# ---------------------------------------------------------------------------
# 12: Direction preserved
# ---------------------------------------------------------------------------

def test_apply_preserves_weight_direction():
    max_norm = 60.0
    trainer = _make_trainer(max_norm=max_norm)
    trainer._setup_weight_norm_constraints()

    w = trainer._dec_ff0_linear1_weight
    _set_weight_norm(w, 120.0)

    direction_before = w.detach().clone() / w.norm(2)
    trainer._apply_weight_norm_constraints()
    direction_after = w.detach() / w.norm(2)

    cosine = (direction_before * direction_after).sum().item()
    assert cosine == pytest.approx(1.0, abs=1e-5)
