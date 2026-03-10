"""
Tests for stop-head gradient isolation via per-parameter clipping.

stop_token_predictor.weight and .bias receive a dedicated per-parameter
norm ceiling (stop_head_spike_clip_norm) applied in _preclip_projection_spikes
*before* the global clip_grad_norm_, preventing a stop-gradient spike from
consuming the mel decoder's gradient budget.

Covered behaviours:
  1.  Config default: stop_head_spike_clip_norm == 1.0
  2.  Custom value survives __post_init__
  3.  Setting ≤ 0 disables the stop-head clip
  4.  Stop-head params are clipped when their norm exceeds the limit
  5.  Mel-projection params are NOT clipped by stop_head_spike_clip_norm
  6.  Return dict records (pre_norm, ceiling) for each clipped param
  7.  Stop-head grad norm after clip is ≤ stop_head_spike_clip_norm
  8.  Mel-projection grad keeps its original norm while stop head is clipped
  9.  With stop_head_spike_clip_norm=0 stop-head grads are left untouched
  10. Both weight and bias of stop head are independently clipped
"""

import math
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from kokoro.training.config import TrainingConfig
from kokoro.training.trainer import KokoroTrainer


# ---------------------------------------------------------------------------
# Minimal model fixture that mirrors the real model's named parameters
# ---------------------------------------------------------------------------

class _MinimalModel(nn.Module):
    """Mirrors only the parameters that _preclip_projection_spikes inspects."""

    def __init__(self, hidden_dim: int = 8, mel_dim: int = 4):
        super().__init__()
        self.mel_projection_in  = nn.Linear(mel_dim,    hidden_dim)
        self.mel_projection_out = nn.Linear(hidden_dim, mel_dim)
        self.stop_token_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # pragma: no cover
        return x


def _build_trainer(
    stop_head_clip: float = 1.0,
    projection_clip: float = 20.0,
    attention_clip: float = 20.0,
    ffn_clip: float = 15.0,
    hidden_dim: int = 8,
    mel_dim: int = 4,
) -> KokoroTrainer:
    """Construct a minimal KokoroTrainer that exercises _preclip_projection_spikes."""
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.model = _MinimalModel(hidden_dim=hidden_dim, mel_dim=mel_dim)
    trainer.config = SimpleNamespace(
        stop_head_spike_clip_norm=stop_head_clip,
        projection_spike_clip_norm=projection_clip,
        attention_spike_clip_norm=attention_clip,
        ffn_spike_clip_norm=ffn_clip,
        encoder_ffn_spike_clip_norm=100.0,
    )
    # Replicate the assignments from _setup_grad_explosion_tracker
    trainer.projection_spike_clip_norm = projection_clip
    trainer.attention_spike_clip_norm  = attention_clip
    trainer.ffn_spike_clip_norm        = ffn_clip
    trainer.stop_head_spike_clip_norm  = stop_head_clip
    return trainer


def _inject_grad(module: nn.Linear, weight_norm: float, bias_norm: float) -> None:
    """Set synthetic gradients with the given L2 norms."""
    with torch.no_grad():
        w = module.weight
        b = module.bias
        w_flat = torch.ones_like(w.view(-1))
        w.grad = (w_flat * (weight_norm / (w_flat.norm() + 1e-12))).view_as(w).clone()
        b_flat = torch.ones_like(b)
        b.grad = (b_flat * (bias_norm / (b_flat.norm() + 1e-12))).clone()


# ---------------------------------------------------------------------------
# 1–3. Config defaults and disable path
# ---------------------------------------------------------------------------
class TestConfigDefaults:

    def test_default_stop_head_clip_norm(self):
        assert TrainingConfig.__dataclass_fields__['stop_head_spike_clip_norm'].default == pytest.approx(1.0)

    def test_custom_value_survives_post_init(self):
        cfg = TrainingConfig(stop_head_spike_clip_norm=5.0)
        assert cfg.stop_head_spike_clip_norm == pytest.approx(5.0)

    def test_zero_disables_stop_head_clip(self):
        """With stop_head_spike_clip_norm=0 the stop-head params must not appear in the clip dict."""
        trainer = _build_trainer(stop_head_clip=0.0)
        _inject_grad(trainer.model.stop_token_predictor, weight_norm=100.0, bias_norm=100.0)
        clipped = trainer._preclip_projection_spikes()
        stop_keys = {k for k in clipped if 'stop_token_predictor' in k}
        assert len(stop_keys) == 0, f"Expected no stop-head clips when norm=0, got {stop_keys}"

    def test_negative_clip_disables_stop_head_clip(self):
        trainer = _build_trainer(stop_head_clip=-1.0)
        _inject_grad(trainer.model.stop_token_predictor, weight_norm=50.0, bias_norm=50.0)
        clipped = trainer._preclip_projection_spikes()
        stop_keys = {k for k in clipped if 'stop_token_predictor' in k}
        assert len(stop_keys) == 0


# ---------------------------------------------------------------------------
# 4–7. Stop-head clipping correctness
# ---------------------------------------------------------------------------
class TestStopHeadClipping:

    def test_stop_head_weight_clipped_when_above_limit(self):
        clip = 1.0
        trainer = _build_trainer(stop_head_clip=clip)
        _inject_grad(trainer.model.stop_token_predictor, weight_norm=50.0, bias_norm=0.1)
        trainer._preclip_projection_spikes()

        grad_norm = trainer.model.stop_token_predictor.weight.grad.norm(2).item()
        assert grad_norm <= clip + 1e-5, (
            f"After clip, stop_token_predictor weight grad norm {grad_norm:.4f} > limit {clip}"
        )

    def test_stop_head_bias_clipped_when_above_limit(self):
        clip = 1.0
        trainer = _build_trainer(stop_head_clip=clip)
        _inject_grad(trainer.model.stop_token_predictor, weight_norm=0.1, bias_norm=20.0)
        trainer._preclip_projection_spikes()

        grad_norm = trainer.model.stop_token_predictor.bias.grad.norm(2).item()
        assert grad_norm <= clip + 1e-5, (
            f"After clip, stop_token_predictor bias grad norm {grad_norm:.4f} > limit {clip}"
        )

    def test_clipped_params_recorded_in_return_dict(self):
        clip = 1.0
        trainer = _build_trainer(stop_head_clip=clip)
        _inject_grad(trainer.model.stop_token_predictor, weight_norm=50.0, bias_norm=20.0)
        clipped = trainer._preclip_projection_spikes()

        assert 'stop_token_predictor.weight' in clipped, "weight should appear in clipped dict"
        assert 'stop_token_predictor.bias'   in clipped, "bias should appear in clipped dict"
        pre_w, ceiling_w = clipped['stop_token_predictor.weight']
        assert pre_w > clip,   f"recorded pre-norm {pre_w:.3f} should exceed ceiling {clip}"
        assert ceiling_w == pytest.approx(clip)

    def test_stop_head_not_clipped_when_below_limit(self):
        clip = 10.0
        trainer = _build_trainer(stop_head_clip=clip)
        _inject_grad(trainer.model.stop_token_predictor, weight_norm=0.5, bias_norm=0.2)

        before_w = trainer.model.stop_token_predictor.weight.grad.clone()
        before_b = trainer.model.stop_token_predictor.bias.grad.clone()
        clipped = trainer._preclip_projection_spikes()

        stop_keys = {k for k in clipped if 'stop_token_predictor' in k}
        assert len(stop_keys) == 0, "Should not clip when grad norms are below limit"
        assert torch.allclose(trainer.model.stop_token_predictor.weight.grad, before_w)
        assert torch.allclose(trainer.model.stop_token_predictor.bias.grad,   before_b)


# ---------------------------------------------------------------------------
# 8. Mel-projection grad is NOT affected by stop_head_spike_clip_norm
# ---------------------------------------------------------------------------
class TestIsolation:

    def test_mel_projection_grad_untouched_by_stop_head_clip(self):
        """The mel-projection weight grad must be identical before/after stop-head clipping."""
        # Give mel_projection_out a norm that's below its own ceiling (20.0)
        # so it won't be clipped for any other reason.
        trainer = _build_trainer(stop_head_clip=1.0, projection_clip=20.0)
        _inject_grad(trainer.model.mel_projection_out, weight_norm=5.0, bias_norm=2.0)
        _inject_grad(trainer.model.stop_token_predictor, weight_norm=100.0, bias_norm=100.0)

        mel_before = trainer.model.mel_projection_out.weight.grad.clone()
        trainer._preclip_projection_spikes()
        mel_after  = trainer.model.mel_projection_out.weight.grad

        assert torch.allclose(mel_before, mel_after), (
            "mel_projection_out.weight grad must not be modified by stop-head clip"
        )

    def test_stop_head_clip_does_not_affect_mel_input_projection(self):
        trainer = _build_trainer(stop_head_clip=1.0, projection_clip=100.0)
        _inject_grad(trainer.model.mel_projection_in, weight_norm=3.0, bias_norm=1.0)
        _inject_grad(trainer.model.stop_token_predictor, weight_norm=200.0, bias_norm=200.0)

        mel_in_before = trainer.model.mel_projection_in.weight.grad.clone()
        trainer._preclip_projection_spikes()
        mel_in_after  = trainer.model.mel_projection_in.weight.grad

        assert torch.allclose(mel_in_before, mel_in_after), (
            "mel_projection_in.weight grad must not be modified by stop-head clip"
        )


# ---------------------------------------------------------------------------
# 9. All-disabled early-exit guard
# ---------------------------------------------------------------------------
class TestEarlyExit:

    def test_all_clips_disabled_returns_empty_dict(self):
        """With all clip norms ≤ 0, _preclip_projection_spikes must return {} immediately."""
        trainer = _build_trainer(
            stop_head_clip=0.0,
            projection_clip=0.0,
            attention_clip=0.0,
            ffn_clip=0.0,
        )
        _inject_grad(trainer.model.stop_token_predictor, weight_norm=1000.0, bias_norm=1000.0)
        result = trainer._preclip_projection_spikes()
        assert result == {}

    def test_only_stop_head_enabled_does_not_early_exit(self):
        """If only stop_head_spike_clip_norm > 0, the function should still run and clip."""
        trainer = _build_trainer(
            stop_head_clip=1.0,
            projection_clip=0.0,
            attention_clip=0.0,
            ffn_clip=0.0,
        )
        _inject_grad(trainer.model.stop_token_predictor, weight_norm=50.0, bias_norm=50.0)
        result = trainer._preclip_projection_spikes()
        stop_keys = {k for k in result if 'stop_token_predictor' in k}
        assert len(stop_keys) > 0, "Expected stop-head params clipped when only stop_head_clip > 0"
