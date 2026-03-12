"""
Unit tests for KokoroTrainer._setup_optimizer encoder/decoder param group split.

After the weight-decay fix, _setup_optimizer creates three AdamW param groups:
  group 0 — encoder params (text_embedding / transformer_encoder_layers / etc.)
             at encoder_lr_multiplier × base_lr, weight_decay=0
  group 1 — decoder/rest biases + norm affine params at base_lr, weight_decay=0
             (biases and LayerNorm scale/shift must not be L2-penalised)
  group 2 — all other decoder/variance-predictor weights at base_lr, weight_decay>0

Background
----------
Adam's 2nd moment (exp_avg_sq) was 10,000× lower in the encoder than the
decoder after 6 training epochs, leaving the encoder functionally frozen.
Root cause: a single param group meant the encoder had to share the same
base LR as the decoder despite having much smaller gradients.  A higher
effective LR for encoder params compensates without destabilising the decoder.

Additionally, applying weight_decay to biases and LayerNorm parameters penalises
learned statistics toward zero, hurting convergence. The no-decay split follows
the standard AdamW decoupled weight-decay pattern.
"""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from kokoro.training.trainer import KokoroTrainer


# ---------------------------------------------------------------------------
# Minimal model fixtures
# ---------------------------------------------------------------------------

class _FakeModel(nn.Module):
    """Model with both encoder-prefixed and decoder-prefixed parameters."""

    def __init__(self):
        super().__init__()
        # Must match encoder_prefixes in KokoroTrainer._setup_optimizer
        self.text_embedding = nn.Embedding(10, 8)
        self.transformer_encoder_layers = nn.Linear(8, 8)
        # Decoder / rest (no encoder prefix)
        self.decoder_layers = nn.Linear(8, 8)
        self.output_projection = nn.Linear(8, 4)


class _DecoderOnlyModel(nn.Module):
    """No parameters match any encoder prefix — triggers single-group fallback."""

    def __init__(self):
        super().__init__()
        self.decoder_layer = nn.Linear(8, 8)
        self.mel_projection = nn.Linear(8, 80)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_trainer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    encoder_lr_multiplier: float = 3.0,
) -> KokoroTrainer:
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.device = torch.device("cpu")
    trainer.device_type = "cpu"
    trainer.model = model
    trainer.config = SimpleNamespace(
        learning_rate=learning_rate,
        weight_decay=0.01,
        adam_eps=1e-8,
        adam_betas=(0.9, 0.999),
        encoder_lr_multiplier=encoder_lr_multiplier,
        use_fused_adamw=False,
        try_fused_adamw_on_mps=False,
    )
    return trainer


# ---------------------------------------------------------------------------
# Group structure
# ---------------------------------------------------------------------------

class TestParamGroupStructure:
    def test_creates_three_param_groups(self):
        trainer = _make_trainer(_FakeModel())
        trainer._setup_optimizer()
        assert len(trainer.optimizer.param_groups) == 3, (
            "Expected 3 groups: encoder, decoder_no_decay, decoder_decay"
        )

    def test_all_params_covered_across_groups(self):
        model = _FakeModel()
        trainer = _make_trainer(model)
        trainer._setup_optimizer()

        group_param_ids = {
            id(p)
            for g in trainer.optimizer.param_groups
            for p in g["params"]
        }
        model_param_ids = {id(p) for p in model.parameters()}
        assert model_param_ids == group_param_ids, (
            "Every model parameter must appear in exactly one optimizer group"
        )

    def test_all_groups_are_pairwise_disjoint(self):
        trainer = _make_trainer(_FakeModel())
        trainer._setup_optimizer()
        all_ids = [
            {id(p) for p in g["params"]}
            for g in trainer.optimizer.param_groups
        ]
        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                assert len(all_ids[i] & all_ids[j]) == 0, (
                    f"Param groups {i} and {j} must not share parameters"
                )

    def test_encoder_params_land_in_group_0(self):
        model = _FakeModel()
        trainer = _make_trainer(model)
        trainer._setup_optimizer()
        g0_ids = {id(p) for p in trainer.optimizer.param_groups[0]["params"]}

        for p in model.text_embedding.parameters():
            assert id(p) in g0_ids, "text_embedding params must be in the encoder group (0)"
        for p in model.transformer_encoder_layers.parameters():
            assert id(p) in g0_ids, "transformer_encoder_layers params must be in the encoder group (0)"

    def test_decoder_params_land_in_decoder_groups(self):
        """Decoder params must be in group 1 (no_decay) or group 2 (decay), not group 0."""
        model = _FakeModel()
        trainer = _make_trainer(model)
        trainer._setup_optimizer()
        decoder_ids = (
            {id(p) for p in trainer.optimizer.param_groups[1]["params"]}
            | {id(p) for p in trainer.optimizer.param_groups[2]["params"]}
        )
        g0_ids = {id(p) for p in trainer.optimizer.param_groups[0]["params"]}

        for p in model.decoder_layers.parameters():
            assert id(p) in decoder_ids, "decoder_layers params must be in a decoder group (1 or 2)"
            assert id(p) not in g0_ids, "decoder_layers params must not be in the encoder group (0)"
        for p in model.output_projection.parameters():
            assert id(p) in decoder_ids, "output_projection params must be in a decoder group (1 or 2)"
            assert id(p) not in g0_ids, "output_projection params must not be in the encoder group (0)"

    def test_bias_params_land_in_no_decay_group(self):
        """Bias tensors must end up in the no-decay group (group 1), not group 2."""
        model = _FakeModel()
        trainer = _make_trainer(model)
        trainer._setup_optimizer()
        g1_ids = {id(p) for p in trainer.optimizer.param_groups[1]["params"]}

        # decoder_layers.bias and output_projection.bias are non-encoder biases
        assert id(model.decoder_layers.bias) in g1_ids, (
            "decoder_layers.bias must be in the no-decay group (1)"
        )
        assert id(model.output_projection.bias) in g1_ids, (
            "output_projection.bias must be in the no-decay group (1)"
        )

    def test_weight_params_land_in_decay_group(self):
        """Non-bias weight matrices must end up in the decay group (group 2)."""
        model = _FakeModel()
        trainer = _make_trainer(model)
        trainer._setup_optimizer()
        g2_ids = {id(p) for p in trainer.optimizer.param_groups[2]["params"]}

        assert id(model.decoder_layers.weight) in g2_ids, (
            "decoder_layers.weight must be in the decay group (2)"
        )
        assert id(model.output_projection.weight) in g2_ids, (
            "output_projection.weight must be in the decay group (2)"
        )

    def test_no_decay_group_has_zero_weight_decay(self):
        trainer = _make_trainer(_FakeModel())
        trainer._setup_optimizer()
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0, (
            "decoder no-decay group (1) must have weight_decay=0"
        )

    def test_decay_group_has_nonzero_weight_decay(self):
        trainer = _make_trainer(_FakeModel())
        trainer._setup_optimizer()
        assert trainer.optimizer.param_groups[2]["weight_decay"] > 0.0, (
            "decoder decay group (2) must have weight_decay > 0"
        )


# ---------------------------------------------------------------------------
# Learning rates
# ---------------------------------------------------------------------------

class TestParamGroupLearningRates:
    def test_encoder_group_lr_equals_base_times_multiplier(self):
        base_lr = 1e-4
        enc_mult = 3.0
        trainer = _make_trainer(_FakeModel(), learning_rate=base_lr, encoder_lr_multiplier=enc_mult)
        trainer._setup_optimizer()
        enc_lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(enc_lr - base_lr * enc_mult) < 1e-12, (
            f"Encoder LR {enc_lr:.3e} != base_lr({base_lr:.3e}) × enc_mult({enc_mult})"
        )

    def test_decoder_no_decay_group_lr_equals_base_lr(self):
        base_lr = 1e-4
        trainer = _make_trainer(_FakeModel(), learning_rate=base_lr)
        trainer._setup_optimizer()
        dec_lr = trainer.optimizer.param_groups[1]["lr"]
        assert abs(dec_lr - base_lr) < 1e-12, (
            f"Decoder no-decay LR {dec_lr:.3e} != base_lr {base_lr:.3e}"
        )

    def test_decoder_decay_group_lr_equals_base_lr(self):
        base_lr = 1e-4
        trainer = _make_trainer(_FakeModel(), learning_rate=base_lr)
        trainer._setup_optimizer()
        dec_lr = trainer.optimizer.param_groups[2]["lr"]
        assert abs(dec_lr - base_lr) < 1e-12, (
            f"Decoder decay LR {dec_lr:.3e} != base_lr {base_lr:.3e}"
        )

    @pytest.mark.parametrize("enc_mult", [1.5, 2.0, 3.0, 5.0])
    def test_lr_ratio_matches_encoder_lr_multiplier(self, enc_mult: float):
        base_lr = 2e-5
        trainer = _make_trainer(_FakeModel(), learning_rate=base_lr, encoder_lr_multiplier=enc_mult)
        trainer._setup_optimizer()
        enc_lr = trainer.optimizer.param_groups[0]["lr"]
        dec_lr = trainer.optimizer.param_groups[2]["lr"]  # decay group; same base_lr as no-decay
        ratio = enc_lr / dec_lr
        assert abs(ratio - enc_mult) < 1e-9, (
            f"enc_lr / dec_lr = {ratio:.5f}, expected encoder_lr_multiplier = {enc_mult}"
        )


# ---------------------------------------------------------------------------
# Stored attribute
# ---------------------------------------------------------------------------

class TestEncoderLrMultiplierAttribute:
    def test_attribute_stored_on_trainer(self):
        enc_mult = 3.0
        trainer = _make_trainer(_FakeModel(), encoder_lr_multiplier=enc_mult)
        trainer._setup_optimizer()
        assert hasattr(trainer, "_encoder_lr_multiplier")
        assert trainer._encoder_lr_multiplier == enc_mult

    @pytest.mark.parametrize("enc_mult", [1.0, 2.0, 3.0, 5.0])
    def test_attribute_matches_config_value(self, enc_mult: float):
        trainer = _make_trainer(_FakeModel(), encoder_lr_multiplier=enc_mult)
        trainer._setup_optimizer()
        assert trainer._encoder_lr_multiplier == enc_mult


# ---------------------------------------------------------------------------
# Fallback: no encoder params identified
# ---------------------------------------------------------------------------

class TestSingleGroupFallback:
    def test_fallback_to_single_group_when_no_encoder_params(self):
        """When no model params match encoder prefixes, a single group is created."""
        trainer = _make_trainer(_DecoderOnlyModel(), encoder_lr_multiplier=3.0)
        trainer._setup_optimizer()
        assert len(trainer.optimizer.param_groups) == 1

    def test_fallback_resets_encoder_lr_multiplier_to_1(self):
        """Single-group fallback must neutralise the multiplier (no phantom LR boost)."""
        trainer = _make_trainer(_DecoderOnlyModel(), encoder_lr_multiplier=3.0)
        trainer._setup_optimizer()
        assert trainer._encoder_lr_multiplier == 1.0

    def test_fallback_covers_all_params(self):
        model = _DecoderOnlyModel()
        trainer = _make_trainer(model)
        trainer._setup_optimizer()
        group_ids = {id(p) for p in trainer.optimizer.param_groups[0]["params"]}
        model_ids = {id(p) for p in model.parameters()}
        assert model_ids == group_ids
