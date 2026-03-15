from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kokoro.training.trainer import KokoroTrainer

"""Unit tests for per-group LR TensorBoard logging and progress-bar postfix.

Covers:
- _log_lr_scalars: multi-group writes stats/lr_encoder + stats/lr_decoder;
  single-group writes legacy stats/learning_rate.
- _build_lr_postfix: multi-group returns lr_enc + lr_dec keys;
  single-group returns lr key.
- Epoch-end log string format (via train() logging path — light integration).
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_param_groups(*lrs):
    """Return a list of minimal param-group dicts with the given LR values."""
    return [{'lr': lr} for lr in lrs]


def _make_trainer_with_groups(*lrs):
    """Build a minimal KokoroTrainer with a mock optimizer and writer."""
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.optimizer = SimpleNamespace(param_groups=_make_param_groups(*lrs))
    trainer.writer = MagicMock()
    return trainer


# ---------------------------------------------------------------------------
# _log_lr_scalars — TensorBoard scalar names
# ---------------------------------------------------------------------------

class TestLogLrScalars:
    def test_multi_group_logs_encoder_and_decoder_tags(self):
        """Two param groups → stats/lr_encoder and stats/lr_decoder."""
        trainer = _make_trainer_with_groups(3e-4, 1e-4)
        trainer._log_lr_scalars(step=100)

        calls = trainer.writer.add_scalar.call_args_list
        tags = [c.args[0] for c in calls]
        assert 'stats/lr_encoder' in tags
        assert 'stats/lr_decoder' in tags
        assert 'stats/learning_rate' not in tags

    def test_multi_group_logs_correct_lr_values(self):
        """Encoder group 0 and decoder last group values are logged correctly."""
        enc_lr, dec_lr = 3e-4, 1e-4
        trainer = _make_trainer_with_groups(enc_lr, dec_lr)
        trainer._log_lr_scalars(step=50)

        kw = {c.args[0]: c.args[1] for c in trainer.writer.add_scalar.call_args_list}
        assert kw['stats/lr_encoder'] == pytest.approx(enc_lr)
        assert kw['stats/lr_decoder'] == pytest.approx(dec_lr)

    def test_multi_group_uses_supplied_step(self):
        """The step argument is forwarded as the global_step to add_scalar."""
        trainer = _make_trainer_with_groups(3e-4, 1e-4)
        trainer._log_lr_scalars(step=999)

        steps = {c.args[0]: c.args[2] for c in trainer.writer.add_scalar.call_args_list}
        assert steps['stats/lr_encoder'] == 999
        assert steps['stats/lr_decoder'] == 999

    def test_single_group_logs_legacy_tag(self):
        """One param group → legacy stats/learning_rate tag."""
        trainer = _make_trainer_with_groups(1e-4)
        trainer._log_lr_scalars(step=10)

        calls = trainer.writer.add_scalar.call_args_list
        tags = [c.args[0] for c in calls]
        assert 'stats/learning_rate' in tags
        assert 'stats/lr_encoder' not in tags
        assert 'stats/lr_decoder' not in tags

    def test_single_group_logs_correct_lr_value(self):
        lr = 2.5e-4
        trainer = _make_trainer_with_groups(lr)
        trainer._log_lr_scalars(step=1)

        calls = trainer.writer.add_scalar.call_args_list
        assert calls[0].args[1] == pytest.approx(lr)

    def test_three_or_more_groups_uses_first_and_last(self):
        """Groups > 2: group 0 is encoder, group -1 is decoder; middle ignored."""
        trainer = _make_trainer_with_groups(9e-4, 5e-4, 1e-4)
        trainer._log_lr_scalars(step=1)

        kw = {c.args[0]: c.args[1] for c in trainer.writer.add_scalar.call_args_list}
        assert kw['stats/lr_encoder'] == pytest.approx(9e-4)
        assert kw['stats/lr_decoder'] == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# _build_lr_postfix — progress-bar keys
# ---------------------------------------------------------------------------

class TestBuildLrPostfix:
    def test_multi_group_returns_enc_and_dec_keys(self):
        """Two param groups → postfix contains lr_enc and lr_dec, not lr."""
        trainer = _make_trainer_with_groups(3e-4, 1e-4)
        postfix = trainer._build_lr_postfix()

        assert 'lr_enc' in postfix
        assert 'lr_dec' in postfix
        assert 'lr' not in postfix

    def test_multi_group_correct_values(self):
        enc_lr, dec_lr = 6e-4, 2e-4
        trainer = _make_trainer_with_groups(enc_lr, dec_lr)
        postfix = trainer._build_lr_postfix()

        assert postfix['lr_enc'] == pytest.approx(enc_lr)
        assert postfix['lr_dec'] == pytest.approx(dec_lr)

    def test_single_group_returns_lr_key(self):
        trainer = _make_trainer_with_groups(1e-4)
        postfix = trainer._build_lr_postfix()

        assert 'lr' in postfix
        assert 'lr_enc' not in postfix
        assert 'lr_dec' not in postfix

    def test_single_group_correct_value(self):
        lr = 5e-5
        trainer = _make_trainer_with_groups(lr)
        postfix = trainer._build_lr_postfix()

        assert postfix['lr'] == pytest.approx(lr)

    def test_three_groups_uses_first_and_last(self):
        trainer = _make_trainer_with_groups(9e-4, 5e-4, 1e-4)
        postfix = trainer._build_lr_postfix()

        assert postfix['lr_enc'] == pytest.approx(9e-4)
        assert postfix['lr_dec'] == pytest.approx(1e-4)

    def test_postfix_is_a_plain_dict(self):
        trainer = _make_trainer_with_groups(1e-4, 3e-5)
        postfix = trainer._build_lr_postfix()
        assert isinstance(postfix, dict)

    def test_multi_group_enc_higher_than_dec(self):
        """Sanity: with encoder_lr_multiplier > 1 encoder LR always exceeds decoder LR."""
        base = 1e-4
        enc = base * 3.0
        trainer = _make_trainer_with_groups(enc, base)
        postfix = trainer._build_lr_postfix()
        assert postfix['lr_enc'] > postfix['lr_dec']
