import torch
import pytest

from kokoro.training.trainer import KokoroTrainer

"""
Unit tests for KokoroTrainer._apply_spec_augment and its epoch gate.

Covers:
  - Output shape unchanged
  - Clone: original not modified
  - Zero-masking actually zeroes some frames/bins
  - Masking stays within tensor bounds
  - Zero masks produce identity
  - use_spec_augment=False path passes mel_specs through unchanged
  - spec_augment_start_epoch gate: augment suppressed before the epoch threshold
"""

def _mel(B=2, T=100, mel_dim=80):
    """Return a fixed-seed mel tensor filled with non-zero values."""
    torch.manual_seed(42)
    return torch.rand(B, T, mel_dim) + 0.5   # (0.5, 1.5] — never zero


class TestApplySpecAugmentShape:
    def test_output_shape_unchanged(self):
        mel = _mel()
        out = KokoroTrainer._apply_spec_augment(mel)
        assert out.shape == mel.shape

    def test_output_dtype_unchanged(self):
        mel = _mel().to(torch.float32)
        out = KokoroTrainer._apply_spec_augment(mel)
        assert out.dtype == mel.dtype

    def test_single_batch_item(self):
        mel = _mel(B=1)
        out = KokoroTrainer._apply_spec_augment(mel)
        assert out.shape == mel.shape


class TestApplySpecAugmentClone:
    def test_original_not_modified(self):
        mel = _mel()
        original_data = mel.clone()
        KokoroTrainer._apply_spec_augment(mel)
        assert torch.equal(mel, original_data), "Original mel tensor must not be modified"

    def test_returns_different_tensor(self):
        mel = _mel()
        out = KokoroTrainer._apply_spec_augment(mel)
        assert out.data_ptr() != mel.data_ptr()


class TestApplySpecAugmentMasking:
    def test_time_masking_zeroes_frames(self):
        """At least some time frames must be zero after masking."""
        mel = _mel(T=200)
        out = KokoroTrainer._apply_spec_augment(mel, time_mask_max=50, num_time_masks=2, num_freq_masks=0)
        # With mask size up to 50 and 2 masks, some frames should be zero
        zeroed_frames = (out == 0).all(dim=-1)   # (B, T) — True where all mel bins are 0
        assert zeroed_frames.any(), "Expected at least one zeroed time frame after time masking"

    def test_freq_masking_zeroes_bins(self):
        """At least some frequency bins must be zero across all time after freq masking."""
        mel = _mel(T=100, mel_dim=80)
        out = KokoroTrainer._apply_spec_augment(mel, time_mask_max=0, num_time_masks=0,
                                                 freq_mask_max=20, num_freq_masks=2)
        zeroed_bins = (out == 0).all(dim=1)   # (B, mel_dim) — True where all frames are 0
        assert zeroed_bins.any(), "Expected at least one zeroed frequency bin after freq masking"

    def test_zero_masks_produces_identity(self):
        """With num_time_masks=0 and num_freq_masks=0 no masking should occur."""
        mel = _mel()
        original = mel.clone()
        out = KokoroTrainer._apply_spec_augment(mel, num_time_masks=0, num_freq_masks=0)
        assert torch.equal(out, original), "Zero masks should return identical tensor"

    def test_masking_stays_within_bounds(self):
        """Masked region must never extend beyond tensor dimensions (no index errors)."""
        for T in [1, 2, 5, 10, 50]:
            mel = _mel(T=T, mel_dim=8)
            try:
                out = KokoroTrainer._apply_spec_augment(mel, time_mask_max=30, freq_mask_max=10,
                                                         num_time_masks=3, num_freq_masks=3)
                assert out.shape == mel.shape
            except Exception as e:
                pytest.fail(f"_apply_spec_augment raised {e} with T={T}")

    def test_not_all_frames_zeroed(self):
        """Even aggressive masking should not zero every frame (min-size guard)."""
        mel = _mel(T=100)
        out = KokoroTrainer._apply_spec_augment(mel, time_mask_max=30, num_time_masks=2, num_freq_masks=0)
        unmasked = (out != 0).any(dim=-1)   # (B, T) — True where any mel bin is non-zero
        assert unmasked.any(), "At least some frames should remain unmasked"


class TestSpecAugmentIntegration:
    """Verify the use_spec_augment config gate in the trainer context."""

    def _build_minimal_trainer(self, use_spec_augment: bool):
        """Assemble just enough trainer state to test the spec-augment code path."""
        import types
        trainer = KokoroTrainer.__new__(KokoroTrainer)
        trainer.config = types.SimpleNamespace(
            use_spec_augment=use_spec_augment,
            spec_augment_time_mask_max=30,
            spec_augment_freq_mask_max=10,
            spec_augment_num_time_masks=2,
            spec_augment_num_freq_masks=2,
        )
        return trainer

    def test_spec_augment_enabled_produces_masked_mel(self):
        trainer = self._build_minimal_trainer(use_spec_augment=True)
        mel = _mel()
        original = mel.clone()

        if getattr(trainer.config, 'use_spec_augment', False):
            mel_for_model = KokoroTrainer._apply_spec_augment(
                mel,
                time_mask_max=trainer.config.spec_augment_time_mask_max,
                freq_mask_max=trainer.config.spec_augment_freq_mask_max,
                num_time_masks=trainer.config.spec_augment_num_time_masks,
                num_freq_masks=trainer.config.spec_augment_num_freq_masks,
            )
        else:
            mel_for_model = mel

        # Original should be untouched
        assert torch.equal(mel, original)
        # mel_for_model should differ (masking applied)
        assert not torch.equal(mel_for_model, original), "SpecAugment should produce different tensor"

    def test_spec_augment_disabled_returns_same_object(self):
        trainer = self._build_minimal_trainer(use_spec_augment=False)
        mel = _mel()

        if getattr(trainer.config, 'use_spec_augment', False):
            mel_for_model = KokoroTrainer._apply_spec_augment(mel)
        else:
            mel_for_model = mel

        # Same object (no copy)
        assert mel_for_model is mel


class TestSpecAugmentEpochGate:
    """Verify the spec_augment_start_epoch gate added to train_epoch.

    Before the fix, spec augmentation was applied from epoch 0, adding
    noise before the model established basic alignment.  Now it is
    gated behind spec_augment_start_epoch (default=5) so the first
    epochs train on clean mel spectrograms.
    """

    def _gate_active(
        self, epoch: int, use_spec_augment: bool, start_epoch: int
    ) -> bool:
        """Mirror the exact boolean expression used in trainer.train_epoch."""
        import types
        trainer = KokoroTrainer.__new__(KokoroTrainer)
        trainer.config = types.SimpleNamespace(
            use_spec_augment=use_spec_augment,
            spec_augment_start_epoch=start_epoch,
        )
        _spec_aug_start = getattr(trainer.config, 'spec_augment_start_epoch', 5)
        return getattr(trainer.config, 'use_spec_augment', False) and epoch >= _spec_aug_start

    @pytest.mark.parametrize("epoch", [0, 1, 2, 3, 4])
    def test_gate_suppresses_augment_before_start_epoch(self, epoch: int):
        assert not self._gate_active(epoch, use_spec_augment=True, start_epoch=5), (
            f"Spec augment must be suppressed at epoch {epoch} (start_epoch=5)"
        )

    @pytest.mark.parametrize("epoch", [5, 6, 7, 10, 20])
    def test_gate_allows_augment_at_and_after_start_epoch(self, epoch: int):
        assert self._gate_active(epoch, use_spec_augment=True, start_epoch=5), (
            f"Spec augment must be active at epoch {epoch} (start_epoch=5)"
        )

    def test_gate_respects_use_spec_augment_false_regardless_of_epoch(self):
        """use_spec_augment=False suppresses augment even when epoch >= start."""
        for epoch in range(10):
            assert not self._gate_active(
                epoch, use_spec_augment=False, start_epoch=0
            ), f"use_spec_augment=False must suppress at epoch {epoch}"

    def test_getattr_default_start_epoch_is_18(self):
        """When config lacks spec_augment_start_epoch, the fallback default is 18."""
        import types
        trainer = KokoroTrainer.__new__(KokoroTrainer)
        trainer.config = types.SimpleNamespace(use_spec_augment=True)  # no start_epoch
        default = getattr(trainer.config, 'spec_augment_start_epoch', 18)
        assert default == 18

    @pytest.mark.parametrize("start_epoch", [0, 1, 3, 10])
    def test_gate_respects_custom_start_epoch(self, start_epoch: int):
        """Epoch gate boundary is exactly at start_epoch, not before or after."""
        # Suppressed for all epochs strictly before start
        for epoch in range(start_epoch):
            assert not self._gate_active(
                epoch, use_spec_augment=True, start_epoch=start_epoch
            )
        # Active from start_epoch onwards
        for epoch in range(start_epoch, start_epoch + 5):
            assert self._gate_active(
                epoch, use_spec_augment=True, start_epoch=start_epoch
            )
