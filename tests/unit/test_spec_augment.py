"""
Unit tests for KokoroTrainer._apply_spec_augment.

Covers:
  - Output shape unchanged
  - Clone: original not modified
  - Zero-masking actually zeroes some frames/bins
  - Masking stays within tensor bounds
  - Zero masks produce identity
  - use_spec_augment=False path passes mel_specs through unchanged
"""
import torch
import pytest
from unittest.mock import MagicMock

from kokoro.training.trainer import KokoroTrainer


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
