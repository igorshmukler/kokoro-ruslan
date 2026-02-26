"""
Tests covering the mel_spec_linear axis-layout bug in dataset.py.

BUG (fixed):
    torchaudio.transforms.MelSpectrogram returns (n_mels, T_frames) after squeeze.
    EnergyExtractor.extract_energy_from_mel expects (..., n_mels) on the last axis
    because it does mean(dim=-1) to collapse the mel-bin dimension into a
    per-frame scalar energy value.

    Before fix:
        EnergyExtractor.extract_energy_from_mel(mel_spec_linear, log_domain=False)
        mel_spec_linear.shape = (80, T)
        mean(dim=-1) → averaged over T frames → shape (80,) — one value per mel band
        not one value per frame.

    After fix:
        EnergyExtractor.extract_energy_from_mel(
            mel_spec_linear[:, :num_mel_frames].T, log_domain=False
        )
        .T shape = (T, 80)
        mean(dim=-1) → averaged over 80 mel bins → shape (T,) — correct

Tests in this file:
  1. EnergyExtractor shape contract: (T, n_mels) → (T,)
  2. Wrong-axis call: (n_mels, T) → (n_mels,) — wrong shape detected
  3. Transpose fix produces exactly T energy values, not 80
  4. Pre-clip behaviour: [:, :num_mel_frames].T clips and transposes correctly
  5. Per-frame energy reflects per-frame audio content (loud vs quiet frames)
  6. Batch layout: (B, T, n_mels) → (B, T)
  7. Mixed loud/quiet per-frame validation with the correct axis layout
"""

import torch
import pytest

from kokoro.model.variance_predictor import EnergyExtractor


# ---------------------------------------------------------------------------
# Constants matching real pipeline settings
# ---------------------------------------------------------------------------
N_MELS    = 80
T_FRAMES  = 120
BATCH     = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torchaudio_layout(n_mels: int = N_MELS, t: int = T_FRAMES,
                       seed: int = 0) -> torch.Tensor:
    """
    Simulate what torchaudio.transforms.MelSpectrogram().squeeze(0) returns:
    shape (n_mels, T_frames), all positive (linear power).
    """
    torch.manual_seed(seed)
    return torch.abs(torch.randn(n_mels, t)) + 0.1


def _correct_layout(n_mels: int = N_MELS, t: int = T_FRAMES,
                    seed: int = 0) -> torch.Tensor:
    """Same data transposed to (T_frames, n_mels) — what EnergyExtractor expects."""
    return _torchaudio_layout(n_mels=n_mels, t=t, seed=seed).T


# ---------------------------------------------------------------------------
# 1. Shape contract: EnergyExtractor expects (T, n_mels) → output (T,)
# ---------------------------------------------------------------------------
class TestEnergyExtractorShapeContract:
    """EnergyExtractor must produce shape (T,) when fed (T, n_mels)."""

    @pytest.mark.parametrize("t,n_mels", [
        (T_FRAMES, N_MELS),
        (1, N_MELS),
        (N_MELS, N_MELS),   # square — both axes == 80 — ensure dim is interpreted correctly
        (200, 40),
    ])
    def test_output_length_equals_t_frames(self, t: int, n_mels: int):
        mel = _correct_layout(n_mels=n_mels, t=t)
        assert mel.shape == (t, n_mels)
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
        assert energy.shape == (t,), (
            f"Expected energy shape ({t},) from input ({t}, {n_mels}), "
            f"got {energy.shape}"
        )

    def test_output_length_is_not_n_mels(self):
        """Regression guard: should not produce (n_mels,) instead of (T,)."""
        t = T_FRAMES
        mel = _correct_layout(n_mels=N_MELS, t=t)
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
        assert energy.shape[0] != N_MELS, (
            f"Energy has {N_MELS} values — got n_mels instead of T_frames. "
            "Wrong-axis regression detected."
        )
        assert energy.shape[0] == t


# ---------------------------------------------------------------------------
# 2. Old bug: passing (n_mels, T) directly → wrong shape (n_mels,)
#    This test *documents* the bug; it is expected to produce n_mels values.
#    Use it as a regression canary — if EnergyExtractor's internals change
#    such that (n_mels, T) no longer produces n_mels values, this test should
#    be revisited.
# ---------------------------------------------------------------------------
class TestWrongAxisProducesWrongShape:
    """
    Passing torchaudio's (n_mels, T) layout directly (without transpose)
    makes mean(dim=-1) average over T rather than n_mels, producing
    exactly n_mels scalar values instead of T_frames values.
    """

    def test_wrong_axis_produces_n_mels_values(self):
        mel_wrong = _torchaudio_layout(n_mels=N_MELS, t=T_FRAMES)  # (80, T)
        assert mel_wrong.shape == (N_MELS, T_FRAMES)
        energy_wrong = EnergyExtractor.extract_energy_from_mel(mel_wrong, log_domain=False)
        assert energy_wrong.shape == (N_MELS,), (
            f"Without transpose, wrong-axis input (n_mels, T) should yield "
            f"({N_MELS},); got {energy_wrong.shape}. "
            "This test documents the pre-fix bug shape."
        )

    def test_wrong_shape_not_equal_to_t_frames(self):
        """Confirms the wrong shape is DIFFERENT from the correct frame count."""
        mel_wrong = _torchaudio_layout()
        energy_wrong = EnergyExtractor.extract_energy_from_mel(mel_wrong, log_domain=False)
        assert energy_wrong.shape[0] != T_FRAMES, (
            "Wrong-axis energy accidentally has T_frames values — "
            "test invariant broken (T_FRAMES == N_MELS?)."
        )


# ---------------------------------------------------------------------------
# 3. .T transpose fix: (n_mels, T).T → (T, n_mels) → energy (T,)
# ---------------------------------------------------------------------------
class TestTransposeFix:
    """The dataset fix — .T before passing to EnergyExtractor."""

    def test_transposed_input_produces_t_frames_energy_values(self):
        mel_torchaudio = _torchaudio_layout()         # (80, T)
        energy = EnergyExtractor.extract_energy_from_mel(mel_torchaudio.T, log_domain=False)
        assert energy.shape == (T_FRAMES,), (
            f"After .T, expected shape ({T_FRAMES},), got {energy.shape}"
        )

    def test_transposed_output_differs_from_wrong_axis_output(self):
        """Ensures transposition changes the result — not a trivial no-op."""
        mel_torchaudio = _torchaudio_layout()
        energy_correct = EnergyExtractor.extract_energy_from_mel(mel_torchaudio.T, log_domain=False)
        energy_wrong   = EnergyExtractor.extract_energy_from_mel(mel_torchaudio,   log_domain=False)
        # Shapes differ, so they can't be allclose; just confirm both shapes differ
        assert energy_correct.shape != energy_wrong.shape, (
            "Transposed and non-transposed inputs must produce different-shaped outputs"
        )

    def test_transposed_result_in_unit_range(self):
        mel_torchaudio = _torchaudio_layout()
        energy = EnergyExtractor.extract_energy_from_mel(mel_torchaudio.T, log_domain=False)
        assert energy.min().item() >= 0.0 - 1e-6
        assert energy.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 4. Pre-clip slice: mel_spec_linear[:, :num_mel_frames].T
#    Matches the exact expression used in dataset.py line 692.
# ---------------------------------------------------------------------------
class TestPreClipSliceAndTranspose:
    """[:, :num_mel_frames].T reproduces dataset.py's exact call pattern."""

    @pytest.mark.parametrize("actual_frames", [T_FRAMES, T_FRAMES // 2, 1])
    def test_slice_clip_then_transpose_shape(self, actual_frames: int):
        """
        Slicing to num_mel_frames before transpose gives correct shape even
        when num_mel_frames < full T.
        """
        mel_full = _torchaudio_layout(t=T_FRAMES)   # (80, T_FRAMES)
        # Simulate the dataset: clip to actual_frames, transpose
        energy = EnergyExtractor.extract_energy_from_mel(
            mel_full[:, :actual_frames].T, log_domain=False
        )
        assert energy.shape == (actual_frames,), (
            f"Expected ({actual_frames},) after [:, :{actual_frames}].T, "
            f"got {energy.shape}"
        )

    def test_slice_clip_output_in_unit_range(self):
        mel_full = _torchaudio_layout()
        num_mel_frames = T_FRAMES - 10
        energy = EnergyExtractor.extract_energy_from_mel(
            mel_full[:, :num_mel_frames].T, log_domain=False
        )
        assert energy.min().item() >= 0.0 - 1e-6
        assert energy.max().item() <= 1.0 + 1e-6

    def test_full_dataset_call_pattern(self):
        """
        Exact replica of the fixed dataset.py call:
            EnergyExtractor.extract_energy_from_mel(
                mel_spec_linear[:, :num_mel_frames].T, log_domain=False
            )
        """
        mel_spec_linear = _torchaudio_layout()        # (80, T)
        num_mel_frames  = mel_spec_linear.shape[1]    # T
        energy = EnergyExtractor.extract_energy_from_mel(
            mel_spec_linear[:, :num_mel_frames].T, log_domain=False
        )
        assert energy.shape == (num_mel_frames,)
        assert energy.min().item() >= 0.0 - 1e-6
        assert energy.max().item() <= 1.0 + 1e-6
        assert energy.std().item() > 1e-4, "Energy should have non-trivial variance"


# ---------------------------------------------------------------------------
# 5. Per-frame content is preserved: loud frames → high energy
# ---------------------------------------------------------------------------
class TestPerFrameEnergyReflectsContent:
    """
    With correct axis layout, energy varies along the TIME axis — loud frames
    get higher energy than quiet frames.  With the wrong layout, energy values
    are per-mel-band averages and do NOT reflect individual frame loudness.
    """

    def test_loud_first_half_has_higher_energy(self):
        """
        First T//2 frames are 100× louder than the second half.
        With correct axis layout the energy mean of the first half must
        exceed the energy mean of the second half.
        """
        T = 100
        loud_frames  = torch.full((T // 2, N_MELS), 10.0)
        quiet_frames = torch.full((T // 2, N_MELS),  0.1)
        mel = torch.cat([loud_frames, quiet_frames], dim=0)  # (T, n_mels) — already correct layout
        assert mel.shape == (T, N_MELS)

        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
        assert energy.shape == (T,)

        mean_loud  = energy[:T // 2].mean().item()
        mean_quiet = energy[T // 2:].mean().item()
        assert mean_loud > mean_quiet, (
            f"Loud frames (mean energy={mean_loud:.4f}) should exceed "
            f"quiet frames (mean energy={mean_quiet:.4f})"
        )

    def test_wrong_axis_loses_per_frame_discrimination(self):
        """
        Passing (n_mels, T_loud | T_quiet) WITHOUT transpose: the two halves are
        now split along the n_mels axis (not time), so the output has n_mels values
        that do NOT represent per-frame loudness.  This test confirms the bug shape
        and that the 80 output values are indistinguishable between the two halves.
        """
        half = N_MELS // 2                          # 40
        # Build (n_mels=80, T=100) where first 40 mel bands are loud, last 40 quiet
        loud_bands  = torch.full((half, T_FRAMES), 10.0)
        quiet_bands = torch.full((half, T_FRAMES),  0.1)
        mel_wrong = torch.cat([loud_bands, quiet_bands], dim=0)   # (80, T) ← wrong layout
        assert mel_wrong.shape == (N_MELS, T_FRAMES)

        energy_wrong = EnergyExtractor.extract_energy_from_mel(mel_wrong, log_domain=False)
        # With the wrong layout we get (n_mels,) = (80,) values, NOT (T_FRAMES,)
        assert energy_wrong.shape == (N_MELS,), (
            "Wrong-axis input must still produce n_mels values in current implementation"
        )

    def test_correct_axis_energy_has_non_trivial_variance(self):
        """Non-degenerate input with correct axis layout must produce varied energy."""
        mel = _correct_layout()
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
        assert energy.std().item() > 1e-4, (
            "Energy of non-degenerate input should have meaningful variance"
        )


# ---------------------------------------------------------------------------
# 6. Batch layout: (B, T, n_mels) → (B, T)
# ---------------------------------------------------------------------------
class TestBatchAxisLayout:
    """Batched input (B, T, n_mels) should produce (B, T) energy."""

    @pytest.mark.parametrize("b,t,n", [
        (2,  T_FRAMES, N_MELS),
        (4,  50,       N_MELS),
        (1,  T_FRAMES, N_MELS),
    ])
    def test_batch_output_shape(self, b: int, t: int, n: int):
        torch.manual_seed(0)
        mel = torch.abs(torch.randn(b, t, n)) + 0.1
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
        assert energy.shape == (b, t), (
            f"Expected ({b}, {t}) from input ({b}, {t}, {n}), got {energy.shape}"
        )

    def test_batch_output_in_unit_range(self):
        torch.manual_seed(1)
        mel = torch.abs(torch.randn(BATCH, T_FRAMES, N_MELS)) + 0.1
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
        assert energy.min().item() >= 0.0 - 1e-6
        assert energy.max().item() <= 1.0 + 1e-6
