"""
Regression tests for EnergyExtractor.extract_energy_from_mel.

Guards against three concrete bugs found in the original implementation:

Bug 1 — Double log-compression:
    Old code did  exp(log_mel) → mean → log1p.
    For log-mel with mean ≈ -8 this yields linear values ≈ 0.00023,
    so log1p(0.00023) ≈ 0.00023 — near-flat energy targets with no
    dynamic range. Fix: average log values directly in log domain.

Bug 2 — Fragile mel-domain heuristic:
    Old code branched on mel_spec.min() < 0.  A log-mel spectrogram
    that has been clipped at 0 (e.g. mel.clamp(min=0)) passes through
    as-is, giving wrong energy in wrong units.
    Fix: use median() < -1 which is robust to a handful of outliers.

Bug 3 — 1-D tensor quantile crash:
    torch.quantile(1-D-tensor, q, dim=-1, keepdim=True) raises for
    dim=-1 on 1-D inputs in some PyTorch versions.
    Fix: branch on energy.dim() before calling quantile.
"""

import torch
import pytest

from kokoro.model.variance_predictor import EnergyExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_mel(n_bins: int = 80, n_frames: int = 50, batch: int = 0,
             mean: float = -8.38, std: float = 4.44,
             seed: int = 0) -> torch.Tensor:
    """Create a synthetic log-mel tensor matching real inference statistics."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    shape = (n_frames, n_bins) if batch == 0 else (batch, n_frames, n_bins)
    return torch.randn(*shape, generator=gen) * std + mean


def _linear_mel(n_bins: int = 80, n_frames: int = 50, batch: int = 0,
                seed: int = 1) -> torch.Tensor:
    """Create a synthetic linear-scale mel tensor (all non-negative)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    shape = (n_frames, n_bins) if batch == 0 else (batch, n_frames, n_bins)
    return torch.abs(torch.randn(*shape, generator=gen)) * 0.5 + 0.1


# ---------------------------------------------------------------------------
# Bug 1 — Double log-compression
# ---------------------------------------------------------------------------

class TestNoDynamicRangeCollapse:
    """
    Ensures that log-mel input produces energy with meaningful dynamic range.
    The old double-compression path collapsed everything near zero.
    """

    def test_log_mel_energy_has_dynamic_range(self):
        """Energy std across frames must be substantially greater than near-zero."""
        mel = _log_mel(n_frames=100)
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)
        # With the old double-compress path, std was ~1e-4; after fix it should be >0.05
        assert energy.std().item() > 0.05, (
            f"Energy std={energy.std().item():.6f} — dynamic range collapsed "
            "(possible double log-compression regression)"
        )

    def test_log_mel_with_realistic_stats_not_flat(self):
        """
        Specifically replicates the observed inference condition: mean ≈ -8.38, std ≈ 4.44.
        Before the fix, all energy values were ~2.3e-4 regardless of frame content.
        """
        mel = _log_mel(mean=-8.3845, std=4.4417, n_frames=129)
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)

        assert energy.min().item() >= 0.0, "Energy below 0"
        assert energy.max().item() <= 1.0, "Energy above 1"
        # Range must span at least 0.1 (i.e. not all values near 0)
        assert (energy.max() - energy.min()).item() > 0.1, (
            "Energy range too small — dynamic range still collapsed"
        )

    def test_louder_frames_get_higher_energy(self):
        """
        Frames with higher log-mel values (louder in log domain) should map to
        higher energy after normalization.
        """
        n_frames, n_bins = 20, 80
        # Two halves: first half 'loud' (mean -2), second 'quiet' (mean -12)
        loud  = torch.full((n_frames // 2, n_bins), -2.0)
        quiet = torch.full((n_frames // 2, n_bins), -12.0)
        mel = torch.cat([loud, quiet], dim=0)

        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)
        mean_loud  = energy[:n_frames // 2].mean().item()
        mean_quiet = energy[n_frames // 2:].mean().item()
        assert mean_loud > mean_quiet, (
            f"Loud frames (mean={mean_loud:.4f}) should have higher energy than "
            f"quiet frames (mean={mean_quiet:.4f})"
        )


# ---------------------------------------------------------------------------
# Bug 2 — Fragile domain heuristic
# ---------------------------------------------------------------------------

class TestDomainDetectionHeuristic:
    """
    Ensures the median-based heuristic correctly identifies log-mel even when
    min() >= 0 (e.g. after clamp), and doesn't mis-classify linear mel.
    """

    def test_clipped_log_mel_detected_as_log(self):
        """
        log-mel clipped at 0 has min() == 0, so old min() < 0 heuristic
        would treat it as linear.  New median() < -1 heuristic should still
        detect it as log.
        """
        mel = _log_mel(mean=-5.0, std=3.0, n_frames=80)
        mel_clipped = mel.clamp(min=0.0)   # min() is now 0 — old heuristic fails

        # Median should still be well below -1 because clamping affects the tail
        # only when mean >> 0; here mean is -5 so median stays negative.
        energy = EnergyExtractor.extract_energy_from_mel(mel_clipped)
        # Dynamic range check: if it was treated as linear-mel it would be
        # range-compressed via log1p -> values cluster near log1p(0..small) ≈ 0
        assert (energy.max() - energy.min()).item() > 0.05, (
            "Clipped log-mel was mis-classified — dynamic range collapsed"
        )

    def test_log_domain_true_overrides_heuristic(self):
        """Explicit log_domain=True must be respected regardless of mel content."""
        mel = _log_mel()
        energy_explicit = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)
        energy_auto     = EnergyExtractor.extract_energy_from_mel(mel)   # auto-detect
        # Both should agree within rounding for a typical log-mel
        assert torch.allclose(energy_explicit, energy_auto, atol=1e-5), (
            "Explicit log_domain=True disagrees with auto-detected value for log-mel input"
        )

    def test_log_domain_false_overrides_heuristic(self):
        """Explicit log_domain=False must be respected even when values are negative."""
        # A contrived linear-like mel that happens to have some negative values
        mel = torch.rand(40, 80) - 0.05   # mostly positive but min < 0
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
        assert energy.min().item() >= 0.0
        assert energy.max().item() <= 1.0

    def test_linear_mel_not_detected_as_log(self):
        """Linear mel (all non-negative, median >> 0) should not be treated as log."""
        mel = _linear_mel(n_frames=60)   # all positive, median ≈ 0.5+
        energy = EnergyExtractor.extract_energy_from_mel(mel)
        # Energy should still be in [0, 1] and have reasonable range
        assert energy.min().item() >= 0.0
        assert energy.max().item() <= 1.0
        assert (energy.max() - energy.min()).item() > 0.05


# ---------------------------------------------------------------------------
# Bug 3 — 1-D tensor quantile crash
# ---------------------------------------------------------------------------

class TestOneDimensionalInput:
    """
    Ensures (n_frames, n_bins) (i.e. no batch dim) does not crash on quantile
    and returns a 1-D energy tensor of the right shape.
    """

    def test_2d_input_returns_1d_energy(self):
        mel = _log_mel(n_frames=50, batch=0)   # (50, 80)
        assert mel.dim() == 2
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)
        assert energy.dim() == 1, f"Expected 1-D output, got shape {energy.shape}"
        assert energy.shape[0] == 50

    def test_3d_input_returns_2d_energy(self):
        mel = _log_mel(n_frames=50, batch=4)   # (4, 50, 80)
        assert mel.dim() == 3
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)
        assert energy.dim() == 2, f"Expected 2-D output, got shape {energy.shape}"
        assert energy.shape == (4, 50)

    def test_single_frame_does_not_crash(self):
        """Edge case: 1 frame. Quantile floor == ceil, so output should be 0 (clamped)."""
        mel = torch.full((1, 80), -5.0)
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)
        assert energy.shape == (1,)
        assert not torch.isnan(energy).any(), "NaN in single-frame energy"

    def test_two_frames_do_not_crash(self):
        mel = torch.tensor([[-8.0] * 80, [-4.0] * 80])  # (2, 80)
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)
        assert energy.shape == (2,)
        assert not torch.isnan(energy).any()


# ---------------------------------------------------------------------------
# Output contract — always [0, 1], no NaN/Inf
# ---------------------------------------------------------------------------

class TestOutputContract:
    """Output must always be in [0, 1] and contain no NaN or Inf."""

    @pytest.mark.parametrize("batch", [0, 1, 4])
    @pytest.mark.parametrize("log_domain", [True, False, None])
    def test_output_bounds_and_no_nan(self, batch, log_domain):
        if log_domain is False:
            mel = _linear_mel(n_frames=40, batch=batch)
        else:
            mel = _log_mel(n_frames=40, batch=batch)

        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=log_domain)

        assert not torch.isnan(energy).any(), "NaN in output"
        assert not torch.isinf(energy).any(), "Inf in output"
        assert energy.min().item() >= -1e-6, f"Output below 0: {energy.min().item()}"
        assert energy.max().item() <= 1.0 + 1e-6, f"Output above 1: {energy.max().item()}"

    def test_constant_mel_does_not_produce_nan(self):
        """When all frames are identical, floor == ceil; result must be 0 not NaN."""
        mel = torch.full((10, 80), -6.0)
        energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)
        assert not torch.isnan(energy).any()
        assert not torch.isinf(energy).any()
