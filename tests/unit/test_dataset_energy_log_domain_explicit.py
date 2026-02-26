"""
Tests covering the fix in dataset.py:
  extract_energy_from_mel is called with explicit log_domain=False when
  mel_spec_linear (pre-log, linear power mel) is passed.

Before fix:  EnergyExtractor.extract_energy_from_mel(mel_spec_linear)
             — relied on the median heuristic to decide the domain
After fix:   EnergyExtractor.extract_energy_from_mel(mel_spec_linear, log_domain=False)
             — domain is explicit, no heuristic needed

These tests verify that:
  a) log_domain=False on linear mel gives a meaningfully different (and correct)
     result compared to incorrectly calling log_domain=True.
  b) The explicit log_domain=False path produces valid [0, 1] energy values.
  c) The heuristic auto-detect path still agrees with the explicit path for
     typical linear mel inputs (median > 0, well above the -1 threshold).
"""
import torch
import pytest

from kokoro.model.variance_predictor import EnergyExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_linear_mel(n_frames: int = 50, n_mels: int = 80, seed: int = 0) -> torch.Tensor:
    """Random positive linear-scale mel spectrogram (n_frames, n_mels)."""
    torch.manual_seed(seed)
    return torch.abs(torch.randn(n_frames, n_mels)) + 0.01   # strictly positive


def _make_log_mel(n_frames: int = 50, n_mels: int = 80, seed: int = 0) -> torch.Tensor:
    """Synthetic log-mel (values in [-11.5, 2.0] range, median << -1)."""
    torch.manual_seed(seed)
    return torch.randn(n_frames, n_mels) * 3.0 - 5.0  # median ≈ -5


# ---------------------------------------------------------------------------
# 1. log_domain=False on linear mel gives output in [0, 1]
# ---------------------------------------------------------------------------
def test_explicit_false_output_in_unit_range():
    mel = _make_linear_mel()
    energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
    assert energy.min().item() >= 0.0 - 1e-6, "Energy must be >= 0"
    assert energy.max().item() <= 1.0 + 1e-6, "Energy must be <= 1"


# ---------------------------------------------------------------------------
# 2. Explicit log_domain=False and auto-detect agree for typical linear mel
#    (auto-detect should route to the linear path since median >> -1)
# ---------------------------------------------------------------------------
def test_explicit_false_agrees_with_auto_detect_for_linear_mel():
    mel = _make_linear_mel()
    energy_explicit = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
    energy_auto = EnergyExtractor.extract_energy_from_mel(mel)  # auto-detect
    assert torch.allclose(energy_explicit, energy_auto, atol=1e-5), (
        "Explicit log_domain=False and auto-detect should agree for typical linear mel "
        "(median >> -1 triggers linear path in heuristic)"
    )


# ---------------------------------------------------------------------------
# 3. Explicit log_domain=False gives DIFFERENT result than log_domain=True
#    (log_domain=True on linear mel is the old bug — uses raw linear as log-mel)
# ---------------------------------------------------------------------------
def test_explicit_false_differs_from_wrong_true_on_linear_mel():
    mel = _make_linear_mel()
    energy_correct = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
    energy_wrong = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)
    # The two paths compute different things; output should not be identical
    assert not torch.allclose(energy_correct, energy_wrong, atol=1e-5), (
        "log_domain=False and log_domain=True must produce different results on "
        "linear mel — they use different computation paths"
    )


# ---------------------------------------------------------------------------
# 4. log_domain=False output shape matches frame count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_frames,n_mels", [(30, 80), (100, 40), (1, 80)])
def test_explicit_false_output_shape(n_frames: int, n_mels: int):
    mel = _make_linear_mel(n_frames=n_frames, n_mels=n_mels)
    energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
    assert energy.shape == (n_frames,), (
        f"Expected energy shape ({n_frames},), got {energy.shape}"
    )


# ---------------------------------------------------------------------------
# 5. log_domain=False is non-trivial (doesn't produce all-zero or all-one output
#    for a non-degenerate mel)
# ---------------------------------------------------------------------------
def test_explicit_false_produces_nontrivial_energy():
    mel = _make_linear_mel(n_frames=80)
    energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
    # Energy should have some variance (not constant)
    assert energy.std().item() > 1e-4, (
        "Energy from non-degenerate linear mel should have non-zero variance"
    )


# ---------------------------------------------------------------------------
# 6. log_domain=False on a batched mel (batch, frames, mels)
# ---------------------------------------------------------------------------
def test_explicit_false_batched_input():
    torch.manual_seed(42)
    mel = torch.abs(torch.randn(3, 50, 80)) + 0.01   # (batch, frames, mels)
    energy = EnergyExtractor.extract_energy_from_mel(mel, log_domain=False)
    assert energy.shape == (3, 50), f"Expected (3, 50), got {energy.shape}"
    assert energy.min().item() >= 0.0 - 1e-6
    assert energy.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 7. Heuristic correctly auto-detects log-mel (median < -1) — auto-detect still OK
# ---------------------------------------------------------------------------
def test_auto_detect_routes_to_log_path_for_log_mel():
    mel = _make_log_mel()   # median ≈ -5, well below -1.0 threshold
    energy_auto = EnergyExtractor.extract_energy_from_mel(mel)
    energy_explicit_log = EnergyExtractor.extract_energy_from_mel(mel, log_domain=True)
    assert torch.allclose(energy_auto, energy_explicit_log, atol=1e-5), (
        "Auto-detect should route to log path for log-mel (median < -1)"
    )
