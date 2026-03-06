import torch
from kokoro.model.variance_predictor import VarianceAdaptor
from kokoro.model.duration_adaptor import SimpleDurationAdaptor


# ===========================================================================
# Duration encoding/decoding round-trip correctness
# ===========================================================================
# Training encodes integer durations d as log1p(d) = log(d + 1).
# Inference must decode with expm1(x) = exp(x) - 1, NOT exp(x).
# Using exp(x) biases every phoneme duration by +1 frame.

def test_log1p_expm1_roundtrip():
    """expm1(log1p(d)) == d  for all non-negative integer durations."""
    durations = torch.tensor([0.0, 1.0, 2.0, 5.0, 7.0, 10.0, 20.0])
    encoded = torch.log1p(durations)
    recovered = torch.expm1(encoded)
    assert torch.allclose(recovered, durations, atol=1e-5), (
        "expm1(log1p(d)) must equal d; current mismatch means inference uses "
        "exp() instead of expm1() and produces durations biased by +1"
    )


def test_exp_wrong_inverse_of_log1p():
    """Proves that exp() is NOT the correct inverse of log1p(): exp(log1p(d)) = d+1."""
    durations = torch.tensor([1.0, 5.0, 7.0])
    encoded = torch.log1p(durations)
    wrong_recovery = torch.exp(encoded)
    # exp(log(d+1)) = d+1, not d
    assert torch.allclose(wrong_recovery, durations + 1, atol=1e-5), (
        "exp(log1p(d)) should equal d+1 (off by +1), confirming the bug "
        "that was present before the fix"
    )


def test_variance_adaptor_inference_duration_uses_expm1():
    """
    VarianceAdaptor inference path must decode durations via expm1, not exp.
    We feed it a duration_pred that equals log1p(known_integers) and verify
    the expanded output has exactly the expected frame count.
    """
    known_durations = torch.tensor([[2.0, 3.0, 5.0]])  # (1, 3) integer frame counts
    log1p_encoded = torch.log1p(known_durations)        # what predictor should output

    adaptor = VarianceAdaptor(n_bins=8)
    B, T, H = 1, 3, adaptor.hidden_dim
    encoder_output = torch.zeros(B, T, H)
    mask = torch.zeros(B, T, dtype=torch.bool)  # no padding

    # Patch the duration predictor: must be an nn.Module (PyTorch requires it)
    import torch.nn as nn

    class _FixedDurationPredictor(nn.Module):
        def forward(self, x, mask=None):
            return log1p_encoded

    adaptor.duration_predictor = _FixedDurationPredictor()

    # Inference (no duration_target supplied)
    adapted, dur_pred, _, _, frame_mask = adaptor(encoder_output, mask=mask)

    expected_frames = int(known_durations.sum().item())  # 2+3+5 = 10
    # expm1 decoding → 10 real frames; exp decoding would give 3+4+6 = 13
    assert adapted.size(1) == expected_frames, (
        f"Expected {expected_frames} frames (expm1 decoding), "
        f"got {adapted.size(1)} — likely still using exp() instead of expm1()"
    )


def test_simple_duration_adaptor_inference_uses_expm1():
    """
    SimpleDurationAdaptor inference path must also decode via expm1.
    """
    known_durations = torch.tensor([[3.0, 4.0]])   # (1, 2) expected frame counts
    log1p_encoded = torch.log1p(known_durations)   # (1, 2)

    import torch.nn as nn
    from kokoro.utils.lengths import length_regulate

    # A predictor that always returns the pre-computed log1p values
    class FixedPredictor(nn.Module):
        def forward(self, x):
            return log1p_encoded

    adaptor = SimpleDurationAdaptor(
        duration_predictor_fn=FixedPredictor(),
        length_regulate_fn=length_regulate,
    )

    T, H = 2, 8
    encoder_output = torch.zeros(1, T, H)
    mask = torch.zeros(1, T, dtype=torch.bool)  # no padding

    expanded, pred_log_durs, _, _, _ = adaptor(encoder_output, mask=mask, inference=True)

    expected_frames = int(known_durations.sum().item())  # 3+4 = 7
    # exp decoding would give 4+5 = 9
    assert expanded.size(1) == expected_frames, (
        f"Expected {expected_frames} frames (expm1 decoding), "
        f"got {expanded.size(1)} — likely still using exp() instead of expm1()"
    )


# ===========================================================================
# Original tests (unchanged)
# ===========================================================================

def test_hz_to_normalized_and_back():
    adaptor = VarianceAdaptor(n_bins=5, pitch_min=50.0, pitch_max=800.0)

    # Normalized sample values and corresponding Hz values
    norm = torch.tensor([0.0, 0.2, 0.5, 1.0])
    hz = norm * (adaptor.pitch_max - adaptor.pitch_min) + adaptor.pitch_min

    # Using internal helper to convert Hz -> normalized
    norm_from_hz = adaptor._hz_to_normalized(hz)

    assert torch.allclose(norm, norm_from_hz, atol=1e-6)


def test_maybe_normalize_and_quantize_consistency():
    adaptor = VarianceAdaptor(n_bins=5, pitch_min=50.0, pitch_max=800.0)

    # Create some normalized values spanning the range
    norm_vals = torch.tensor([0.0, 0.2, 0.5, 0.9])
    hz_vals = norm_vals * (adaptor.pitch_max - adaptor.pitch_min) + adaptor.pitch_min

    # Quantize normalized directly
    q_norm = adaptor.quantize_pitch(norm_vals)

    # Quantize after converting from Hz using the helper
    q_hz_manual = adaptor.quantize_pitch(adaptor._hz_to_normalized(hz_vals))

    # Using the heuristic normalizer should yield the same result
    q_hz_maybe = adaptor.quantize_pitch(adaptor._maybe_normalize_pitch(hz_vals))

    assert torch.equal(q_norm, q_hz_manual)
    assert torch.equal(q_norm, q_hz_maybe)


def test_forward_accepts_hz_targets():
    adaptor = VarianceAdaptor(n_bins=8, pitch_min=50.0, pitch_max=800.0)

    batch = 2
    seq_len = 6
    hidden = adaptor.hidden_dim
    frames_per_phoneme = 3
    expected_frames = seq_len * frames_per_phoneme

    encoder_output = torch.randn(batch, seq_len, hidden)

    base_norm = torch.linspace(0.1, 0.9, steps=seq_len)
    hz_targets = base_norm * (adaptor.pitch_max - adaptor.pitch_min) + adaptor.pitch_min
    pitch_target_hz = hz_targets.unsqueeze(0).repeat(batch, 1)

    # Must supply duration_target — without it, random predicted durations
    # make the output frame count unpredictable and the shape assertion meaningless
    duration_target = torch.full((batch, seq_len), frames_per_phoneme, dtype=torch.float)

    adapted, dur_pred, pitch_pred, energy_pred, frame_mask = adaptor(
        encoder_output, None,
        pitch_target=pitch_target_hz,
        energy_target=None,
        duration_target=duration_target,
    )

    assert adapted.shape == (batch, expected_frames, hidden)
    assert dur_pred.shape == (batch, seq_len)        # duration pred is token-level
    assert pitch_pred.shape == (batch, expected_frames)   # pitch/energy are frame-level
    assert energy_pred.shape == (batch, expected_frames)


def test_energy_normalization_and_quantize():
    adaptor = VarianceAdaptor(n_bins=6, pitch_min=50.0, pitch_max=800.0, energy_min=0.0, energy_max=100.0)

    # Create normalized energy and corresponding 'raw' energy values
    norm = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    hz = norm * (adaptor.energy_max - adaptor.energy_min) + adaptor.energy_min

    q_norm = adaptor.quantize_energy(norm)
    q_from_hz = adaptor.quantize_energy(adaptor._energy_to_normalized(hz))

    assert torch.equal(q_norm, q_from_hz)


def test_mask_zeroing_embeddings():
    adaptor = VarianceAdaptor(n_bins=8, pitch_min=50.0, pitch_max=800.0,
                              energy_min=0.0, energy_max=100.0)

    batch = 2
    seq_len = 5
    hidden = adaptor.hidden_dim

    encoder_output = torch.zeros(batch, seq_len, hidden)

    base_norm = torch.linspace(0.1, 0.9, steps=seq_len)
    pitch_target = (base_norm * (adaptor.pitch_max - adaptor.pitch_min)
                    + adaptor.pitch_min).unsqueeze(0).repeat(batch, 1)
    energy_target = (base_norm * (adaptor.energy_max - adaptor.energy_min)
                     + adaptor.energy_min).unsqueeze(0).repeat(batch, 1)

    # Explicit durations: 2 frames each for first 4 tokens, 0 for the last
    # (last token is "masked/padding" — zero duration means it contributes
    # no frames, so its position simply won't appear in the output)
    duration_target = torch.tensor([[2, 2, 2, 2, 0],
                                    [2, 2, 2, 2, 0]], dtype=torch.float)

    phoneme_mask = torch.zeros(batch, seq_len).bool()
    phoneme_mask[:, -1] = True  # last phoneme is padding

    adapted, dur_pred, pitch_pred, energy_pred, frame_mask = adaptor(
        encoder_output, phoneme_mask,
        pitch_target=pitch_target,
        energy_target=energy_target,
        duration_target=duration_target,
    )

    # encoder_output is all zeros, so adapted IS the embedding signal
    # frame_mask marks padding frames in the expanded output
    delta = adapted  # encoder is zero so delta == adapted

    # Masked (padding) frame positions must be zeroed
    assert torch.allclose(delta[frame_mask], torch.zeros_like(delta[frame_mask])), \
        "Padded frame positions should be zero"

    # Unmasked frame positions must have non-zero embeddings
    assert torch.any(delta[~frame_mask]), \
        "Real frame positions should have non-zero embeddings"
