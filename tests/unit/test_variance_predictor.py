import torch
from kokoro.model.variance_predictor import VarianceAdaptor


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
