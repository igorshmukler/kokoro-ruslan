import torch
import pytest

from kokoro.model.model import KokoroModel


def make_simple_inputs(batch=1, phonemes=5, mel_len=12, mel_dim=80):
    phoneme_indices = torch.randint(1, 10, (batch, phonemes))
    # ensure some padding position 0 occasionally
    phoneme_indices[0, -1] = 0
    stress_indices = None
    text_padding_mask = (phoneme_indices == 0)

    # simple duration assignment that sums to mel_len
    base = mel_len // phonemes
    extras = mel_len - base * phonemes
    durations = torch.full((batch, phonemes), base, dtype=torch.long)
    for i in range(extras):
        durations[:, i] += 1

    # frame-level pitch/energy targets
    pitch_targets = torch.rand(batch, mel_len)
    energy_targets = torch.rand(batch, mel_len)

    mel_specs = torch.randn(batch, mel_len, mel_dim)

    return phoneme_indices, stress_indices, text_padding_mask, durations, pitch_targets, energy_targets, mel_specs


def test_encode_and_expand_with_variance_predictor():
    # Small model with variance adaptor enabled
    model = KokoroModel(vocab_size=20, hidden_dim=64, n_encoder_layers=1, use_variance_predictor=True)
    model.eval()

    phoneme_indices, stress_indices, text_padding_mask, durations, pitch_targets, energy_targets, mel_specs = make_simple_inputs()

    # Training mode: provide durations and frame-level targets
    (expanded_encoder_outputs,
     encoder_output_padding_mask,
     predicted_log_durations,
     predicted_pitch,
     predicted_energy) = model._encode_and_expand(
        phoneme_indices=phoneme_indices,
        stress_indices=stress_indices,
        text_padding_mask=text_padding_mask,
        pitch_targets=pitch_targets,
        energy_targets=energy_targets,
        phoneme_durations=durations,
        inference=False,
    )

    # Basic sanity checks
    assert expanded_encoder_outputs.ndim == 3
    assert encoder_output_padding_mask.ndim == 2
    assert predicted_log_durations.shape == (phoneme_indices.size(0), phoneme_indices.size(1))
    assert predicted_pitch is not None and predicted_energy is not None
    assert predicted_pitch.shape[0] == phoneme_indices.size(0)
    assert predicted_energy.shape[0] == phoneme_indices.size(0)


def test_encode_and_expand_without_variance_predictor():
    # Model with variance adaptor disabled
    model = KokoroModel(vocab_size=20, hidden_dim=64, n_encoder_layers=1, use_variance_predictor=False)
    model.eval()

    phoneme_indices, stress_indices, text_padding_mask, durations, pitch_targets, energy_targets, mel_specs = make_simple_inputs()

    (expanded_encoder_outputs,
     encoder_output_padding_mask,
     predicted_log_durations,
     predicted_pitch,
     predicted_energy) = model._encode_and_expand(
        phoneme_indices=phoneme_indices,
        stress_indices=stress_indices,
        text_padding_mask=text_padding_mask,
        pitch_targets=None,
        energy_targets=None,
        phoneme_durations=durations,
        inference=False,
    )

    # Basic sanity checks
    assert expanded_encoder_outputs.ndim == 3
    assert encoder_output_padding_mask.ndim == 2
    assert predicted_log_durations.shape == (phoneme_indices.size(0), phoneme_indices.size(1))
    assert predicted_pitch is None and predicted_energy is None


if __name__ == '__main__':
    pytest.main([__file__])
