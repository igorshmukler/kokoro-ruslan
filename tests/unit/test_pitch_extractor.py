import math
import torch
from kokoro.model.variance_predictor import PitchExtractor


def test_extract_pitch_single_sine():
    sr = 22050
    freq = 220.0
    duration = 1.0
    t = torch.arange(int(sr * duration), dtype=torch.float32)
    waveform = torch.sin(2 * math.pi * freq * (t / sr))

    pitch = PitchExtractor.extract_pitch(waveform, sample_rate=sr, hop_length=256, fmin=50.0, fmax=800.0)

    # Expect a 1-D tensor of frame-wise normalized pitch values
    assert pitch.dim() == 1
    nonzero = pitch[pitch > 0]
    assert nonzero.numel() > 0

    expected = (freq - 50.0) / (800.0 - 50.0)
    mean_est = nonzero.mean().item()

    # Allow a reasonable tolerance for estimator error
    assert abs(mean_est - expected) < 0.06


def test_extract_pitch_batch_sines():
    sr = 22050
    freqs = [110.0, 440.0]
    duration = 1.0
    t = torch.arange(int(sr * duration), dtype=torch.float32)

    waves = torch.stack([torch.sin(2 * math.pi * f * (t / sr)) for f in freqs], dim=0)

    pitches = PitchExtractor.extract_pitch(waves, sample_rate=sr, hop_length=256, fmin=50.0, fmax=800.0)

    assert pitches.dim() == 2
    for i, f in enumerate(freqs):
        nonzero = pitches[i][pitches[i] > 0]
        assert nonzero.numel() > 0
        expected = (f - 50.0) / (800.0 - 50.0)
        mean_est = nonzero.mean().item()
        assert abs(mean_est - expected) < 0.06


def test_short_waveform_returns_values_in_range():
    sr = 22050
    t = torch.arange(100, dtype=torch.float32)
    wave = torch.sin(2 * math.pi * 220 * (t / sr))

    pitch = PitchExtractor.extract_pitch(wave, sample_rate=sr, hop_length=256, fmin=50.0, fmax=800.0)

    # Should return at least one frame and values in [0,1]
    assert pitch.numel() >= 1
    assert ((pitch >= 0.0) & (pitch <= 1.0)).all()
