import torch

from kokoro.data.dataset import RuslanDataset


def test_fallback_durations_sum_matches_mel_frames_when_phonemes_exceed_frames():
    durations = RuslanDataset._build_fallback_durations(num_phonemes=7, num_mel_frames=3)

    assert durations.shape == (7,)
    assert durations.dtype == torch.long
    assert int(durations.sum().item()) == 3
    assert (durations >= 0).all()
    assert (durations == 0).any()  # short utterance must allow zero-duration tail phonemes


def test_fallback_durations_sum_matches_mel_frames_when_frames_exceed_phonemes():
    durations = RuslanDataset._build_fallback_durations(num_phonemes=4, num_mel_frames=11)

    assert durations.shape == (4,)
    assert int(durations.sum().item()) == 11
    assert (durations >= 0).all()


def test_fallback_durations_handles_zero_phonemes():
    durations = RuslanDataset._build_fallback_durations(num_phonemes=0, num_mel_frames=9)

    assert durations.shape == (0,)
    assert int(durations.sum().item()) == 0


def test_fallback_durations_handles_zero_mel_frames():
    durations = RuslanDataset._build_fallback_durations(num_phonemes=5, num_mel_frames=0)

    assert durations.shape == (5,)
    assert int(durations.sum().item()) == 0
    assert (durations == 0).all()
