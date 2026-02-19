import numpy as np
import torch
from scipy.io import wavfile

from kokoro.data.dataset import RuslanDataset
from kokoro.model.variance_predictor import EnergyExtractor, PitchExtractor
from kokoro.training.config import TrainingConfig


def _create_minimal_corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    wavs_dir = corpus_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 22050
    duration_seconds = 0.15
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = (0.15 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    wav_path = wavs_dir / "000000_RUSLAN.wav"
    wavfile.write(wav_path, sample_rate, (waveform * 32767.0).astype(np.int16))

    metadata = corpus_dir / "metadata_RUSLAN_22200.csv"
    metadata.write_text("000000_RUSLAN|тест\n", encoding="utf-8")

    return corpus_dir


def test_dataset_uses_extractor_pitch_hz_bounds_not_normalized_bins(tmp_path, monkeypatch):
    corpus_dir = _create_minimal_corpus(tmp_path)

    config = TrainingConfig(
        data_dir=str(corpus_dir),
        use_mfa=False,
        use_feature_cache=False,
        use_variance_predictor=True,
    )

    config.pitch_extract_fmin = 70.0
    config.pitch_extract_fmax = 420.0
    config.pitch_min = 0.0
    config.pitch_max = 1.0

    observed = {}

    def _mock_pitch(waveform, sample_rate, hop_length, fmin, fmax, win_length=None):
        observed["fmin"] = fmin
        observed["fmax"] = fmax
        length = max(1, int(np.ceil(waveform.shape[-1] / hop_length)))
        return torch.full((length,), 0.5, dtype=torch.float32)

    monkeypatch.setattr(PitchExtractor, "extract_pitch", staticmethod(_mock_pitch))

    dataset = RuslanDataset(str(corpus_dir), config, use_mfa=False)
    sample = dataset[0]

    assert observed["fmin"] == config.pitch_extract_fmin
    assert observed["fmax"] == config.pitch_extract_fmax
    assert float(sample["pitch"].min()) >= 0.0
    assert float(sample["pitch"].max()) <= 1.0


def test_dataset_passes_linear_mel_to_energy_extractor(tmp_path, monkeypatch):
    corpus_dir = _create_minimal_corpus(tmp_path)

    config = TrainingConfig(
        data_dir=str(corpus_dir),
        use_mfa=False,
        use_feature_cache=False,
        use_variance_predictor=True,
    )

    monkeypatch.setattr(
        PitchExtractor,
        "extract_pitch",
        staticmethod(lambda waveform, sample_rate, hop_length, fmin, fmax, win_length=None: torch.zeros(max(1, int(np.ceil(waveform.shape[-1] / hop_length))))),
    )

    observed = {}

    def _mock_energy(mel_spec):
        observed["min"] = float(mel_spec.min())
        observed["max"] = float(mel_spec.max())
        return torch.zeros(mel_spec.shape[-1], dtype=torch.float32)

    monkeypatch.setattr(EnergyExtractor, "extract_energy_from_mel", staticmethod(_mock_energy))

    dataset = RuslanDataset(str(corpus_dir), config, use_mfa=False)
    _ = dataset[0]

    assert observed["min"] >= 0.0
    assert observed["max"] > 0.0


def test_energy_extractor_normalizes_linear_and_log_mel_to_unit_interval():
    torch.manual_seed(7)
    linear_mel = torch.rand(80, 300) * 20.0
    log_mel = torch.log(linear_mel + 1e-9)

    energy_linear = EnergyExtractor.extract_energy_from_mel(linear_mel)
    energy_log = EnergyExtractor.extract_energy_from_mel(log_mel)

    assert float(energy_linear.min()) >= 0.0
    assert float(energy_linear.max()) <= 1.0
    assert float(energy_log.min()) >= 0.0
    assert float(energy_log.max()) <= 1.0

    assert torch.allclose(energy_linear, energy_log, atol=1e-5)
