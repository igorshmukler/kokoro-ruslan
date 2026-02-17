import logging

import numpy as np
import torch
from scipy.io import wavfile

from kokoro.data.dataset import RuslanDataset
from kokoro.training.config import TrainingConfig


def _create_minimal_corpus(tmp_path, sample_rate=22050, n_files=1):
    corpus_dir = tmp_path / "corpus"
    wavs_dir = corpus_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    duration_seconds = 0.12
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = (0.12 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    metadata_lines = []
    for index in range(n_files):
        stem = f"{index:06d}_RUSLAN"
        wav_path = wavs_dir / f"{stem}.wav"
        wavfile.write(wav_path, sample_rate, (waveform * 32767.0).astype(np.int16))
        metadata_lines.append(f"{stem}|тест {index}")

    metadata = corpus_dir / "metadata_RUSLAN_22200.csv"
    metadata.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    return corpus_dir


def test_resampler_is_cached_by_source_rate(tmp_path, monkeypatch):
    corpus_dir = _create_minimal_corpus(tmp_path, sample_rate=16000, n_files=1)

    config = TrainingConfig(
        data_dir=str(corpus_dir),
        use_mfa=False,
        use_feature_cache=False,
        use_variance_predictor=False,
        sample_rate=22050,
    )

    init_calls = {"count": 0}

    class _FakeResample:
        def __init__(self, src, dst):
            init_calls["count"] += 1
            self.src = src
            self.dst = dst

        def __call__(self, audio):
            return audio

    monkeypatch.setattr("torchaudio.transforms.Resample", _FakeResample)

    dataset = RuslanDataset(str(corpus_dir), config, use_mfa=False)
    _ = dataset[0]
    _ = dataset[0]

    assert init_calls["count"] == 1


def test_feature_cache_lru_respects_entry_limit(tmp_path):
    corpus_dir = _create_minimal_corpus(tmp_path, sample_rate=22050, n_files=3)

    config = TrainingConfig(
        data_dir=str(corpus_dir),
        use_mfa=False,
        use_feature_cache=True,
        use_variance_predictor=False,
    )
    config.feature_cache_max_entries = 2
    config.feature_cache_max_mb = 512.0

    dataset = RuslanDataset(str(corpus_dir), config, use_mfa=False)

    _ = dataset[0]
    _ = dataset[1]
    _ = dataset[2]

    assert len(dataset.feature_cache) <= 2
    keys = list(dataset.feature_cache.keys())
    assert "000000_RUSLAN" not in keys
    assert "000001_RUSLAN" in keys and "000002_RUSLAN" in keys


def test_feature_cache_lru_respects_memory_limit(tmp_path):
    corpus_dir = _create_minimal_corpus(tmp_path, sample_rate=22050, n_files=1)

    config = TrainingConfig(
        data_dir=str(corpus_dir),
        use_mfa=False,
        use_feature_cache=True,
        use_variance_predictor=False,
    )
    config.feature_cache_max_entries = 100
    config.feature_cache_max_mb = 0.002  # ~2KB

    dataset = RuslanDataset(str(corpus_dir), config, use_mfa=False)

    small = {"mel_spec": torch.zeros(64, dtype=torch.float32), "_cache_version": 2}
    big = {"mel_spec": torch.zeros(2000, dtype=torch.float32), "_cache_version": 2}

    dataset._put_feature_in_memory_cache("small", small)
    assert "small" in dataset.feature_cache

    dataset._put_feature_in_memory_cache("big", big)

    assert dataset.feature_cache_total_bytes <= dataset.feature_cache_max_bytes


def test_verbose_cache_runtime_logging(tmp_path, caplog):
    corpus_dir = _create_minimal_corpus(tmp_path, sample_rate=22050, n_files=1)

    config = TrainingConfig(
        data_dir=str(corpus_dir),
        use_mfa=False,
        use_feature_cache=True,
        use_variance_predictor=False,
        verbose=True,
    )
    config.feature_cache_log_interval = 1

    dataset = RuslanDataset(str(corpus_dir), config, use_mfa=False)

    with caplog.at_level(logging.INFO):
        _ = dataset[0]  # miss
        _ = dataset[0]  # memory hit

    assert any("Feature cache stats:" in record.getMessage() for record in caplog.records)
