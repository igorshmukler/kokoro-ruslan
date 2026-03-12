import pytest
import torch

from src.kokoro.utils import utils


def test_format_model_size_and_memory():
    assert utils.format_model_size(10) == '10'
    assert utils.format_model_size(1500) == '1.5K'
    assert utils.format_model_size(2_500_000) == '2.5M'
    assert utils.format_model_size(3_000_000_000) == '3.0B'

    mb = utils.calculate_model_memory(1_000_000, precision='float32')
    assert mb > 0
    mb2 = utils.calculate_model_memory(1_000_000, precision='float16')
    assert mb2 * 2 == pytest.approx(mb)


def test_estimate_training_time_and_logging(monkeypatch, caplog):
    s = utils.estimate_training_time(1000, batch_size=100, seconds_per_batch=2.0, num_epochs=2)
    assert '~' in s

    # monkeypatch device backends
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    info = utils.get_device_info()
    assert info['recommended_device'] == 'cpu'

    # test validate_training_config with invalid config
    class C:
        pass

    cfg = C()
    cfg.batch_size = 0
    cfg.learning_rate = -1
    cfg.num_epochs = 0
    cfg.sample_rate = 0
    cfg.n_mels = -1
    cfg.hop_length = 0

    ok = utils.validate_training_config(cfg)
    assert ok is False


def test_cache_manager_get_and_clear(tmp_path, monkeypatch, capsys):
    from src.kokoro.utils.cache_manager import get_cache_status, clear_cache

    # non-existent dir
    status = get_cache_status(tmp_path / 'nope')
    assert status['exists'] is False

    # create cache dir with .pt files
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    f1 = cache_dir / 'a.pt'
    f2 = cache_dir / 'b.pt'
    f1.write_bytes(b'123')
    f2.write_bytes(b'abcd')

    status2 = get_cache_status(cache_dir)
    assert status2['exists'] is True
    assert status2['num_files'] == 2

    # clear cache without confirmation
    clear_cache(cache_dir, confirm=False)
    assert not cache_dir.exists()
