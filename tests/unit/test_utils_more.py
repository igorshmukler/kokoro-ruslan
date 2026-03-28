import torch
import logging
from src.kokoro.utils import utils


def test_clear_gpu_cache_mps(monkeypatch):
    calls = {}

    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)

    def fake_empty():
        calls['mps'] = True

    monkeypatch.setattr(torch.mps, 'empty_cache', fake_empty)
    utils.clear_gpu_cache()
    assert calls.get('mps', False) is True


def test_setup_training_environment_and_log(monkeypatch, caplog):
    # simulate CUDA available
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda i: 'FakeGPU')

    caplog.set_level(logging.INFO)
    utils.setup_training_environment()
    assert 'Optimizing for CUDA backend' in caplog.text or 'Using CPU backend' in caplog.text


def test_log_device_info_cpu(monkeypatch, caplog):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    caplog.set_level(logging.INFO)
    utils.log_device_info()
    assert 'MPS Available' in caplog.text
