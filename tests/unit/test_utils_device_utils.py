import torch
import contextlib
from src.kokoro.utils.device_utils import check_mps_mixed_precision_support


def test_check_mps_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert check_mps_mixed_precision_support() is False


def test_check_mps_version_old(monkeypatch):
    # Pretend MPS is available but torch < 2.0
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(torch, '__version__', '1.8.0', raising=False)
    assert check_mps_mixed_precision_support() is False


def test_check_mps_autocast_works(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(torch, '__version__', '2.1.0', raising=False)

    # Ensure device creation forces CPU tensors without causing recursion
    original_device = torch.device
    monkeypatch.setattr(torch, 'device', lambda s: original_device('cpu'))

    @contextlib.contextmanager
    def fake_autocast(**_kwargs):
        yield

    monkeypatch.setattr(torch, 'autocast', fake_autocast)

    assert check_mps_mixed_precision_support() is True
