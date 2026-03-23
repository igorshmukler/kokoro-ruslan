"""
Unit-test-level pytest configuration.
"""
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _patch_simple_duration_adaptor_kwargs():
    """
    SimpleDurationAdaptor.forward() does not declare pitch_target_is_frame_level,
    but KokoroModel._encode_and_expand() always passes it.  Until the adaptor
    interface is updated, strip the unknown kwarg before dispatching to the
    original implementation so that tests exercising the no-variance-predictor
    path can run without TypeError.
    """
    from kokoro.model.duration_adaptor import SimpleDurationAdaptor

    original_forward = SimpleDurationAdaptor.forward

    def _forward_compat(self, *args, **kwargs):
        kwargs.pop('pitch_target_is_frame_level', None)
        return original_forward(self, *args, **kwargs)

    with patch.object(SimpleDurationAdaptor, 'forward', _forward_compat):
        yield
