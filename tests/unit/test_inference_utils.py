import torch
import types
import pytest

from kokoro.inference.inference import KokoroTTS


def make_tts_instance():
    # Create an instance without calling __init__ to avoid heavy model loading.
    inst = object.__new__(KokoroTTS)
    # Provide minimal attributes used by the tested methods
    inst.mel_mean = None
    inst.mel_std = None
    inst.inference_max_len = None
    inst.inference_stop_threshold = None
    inst.inference_min_len_ratio = None
    inst.inference_min_len_floor = None
    inst._explicit_inference_max_len = False
    inst._explicit_inference_stop_threshold = False
    inst._explicit_inference_min_len_ratio = False
    inst._explicit_inference_min_len_floor = False
    return inst


def test_split_text_basic_sentences():
    tts = make_tts_instance()
    text = "Привет. Как дела? Всё хорошо! Новая строка\nи продолжение."
    chunks = tts.split_text(text, max_chars=50)
    # Expect multiple chunks and that each chunk ends with punctuation
    assert isinstance(chunks, list)
    assert all(len(c) <= 50 or '\n' in c for c in chunks)
    assert any('Привет.' in c for c in chunks)


@pytest.mark.parametrize("value,default,min_val,expected", [
    (10, 5, 1, 10),
    ('20', 5, 1, 20),
    (None, 7, 1, 7),
    ('notint', 6, 2, 6),
])
def test_safe_int_various(value, default, min_val, expected):
    tts = make_tts_instance()
    assert tts._safe_int(value, default, min_val) == expected


@pytest.mark.parametrize("value,default,min_val,max_val,expected", [
    (0.5, 0.1, 0.0, 1.0, 0.5),
    ('0.25', 0.1, 0.0, 1.0, 0.25),
    (None, 0.2, 0.0, 1.0, 0.2),
    ('bad', 0.3, 0.0, 1.0, 0.3),
    (5.0, 0.3, 0.0, 1.0, 1.0),
    (-1.0, 0.3, 0.0, 1.0, 0.0),
])
def test_safe_float_various(value, default, min_val, max_val, expected):
    tts = make_tts_instance()
    assert tts._safe_float(value, default, min_val, max_val) == pytest.approx(expected)


def test_denormalize_mel_with_stats_and_fallback():
    tts = make_tts_instance()

    mel = torch.tensor([[0.0, 1.0], [2.0, -1.0]])

    # Explicit stats
    tts.mel_mean = torch.tensor(5.0)
    tts.mel_std = torch.tensor(2.0)
    out = tts.denormalize_mel(mel)
    assert torch.allclose(out, (mel * 2.0) + 5.0)

    # Fallback when stats missing
    tts.mel_mean = None
    tts.mel_std = None
    out2 = tts.denormalize_mel(mel)
    assert torch.allclose(out2, (mel * 2.0) - 5.5)


def test_apply_checkpoint_inference_controls_metadata_precedence():
    tts = make_tts_instance()

    # Ensure defaults are applied when no metadata and no explicit flags
    checkpoint = {}
    tts._apply_checkpoint_inference_controls(checkpoint)
    assert tts.inference_max_len == 1200
    assert pytest.approx(tts.inference_stop_threshold, 0.01) == 0.45

    # Metadata should override defaults
    checkpoint = {
        'model_metadata': {
            'inference_controls': {
                'max_len': 2000,
                'stop_threshold': 0.9,
                'min_len_ratio': 0.5,
                'min_len_floor': 3,
            }
        }
    }
    tts._explicit_inference_max_len = False
    tts._explicit_inference_stop_threshold = False
    tts._explicit_inference_min_len_ratio = False
    tts._explicit_inference_min_len_floor = False
    tts._apply_checkpoint_inference_controls(checkpoint)
    assert tts.inference_max_len == 2000
    assert pytest.approx(tts.inference_stop_threshold, 0.001) == 0.9

    # Explicit flags should prevent overrides
    tts.inference_max_len = 777
    tts._explicit_inference_max_len = True
    tts._apply_checkpoint_inference_controls(checkpoint)
    assert tts.inference_max_len == 777
