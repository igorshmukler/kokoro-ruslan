import torch


def test_forward_inference_receives_post_expected_threshold(monkeypatch):
    """Ensure explicit stop thresholds (init or per-call) set post_expected_stop_threshold."""
    import kokoro.inference.inference as inference_mod

    class DummyProcessor:
        def __init__(self):
            self.phoneme_to_id = {'a': 1, '<sil>': 2, '<unk>': 0}

        def process_text(self, text):
            # single word with one phoneme
            return [(text, ['a'], None)]

    class DummyModel:
        def __init__(self):
            self.last_forward_kwargs = None
            self.mel_dim = 80

        def forward_inference(self, **kwargs):
            # Record what was passed and return a minimal mel tensor
            self.last_forward_kwargs = kwargs
            return torch.zeros(1, 10, 80)

    class DummyVocoderManager:
        def __init__(self, *args, **kwargs):
            pass

        def mel_to_audio(self, mel):
            # Return a short 1-D audio tensor
            return torch.zeros(100)

    # Patch TTS internals to use our lightweight dummies
    monkeypatch.setattr(inference_mod.KokoroTTS, '_load_phoneme_processor', lambda self: DummyProcessor())
    monkeypatch.setattr(inference_mod.KokoroTTS, '_load_model', lambda self: DummyModel())
    monkeypatch.setattr(inference_mod, 'VocoderManager', DummyVocoderManager)

    # 1) Per-call explicit stop_threshold should inject post_expected_stop_threshold
    tts = inference_mod.KokoroTTS(model_dir='my_model', device='cpu')
    tts.text_to_speech('hello', output_path=None, stop_threshold=0.8)
    assert hasattr(tts.model, 'last_forward_kwargs')
    assert 'post_expected_stop_threshold' in tts.model.last_forward_kwargs
    assert abs(float(tts.model.last_forward_kwargs['post_expected_stop_threshold']) - 0.8) < 1e-6

    # 2) Init-time explicit stop threshold should also inject post_expected_stop_threshold
    tts2 = inference_mod.KokoroTTS(model_dir='my_model', device='cpu', inference_stop_threshold=0.75)
    # call without per-call override
    tts2.text_to_speech('hello', output_path=None)
    assert hasattr(tts2.model, 'last_forward_kwargs')
    assert 'post_expected_stop_threshold' in tts2.model.last_forward_kwargs
    assert abs(float(tts2.model.last_forward_kwargs['post_expected_stop_threshold']) - 0.75) < 1e-6
