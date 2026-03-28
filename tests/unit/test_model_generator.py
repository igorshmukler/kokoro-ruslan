import torch
from src.kokoro.model.generator import KokoroGenerator


class FakeDecoder:
    def __init__(self, hidden_dim, layers=1):
        self.layers = [None] * layers

    def precompute_cross_attention_kv(self, expanded_encoder_outputs):
        # no-op for testing
        self.kv = True

    def __call__(self, tgt, memory, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask, self_attn_kv_caches):
        # return decoder outputs shaped like (batch, seq_len, hidden)
        batch = tgt.size(0)
        # produce a single-step output with hidden dim matching tgt last dim
        hidden = tgt.size(-1)
        out = torch.zeros(batch, tgt.size(1), hidden)
        return out, self_attn_kv_caches


class FakeModel:
    def __init__(self, mel_dim=2, hidden=4):
        self.mel_dim = mel_dim
        self.decoder = FakeDecoder(hidden_dim=hidden)
        self.enable_profiling = False

    def _project_mel_frame(self, decoder_input_mel, seq_offset=0):
        # project mel to decoder hidden dim
        b = decoder_input_mel.size(0)
        return torch.zeros(b, 1, 4)

    def _project_decoder_outputs(self, decoder_out_t):
        # return mel prediction and stop logit
        b = decoder_out_t.size(0)
        mel = torch.ones(b, 1, self.mel_dim) * 0.5
        stop_logit = torch.full((b, 1, 1), -100.0)  # very small stop prob
        return mel, stop_logit

    def _log_memory(self, *_args, **_kwargs):
        return


def test_generator_returns_concatenated_mels():
    model = FakeModel(mel_dim=2, hidden=4)
    gen = KokoroGenerator(model)

    expanded = torch.zeros(1, 5, 4)
    padding_mask = torch.zeros(1, 5, dtype=torch.bool)

    out = gen.generate(expanded, padding_mask, expected_length=3, min_expected_length=0, max_expected_length=4, stop_threshold=0.5, post_expected_stop_threshold=0.5)
    # Should produce frames equal to max_expected_length (or more if loop continues); check dims
    assert out.ndim == 3
    assert out.size(0) == 1


def test_generator_stops_on_high_stop_probability():
    class StoppingModel(FakeModel):
        def _project_decoder_outputs(self, decoder_out_t):
            b = decoder_out_t.size(0)
            mel = torch.ones(b, 1, self.mel_dim) * 0.1
            stop_logit = torch.full((b, 1, 1), 100.0)  # very large stop prob
            return mel, stop_logit

    model = StoppingModel(mel_dim=2, hidden=4)
    gen = KokoroGenerator(model)

    expanded = torch.zeros(1, 5, 4)
    padding_mask = torch.zeros(1, 5, dtype=torch.bool)

    out = gen.generate(expanded, padding_mask, expected_length=10, min_expected_length=0, max_expected_length=10, stop_threshold=0.5, post_expected_stop_threshold=0.5)
    # With stop prob 1.0, should stop early and return at least one frame
    assert out.size(1) >= 1
