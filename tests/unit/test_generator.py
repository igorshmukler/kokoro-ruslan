import torch
import pytest

from kokoro.model.model import KokoroModel, KokoroGenerator


import torch.nn as nn


class MockDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Provide a layers attribute length used by generator
        self.layers = nn.ModuleList([nn.Module()])
        self.hidden_dim = hidden_dim

    def precompute_cross_attention_kv(self, memory):
        # noop for mock
        return None

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None,
                tgt_key_padding_mask=None, self_attn_kv_caches=None):
        # Echo back tgt as decoder outputs; keep caches unchanged
        # Return shape: (B, T, H)
        out = tgt.clone()
        return out, self_attn_kv_caches

    def clear_cross_attention_cache(self):
        return None


def test_generator_runs_to_max_length():
    model = KokoroModel(vocab_size=30, hidden_dim=48, n_encoder_layers=1, use_variance_predictor=False)
    model.eval()

    # Replace decoder with mock implementation
    model.decoder = MockDecoder(hidden_dim=model.hidden_dim)

    generator = KokoroGenerator(model)

    batch = 1
    expected_length = 5
    min_expected_length = 1
    max_expected_length = 3

    # Provide small expanded encoder outputs
    expanded_encoder_outputs = torch.zeros(batch, expected_length, model.hidden_dim)
    encoder_output_padding_mask = torch.zeros(batch, expected_length, dtype=torch.bool)

    # Force project_decoder_outputs to return zero mel frames and very low stop logits
    def fake_project_decoder_outputs(dec_out):
        batch_size = dec_out.size(0)
        mel = torch.zeros(batch_size, 1, model.mel_dim)
        stop_logits = torch.full((batch_size, 1), -100.0)  # sigmoid ~ 0 -> no early stop
        return mel, stop_logits

    model._project_decoder_outputs = fake_project_decoder_outputs

    mel_output = generator.generate(
        expanded_encoder_outputs=expanded_encoder_outputs,
        encoder_output_padding_mask=encoder_output_padding_mask,
        expected_length=expected_length,
        min_expected_length=min_expected_length,
        max_expected_length=max_expected_length,
        stop_threshold=0.5,
        post_expected_stop_threshold=0.2,
    )

    assert mel_output.shape == (batch, max_expected_length, model.mel_dim)


def test_generator_early_stop_triggers():
    model = KokoroModel(vocab_size=30, hidden_dim=48, n_encoder_layers=1, use_variance_predictor=False)
    model.eval()

    model.decoder = MockDecoder(hidden_dim=model.hidden_dim)
    generator = KokoroGenerator(model)

    batch = 1
    expected_length = 5
    min_expected_length = 1
    max_expected_length = 10

    expanded_encoder_outputs = torch.zeros(batch, expected_length, model.hidden_dim)
    encoder_output_padding_mask = torch.zeros(batch, expected_length, dtype=torch.bool)

    # First call returns low stop prob, second returns very high to trigger early stop
    call_count = {'n': 0}

    def fake_project_decoder_outputs(dec_out):
        call_count['n'] += 1
        batch_size = dec_out.size(0)
        mel = torch.zeros(batch_size, 1, model.mel_dim)
        if call_count['n'] >= 2:
            stop_logits = torch.full((batch_size, 1), 100.0)  # sigmoid ~1 -> stop
        else:
            stop_logits = torch.full((batch_size, 1), -100.0)
        return mel, stop_logits

    model._project_decoder_outputs = fake_project_decoder_outputs

    mel_output = generator.generate(
        expanded_encoder_outputs=expanded_encoder_outputs,
        encoder_output_padding_mask=encoder_output_padding_mask,
        expected_length=expected_length,
        min_expected_length=min_expected_length,
        max_expected_length=max_expected_length,
        stop_threshold=0.5,
        post_expected_stop_threshold=0.2,
    )

    # Should have stopped after second frame -> length 2
    assert mel_output.shape[1] == 2
