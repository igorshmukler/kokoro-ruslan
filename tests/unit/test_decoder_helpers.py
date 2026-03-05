import torch

from kokoro.model.model import KokoroModel


def test_prepare_training_decoder_inputs_shapes():
    model = KokoroModel(vocab_size=30, hidden_dim=64, n_encoder_layers=1, use_variance_predictor=False)
    model.train()

    batch = 2
    mel_len = 10
    mel_dim = model.mel_dim
    mel_specs = torch.randn(batch, mel_len, mel_dim)
    mel_padding_mask = torch.zeros(batch, mel_len, dtype=torch.bool)

    projected_with_pe, tgt_mask, mask = model._prepare_training_decoder_inputs(mel_specs, mel_padding_mask)

    assert projected_with_pe.shape[0] == batch
    assert projected_with_pe.shape[1] == mel_len
    assert tgt_mask.shape[0] == mel_len and tgt_mask.shape[1] == mel_len
    assert mask.shape == mel_padding_mask.shape


def test_project_mel_frame_and_project_decoder_outputs():
    model = KokoroModel(vocab_size=30, hidden_dim=64, n_encoder_layers=1, use_variance_predictor=False)
    model.eval()

    batch = 1
    mel_frame = torch.zeros(batch, 1, model.mel_dim)

    mel_with_pe = model._project_mel_frame(mel_frame, seq_offset=5)
    assert mel_with_pe.shape == (batch, 1, model.hidden_dim)

    # Fake decoder output sequence
    dec_out = torch.randn(batch, 3, model.hidden_dim)
    pred_mel, stop_logits = model._project_decoder_outputs(dec_out)
    assert pred_mel.shape == (batch, 3, model.mel_dim)
    assert stop_logits.shape == (batch, 3)
