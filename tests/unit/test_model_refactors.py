import pytest
import torch
import types

from kokoro.model.model import KokoroModel


def test_causal_mask_cache_size_and_device_separation():
    model = KokoroModel(vocab_size=10)

    dev_cpu = torch.device('cpu')
    m1 = model._get_causal_mask(16, dev_cpu)
    m2 = model._get_causal_mask(16, dev_cpu)
    # same size/device should return cached tensor (same object)
    assert m1 is m2

    # different size should yield a different tensor
    m3 = model._get_causal_mask(20, dev_cpu)
    assert not torch.equal(m1, m3)

    # if MPS is available, mask for a different device must be different/cache separate
    if torch.backends.mps.is_available():
        dev_mps = torch.device('mps')
        mm = model._get_causal_mask(16, dev_mps)
        assert not torch.equal(m1, mm)
    else:
        pytest.skip("MPS not available; device separation test skipped")


def test_encode_and_expand_calls_length_regulate_when_variance_disabled(monkeypatch):
    # Create model with variance predictor disabled
    model = KokoroModel(vocab_size=10, use_variance_predictor=False)

    # Prepare dummy inputs
    phonemes = torch.randint(1, 9, (1, 5))
    stress = None
    text_mask = (phonemes == 0)
    # Provide phoneme durations for training path
    durations = torch.tensor([[2, 1, 3, 1, 2]])

    called = {'count': 0}

    def fake_length_regulate(encoder_outputs, durations_arg, text_padding_mask):
        called['count'] += 1
        # Return a dummy expanded tensor and mask consistent with durations sum
        total = int(durations_arg.sum().item())
        bsz = encoder_outputs.size(0)
        hidden = encoder_outputs.size(2)
        expanded = torch.zeros(bsz, total, hidden)
        mask = torch.zeros(bsz, total, dtype=torch.bool)
        return expanded, mask

    # Monkeypatch the model's _length_regulate (SimpleDurationAdaptor should use bound method)
    monkeypatch.setattr(model, '_length_regulate', fake_length_regulate)

    # Call private helper
    expanded, mask, pred_log_durs, pred_pitch, pred_energy = model._encode_and_expand(
        phoneme_indices=phonemes,
        stress_indices=stress,
        text_padding_mask=text_mask,
        pitch_targets=None,
        energy_targets=None,
        phoneme_durations=durations,
        inference=False,
    )

    assert called['count'] >= 1, "_length_regulate must be called when use_variance_predictor=False"
    assert expanded.size(1) == int(durations.sum().item())


def test_checkpoint_encoder_layers_same_output_with_and_without_checkpointing():
    model = KokoroModel(vocab_size=10)

    # Create a few simple layers that deterministically transform the input
    class AddLayer(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, x, src_key_padding_mask=None):
            return x + self.v

    layers = torch.nn.ModuleList([AddLayer(0.5), AddLayer(1.25), AddLayer(-0.75)])

    x = torch.randn(2, 6, model.hidden_dim)

    # Without checkpointing
    model.disable_gradient_checkpointing()
    model.eval()
    out_no_gc = model._checkpoint_encoder_layers(x.clone(), layers, mask=None)

    # With checkpointing enabled and in training mode
    model.enable_gradient_checkpointing()
    model.train()
    out_gc = model._checkpoint_encoder_layers(x.clone(), layers, mask=None)

    assert torch.allclose(out_no_gc, out_gc, atol=1e-6), "Checkpointed and non-checkpointed outputs must match"
