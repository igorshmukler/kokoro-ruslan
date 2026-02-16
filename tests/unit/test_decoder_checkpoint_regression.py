"""Regression tests for decoder checkpointing behavior."""

import torch

from kokoro.model.model import KokoroModel
from kokoro.model import model as model_module
from kokoro.model import transformers as transformers_module


def test_decoder_checkpointing_is_not_nested(monkeypatch):
    """
    Ensure decoder checkpointing uses only per-layer checkpoints.

    Thesis being validated:
    - Nested checkpointing (outer decoder checkpoint + inner per-layer checkpoints)
      causes excessive recomputation pressure and was a likely MPS backward crash trigger.
    - Fix removes the outer wrapper and keeps only per-layer checkpointing.
    """
    original_checkpoint = torch.utils.checkpoint.checkpoint
    checkpoint_calls = {"model": 0, "transformers": 0}

    def model_checkpoint_wrapper(*args, **kwargs):
        checkpoint_calls["model"] += 1
        return original_checkpoint(*args, **kwargs)

    def transformers_checkpoint_wrapper(*args, **kwargs):
        checkpoint_calls["transformers"] += 1
        return original_checkpoint(*args, **kwargs)

    monkeypatch.setattr(model_module, "checkpoint", model_checkpoint_wrapper)
    monkeypatch.setattr(transformers_module, "checkpoint", transformers_checkpoint_wrapper)

    n_decoder_layers = 3
    hidden_dim = 64

    model = KokoroModel(
        vocab_size=64,
        mel_dim=80,
        hidden_dim=hidden_dim,
        n_encoder_layers=1,
        n_heads=4,
        encoder_ff_dim=128,
        n_decoder_layers=n_decoder_layers,
        decoder_ff_dim=128,
        gradient_checkpointing=True,
        use_variance_predictor=False,
    )
    model.train()

    batch_size = 2
    seq_len = 12
    decoder_input = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    memory = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

    output = model._checkpoint_decoder_forward(decoder_input, memory)
    output.sum().backward()

    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert checkpoint_calls["model"] == 0, (
        "Outer decoder checkpoint should not be used; this indicates nested checkpointing."
    )
    assert checkpoint_calls["transformers"] == n_decoder_layers, (
        "Per-layer decoder checkpointing should still be active for each decoder layer."
    )
