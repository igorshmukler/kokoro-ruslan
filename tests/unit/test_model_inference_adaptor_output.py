"""
Tests covering the fix in KokoroModel.forward_inference:
  - Before fix: variance_adaptor was called only to extract predicted_log_durations;
    adapted_output (with pitch/energy embeddings) was discarded, and
    _length_regulate was called again on bare text_encoded.
  - After fix: adapted_output (encoder + pitch_embed + energy_embed) is used
    directly as expanded_encoder_outputs — _length_regulate is NOT called
    when use_variance_predictor=True.
"""
import torch
import pytest
from unittest.mock import patch

from kokoro.model.model import KokoroModel


# ---------------------------------------------------------------------------
# Shared tiny model factory
# ---------------------------------------------------------------------------
def _tiny_model(use_variance_predictor: bool = True) -> KokoroModel:
    """Return a minimal KokoroModel suitable for fast unit tests."""
    return KokoroModel(
        vocab_size=20,
        mel_dim=4,
        hidden_dim=16,
        n_encoder_layers=1,
        n_heads=2,
        encoder_ff_dim=32,
        encoder_dropout=0.0,
        n_decoder_layers=1,
        decoder_ff_dim=32,
        max_decoder_seq_len=100,
        use_variance_predictor=use_variance_predictor,
        variance_filter_size=16,
        variance_kernel_size=3,
        n_variance_bins=8,
        pitch_min=0.0,
        pitch_max=1.0,
        energy_min=0.0,
        energy_max=1.0,
        gradient_checkpointing=False,
        use_stochastic_depth=False,
    )


# A fixed-output phoneme sequence (batch=1, no padding tokens)
_PHONEMES = torch.tensor([[1, 2, 3, 4]])


# ---------------------------------------------------------------------------
# 1. _length_regulate NOT called when use_variance_predictor=True (key assertion)
# ---------------------------------------------------------------------------
def test_length_regulate_not_called_when_variance_predictor_enabled():
    """
    After the fix, VarianceAdaptor handles both expansion AND embedding.
    _length_regulate must not be invoked in the use_variance_predictor=True path.
    """
    model = _tiny_model(use_variance_predictor=True)

    with patch.object(model, '_length_regulate', wraps=model._length_regulate) as mock_lr:
        model.forward_inference(_PHONEMES, max_len=5, stop_threshold=2.0)
        assert mock_lr.call_count == 0, (
            "_length_regulate must NOT be called when use_variance_predictor=True; "
            f"it was called {mock_lr.call_count} time(s)"
        )


# ---------------------------------------------------------------------------
# 2. _length_regulate IS called in the fallback (no variance predictor) path
# ---------------------------------------------------------------------------
def test_length_regulate_called_when_variance_predictor_disabled():
    """
    When use_variance_predictor=False, the fallback path must use _length_regulate.
    """
    model = _tiny_model(use_variance_predictor=False)

    with patch.object(model, '_length_regulate', wraps=model._length_regulate) as mock_lr:
        model.forward_inference(_PHONEMES, max_len=5, stop_threshold=2.0)
        assert mock_lr.call_count >= 1, (
            "_length_regulate must be called when use_variance_predictor=False"
        )


# ---------------------------------------------------------------------------
# 3. forward_inference returns mel tensor with the correct last dimension
# ---------------------------------------------------------------------------
def test_forward_inference_mel_output_has_correct_mel_dim():
    model = _tiny_model(use_variance_predictor=True)
    mel_out = model.forward_inference(_PHONEMES, max_len=5, stop_threshold=2.0)
    assert mel_out.dim() == 3, "Output should be (batch, frames, mel_dim)"
    assert mel_out.shape[0] == 1, "Batch size should be 1"
    assert mel_out.shape[2] == model.mel_dim, (
        f"Last dim should be mel_dim={model.mel_dim}, got {mel_out.shape[2]}"
    )


# ---------------------------------------------------------------------------
# 4. Pitch/energy embeddings contributed to the decoder input
#    (output changes when embeddings are zeroed out)
# ---------------------------------------------------------------------------
def test_pitch_energy_embeddings_affect_inference_output():
    """
    Zeroing the pitch and energy embedding tables must produce a different
    mel output, confirming that the embeddings are wired into the decoder
    input path (adapted_output used, not bare _length_regulate output).
    """
    torch.manual_seed(0)
    model = _tiny_model(use_variance_predictor=True)

    # Baseline output
    mel_with_embeddings = model.forward_inference(
        _PHONEMES, max_len=5, stop_threshold=2.0
    ).detach().clone()

    # Zero out pitch and energy embeddings
    with torch.no_grad():
        model.variance_adaptor.pitch_embedding.weight.zero_()
        model.variance_adaptor.energy_embedding.weight.zero_()

    mel_without_embeddings = model.forward_inference(
        _PHONEMES, max_len=5, stop_threshold=2.0
    ).detach()

    assert not torch.allclose(mel_with_embeddings, mel_without_embeddings, atol=1e-6), (
        "Zeroing pitch/energy embeddings should change the inference output. "
        "If they match, embeddings are not being added to the decoder input."
    )


# ---------------------------------------------------------------------------
# 5. Output is clamped to [-11.5, 2.0]
# ---------------------------------------------------------------------------
def test_forward_inference_output_is_clamped():
    model = _tiny_model(use_variance_predictor=True)
    mel_out = model.forward_inference(_PHONEMES, max_len=5, stop_threshold=2.0)
    assert mel_out.min().item() >= -11.5 - 1e-6
    assert mel_out.max().item() <= 2.0 + 1e-6


# ---------------------------------------------------------------------------
# 6. VarianceAdaptor.forward called exactly once during inference
# ---------------------------------------------------------------------------
def test_variance_adaptor_called_exactly_once_during_inference():
    """
    The refactored path calls variance_adaptor once (for both duration + embeddings).
    Calling it twice would be wasteful and indicates a regression.
    """
    model = _tiny_model(use_variance_predictor=True)

    with patch.object(
        model.variance_adaptor, 'forward', wraps=model.variance_adaptor.forward
    ) as mock_va:
        model.forward_inference(_PHONEMES, max_len=5, stop_threshold=2.0)
        assert mock_va.call_count == 1, (
            f"variance_adaptor should be called exactly once, got {mock_va.call_count}"
        )
