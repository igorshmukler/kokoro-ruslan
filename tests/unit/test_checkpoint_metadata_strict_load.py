"""Unit tests for strict checkpoint metadata requirements on load paths."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kokoro.inference.inference import KokoroTTS
from kokoro.model.model import KokoroModel
from kokoro.training.checkpoint_manager import load_checkpoint


def test_training_resume_requires_model_metadata(tmp_path):
    """Training resume should fail fast if checkpoint metadata is missing."""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    checkpoint_path = tmp_path / "checkpoint_epoch_1.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": 0,
            "loss": 1.0,
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="missing required 'model_metadata.architecture'"):
        load_checkpoint(
            str(checkpoint_path),
            model,
            optimizer,
            scheduler,
            str(tmp_path),
        )


def test_inference_load_requires_model_metadata(tmp_path):
    """Inference model loading should fail fast if checkpoint metadata is missing."""
    model_path = tmp_path / "kokoro_russian_final.pth"
    torch.save({"model_state_dict": {}}, model_path)

    tts = KokoroTTS.__new__(KokoroTTS)
    tts.model_dir = tmp_path
    tts.device = torch.device("cpu")
    tts.phoneme_processor = SimpleNamespace(phoneme_to_id={"a": 0})

    with pytest.raises(RuntimeError, match="missing required 'model_metadata.architecture'"):
        tts._load_model()


def test_training_resume_succeeds_with_valid_model_metadata(tmp_path):
    """Training resume should succeed when checkpoint metadata is present and matches."""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    checkpoint_path = tmp_path / "checkpoint_epoch_1.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": 0,
            "loss": 0.5,
            "phoneme_processor": {"dummy": True},
            "model_metadata": {
                "architecture": {
                    "mel_dim": 0,
                    "hidden_dim": 0,
                    "n_encoder_layers": 0,
                    "n_decoder_layers": 0,
                    "max_decoder_seq_len": 0,
                    "use_variance_predictor": False,
                }
            },
        },
        checkpoint_path,
    )

    start_epoch, best_loss, phoneme_processor = load_checkpoint(
        str(checkpoint_path),
        model,
        optimizer,
        scheduler,
        str(tmp_path),
    )

    assert start_epoch == 1
    assert best_loss == 0.5
    assert phoneme_processor == {"dummy": True}


def test_inference_load_succeeds_with_valid_model_metadata(tmp_path):
    """Inference strict load should succeed when metadata and state dict are aligned."""
    vocab_size = 5
    model_kwargs = {
        "vocab_size": vocab_size,
        "mel_dim": 80,
        "hidden_dim": 16,
        "n_encoder_layers": 1,
        "n_heads": 2,
        "encoder_ff_dim": 32,
        "encoder_dropout": 0.1,
        "n_decoder_layers": 1,
        "decoder_ff_dim": 32,
        "max_decoder_seq_len": 64,
        "use_variance_predictor": True,
        "variance_filter_size": 16,
        "variance_kernel_size": 3,
        "variance_dropout": 0.1,
        "n_variance_bins": 16,
        "pitch_min": 0.0,
        "pitch_max": 1.0,
        "energy_min": 0.0,
        "energy_max": 1.0,
        "use_stochastic_depth": False,
        "stochastic_depth_rate": 0.0,
        "enable_profiling": False,
    }
    model = KokoroModel(**model_kwargs)

    model_path = tmp_path / "kokoro_russian_final.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_metadata": {
                "architecture": {
                    "vocab_size": vocab_size,
                    "mel_dim": model_kwargs["mel_dim"],
                    "hidden_dim": model_kwargs["hidden_dim"],
                    "n_encoder_layers": model_kwargs["n_encoder_layers"],
                    "n_decoder_layers": model_kwargs["n_decoder_layers"],
                    "n_heads": model_kwargs["n_heads"],
                    "encoder_ff_dim": model_kwargs["encoder_ff_dim"],
                    "decoder_ff_dim": model_kwargs["decoder_ff_dim"],
                    "encoder_dropout": model_kwargs["encoder_dropout"],
                    "max_decoder_seq_len": model_kwargs["max_decoder_seq_len"],
                    "use_variance_predictor": model_kwargs["use_variance_predictor"],
                    "variance_filter_size": model_kwargs["variance_filter_size"],
                    "variance_kernel_size": model_kwargs["variance_kernel_size"],
                    "variance_dropout": model_kwargs["variance_dropout"],
                    "n_variance_bins": model_kwargs["n_variance_bins"],
                    "pitch_min": model_kwargs["pitch_min"],
                    "pitch_max": model_kwargs["pitch_max"],
                    "energy_min": model_kwargs["energy_min"],
                    "energy_max": model_kwargs["energy_max"],
                    "use_stochastic_depth": model_kwargs["use_stochastic_depth"],
                    "stochastic_depth_rate": model_kwargs["stochastic_depth_rate"],
                }
            },
        },
        model_path,
    )

    tts = KokoroTTS.__new__(KokoroTTS)
    tts.model_dir = tmp_path
    tts.device = torch.device("cpu")
    tts.phoneme_processor = SimpleNamespace(phoneme_to_id={f"p{i}": i for i in range(vocab_size)})

    loaded_model = tts._load_model()

    assert isinstance(loaded_model, KokoroModel)
    assert loaded_model.vocab_size == vocab_size


def test_inference_auto_tunes_controls_from_checkpoint_metadata(tmp_path):
    """Inference controls should auto-tune from checkpoint metadata when not explicitly set."""
    vocab_size = 5
    model_kwargs = {
        "vocab_size": vocab_size,
        "mel_dim": 80,
        "hidden_dim": 16,
        "n_encoder_layers": 1,
        "n_heads": 2,
        "encoder_ff_dim": 32,
        "encoder_dropout": 0.1,
        "n_decoder_layers": 1,
        "decoder_ff_dim": 32,
        "max_decoder_seq_len": 64,
        "use_variance_predictor": True,
        "variance_filter_size": 16,
        "variance_kernel_size": 3,
        "variance_dropout": 0.1,
        "n_variance_bins": 16,
        "pitch_min": 0.0,
        "pitch_max": 1.0,
        "energy_min": 0.0,
        "energy_max": 1.0,
        "use_stochastic_depth": False,
        "stochastic_depth_rate": 0.0,
        "enable_profiling": False,
    }
    model = KokoroModel(**model_kwargs)

    checkpoint_path = tmp_path / "kokoro_russian_final.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_metadata": {
                "architecture": {
                    "vocab_size": vocab_size,
                    "mel_dim": model_kwargs["mel_dim"],
                    "hidden_dim": model_kwargs["hidden_dim"],
                    "n_encoder_layers": model_kwargs["n_encoder_layers"],
                    "n_decoder_layers": model_kwargs["n_decoder_layers"],
                    "n_heads": model_kwargs["n_heads"],
                    "encoder_ff_dim": model_kwargs["encoder_ff_dim"],
                    "decoder_ff_dim": model_kwargs["decoder_ff_dim"],
                    "encoder_dropout": model_kwargs["encoder_dropout"],
                    "max_decoder_seq_len": model_kwargs["max_decoder_seq_len"],
                    "use_variance_predictor": model_kwargs["use_variance_predictor"],
                    "variance_filter_size": model_kwargs["variance_filter_size"],
                    "variance_kernel_size": model_kwargs["variance_kernel_size"],
                    "variance_dropout": model_kwargs["variance_dropout"],
                    "n_variance_bins": model_kwargs["n_variance_bins"],
                    "pitch_min": model_kwargs["pitch_min"],
                    "pitch_max": model_kwargs["pitch_max"],
                    "energy_min": model_kwargs["energy_min"],
                    "energy_max": model_kwargs["energy_max"],
                    "use_stochastic_depth": model_kwargs["use_stochastic_depth"],
                    "stochastic_depth_rate": model_kwargs["stochastic_depth_rate"],
                },
                "inference_controls": {
                    "max_len": 900,
                    "stop_threshold": 0.38,
                    "min_len_ratio": 0.62,
                    "min_len_floor": 10,
                },
            },
        },
        checkpoint_path,
    )

    tts = KokoroTTS.__new__(KokoroTTS)
    tts.model_dir = tmp_path
    tts.device = torch.device("cpu")
    tts.phoneme_processor = SimpleNamespace(phoneme_to_id={f"p{i}": i for i in range(vocab_size)})
    tts.enable_profiling = False

    tts.inference_max_len = None
    tts.inference_stop_threshold = None
    tts.inference_min_len_ratio = None
    tts.inference_min_len_floor = None
    tts._explicit_inference_max_len = False
    tts._explicit_inference_stop_threshold = False
    tts._explicit_inference_min_len_ratio = False
    tts._explicit_inference_min_len_floor = False

    tts._load_model()

    assert tts.inference_max_len == 900
    assert tts.inference_stop_threshold == pytest.approx(0.38)
    assert tts.inference_min_len_ratio == pytest.approx(0.62)
    assert tts.inference_min_len_floor == 10
