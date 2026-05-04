import sys

from kokoro.cli.cli import create_config_from_args, parse_arguments
from kokoro.training.config import TrainingConfig


def test_training_config_pitch_extraction_defaults():
    config = TrainingConfig()

    assert config.pitch_extract_fmin == 50.0
    assert config.pitch_extract_fmax == 800.0

    assert config.pitch_min == 0.0
    assert config.pitch_max == 1.0


def test_training_config_convergence_fix_defaults():
    """Verify convergence-fix hyperparameter defaults introduced to correct encoder freezing."""
    config = TrainingConfig()

    # Peak LR raised to allow the model to escape the loss plateau seen at 2e-4
    assert config.max_lr_multiplier == 1.0, (
        f"max_lr_multiplier should be 1.0; got {config.max_lr_multiplier}"
    )

    # Encoder gets a separate higher LR to compensate for its much smaller Adam 2nd moments
    assert config.encoder_lr_multiplier == 0.65, (
        f"encoder_lr_multiplier should be 0.65; got {config.encoder_lr_multiplier}"
    )

    # Duration signal raised so the encoder receives stronger gradient from alignment
    assert config.duration_loss_weight == 0.35, (
        f"duration_loss_weight should be 0.35 (was 0.1); got {config.duration_loss_weight}"
    )

    assert config.spec_augment_start_epoch == 1, (
        f"spec_augment_start_epoch should be 1; got {config.spec_augment_start_epoch}"
    )

    # Encoder FFN pre-clip loosened — the old 10.0 was zeroing microscopic-but-valid gradients
    assert config.encoder_ffn_spike_clip_norm == 8.0, (
        f"encoder_ffn_spike_clip_norm should be 8.0; got {config.encoder_ffn_spike_clip_norm}"
    )


def test_cli_create_config_preserves_pitch_extraction_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training.py",
            "--corpus",
            "./ruslan_corpus",
            "--output",
            "./tmp_model",
        ],
    )

    args = parse_arguments()
    config = create_config_from_args(args)

    assert config.pitch_extract_fmin == 50.0
    assert config.pitch_extract_fmax == 800.0

    assert config.pitch_min == 0.0
    assert config.pitch_max == 1.0
