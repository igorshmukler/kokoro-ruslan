import sys

from kokoro.cli.cli import create_config_from_args, parse_arguments
from kokoro.training.config import TrainingConfig


def test_training_config_pitch_extraction_defaults():
    config = TrainingConfig()

    assert config.pitch_extract_fmin == 50.0
    assert config.pitch_extract_fmax == 800.0

    assert config.pitch_min == 0.0
    assert config.pitch_max == 1.0


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
