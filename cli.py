#!/usr/bin/env python3
"""
Command line interface for Kokoro training
"""

import argparse
from config import TrainingConfig


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Kokoro Language Model Training Script for Russian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with custom directories
  python training.py --corpus /path/to/corpus --output ./my_model

  # Resume training from latest checkpoint
  python training.py --corpus /path/to/corpus --output ./my_model --resume auto

  # Resume from specific checkpoint
  python training.py --corpus /path/to/corpus --output ./my_model --resume /path/to/checkpoint.pth

  # Training with all custom options
  python training.py --corpus /path/to/corpus --output ./my_model --batch-size 16 --epochs 50
        """
    )

    parser.add_argument(
        '--corpus', '-c',
        type=str,
        default='./ruslan_corpus',
        help='Path to the corpus directory (default: ./ruslan_corpus)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./kokoro_russian_model',
        help='Path to the output model directory (default: ./kokoro_russian_model)'
    )

    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Resume training from checkpoint. Use "auto" to auto-detect latest checkpoint, or provide path to specific checkpoint file'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Batch size for training (default: 8)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )

    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )

    parser.add_argument(
        '--save-every',
        type=int,
        default=2,
        help='Save checkpoint every N epochs (default: 2)'
    )

    return parser.parse_args()


def create_config_from_args(args) -> TrainingConfig:
    """Create TrainingConfig from parsed arguments"""
    return TrainingConfig(
        data_dir=args.corpus,
        output_dir=args.output,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        n_fft=1024,
        n_mels=80,
        f_min=0.0,
        f_max=8000.0,
        save_every=args.save_every,
        use_mixed_precision=True,
        num_workers=0,  # Important for MPS
        pin_memory=False,  # Important for MPS
        resume_checkpoint=args.resume
    )
