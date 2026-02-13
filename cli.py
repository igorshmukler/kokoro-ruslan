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

    parser.add_argument(
        '--mfa-alignments',
        type=str,
        default=None,
        help='Path to MFA alignment directory (default: auto-detect from ./mfa_output/alignments)'
    )

    parser.add_argument(
        '--no-mfa',
        action='store_true',
        help='Disable MFA alignments and use estimated durations'
    )

    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1 = 10%%)'
    )

    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable validation (use all data for training)'
    )

    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Early stopping patience in epochs (default: 10)'
    )

    parser.add_argument(
        '--validation-interval',
        type=int,
        default=1,
        help='Run validation every N epochs (default: 1)'
    )

    parser.add_argument(
        '--dynamic-batching',
        action='store_true',
        default=True,
        help='Use dynamic frame-based batching (default: True)'
    )

    parser.add_argument(
        '--no-dynamic-batching',
        action='store_false',
        dest='dynamic_batching',
        help='Disable dynamic batching, use fixed batch size'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=20000,
        help='Maximum mel frames per batch for dynamic batching (default: 20000)'
    )

    parser.add_argument(
        '--min-batch-size',
        type=int,
        default=4,
        help='Minimum batch size for dynamic batching (default: 4)'
    )

    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=32,
        help='Maximum batch size for dynamic batching (default: 32)'
    )

    parser.add_argument(
        '--profile-amp',
        action='store_true',
        help='Profile AMP benefits before training (compare speed with/without mixed precision)'
    )

    parser.add_argument(
        '--profile-amp-batches',
        type=int,
        default=10,
        help='Number of batches to use for AMP profiling (default: 10)'
    )

    return parser.parse_args()


def create_config_from_args(args) -> TrainingConfig:
    """Create TrainingConfig from parsed arguments"""

    # Determine MFA alignment directory
    mfa_alignment_dir = "./mfa_output/alignments"
    if args.mfa_alignments:
        mfa_alignment_dir = args.mfa_alignments

    use_mfa = not args.no_mfa

    # Validation settings
    validation_split = 0.0 if args.no_validation else args.val_split

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
        # Dynamic batching settings
        use_dynamic_batching=args.dynamic_batching,
        max_frames_per_batch=args.max_frames,
        min_batch_size=args.min_batch_size,
        max_batch_size=args.max_batch_size,
        use_mfa=use_mfa,
        mfa_alignment_dir=mfa_alignment_dir,
        num_workers=0,  # Important for MPS
        pin_memory=False,  # Important for MPS
        resume_checkpoint=args.resume,
        validation_split=validation_split,
        validation_interval=args.validation_interval,
        early_stopping_patience=args.early_stopping_patience
    )
