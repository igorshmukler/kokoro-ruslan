#!/usr/bin/env python3
"""
Pre-compute and cache mel spectrograms, pitch, and energy features for faster training.

This script processes all audio files in the dataset and saves the computed features
to disk, significantly speeding up training by avoiding redundant audio processing.

Usage:
    kokoro-precompute --corpus ./ruslan_corpus --config path/to/config.yaml
    python3 -m kokoro.cli.precompute_features --corpus ./ruslan_corpus
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch

from kokoro.training.config import TrainingConfig
from kokoro.data.dataset import RuslanDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_features(data_dir: str, config: TrainingConfig, force_recompute: bool = False):
    """
    Pre-compute and cache all features for the dataset.

    Args:
        data_dir: Path to the corpus directory
        config: Training configuration
        force_recompute: If True, recompute even if cache exists
    """
    logger.info(f"Starting feature pre-computation for: {data_dir}")
    logger.info(f"Feature cache directory: {config.feature_cache_dir}")

    # Ensure feature caching is enabled
    config.use_feature_cache = True

    # Create temporary dataset to access all samples
    dataset = RuslanDataset(data_dir, config)

    cache_dir = Path(config.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    total_samples = len(dataset)
    cached_count = 0
    computed_count = 0
    failed_count = 0

    logger.info(f"Processing {total_samples} samples...")

    # Process all samples
    for idx in tqdm(range(total_samples), desc="Pre-computing features"):
        sample = dataset.samples[idx]
        audio_file = sample['audio_file']
        cache_path = cache_dir / f"{audio_file}.pt"

        # Skip if already cached and not forcing recompute
        if cache_path.exists() and not force_recompute:
            cached_count += 1
            continue

        try:
            # Access the item - this will compute and cache it
            _ = dataset[idx]
            computed_count += 1
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            failed_count += 1

    # Print summary
    logger.info("="*60)
    logger.info("PRE-COMPUTATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Already cached: {cached_count}")
    logger.info(f"Newly computed: {computed_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Cache directory: {cache_dir}")

    # Calculate cache size
    try:
        cache_size_mb = sum(f.stat().st_size for f in cache_dir.glob("*.pt")) / (1024**2)
        logger.info(f"Total cache size: {cache_size_mb:.1f} MB")
    except Exception as e:
        logger.warning(f"Could not calculate cache size: {e}")

    logger.info("="*60)
    logger.info("Training will now be 5-10x faster on subsequent epochs!")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute and cache mel spectrograms for faster training"
    )
    parser.add_argument(
        "--corpus", "-c",
        type=str,
        default="./ruslan_corpus",
        help="Path to the corpus directory"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory (default: {corpus}/.feature_cache)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cache exists"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Audio sample rate"
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Number of mel frequency bins"
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="STFT hop length"
    )
    parser.add_argument(
        "--no-variance",
        action="store_true",
        help="Disable pitch/energy extraction"
    )
    parser.add_argument(
        "--use-mfa",
        action="store_true",
        default=True,
        help="Use MFA alignments if available"
    )
    parser.add_argument(
        "--mfa-alignment-dir",
        type=str,
        default=None,
        help="MFA alignment directory"
    )

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        data_dir=args.corpus,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        use_variance_predictor=not args.no_variance,
        use_feature_cache=True,
        use_mfa=args.use_mfa
    )

    # Set cache directory
    if args.cache_dir:
        config.feature_cache_dir = args.cache_dir
    else:
        config.feature_cache_dir = str(Path(args.corpus) / ".feature_cache")

    # Set MFA alignment directory if provided
    if args.mfa_alignment_dir:
        config.mfa_alignment_dir = args.mfa_alignment_dir
    elif args.use_mfa:
        # Default MFA location
        config.mfa_alignment_dir = "./mfa_output/alignments"

    # Run pre-computation
    try:
        precompute_features(args.corpus, config, force_recompute=args.force)
    except KeyboardInterrupt:
        logger.info("\nPre-computation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pre-computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
