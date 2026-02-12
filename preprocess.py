#!/usr/bin/env python3
"""
Preprocessing pipeline for Kokoro-Ruslan TTS
Runs Montreal Forced Aligner and prepares data for training
"""

import argparse
import logging
import sys
from pathlib import Path

from mfa_integration import setup_mfa_for_corpus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Preprocess corpus for Kokoro-Ruslan TTS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing with MFA alignment
  python preprocess.py --corpus ./ruslan_corpus

  # Custom output directory and more parallel jobs
  python preprocess.py --corpus ./ruslan_corpus --output ./preprocessed --jobs 8

  # Skip MFA alignment (for testing)
  python preprocess.py --corpus ./ruslan_corpus --skip-mfa
        """
    )

    parser.add_argument(
        '--corpus',
        type=str,
        default='./ruslan_corpus',
        help='Path to the corpus directory (default: ./ruslan_corpus)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./mfa_output',
        help='Path to the output directory for alignments (default: ./mfa_output)'
    )

    parser.add_argument(
        '--metadata',
        type=str,
        default='metadata_RUSLAN_22200.csv',
        help='Metadata filename (default: metadata_RUSLAN_22200.csv)'
    )

    parser.add_argument(
        '--jobs',
        type=int,
        default=4,
        help='Number of parallel jobs for MFA (default: 4)'
    )

    parser.add_argument(
        '--skip-mfa',
        action='store_true',
        help='Skip MFA alignment (for testing or if already done)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing alignments without running MFA'
    )

    return parser.parse_args()


def validate_corpus(corpus_dir: Path, metadata_file: str) -> bool:
    """Validate that the corpus structure is correct"""
    logger.info("Validating corpus structure...")

    # Check corpus directory exists
    if not corpus_dir.exists():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        return False

    # Check metadata file exists
    metadata_path = corpus_dir / metadata_file
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return False

    # Check wavs directory exists
    wavs_dir = corpus_dir / "wavs"
    if not wavs_dir.exists():
        logger.error(f"Wavs directory not found: {wavs_dir}")
        logger.error("Expected structure: corpus/wavs/*.wav")
        return False

    # Count audio files
    wav_files = list(wavs_dir.glob("*.wav"))
    if len(wav_files) == 0:
        logger.error(f"No .wav files found in {wavs_dir}")
        return False

    # Count metadata entries
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_lines = [line.strip() for line in f if line.strip()]

    logger.info(f"✓ Found {len(wav_files)} audio files")
    logger.info(f"✓ Found {len(metadata_lines)} metadata entries")

    # Check if counts match
    if abs(len(wav_files) - len(metadata_lines)) > 10:
        logger.warning(
            f"Mismatch between audio files ({len(wav_files)}) and "
            f"metadata entries ({len(metadata_lines)})"
        )

    return True


def main():
    """Main preprocessing pipeline"""
    args = parse_args()

    corpus_dir = Path(args.corpus)
    output_dir = Path(args.output)

    logger.info("="*70)
    logger.info("Kokoro-Ruslan TTS Preprocessing Pipeline")
    logger.info("="*70)

    # Validate corpus structure
    if not validate_corpus(corpus_dir, args.metadata):
        logger.error("Corpus validation failed!")
        return 1

    logger.info("\n✓ Corpus validation passed")

    # Skip MFA if requested
    if args.skip_mfa:
        logger.info("\nSkipping MFA alignment (--skip-mfa specified)")
        logger.info("Training will use estimated durations")
        return 0

    # Validate-only mode
    if args.validate_only:
        logger.info("\nValidate-only mode (--validate-only specified)")
        from mfa_integration import MFAIntegration

        mfa = MFAIntegration(str(corpus_dir), str(output_dir))
        stats = mfa.validate_alignments(str(corpus_dir / args.metadata))

        logger.info("\n" + "="*70)
        logger.info("Validation Results:")
        logger.info("="*70)
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Aligned files: {stats['aligned_files']}")
        logger.info(f"Failed files: {stats['failed_files']}")
        logger.info(f"Alignment rate: {stats['alignment_rate']*100:.1f}%")
        logger.info(f"Average duration: {stats['avg_duration_frames']:.1f} frames")

        if stats['alignment_rate'] < 0.90:
            logger.warning("\n⚠ Alignment rate below 90% - consider investigating failures")
            return 1

        logger.info("\n✓ Validation successful!")
        return 0

    # Run MFA alignment
    logger.info("\n" + "="*70)
    logger.info("Running Montreal Forced Aligner")
    logger.info("="*70)
    logger.info(f"Corpus: {corpus_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Metadata: {args.metadata}")
    logger.info(f"Parallel jobs: {args.jobs}")
    logger.info("")

    mfa = setup_mfa_for_corpus(
        str(corpus_dir),
        str(output_dir),
        args.metadata,
        args.jobs
    )

    if mfa is None:
        logger.error("\n✗ MFA alignment failed!")
        logger.error("Check the logs above for errors")
        logger.error("\nCommon issues:")
        logger.error("  1. MFA not installed: conda install -c conda-forge montreal-forced-aligner")
        logger.error("  2. Audio/text mismatch in corpus")
        logger.error("  3. Incorrect metadata format")
        return 1

    logger.info("\n" + "="*70)
    logger.info("✓ Preprocessing Complete!")
    logger.info("="*70)
    logger.info(f"\nAlignments saved to: {mfa.alignment_dir}")
    logger.info("\nYou can now start training with:")
    logger.info(f"  python training.py --corpus {corpus_dir} --mfa-alignments {mfa.alignment_dir}")
    logger.info("\nOr with default settings:")
    logger.info("  python training.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
