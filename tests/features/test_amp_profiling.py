#!/usr/bin/env python3
"""
Test script for AMP profiling functionality
Demonstrates how to profile AMP benefits before training
"""

import torch
import logging
from kokoro.training.config import TrainingConfig
from kokoro.training.trainer import KokoroTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_amp_profiling():
    """Test AMP profiling with a small dataset"""

    # Create a minimal config for testing
    config = TrainingConfig(
        data_dir='./ruslan_corpus',
        output_dir='./test_amp_output',
        batch_size=8,
        learning_rate=1e-4,
        num_epochs=1,  # We're just profiling, not training
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        n_fft=1024,
        n_mels=80,
        use_mixed_precision=True,  # Start with AMP enabled
        validation_split=0.0,  # No validation needed for profiling
        use_dynamic_batching=False,  # Use fixed batch size for consistent profiling
    )

    logger.info("="*70)
    logger.info("AMP PROFILING TEST")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Mixed precision: {config.use_mixed_precision}")
    logger.info("")

    # Initialize trainer
    try:
        logger.info("Initializing trainer...")
        trainer = KokoroTrainer(config)

        # Run AMP profiling
        logger.info("Running AMP profiling (this will take a minute)...\n")
        results = trainer.profile_amp_benefits(num_batches=10)

        # Display results
        logger.info("\n" + "="*70)
        logger.info("PROFILING COMPLETE")
        logger.info("="*70)

        if results['supported']:
            logger.info(f"Without AMP: {results['without_amp']:.2f}s")
            logger.info(f"With AMP:    {results['with_amp']:.2f}s")
            logger.info(f"Speedup:     {results['speedup']:.2f}x")

            # Recommendations
            logger.info("\nRECOMMENDATIONS:")
            if results['speedup'] > 1.2:
                logger.info("✓ AMP provides significant speedup!")
                logger.info("  Keep use_mixed_precision=True in your config")
            elif results['speedup'] > 1.05:
                logger.info("✓ AMP provides modest speedup")
                logger.info("  Keep use_mixed_precision=True in your config")
            elif results['speedup'] > 0.95:
                logger.info("≈ AMP has minimal impact")
                logger.info("  You can keep it enabled (no harm)")
            else:
                logger.info("⚠ AMP is actually slower on your hardware")
                logger.info("  Consider setting use_mixed_precision=False")
        else:
            logger.info("AMP is not supported on this device (CPU)")
            logger.info("Set use_mixed_precision=False in your config")

        logger.info("="*70)

    except Exception as e:
        logger.error(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Main function"""
    logger.info("\nThis script tests AMP profiling functionality")
    logger.info("It will compare training speed with and without mixed precision\n")

    # Check device availability
    if torch.backends.mps.is_available():
        logger.info("✓ MPS (Apple Silicon GPU) is available")
    elif torch.cuda.is_available():
        logger.info("✓ CUDA (NVIDIA GPU) is available")
    else:
        logger.info("ℹ Running on CPU (AMP profiling will be skipped)")

    logger.info("")

    # Run profiling test
    success = test_amp_profiling()

    if success:
        logger.info("\n✓ AMP profiling test completed successfully!")
        logger.info("\nTo use AMP profiling in your training:")
        logger.info("  python3 training.py --corpus ./ruslan_corpus --profile-amp")
        logger.info("\nOr with custom settings:")
        logger.info("  python3 training.py --corpus ./ruslan_corpus --profile-amp --profile-amp-batches 20")
    else:
        logger.info("\n✗ AMP profiling test failed")


if __name__ == "__main__":
    main()
