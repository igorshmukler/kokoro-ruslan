#!/usr/bin/env python3
"""
Test and benchmark dynamic frame-based batching vs fixed batch size
"""

import torch
import time
import logging
from pathlib import Path

from kokoro.training.config import TrainingConfig
from kokoro.data.dataset import RuslanDataset, DynamicFrameBatchSampler, LengthBasedBatchSampler, collate_fn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_batch_sampler(dataset, sampler_type='dynamic', **kwargs):
    """Test a batch sampler and return statistics"""

    if sampler_type == 'dynamic':
        logger.info("\n" + "="*60)
        logger.info("Testing Dynamic Frame-Based Batching")
        logger.info("="*60)

        sampler = DynamicFrameBatchSampler(
            dataset=dataset,
            max_frames=kwargs.get('max_frames', 20000),
            min_batch_size=kwargs.get('min_batch_size', 4),
            max_batch_size=kwargs.get('max_batch_size', 32),
            drop_last=True,
            shuffle=False  # Don't shuffle for consistent testing
        )
    else:
        logger.info("\n" + "="*60)
        logger.info("Testing Fixed Batch Size")
        logger.info("="*60)

        sampler = LengthBasedBatchSampler(
            dataset=dataset,
            batch_size=kwargs.get('batch_size', 16),
            drop_last=True,
            shuffle=False
        )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing overhead
        pin_memory=False
    )

    # Collect statistics
    batch_sizes = []
    batch_frames = []
    total_padding = 0
    total_frames = 0

    logger.info(f"\nProcessing {len(dataloader)} batches...")

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        mel_specs = batch['mel_specs']
        mel_lengths = batch['mel_lengths']

        batch_size = mel_specs.shape[0]
        max_len = mel_specs.shape[1]

        # Calculate actual frames and padding
        actual_frames = mel_lengths.sum().item()
        padded_frames = batch_size * max_len
        padding = padded_frames - actual_frames

        batch_sizes.append(batch_size)
        batch_frames.append(actual_frames)
        total_padding += padding
        total_frames += actual_frames

        if batch_idx < 5:  # Show first 5 batches
            logger.info(f"  Batch {batch_idx}: size={batch_size}, frames={actual_frames}, "
                       f"max_len={max_len}, padding={padding} ({padding/padded_frames*100:.1f}%)")

    elapsed_time = time.time() - start_time

    # Calculate statistics
    num_batches = len(batch_sizes)
    avg_batch_size = sum(batch_sizes) / num_batches
    avg_frames = sum(batch_frames) / num_batches
    padding_ratio = total_padding / (total_frames + total_padding)

    logger.info("\n" + "-"*60)
    logger.info("Statistics:")
    logger.info("-"*60)
    logger.info(f"Total batches: {num_batches}")
    logger.info(f"Batch sizes - Min: {min(batch_sizes)}, Max: {max(batch_sizes)}, Avg: {avg_batch_size:.1f}")
    logger.info(f"Frames per batch - Min: {min(batch_frames)}, Max: {max(batch_frames)}, Avg: {avg_frames:.1f}")
    logger.info(f"Total frames processed: {total_frames}")
    logger.info(f"Total padding: {total_padding} ({padding_ratio*100:.1f}%)")
    logger.info(f"Processing time: {elapsed_time:.2f}s ({elapsed_time/num_batches*1000:.1f}ms/batch)")
    logger.info(f"Throughput: {total_frames/elapsed_time:.0f} frames/sec")

    return {
        'num_batches': num_batches,
        'avg_batch_size': avg_batch_size,
        'avg_frames': avg_frames,
        'padding_ratio': padding_ratio,
        'total_frames': total_frames,
        'elapsed_time': elapsed_time,
        'throughput': total_frames / elapsed_time
    }


def compare_batching_strategies():
    """Compare dynamic vs fixed batching"""

    logger.info("\n" + "="*60)
    logger.info("Dynamic Batching Comparison Test")
    logger.info("="*60)

    # Create test configuration
    config = TrainingConfig(
        data_dir="./ruslan_corpus",
        batch_size=16,
        use_dynamic_batching=True,
        max_frames_per_batch=20000,
        min_batch_size=4,
        max_batch_size=32
    )

    # Check if dataset exists
    corpus_path = Path(config.data_dir)
    if not corpus_path.exists():
        logger.error(f"Corpus not found at {corpus_path}")
        logger.info("Please run this test from the project root with the ruslan_corpus available")
        return

    # Load dataset (use a subset for faster testing)
    logger.info(f"\nLoading dataset from {config.data_dir}...")
    full_dataset = RuslanDataset(config.data_dir, config)

    # Use first 1000 samples for testing
    test_size = min(1000, len(full_dataset.samples))
    logger.info(f"Using {test_size} samples for testing")

    # Create subset dataset
    dataset = RuslanDataset(config.data_dir, config, indices=list(range(test_size)))

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Test 1: Fixed batch size (baseline)
    fixed_stats = test_batch_sampler(
        dataset,
        sampler_type='fixed',
        batch_size=16
    )

    # Test 2: Dynamic batching with default settings
    dynamic_stats = test_batch_sampler(
        dataset,
        sampler_type='dynamic',
        max_frames=20000,
        min_batch_size=4,
        max_batch_size=32
    )

    # Test 3: Dynamic batching with smaller frame budget
    dynamic_small_stats = test_batch_sampler(
        dataset,
        sampler_type='dynamic',
        max_frames=10000,
        min_batch_size=2,
        max_batch_size=24
    )

    # Compare results
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)

    logger.info("\n1. Fixed Batch Size (16):")
    logger.info(f"   Padding: {fixed_stats['padding_ratio']*100:.1f}%")
    logger.info(f"   Throughput: {fixed_stats['throughput']:.0f} frames/sec")

    logger.info("\n2. Dynamic Batching (max_frames=20000):")
    logger.info(f"   Padding: {dynamic_stats['padding_ratio']*100:.1f}%")
    logger.info(f"   Throughput: {dynamic_stats['throughput']:.0f} frames/sec")
    logger.info(f"   Speedup: {dynamic_stats['throughput']/fixed_stats['throughput']:.2f}x")
    logger.info(f"   Padding reduction: {(fixed_stats['padding_ratio']-dynamic_stats['padding_ratio'])*100:.1f}%")

    logger.info("\n3. Dynamic Batching (max_frames=10000):")
    logger.info(f"   Padding: {dynamic_small_stats['padding_ratio']*100:.1f}%")
    logger.info(f"   Throughput: {dynamic_small_stats['throughput']:.0f} frames/sec")
    logger.info(f"   Speedup: {dynamic_small_stats['throughput']/fixed_stats['throughput']:.2f}x")

    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)

    if dynamic_stats['throughput'] > fixed_stats['throughput']:
        improvement = (dynamic_stats['throughput'] / fixed_stats['throughput'] - 1) * 100
        logger.info(f"✓ Dynamic batching is {improvement:.1f}% faster!")
        logger.info(f"✓ Recommended: Use --max-frames {config.max_frames_per_batch}")
    else:
        logger.info("! Fixed batching performed better on this dataset")
        logger.info("! Consider using --no-dynamic-batching")

    logger.info("\nTo enable dynamic batching in training:")
    logger.info(f"  python training.py --max-frames {config.max_frames_per_batch} " +
                f"--min-batch-size {config.min_batch_size} --max-batch-size {config.max_batch_size}")
    logger.info("\nTo disable dynamic batching:")
    logger.info("  python training.py --no-dynamic-batching --batch-size 16")


if __name__ == "__main__":
    try:
        compare_batching_strategies()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        logger.info("\nMake sure you have:")
        logger.info("  1. The ruslan_corpus directory in the current path")
        logger.info("  2. All dependencies installed (pip install -r requirements.txt)")
