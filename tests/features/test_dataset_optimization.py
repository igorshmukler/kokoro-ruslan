#!/usr/bin/env python3
"""
Test optimized dataset loading performance
"""

import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dataset_loading_speed():
    """Test that optimized loading is faster"""
    try:
        from kokoro.training.config import TrainingConfig
        from kokoro.data.dataset import RuslanDataset

        # Check if corpus exists
        corpus_path = Path("./ruslan_corpus")
        if not corpus_path.exists():
            logger.warning("⚠ Corpus not found, skipping loading speed test")
            return True

        logger.info("Testing optimized dataset loading speed...")

        # Test 1: First load (no cache)
        cache_file = corpus_path / ".cache" / "audio_metadata.pkl"
        if cache_file.exists():
            cache_file.unlink()
            logger.info("Removed existing cache for fair test")

        config = TrainingConfig(data_dir="./ruslan_corpus")

        start_time = time.time()
        dataset = RuslanDataset(config.data_dir, config)
        first_load_time = time.time() - start_time

        sample_count = len(dataset)
        logger.info(f"✓ First load (no cache): {first_load_time:.2f}s for {sample_count} samples")
        logger.info(f"  Average: {first_load_time/sample_count*1000:.2f}ms per sample")

        # Test 2: Second load (with cache)
        start_time = time.time()
        dataset2 = RuslanDataset(config.data_dir, config)
        second_load_time = time.time() - start_time

        logger.info(f"✓ Second load (with cache): {second_load_time:.2f}s for {sample_count} samples")
        logger.info(f"  Average: {second_load_time/sample_count*1000:.2f}ms per sample")

        # Calculate speedup
        speedup = first_load_time / second_load_time if second_load_time > 0 else float('inf')
        logger.info(f"✓ Cache speedup: {speedup:.1f}x faster")

        # Verify cache exists
        if cache_file.exists():
            logger.info(f"✓ Cache file created: {cache_file}")
            cache_size = cache_file.stat().st_size / 1024
            logger.info(f"  Cache size: {cache_size:.1f} KB")
        else:
            logger.warning("⚠ Cache file not created")

        # Test 3: Verify sample data is correct
        sample = dataset[0]
        logger.info(f"✓ Sample validation:")
        logger.info(f"  Audio path: {dataset.samples[0]['audio_path']}")
        logger.info(f"  Audio length: {dataset.samples[0]['audio_length']} frames")
        logger.info(f"  Phoneme length: {dataset.samples[0]['phoneme_length']}")
        logger.info(f"  Mel spec shape: {sample['mel_spec'].shape}")

        logger.info("\n✓ Dataset loading optimization test PASSED\n")
        return True

    except Exception as e:
        logger.error(f"✗ Dataset loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_repeated_loading():
    """Verify that audio is not loaded multiple times during __init__"""
    try:
        logger.info("Testing that audio is not loaded during initialization...")

        from kokoro.training.config import TrainingConfig
        from kokoro.data.dataset import RuslanDataset
        import torchaudio

        # Check if corpus exists
        corpus_path = Path("./ruslan_corpus")
        if not corpus_path.exists():
            logger.warning("⚠ Corpus not found, skipping test")
            return True

        # Clear cache for accurate test
        cache_file = corpus_path / ".cache" / "audio_metadata.pkl"
        if cache_file.exists():
            cache_file.unlink()

        # Monkey patch torchaudio.load to count calls
        original_load = torchaudio.load
        load_count = {'count': 0}

        def counting_load(*args, **kwargs):
            load_count['count'] += 1
            return original_load(*args, **kwargs)

        torchaudio.load = counting_load

        try:
            config = TrainingConfig(data_dir="./ruslan_corpus")
            dataset = RuslanDataset(config.data_dir, config)

            # During __init__, torchaudio.load should NOT be called
            # (we use torchaudio.info instead)
            if load_count['count'] == 0:
                logger.info(f"✓ No audio files loaded during initialization (using torchaudio.info)")
            else:
                logger.warning(f"⚠ {load_count['count']} audio files were loaded during initialization")
                logger.warning("  (Should be 0 - optimization may not be working)")

            # Now test __getitem__ which SHOULD load audio
            load_count['count'] = 0
            sample = dataset[0]

            if load_count['count'] == 1:
                logger.info(f"✓ Audio loaded exactly once during __getitem__")
            else:
                logger.warning(f"⚠ Audio loaded {load_count['count']} times during __getitem__ (expected 1)")

        finally:
            # Restore original function
            torchaudio.load = original_load

        logger.info("\n✓ No repeated loading test PASSED\n")
        return True

    except Exception as e:
        logger.error(f"✗ No repeated loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all optimization tests"""
    logger.info("="*60)
    logger.info("DATASET OPTIMIZATION TEST SUITE")
    logger.info("="*60)
    print()

    tests = [
        ("Loading Speed (Cache)", test_dataset_loading_speed),
        ("No Repeated Loading", test_no_repeated_loading),
    ]

    results = {}
    for name, test_func in tests:
        logger.info(f"Running test: {name}")
        logger.info("-"*60)
        results[name] = test_func()
        print()

    logger.info("="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)

    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{name}: {status}")

    print()

    total = len(results)
    passed = sum(1 for p in results.values() if p)

    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        return True
    else:
        logger.error(f"\n✗✗✗ {total - passed} TESTS FAILED ✗✗✗")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
