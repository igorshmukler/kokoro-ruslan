#!/usr/bin/env python3
"""
Quick test to verify validation loop implementation
Tests dataset splitting and validation without full training
"""

import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dataset_split():
    """Test that dataset splitting works correctly"""
    try:
        from kokoro.training.config import TrainingConfig
        from kokoro.data.dataset import RuslanDataset

        logger.info("Testing dataset split functionality...")

        # Create config with validation
        config = TrainingConfig(
            data_dir="./ruslan_corpus",
            validation_split=0.1
        )

        # Load full dataset
        full_dataset = RuslanDataset(config.data_dir, config)
        total_samples = len(full_dataset.samples)
        logger.info(f"✓ Full dataset loaded: {total_samples} samples")

        # Create split indices
        import random
        random.seed(42)
        indices = list(range(total_samples))
        random.shuffle(indices)

        split_idx = int(total_samples * 0.9)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        logger.info(f"✓ Split created: {len(train_indices)} train, {len(val_indices)} val")

        # Create train dataset
        train_dataset = RuslanDataset(config.data_dir, config, indices=train_indices)
        logger.info(f"✓ Train dataset: {len(train_dataset)} samples")

        # Create validation dataset
        val_dataset = RuslanDataset(config.data_dir, config, indices=val_indices)
        logger.info(f"✓ Validation dataset: {len(val_dataset)} samples")

        # Verify no overlap
        train_samples = set(s['audio_file'] for s in train_dataset.samples)
        val_samples = set(s['audio_file'] for s in val_dataset.samples)
        overlap = train_samples & val_samples

        if overlap:
            logger.error(f"✗ Found {len(overlap)} overlapping samples!")
            return False
        else:
            logger.info("✓ No overlap between train and validation sets")

        # Verify total count
        if len(train_dataset) + len(val_dataset) == total_samples:
            logger.info("✓ Total sample count matches")
        else:
            logger.error("✗ Sample count mismatch!")
            return False

        logger.info("\n✓ Dataset split test PASSED\n")
        return True

    except Exception as e:
        logger.error(f"✗ Dataset split test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """Test that config validation parameters work"""
    try:
        from kokoro.training.config import TrainingConfig

        logger.info("Testing validation configuration...")

        config = TrainingConfig(
            validation_split=0.15,
            validation_interval=2,
            early_stopping_patience=5,
            early_stopping_min_delta=0.002
        )

        assert config.validation_split == 0.15, "validation_split not set"
        assert config.validation_interval == 2, "validation_interval not set"
        assert config.early_stopping_patience == 5, "early_stopping_patience not set"
        assert config.early_stopping_min_delta == 0.002, "early_stopping_min_delta not set"

        logger.info("✓ All validation config parameters accessible")
        logger.info(f"  - validation_split: {config.validation_split}")
        logger.info(f"  - validation_interval: {config.validation_interval}")
        logger.info(f"  - early_stopping_patience: {config.early_stopping_patience}")
        logger.info(f"  - early_stopping_min_delta: {config.early_stopping_min_delta}")

        logger.info("\n✓ Config validation test PASSED\n")
        return True

    except Exception as e:
        logger.error(f"✗ Config validation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_arguments():
    """Test that CLI arguments work"""
    try:
        from cli import parse_arguments, create_config_from_args
        import sys

        logger.info("Testing CLI validation arguments...")

        # Simulate command line arguments
        test_args = [
            'training.py',
            '--corpus', './ruslan_corpus',
            '--val-split', '0.2',
            '--early-stopping-patience', '15',
            '--validation-interval', '2'
        ]

        sys.argv = test_args
        args = parse_arguments()

        assert args.val_split == 0.2, "val_split not parsed"
        assert args.early_stopping_patience == 15, "early_stopping_patience not parsed"
        assert args.validation_interval == 2, "validation_interval not parsed"

        logger.info("✓ All CLI arguments parsed correctly")
        logger.info(f"  - val_split: {args.val_split}")
        logger.info(f"  - early_stopping_patience: {args.early_stopping_patience}")
        logger.info(f"  - validation_interval: {args.validation_interval}")

        # Test config creation
        config = create_config_from_args(args)
        assert config.validation_split == 0.2, "validation_split not in config"
        assert config.early_stopping_patience == 15, "early_stopping_patience not in config"

        logger.info("✓ Config created from CLI args successfully")

        logger.info("\n✓ CLI arguments test PASSED\n")
        return True

    except Exception as e:
        logger.error(f"✗ CLI arguments test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_initialization():
    """Test that trainer initializes with validation"""
    try:
        # Check if corpus exists
        corpus_path = Path("./ruslan_corpus")
        if not corpus_path.exists():
            logger.warning("⚠ Corpus not found, skipping trainer initialization test")
            return True

        from kokoro.training.config import TrainingConfig
        from kokoro.training.trainer import KokoroTrainer

        logger.info("Testing trainer initialization with validation...")

        config = TrainingConfig(
            data_dir="./ruslan_corpus",
            output_dir="./test_model",
            num_epochs=2,
            batch_size=2,
            validation_split=0.1
        )

        # Initialize trainer (this will create the validation split)
        trainer = KokoroTrainer(config)

        # Check that validation dataloader was created
        assert trainer.val_dataloader is not None, "Validation dataloader not created"
        assert trainer.val_dataset is not None, "Validation dataset not created"

        logger.info(f"✓ Trainer initialized with validation")
        logger.info(f"  - Training samples: {len(trainer.dataset)}")
        logger.info(f"  - Validation samples: {len(trainer.val_dataset)}")
        logger.info(f"  - Validation dataloader batches: {len(trainer.val_dataloader)}")

        # Check validation state variables
        assert hasattr(trainer, 'best_val_loss'), "best_val_loss not initialized"
        assert hasattr(trainer, 'epochs_without_improvement'), "epochs_without_improvement not initialized"
        assert hasattr(trainer, 'validation_losses'), "validation_losses not initialized"

        logger.info("✓ Validation state variables initialized")

        logger.info("\n✓ Trainer initialization test PASSED\n")
        return True

    except Exception as e:
        logger.error(f"✗ Trainer initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests"""
    logger.info("="*60)
    logger.info("VALIDATION IMPLEMENTATION TEST SUITE")
    logger.info("="*60)
    print()  # Empty line

    tests = [
        ("Config Validation", test_config_validation),
        ("CLI Arguments", test_cli_arguments),
        ("Dataset Split", test_dataset_split),
        ("Trainer Initialization", test_trainer_initialization),
    ]

    results = {}
    for name, test_func in tests:
        logger.info(f"Running test: {name}")
        logger.info("-"*60)
        results[name] = test_func()
        print()  # Empty line

    logger.info("="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)

    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{name}: {status}")

    print()  # Empty line

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
