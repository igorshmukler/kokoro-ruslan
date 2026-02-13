#!/usr/bin/env python3
"""
Example usage of validation loop and overfitting monitoring

This script demonstrates different validation configurations
"""

from config import TrainingConfig
from trainer import KokoroTrainer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Example 1: Standard training with validation (recommended)
def example_standard_validation():
    """
    Standard setup with 10% validation and early stopping
    This is the recommended configuration for most use cases
    """
    config = TrainingConfig(
        data_dir="./ruslan_corpus",
        output_dir="./kokoro_russian_model",
        num_epochs=100,
        batch_size=8,
        learning_rate=1e-4,

        # Validation settings
        validation_split=0.1,  # 10% for validation
        validation_interval=1,  # Validate every epoch
        early_stopping_patience=10,  # Stop after 10 epochs without improvement
        early_stopping_min_delta=0.001,  # Minimum improvement threshold

        # Other settings
        use_mfa=True,
        save_every=2
    )

    trainer = KokoroTrainer(config)
    trainer.train()


# Example 2: More aggressive validation for small datasets
def example_small_dataset_validation():
    """
    For smaller datasets, use larger validation split and more patience
    """
    config = TrainingConfig(
        data_dir="./ruslan_corpus",
        output_dir="./kokoro_russian_model_small",
        num_epochs=100,
        batch_size=8,
        learning_rate=1e-4,

        # Larger validation split for small datasets
        validation_split=0.2,  # 20% for validation
        validation_interval=1,  # Validate every epoch
        early_stopping_patience=15,  # More patience for noisy small datasets
        early_stopping_min_delta=0.002,  # Slightly higher threshold

        use_mfa=True,
        save_every=2
    )

    trainer = KokoroTrainer(config)
    trainer.train()


# Example 3: Fast iteration with frequent validation
def example_fast_iteration():
    """
    For development/debugging: Validate frequently to catch issues early
    """
    config = TrainingConfig(
        data_dir="./ruslan_corpus",
        output_dir="./kokoro_russian_model_fast",
        num_epochs=50,
        batch_size=16,  # Larger batches for faster training
        learning_rate=2e-4,  # Higher LR for faster convergence

        # Quick iteration settings
        validation_split=0.1,
        validation_interval=1,  # Validate every epoch
        early_stopping_patience=5,  # Stop quickly if not improving
        early_stopping_min_delta=0.005,  # Higher threshold for quick decisions

        use_mfa=True,
        save_every=1  # Save more frequently
    )

    trainer = KokoroTrainer(config)
    trainer.train()


# Example 4: Final training without validation
def example_final_training():
    """
    After hyperparameter tuning, train final model on all data
    Disable validation to use 100% of data for training
    """
    config = TrainingConfig(
        data_dir="./ruslan_corpus",
        output_dir="./kokoro_russian_model_final",
        num_epochs=100,
        batch_size=8,
        learning_rate=1e-4,

        # Disable validation for final training
        validation_split=0.0,  # No validation

        use_mfa=True,
        save_every=2
    )

    trainer = KokoroTrainer(config)
    trainer.train()


# Example 5: Large dataset with less frequent validation
def example_large_dataset():
    """
    For very large datasets, validate less frequently to save time
    """
    config = TrainingConfig(
        data_dir="./ruslan_corpus",
        output_dir="./kokoro_russian_model_large",
        num_epochs=100,
        batch_size=32,  # Larger batches for large datasets
        learning_rate=1e-4,

        # Less frequent validation for speed
        validation_split=0.05,  # Only 5% needed for large datasets
        validation_interval=2,  # Validate every 2 epochs
        early_stopping_patience=8,
        early_stopping_min_delta=0.001,

        use_mfa=True,
        save_every=2
    )

    trainer = KokoroTrainer(config)
    trainer.train()


# Example 6: Conservative training with strict early stopping
def example_conservative():
    """
    Conservative approach: Stop at first sign of overfitting
    """
    config = TrainingConfig(
        data_dir="./ruslan_corpus",
        output_dir="./kokoro_russian_model_conservative",
        num_epochs=100,
        batch_size=8,
        learning_rate=1e-4,

        # Strict early stopping
        validation_split=0.15,  # More validation data
        validation_interval=1,  # Validate every epoch
        early_stopping_patience=5,  # Stop quickly
        early_stopping_min_delta=0.0005,  # Very small threshold

        use_mfa=True,
        save_every=1  # Save frequently in case of early stop
    )

    trainer = KokoroTrainer(config)
    trainer.train()


if __name__ == "__main__":
    print("Validation Examples")
    print("=" * 60)
    print()
    print("Available examples:")
    print("  1. Standard validation (recommended)")
    print("  2. Small dataset validation")
    print("  3. Fast iteration (development)")
    print("  4. Final training (no validation)")
    print("  5. Large dataset validation")
    print("  6. Conservative early stopping")
    print()

    # Run the standard example
    print("Running example 1: Standard validation...")
    print()
    example_standard_validation()
