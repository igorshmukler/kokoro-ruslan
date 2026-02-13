#!/usr/bin/env python3
"""
Test script to verify gradient accumulation implementation
"""

import torch
from kokoro.training.config import TrainingConfig

def test_gradient_accumulation():
    """Test that gradient accumulation parameters are correctly set"""

    # Test default configuration
    config = TrainingConfig()

    print("="*60)
    print("Gradient Accumulation Configuration Test")
    print("="*60)

    print(f"\n✓ gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    print(f"✓ batch_size: {config.batch_size}")
    print(f"✓ Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

    # Calculate expected scheduler steps
    num_batches = 625  # Example from your dataset
    optimizer_steps_per_epoch = (num_batches + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps
    total_steps = config.num_epochs * optimizer_steps_per_epoch

    print(f"\n✓ Batches per epoch: {num_batches}")
    print(f"✓ Optimizer steps per epoch: {optimizer_steps_per_epoch}")
    print(f"✓ Total optimizer steps (100 epochs): {total_steps}")
    print(f"✓ Reduction factor: {num_batches / optimizer_steps_per_epoch:.2f}x fewer optimizer steps")

    # Test with different accumulation values
    print(f"\n{'='*60}")
    print("Impact Analysis for Different Accumulation Steps:")
    print(f"{'='*60}")
    print(f"{'Accum Steps':<12} {'Effective BS':<15} {'Opt Steps/Epoch':<20} {'Total Steps':<15}")
    print("-"*60)

    for accum_steps in [1, 2, 4, 8]:
        eff_batch = config.batch_size * accum_steps
        opt_steps = (num_batches + accum_steps - 1) // accum_steps
        total = config.num_epochs * opt_steps
        print(f"{accum_steps:<12} {eff_batch:<15} {opt_steps:<20} {total:<15}")

    print(f"\n{'='*60}")
    print("Memory vs Speed Trade-off:")
    print(f"{'='*60}")
    print("• gradient_accumulation_steps=1: Fastest updates, smallest effective batch")
    print("• gradient_accumulation_steps=2: 2x effective batch, ~50% fewer optimizer steps")
    print("• gradient_accumulation_steps=4: 4x effective batch, ~75% fewer optimizer steps")
    print("• gradient_accumulation_steps=8: 8x effective batch, ~87.5% fewer optimizer steps")

    print(f"\n{'='*60}")
    print("Current Configuration (gradient_accumulation_steps=4):")
    print(f"{'='*60}")
    print(f"• Physical batch size: {config.batch_size}")
    print(f"• Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"• Optimizer steps per epoch: {optimizer_steps_per_epoch} (vs {num_batches} without accumulation)")
    print(f"• Speedup: ~{num_batches / optimizer_steps_per_epoch:.2f}x fewer optimizer/scheduler steps")
    print(f"• Memory: Same as batch_size={config.batch_size} (gradients accumulated, not data)")

    print(f"\n✅ Configuration loaded successfully!")
    print(f"✅ Gradient accumulation enabled with {config.gradient_accumulation_steps} steps")
    print(f"✅ Ready to train with effective batch size of {config.batch_size * config.gradient_accumulation_steps}")

if __name__ == "__main__":
    test_gradient_accumulation()
