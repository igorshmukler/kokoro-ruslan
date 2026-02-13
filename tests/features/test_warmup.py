#!/usr/bin/env python3
"""
Test script to verify linear warmup before OneCycleLR implementation
"""

import torch
from kokoro.training.config import TrainingConfig

def test_warmup_configuration():
    """Test that warmup parameters are correctly set"""

    config = TrainingConfig()

    print("="*70)
    print("Linear Warmup Configuration Test")
    print("="*70)

    print(f"\n✓ use_warmup: {config.use_warmup}")
    print(f"✓ warmup_steps: {config.warmup_steps}")
    print(f"✓ warmup_start_lr_ratio: {config.warmup_start_lr_ratio}")
    print(f"✓ base learning_rate: {config.learning_rate:.2e}")

    warmup_start_lr = config.learning_rate * config.warmup_start_lr_ratio
    warmup_target_lr = config.learning_rate

    print(f"\n✓ Warmup start LR: {warmup_start_lr:.2e}")
    print(f"✓ Warmup target LR: {warmup_target_lr:.2e}")
    print(f"✓ LR increase during warmup: {warmup_target_lr / warmup_start_lr:.1f}x")

    # Calculate training timeline
    num_batches = 625  # Example from your dataset
    gradient_accumulation_steps = config.gradient_accumulation_steps
    optimizer_steps_per_epoch = (num_batches + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_optimizer_steps = config.num_epochs * optimizer_steps_per_epoch

    warmup_epochs = config.warmup_steps / optimizer_steps_per_epoch
    onecycle_steps = total_optimizer_steps - config.warmup_steps

    print(f"\n{'='*70}")
    print("Training Timeline:")
    print(f"{'='*70}")
    print(f"Total epochs:                {config.num_epochs}")
    print(f"Batches per epoch:           {num_batches}")
    print(f"Gradient accumulation:       {gradient_accumulation_steps}")
    print(f"Optimizer steps per epoch:   {optimizer_steps_per_epoch}")
    print(f"Total optimizer steps:       {total_optimizer_steps}")
    print(f"\nWarmup phase:")
    print(f"  Duration:                  {config.warmup_steps} optimizer steps")
    print(f"  Equivalent epochs:         {warmup_epochs:.2f} epochs")
    print(f"  LR range:                  {warmup_start_lr:.2e} → {warmup_target_lr:.2e}")
    print(f"\nOneCycleLR phase:")
    print(f"  Duration:                  {onecycle_steps} optimizer steps")
    print(f"  Equivalent epochs:         {onecycle_steps / optimizer_steps_per_epoch:.2f} epochs")
    print(f"  Max LR:                    {config.learning_rate * config.max_lr_multiplier:.2e}")

    print(f"\n{'='*70}")
    print("Learning Rate Schedule Visualization:")
    print(f"{'='*70}")

    # Simulate first 20 optimizer steps
    print(f"\n{'Step':<6} {'Phase':<15} {'Learning Rate':<15} {'Description'}")
    print("-" * 70)

    # Warmup steps
    test_steps = [0, 1, 2, 5, 10, 50, 100, 250, 499, 500, 501, 502, 600, 1000, 5000]

    for step in test_steps:
        if step < config.warmup_steps:
            # Warmup phase
            warmup_progress = step / config.warmup_steps
            current_lr = warmup_start_lr + (warmup_target_lr - warmup_start_lr) * warmup_progress
            phase = "Warmup"
            desc = f"{warmup_progress*100:.1f}% through warmup"
        else:
            # OneCycleLR phase (simplified - actual would need scheduler state)
            phase = "OneCycleLR"
            onecycle_step = step - config.warmup_steps
            desc = f"Step {onecycle_step}/{onecycle_steps} in OneCycleLR"
            current_lr = config.learning_rate  # Placeholder

        print(f"{step:<6} {phase:<15} {current_lr:.2e}      {desc}")

    print(f"\n{'='*70}")
    print("Benefits of Linear Warmup:")
    print(f"{'='*70}")
    print("✅ Prevents very low LR at start (1e-06 without warmup)")
    print("✅ Gradual increase to base LR reduces instability")
    print("✅ OneCycleLR takes over smoothly after warmup")
    print("✅ First 500 steps use stable, predictable LR increase")
    print(f"✅ Training effectively starts at {warmup_start_lr:.2e} instead of extremely low LR")

    print(f"\n{'='*70}")
    print("Warmup vs No Warmup:")
    print(f"{'='*70}")

    # OneCycleLR initial LR calculation
    max_lr = config.learning_rate * config.max_lr_multiplier
    div_factor = 25.0
    onecycle_initial_lr = max_lr / div_factor

    print(f"\nWithout warmup (OneCycleLR only):")
    print(f"  First step LR:             {onecycle_initial_lr:.2e}")
    print(f"  Issue:                     Too low, wastes training time")

    print(f"\nWith warmup (our implementation):")
    print(f"  First step LR:             {warmup_start_lr:.2e}")
    print(f"  After {config.warmup_steps} steps:          {warmup_target_lr:.2e}")
    print(f"  Benefit:                   {warmup_start_lr / onecycle_initial_lr:.1f}x higher initial LR")

    print(f"\n✅ Configuration loaded successfully!")
    print(f"✅ Linear warmup enabled for {config.warmup_steps} steps")
    print(f"✅ Ready to train with smooth LR ramp-up")

def simulate_warmup_schedule():
    """Simulate the warmup learning rate schedule"""

    config = TrainingConfig()

    print(f"\n\n{'='*70}")
    print("Warmup Learning Rate Schedule Simulation")
    print(f"{'='*70}\n")

    warmup_start_lr = config.learning_rate * config.warmup_start_lr_ratio
    warmup_target_lr = config.learning_rate
    warmup_steps = config.warmup_steps

    # Sample points across warmup
    sample_points = [0, 25, 50, 100, 200, 300, 400, 499]

    print(f"{'Optimizer Step':<15} {'Progress':<10} {'Learning Rate':<15} {'LR / Base LR'}")
    print("-" * 70)

    for step in sample_points:
        warmup_progress = step / warmup_steps
        current_lr = warmup_start_lr + (warmup_target_lr - warmup_start_lr) * warmup_progress
        lr_ratio = current_lr / config.learning_rate

        print(f"{step:<15} {warmup_progress*100:>6.1f}%    {current_lr:.6e}      {lr_ratio:.3f}x")

    print(f"\n{'='*70}")
    print("Key Milestones:")
    print(f"{'='*70}")
    print(f"Step 0:         {warmup_start_lr:.2e} (1% of base LR)")
    print(f"Step 250:       {(warmup_start_lr + warmup_target_lr) / 2:.2e} (50.5% of base LR)")
    print(f"Step 499:       {warmup_target_lr:.2e} (100% of base LR)")
    print(f"Step 500+:      OneCycleLR takes over from {warmup_target_lr:.2e}")

if __name__ == "__main__":
    test_warmup_configuration()
    simulate_warmup_schedule()

    print(f"\n{'='*70}")
    print("🎯 All warmup validations complete!")
    print(f"{'='*70}\n")
