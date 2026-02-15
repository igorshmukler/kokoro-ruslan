#!/usr/bin/env python3
"""
Test to verify pitch/energy normalization is actually working
"""
import sys
sys.path.insert(0, 'src')

import torch
from kokoro.data.dataset import RuslanDataset
from kokoro.training.config import TrainingConfig
from torch.utils.data import DataLoader
from kokoro.data.dataset import collate_fn, DynamicFrameBatchSampler


def test_pitch_energy_normalization():
    print("\n" + "="*80)
    print("TESTING PITCH/ENERGY NORMALIZATION")
    print("="*80)

    # Load config
    config = TrainingConfig()
    config.data_dir = "/Users/ishmukle/Projects/kokoro-ruslan/ruslan_corpus"
    config.use_mfa = True

    print("\nLoading dataset...")
    dataset = RuslanDataset(
        data_dir=config.data_dir,
        config=config,
        use_mfa=True
    )

    # Create dataloader
    batch_sampler = DynamicFrameBatchSampler(
        dataset=dataset,
        max_frames=config.max_frames_per_batch,
        min_batch_size=config.min_batch_size,
        max_batch_size=config.max_batch_size,
        drop_last=True,
        shuffle=False
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=0
    )

    print("\n" + "-"*80)
    print("Checking pitch/energy values in batches...")
    print("-"*80)

    unnormalized_count = 0
    batch_count = 0
    max_pitch_seen = 0.0
    max_energy_seen = 0.0

    # Check first 10 batches and batch 281, 816
    target_batches = list(range(10)) + [281, 816] if len(dataloader) > 816 else list(range(10)) + [281]

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx not in target_batches:
            if batch_idx > max(target_batches):
                break
            continue

        batch_count += 1

        pitches = batch['pitches']
        energies = batch['energies']

        pitch_min = pitches.min().item()
        pitch_max = pitches.max().item()
        energy_min = energies.min().item()
        energy_max = energies.max().item()

        max_pitch_seen = max(max_pitch_seen, pitch_max)
        max_energy_seen = max(max_energy_seen, energy_max)

        print(f"\nBatch {batch_idx}:")
        print(f"  Pitch:  min={pitch_min:.4f}, max={pitch_max:.4f}, mean={pitches.mean().item():.4f}")
        print(f"  Energy: min={energy_min:.4f}, max={energy_max:.4f}, mean={energies.mean().item():.4f}")

        # Check if values are properly normalized [0, 1]
        if pitch_max > 1.5 or energy_max > 1.5:
            print(f"  ❌ UNNORMALIZED! Pitch max={pitch_max:.2f}, Energy max={energy_max:.2f}")
            unnormalized_count += 1
        elif pitch_max > 1.0 or energy_max > 1.0:
            print(f"  ⚠️  Values slightly exceed 1.0 (expected range [0, 1])")
        else:
            print(f"  ✅ Properly normalized to [0, 1]")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Batches checked: {batch_count}")
    print(f"Maximum pitch value seen: {max_pitch_seen:.4f}")
    print(f"Maximum energy value seen: {max_energy_seen:.4f}")

    if unnormalized_count > 0:
        print(f"\n❌ PROBLEM FOUND: {unnormalized_count} batches have UNNORMALIZED values!")
        print("   This will cause loss explosion and INT_MAX crash!")
        print("\n   FIX NEEDED: Ensure PitchExtractor and EnergyExtractor normalize to [0, 1]")
        return False
    elif max_pitch_seen > 1.0 or max_energy_seen > 1.0:
        print(f"\n⚠️  WARNING: Values slightly exceed 1.0")
        print(f"   Max pitch: {max_pitch_seen:.4f}, Max energy: {max_energy_seen:.4f}")
        print("   Should be clamped to [0, 1] range")
        return False
    else:
        print(f"\n✅ SUCCESS: All values properly normalized to [0, 1]")
        print("   No risk of loss explosion or INT_MAX crash")
        return True


if __name__ == "__main__":
    success = test_pitch_energy_normalization()
    sys.exit(0 if success else 1)
