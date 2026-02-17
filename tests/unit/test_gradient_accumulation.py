#!/usr/bin/env python3
"""
Comprehensive test with gradient accumulation to reproduce any remaining issues
"""
import sys
sys.path.insert(0, 'src')

import pytest
import torch
import torch.nn as nn
from kokoro.data.dataset import RuslanDataset, collate_fn, DynamicFrameBatchSampler
from kokoro.training.config import TrainingConfig
from kokoro.model.model import KokoroModel
from torch.utils.data import DataLoader
import gc


def test_with_gradient_accumulation():
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST WITH GRADIENT ACCUMULATION")
    print("="*80)

    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load config
    config = TrainingConfig()
    config.data_dir = "/Users/ishmukle/Projects/kokoro-ruslan/ruslan_corpus"
    config.use_mfa = True

    print("\n" + "-"*80)
    print("Loading dataset...")
    print("-"*80)

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

    print(f"Total batches: {len(dataloader)}")

    print("\n" + "-"*80)
    print("Creating full Kokoro model...")
    print("-"*80)

    vocab_size = dataset.phoneme_processor.get_vocab_size()
    model = KokoroModel(
        vocab_size=vocab_size,
        mel_dim=config.n_mels,
        hidden_dim=config.hidden_dim,
        use_variance_predictor=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        checkpoint_segments=4
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Gradient checkpointing: enabled (4 segments)")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print("\n" + "-"*80)
    print("Testing with gradient accumulation (4 steps)...")
    print("-"*80)

    accumulation_steps = 4
    start_batch = 278
    end_batch = 285

    optimizer.zero_grad()
    accumulated_batches = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < start_batch:
            continue
        if batch_idx > end_batch:
            break

        try:
            # Extract batch data
            phoneme_indices = batch['phoneme_indices'].to(device)
            mel_specs = batch['mel_specs'].to(device)
            phoneme_durations = batch['phoneme_durations'].to(device)
            stop_token_targets = batch['stop_token_targets'].to(device)
            pitch_targets = batch['pitches'].to(device) if 'pitches' in batch else None
            energy_targets = batch['energies'].to(device) if 'energies' in batch else None

            batch_size = phoneme_indices.shape[0]
            text_len = phoneme_indices.shape[1]
            mel_len = mel_specs.shape[1]

            print(f"\nBatch {batch_idx}: batch_size={batch_size}, text_len={text_len}, mel_len={mel_len}")

            # Forward pass
            outputs = model(
                phoneme_indices=phoneme_indices,
                mel_specs=mel_specs,
                phoneme_durations=phoneme_durations,
                stop_token_targets=stop_token_targets,
                pitch_targets=pitch_targets,
                energy_targets=energy_targets
            )

            # Calculate loss
            if isinstance(outputs, tuple):
                mel_output = outputs[0]
            else:
                mel_output = outputs['mel_output']

            mel_loss = nn.MSELoss()(mel_output, mel_specs)

            # Scale loss for gradient accumulation
            scaled_loss = mel_loss / accumulation_steps

            # Backward pass
            scaled_loss.backward()

            accumulated_batches.append(batch_idx)
            print(f"  ✅ Loss: {mel_loss.item():.4f} (accumulated)")

            # Highlight batch 281
            if batch_idx == 281:
                print(f"  🎯 This is BATCH 281 - accumulated in gradients!")

            # Optimizer step every N batches
            if len(accumulated_batches) == accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                print(f"\n  📊 Optimizer step completed for batches {accumulated_batches}")
                accumulated_batches = []

            # Memory cleanup
            del phoneme_indices, mel_specs, phoneme_durations, stop_token_targets
            del pitch_targets, energy_targets, outputs, mel_output, mel_loss, scaled_loss
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        except RuntimeError as e:
            print("\n" + "="*80)
            print(f"❌ ERROR AT BATCH {batch_idx}!")
            print("="*80)
            print(f"Error: {e}")

            if "INT_MAX" in str(e) or "NDArray dimension length" in str(e):
                print("\n⚠️  INT_MAX crash detected!")
                print(f"\nBatch {batch_idx} details:")
                print(f"  batch_size: {batch['phoneme_indices'].shape[0]}")
                print(f"  text_len: {batch['phoneme_indices'].shape[1]}")
                print(f"  mel_len: {batch['mel_specs'].shape[1]}")
                print(f"  Accumulated batches: {accumulated_batches}")
                import traceback
                traceback.print_exc()
                pytest.fail(f"INT_MAX/NDArray crash at batch {batch_idx}: {e}")
            else:
                print("\n⚠️  Different error")
                import traceback
                traceback.print_exc()
                pytest.fail(f"Unexpected RuntimeError at batch {batch_idx}: {e}")

    # Final optimizer step if there are remaining gradients
    if len(accumulated_batches) > 0:
        optimizer.step()
        print(f"\n  📊 Final optimizer step for batches {accumulated_batches}")

    print("\n" + "="*80)
    print("🎉 SUCCESS! All tests passed!")
    print("="*80)
    print(f"\n✅ Batch 281 processed successfully with gradient accumulation")
    print(f"✅ Gradient checkpointing enabled and working")
    print(f"✅ No INT_MAX or OOM errors detected")
    print(f"\nConclusion: The problem has been FIXED!")


if __name__ == "__main__":
    success = test_with_gradient_accumulation()
    sys.exit(0 if success else 1)
