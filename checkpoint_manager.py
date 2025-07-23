#!/usr/bin/env python3
"""
Checkpoint management utilities
"""

import os
import torch
import pickle
from pathlib import Path
from typing import Optional, Tuple
import logging

from config import TrainingConfig
from russian_phoneme_processor import RussianPhonemeProcessor

logger = logging.getLogger(__name__)


def save_phoneme_processor(processor: RussianPhonemeProcessor, output_dir: str):
    """Save phoneme processor separately as pickle file"""
    processor_path = os.path.join(output_dir, "phoneme_processor.pkl")
    with open(processor_path, 'wb') as f:
        pickle.dump(processor.to_dict(), f)
    logger.info(f"Phoneme processor saved: {processor_path}")


def load_phoneme_processor(output_dir: str) -> RussianPhonemeProcessor:
    """Load phoneme processor from pickle file"""
    processor_path = os.path.join(output_dir, "phoneme_processor.pkl")
    with open(processor_path, 'rb') as f:
        processor_data = pickle.load(f)
    processor = RussianPhonemeProcessor.from_dict(processor_data)
    logger.info(f"Phoneme processor loaded: {processor_path}")
    return processor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    config: TrainingConfig,
    output_dir: str
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config
    }
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    output_dir: str
) -> Tuple[int, float, RussianPhonemeProcessor]:
    """Load checkpoint and return starting epoch, best loss, and phoneme processor"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Add safe globals for our custom classes
    torch.serialization.add_safe_globals([TrainingConfig, RussianPhonemeProcessor])

    # Helper function to load and process state_dict
    def _load_and_process_state_dict(model_state_dict, current_model):
        model_keys = current_model.state_dict().keys()
        new_state_dict = {}
        for k, v in model_state_dict.items():
            if k in model_keys:
                current_param = current_model.state_dict()[k]
                current_shape = current_param.shape

                # Specific handling for positional encoding where we can truncate if needed
                if "positional_encoding.pe" in k: # Catches both encoder and decoder PEs
                    if v.shape == current_shape:
                        new_state_dict[k] = v
                    elif v.dim() == 3 and len(current_shape) == 3 and v.shape[0] == current_shape[0] and v.shape[2] == current_shape[2]:
                        # Check if it's a difference in sequence length (second dimension)
                        if v.shape[1] > current_shape[1]:
                            logger.warning(
                                f"Truncating checkpoint's '{k}' from {v.shape} to "
                                f"{current_shape}. Ensure 'params.max_seq_len' is correctly set."
                            )
                            new_state_dict[k] = v[:, :current_shape[1], :]
                        else: # v.shape[1] < current_shape[1] - cannot simply pad, usually indicates config error
                            logger.error(
                                f"Size mismatch for {k}: Checkpoint has {v.shape}, current model has {current_shape}. "
                                "Cannot extend positional encoding. Check 'params.max_seq_len' in your config. "
                                "Skipping this key, model might not load correctly without it."
                            )
                            # Do not add to new_state_dict, allowing load_state_dict to report missing key
                            # or you can load with strict=False later.
                    else: # Positional encoding has mismatched dimensions beyond just sequence length
                        logger.error(f"Complex size mismatch for positional encoding {k}: Checkpoint has {v.shape}, current model has {current_shape}. Skipping this key.")
                else:
                    # For all other layers, shapes must match exactly
                    if v.shape == current_shape:
                        new_state_dict[k] = v
                    else:
                        logger.error(
                            f"Size mismatch for {k}: Checkpoint has {v.shape}, current model has {current_shape}. "
                            "This indicates a significant architectural change. "
                            "Consider retraining or adjusting model definition. Skipping this key."
                        )
            else:
                logger.warning(f"Skipping unexpected key in checkpoint: {k}")
        return new_state_dict


    try:
        # Try loading with weights_only=True first (new default)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Process and load model state dict
        filtered_model_state_dict = _load_and_process_state_dict(checkpoint['model_state_dict'], model)
        # We use strict=False here because _load_and_process_state_dict might skip keys it can't handle
        # or if the current model has new parameters not in the old checkpoint.
        model.load_state_dict(filtered_model_state_dict, strict=False)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']

        # Try to load phoneme processor from checkpoint first
        if 'phoneme_processor' in checkpoint:
            phoneme_processor = checkpoint['phoneme_processor']
        else:
            # Fall back to loading from separate file
            phoneme_processor = load_phoneme_processor(output_dir)

        logger.info(f"Resumed from epoch {start_epoch} with loss {best_loss:.4f}")
        return start_epoch, best_loss, phoneme_processor

    except Exception as e:
        logger.warning(f"Loading with weights_only=True failed: {e}")
        logger.info("Trying to load with weights_only=False for compatibility...")

        try:
            # Try loading with weights_only=False for older checkpoints
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Process and load model state dict
            filtered_model_state_dict = _load_and_process_state_dict(checkpoint['model_state_dict'], model)
            model.load_state_dict(filtered_model_state_dict, strict=False) # Use strict=False here

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['loss']

            if 'phoneme_processor' in checkpoint:
                phoneme_processor = checkpoint['phoneme_processor']
            else:
                phoneme_processor = load_phoneme_processor(output_dir)

            logger.info(f"Resumed from epoch {start_epoch} with loss {best_loss:.4f}")
            return start_epoch, best_loss, phoneme_processor

        except Exception as e2:
            logger.error(f"Error loading checkpoint even with weights_only=False: {e2}")
            raise e2


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the output directory"""
    checkpoint_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None

    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    latest_checkpoint = checkpoint_files[-1]

    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)


def save_final_model(model: torch.nn.Module, config: TrainingConfig, output_dir: str):
    """Save final model"""
    final_model_path = os.path.join(output_dir, "kokoro_russian_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
