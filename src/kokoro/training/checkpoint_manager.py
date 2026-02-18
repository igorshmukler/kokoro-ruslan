#!/usr/bin/env python3
"""
Checkpoint management utilities
"""

import os
import torch
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

from kokoro.training.config import TrainingConfig
from kokoro.data.russian_phoneme_processor import RussianPhonemeProcessor

logger = logging.getLogger(__name__)


def build_model_metadata(config: TrainingConfig, model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """Build explicit architecture metadata for robust strict loading."""
    metadata = {
        'schema_version': 2,
        'architecture': {
            'mel_dim': int(getattr(config, 'n_mels', 80)),
            'hidden_dim': int(getattr(config, 'hidden_dim', 512)),
            'n_encoder_layers': int(getattr(config, 'n_encoder_layers', 6)),
            'n_decoder_layers': int(getattr(config, 'n_decoder_layers', 6)),
            'n_heads': int(getattr(config, 'n_heads', 8)),
            'encoder_ff_dim': int(getattr(config, 'encoder_ff_dim', 2048)),
            'decoder_ff_dim': int(getattr(config, 'decoder_ff_dim', 2048)),
            'encoder_dropout': float(getattr(config, 'encoder_dropout', 0.1)),
            'max_decoder_seq_len': int(getattr(config, 'max_decoder_seq_len', 4000)),
            'use_variance_predictor': bool(getattr(config, 'use_variance_predictor', True)),
            'variance_filter_size': int(getattr(config, 'variance_filter_size', 256)),
            'variance_kernel_size': int(getattr(config, 'variance_kernel_size', 3)),
            'variance_dropout': float(getattr(config, 'variance_dropout', 0.1)),
            'n_variance_bins': int(getattr(config, 'n_variance_bins', 256)),
            'pitch_min': float(getattr(config, 'pitch_min', 0.0)),
            'pitch_max': float(getattr(config, 'pitch_max', 1.0)),
            'energy_min': float(getattr(config, 'energy_min', 0.0)),
            'energy_max': float(getattr(config, 'energy_max', 1.0)),
            'use_stochastic_depth': bool(getattr(config, 'use_stochastic_depth', True)),
            'stochastic_depth_rate': float(getattr(config, 'stochastic_depth_rate', 0.1)),
        },
        'inference_controls': {
            'max_len': int(getattr(config, 'inference_max_len', 1200)),
            'stop_threshold': float(getattr(config, 'inference_stop_threshold', 0.45)),
            'min_len_ratio': float(getattr(config, 'inference_min_len_ratio', 0.7)),
            'min_len_floor': int(getattr(config, 'inference_min_len_floor', 12)),
        },
    }

    if model is not None:
        metadata['architecture']['vocab_size'] = int(getattr(model, 'vocab_size', 0))

    return metadata


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
    model_metadata = build_model_metadata(config, model)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config,
        'model_metadata': model_metadata,
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

    def _extract_architecture_metadata(checkpoint_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(checkpoint_dict, dict):
            return {}
        model_metadata = checkpoint_dict.get('model_metadata')
        if not isinstance(model_metadata, dict):
            return {}
        architecture = model_metadata.get('architecture')
        return architecture if isinstance(architecture, dict) else {}

    def _validate_architecture_metadata(architecture: Dict[str, Any], current_model: torch.nn.Module):
        required_fields = [
            'mel_dim', 'hidden_dim', 'n_encoder_layers', 'n_decoder_layers',
            'max_decoder_seq_len', 'use_variance_predictor'
        ]
        missing_fields = [field for field in required_fields if field not in architecture]
        if missing_fields:
            raise RuntimeError(
                f"Checkpoint metadata is incomplete. Missing fields: {missing_fields}. "
                "Please resume from a compatible checkpoint with full metadata."
            )

        current_snapshot = {
            'vocab_size': int(getattr(current_model, 'vocab_size', 0)),
            'mel_dim': int(getattr(current_model, 'mel_dim', 0)),
            'hidden_dim': int(getattr(current_model, 'hidden_dim', 0)),
            'n_encoder_layers': int(len(getattr(current_model, 'transformer_encoder_layers', []))),
            'n_decoder_layers': int(getattr(getattr(current_model, 'decoder', None), 'num_layers', 0)),
            'max_decoder_seq_len': int(getattr(current_model, 'max_decoder_seq_len', 0)),
            'use_variance_predictor': bool(getattr(current_model, 'use_variance_predictor', False)),
        }

        compared_fields = [
            'vocab_size', 'mel_dim', 'hidden_dim', 'n_encoder_layers',
            'n_decoder_layers', 'max_decoder_seq_len', 'use_variance_predictor'
        ]
        mismatches = []
        for field in compared_fields:
            if field not in architecture:
                continue
            expected_value = architecture[field]
            current_value = current_snapshot[field]
            if isinstance(current_value, bool):
                expected_cast = bool(expected_value)
            elif isinstance(current_value, int):
                expected_cast = int(expected_value)
            else:
                expected_cast = expected_value
            if expected_cast != current_value:
                mismatches.append((field, expected_cast, current_value))

        if mismatches:
            mismatch_summary = ', '.join(
                [f"{field}: checkpoint={expected} current={current}" for field, expected, current in mismatches]
            )
            raise RuntimeError(
                "Checkpoint architecture metadata mismatch. "
                f"{mismatch_summary}. "
                "Use a checkpoint produced by the same model/config."
            )

    def _strict_resume(checkpoint_dict: Dict[str, Any]) -> Tuple[int, float, RussianPhonemeProcessor]:
        if not isinstance(checkpoint_dict, dict):
            raise RuntimeError("Checkpoint payload is not a dictionary.")

        architecture = _extract_architecture_metadata(checkpoint_dict)
        if architecture:
            _validate_architecture_metadata(architecture, model)
        else:
            raise RuntimeError(
                "Checkpoint is missing required 'model_metadata.architecture'. "
                "Legacy checkpoints without metadata are not supported for strict resume. "
                "Use a checkpoint saved with metadata from the same training run."
            )

        if 'model_state_dict' in checkpoint_dict:
            model_state_dict = checkpoint_dict['model_state_dict']
        elif 'model' in checkpoint_dict:
            model_state_dict = checkpoint_dict['model']
        else:
            # Legacy raw state_dict fallback
            if all(isinstance(v, torch.Tensor) for v in checkpoint_dict.values()):
                model_state_dict = checkpoint_dict
            else:
                raise RuntimeError(
                    "Checkpoint does not contain a recognized model state dictionary. "
                    "Expected key 'model_state_dict' or 'model'."
                )

        try:
            model.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                "Strict checkpoint model load failed due to architecture/state mismatch. "
                f"Original error: {e}"
            ) from e

        if 'optimizer_state_dict' not in checkpoint_dict or 'scheduler_state_dict' not in checkpoint_dict:
            raise RuntimeError(
                "Checkpoint is missing optimizer/scheduler state required for training resume."
            )

        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])

        if 'epoch' not in checkpoint_dict or 'loss' not in checkpoint_dict:
            raise RuntimeError("Checkpoint is missing required 'epoch' or 'loss' fields.")

        start_epoch = checkpoint_dict['epoch'] + 1
        best_loss = checkpoint_dict['loss']

        if 'phoneme_processor' in checkpoint_dict:
            phoneme_processor = checkpoint_dict['phoneme_processor']
        else:
            phoneme_processor = load_phoneme_processor(output_dir)

        logger.info(f"Resumed from epoch {start_epoch} with loss {best_loss:.4f}")
        return start_epoch, best_loss, phoneme_processor


    try:
        # Try loading with weights_only=True first (new default)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        return _strict_resume(checkpoint)

    except Exception as e:
        logger.warning(f"Loading with weights_only=True failed: {e}")
        logger.info("Trying to load with weights_only=False for compatibility...")

        try:
            # Try loading with weights_only=False for older checkpoints
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            return _strict_resume(checkpoint)

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
    model_metadata = build_model_metadata(config, model)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_metadata': model_metadata,
    }, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
