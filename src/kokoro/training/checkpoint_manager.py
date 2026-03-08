#!/usr/bin/env python3
"""
Checkpoint management utilities
"""

import math
import os
import torch
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

from torch.utils.tensorboard import SummaryWriter

from kokoro.training.config import TrainingConfig
from kokoro.data.russian_phoneme_processor import RussianPhonemeProcessor

logger = logging.getLogger(__name__)


def build_model_metadata(config: TrainingConfig, model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """Build explicit architecture metadata for robust strict loading."""
    metadata = {
        'schema_version': 2,
        'architecture': {
            'mel_dim': int(getattr(config, 'n_mels', 80)),
            'hidden_dim': int(getattr(config, 'hidden_dim', 768)),
            'n_encoder_layers': int(getattr(config, 'n_encoder_layers', 6)),
            'n_decoder_layers': int(getattr(config, 'n_decoder_layers', 6)),
            'n_heads': int(getattr(config, 'n_heads', 8)),
            # Derive ff_dim from the actual model weights (GLU: linear1 outputs ff_dim*2).
            # This is the authoritative source; config values can diverge from what was
            # actually used to build the model.
            'encoder_ff_dim': int(getattr(config, 'encoder_ff_dim', 3072)),
            'decoder_ff_dim': int(getattr(config, 'decoder_ff_dim', 3072)),
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

        # Override ff_dim with values derived from the actual model weights.
        # The GLU-style FFN projects to ff_dim*2 in linear1 and back from ff_dim in linear2,
        # so the true ff_dim = linear1.out_features // 2.  This overrides any stale config
        # value and is the authoritative record of what was actually trained.
        enc_layers = getattr(model, 'transformer_encoder_layers', None)
        if enc_layers and len(enc_layers) > 0:
            linear1 = getattr(enc_layers[0], 'linear1', None)
            if linear1 is not None and hasattr(linear1, 'out_features'):
                metadata['architecture']['encoder_ff_dim'] = int(linear1.out_features // 2)

        decoder = getattr(model, 'decoder', None)
        dec_layers = getattr(decoder, 'layers', None) if decoder is not None else None
        if dec_layers and len(dec_layers) > 0:
            linear1 = getattr(dec_layers[0], 'linear1', None)
            if linear1 is not None and hasattr(linear1, 'out_features'):
                metadata['architecture']['decoder_ff_dim'] = int(linear1.out_features // 2)

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
            # If the checkpoint indicates variance predictors were used, ensure
            # that the saved pitch/energy normalization bounds correspond to
            # normalized [0,1] targets. Loading a checkpoint trained with
            # different normalization without adjusting the current config can
            # silently corrupt training/inference, so fail-fast here.
            if architecture.get('use_variance_predictor', False):
                p_min = float(architecture.get('pitch_min', float('nan')))
                p_max = float(architecture.get('pitch_max', float('nan')))
                e_min = float(architecture.get('energy_min', float('nan')))
                e_max = float(architecture.get('energy_max', float('nan')))

                if not (p_min == 0.0 and p_max == 1.0 and e_min == 0.0 and e_max == 1.0):
                    raise RuntimeError(
                        "Checkpoint variance targets are not normalized to [0,1]. "
                        "Aborting strict resume/load to avoid predictor-target mismatch. "
                        "If this checkpoint was intentionally trained with different "
                        "normalization, either resave with normalized targets or set "
                        "use_variance_predictor=False in the current config and handle "
                        "normalization manually."
                    )
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
            # Check whether ALL missing keys belong to variance_adaptor sub-components
            # that were added in a later architecture revision. If so, allow a partial
            # load so that training can resume with those weights freshly initialised.
            error_str = str(e)
            import re as _re
            missing_match = _re.search(r'Missing key\(s\) in state_dict:\s*(.*?)(?:\.\s*Unexpected|\Z)', error_str, _re.DOTALL)
            unexpected_match = _re.search(r'Unexpected key\(s\) in state_dict:\s*(.*?)(?:\.\s*Missing|\Z)', error_str, _re.DOTALL)

            # Collect missing keys
            missing_keys: list = []
            if missing_match:
                missing_keys = [k.strip().strip('"') for k in missing_match.group(1).split(',') if k.strip()]

            # Collect unexpected keys (keys in ckpt but not in model)
            unexpected_keys: list = []
            if unexpected_match:
                unexpected_keys = [k.strip().strip('"') for k in unexpected_match.group(1).split(',') if k.strip()]

            # Define the namespace prefixes that represent newly-added sub-modules
            _NEW_VARIANCE_PREFIX = 'duration_adaptor.variance_adaptor.'

            all_missing_are_new_variance = missing_keys and all(
                k.startswith(_NEW_VARIANCE_PREFIX) for k in missing_keys
            )
            no_unexpected = not unexpected_keys

            if all_missing_are_new_variance and no_unexpected:
                logger.warning(
                    "Checkpoint is missing variance_adaptor sub-module weights "
                    f"({len(missing_keys)} keys). This is an expected architecture "
                    "migration (old checkpoint predates the restructured VarianceAdaptor). "
                    "Loading shared weights and initialising missing keys from scratch."
                )
                # Load the compatible weights; missing keys keep their randomly-initialised values
                incompatible = model.load_state_dict(model_state_dict, strict=False)
                if incompatible.unexpected_keys:
                    raise RuntimeError(
                        "Partial checkpoint load produced unexpected keys, aborting. "
                        f"Unexpected: {incompatible.unexpected_keys}"
                    )
                logger.info(
                    f"Partial load succeeded. {len(incompatible.missing_keys)} variance_adaptor "
                    "keys will train from random initialisation."
                )
            else:
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


def resume_from_checkpoint(trainer, *, _load_checkpoint_fn=None, _SummaryWriter=None) -> None:
    """Resume all training state from a checkpoint into *trainer*.

    Handles model weights, optimizer, scheduler, scaler (AMP), EMA, step
    counters, TensorBoard writer purge, and OneCycleLR reconstruction.
    Mutates *trainer* in-place; no return value.

    Args:
        trainer: A ``KokoroTrainer`` instance (typed as ``Any`` to avoid a
            circular import — only attribute access is required).
        _load_checkpoint_fn: Override for ``load_checkpoint``; used by tests
            to inject a monkeypatched version without altering this module.
        _SummaryWriter: Override for ``SummaryWriter``; same rationale.
    """
    _lc = _load_checkpoint_fn if _load_checkpoint_fn is not None else load_checkpoint
    _SW = _SummaryWriter if _SummaryWriter is not None else SummaryWriter

    config = trainer.config

    if not config.resume_checkpoint:
        logger.info("No resume checkpoint specified, starting from scratch.")
        return

    checkpoint_path = None
    if config.resume_checkpoint.lower() == 'auto':
        checkpoint_path = find_latest_checkpoint(config.output_dir)
        if not checkpoint_path:
            logger.info("No checkpoint found for auto-resume, starting from scratch.")
            return
    else:
        checkpoint_path = config.resume_checkpoint
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    trainer.start_epoch, trainer.best_loss, phoneme_processor = _lc(
        checkpoint_path, trainer.model, trainer.optimizer, trainer.scheduler, config.output_dir
    )

    checkpoint = None
    try:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    except Exception as e:
        logger.warning(f"Could not load raw checkpoint metadata: {e}")

    # Load scaler state if available
    if trainer.use_mixed_precision and trainer.scaler:
        try:
            if checkpoint is None:
                checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
            if 'scaler' in checkpoint:
                trainer.scaler.load_state_dict(checkpoint['scaler'])
                logger.info(f"Loaded {trainer.device_type.upper()} scaler state from checkpoint")
            else:
                logger.info(f"No scaler state found in checkpoint, using default for {trainer.device_type}")
        except Exception as e:
            logger.warning(f"Could not load scaler state: {e}")

    # Load EMA model state if available
    if trainer.use_ema and trainer.ema_model is not None:
        try:
            if checkpoint is None:
                checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
            if 'ema_model_state_dict' in checkpoint:
                try:
                    trainer.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                except RuntimeError as _ema_err:
                    # Architecture migration: if the only missing keys are the
                    # newly-restructured variance_adaptor sub-modules, do a
                    # partial load and fill the missing keys from the current
                    # model (which already has them freshly initialised).
                    _NEW_VP = 'duration_adaptor.variance_adaptor.'
                    _incompat = trainer.ema_model.load_state_dict(
                        checkpoint['ema_model_state_dict'], strict=False
                    )
                    if _incompat.unexpected_keys or not all(
                        k.startswith(_NEW_VP) for k in _incompat.missing_keys
                    ):
                        raise
                    # Copy freshly-initialised weights from current model into EMA
                    _current_sd = trainer.model.state_dict()
                    _ema_sd = trainer.ema_model.state_dict()
                    for k in _incompat.missing_keys:
                        if k in _current_sd:
                            _ema_sd[k] = _current_sd[k].clone()
                    trainer.ema_model.load_state_dict(_ema_sd)
                    logger.warning(
                        f"EMA partial load: {len(_incompat.missing_keys)} variance_adaptor "
                        "keys initialised from current model (architecture migration)."
                    )
                if 'ema_updates' in checkpoint:
                    trainer.ema_updates = checkpoint['ema_updates']
                logger.info(f"Loaded EMA model state from checkpoint (updates: {trainer.ema_updates})")
            else:
                logger.info("No EMA state found in checkpoint, initializing EMA from current model")
                trainer.ema_model.load_state_dict(trainer.model.state_dict())
        except Exception as e:
            logger.warning(f"Could not load EMA state: {e}")

    if checkpoint is not None:
        if 'current_optimizer_step' in checkpoint:
            trainer.current_optimizer_step = int(checkpoint['current_optimizer_step'])
            logger.info(f"Restored current_optimizer_step={trainer.current_optimizer_step}")
        else:
            logger.info("No current_optimizer_step found in checkpoint, using default counter state")

        if 'optimizer_steps_completed' in checkpoint:
            trainer.optimizer_steps_completed = int(checkpoint['optimizer_steps_completed'])
            logger.info(f"Restored optimizer_steps_completed={trainer.optimizer_steps_completed}")
        else:
            logger.info("No optimizer_steps_completed found in checkpoint, using default counter state")

        # Restore best validation loss so early-stopping and best-model saving
        # resume from the correct baseline rather than float('inf').
        # load_checkpoint() only sets trainer.best_loss (from the 'loss'/train-loss
        # key); the training loop checks trainer.best_val_loss, so without this
        # the first post-resume epoch always "improves by inf" and overwrites the
        # best checkpoint even when the new validation loss is actually worse.
        if 'val_loss' in checkpoint:
            trainer.best_val_loss = float(checkpoint['val_loss'])
            logger.info(f"Restored best_val_loss={trainer.best_val_loss:.4f} from checkpoint")
        else:
            logger.info("No val_loss in checkpoint; best_val_loss remains inf (first epoch will always save)")

    trainer.dataset.phoneme_processor = phoneme_processor

    logger.info(f"Resumed from epoch {trainer.start_epoch}, best loss {trainer.best_loss:.4f}")

    # Purge stale TensorBoard events from epochs after the resume point.
    purge_step = trainer.current_optimizer_step + 1
    trainer.writer.close()
    trainer.writer = _SW(log_dir=trainer.log_dir, purge_step=purge_step)
    logger.info(
        f"TensorBoard writer reopened with purge_step={purge_step} "
        f"(hiding stale events from optimizer_step >= {purge_step})"
    )

    # Reconstruct the OneCycleLR from the CURRENT config to avoid stale phase
    # definitions loaded via load_state_dict() from a checkpoint that may have
    # been saved with different max_lr / pct_start values.
    if trainer.scheduler_per_batch and isinstance(trainer.scheduler, torch.optim.lr_scheduler.OneCycleLR):
        onecycle_steps = getattr(trainer, '_onecycle_steps', None)
        max_lr = getattr(trainer, '_onecycle_max_lr', None)
        pct_start = getattr(trainer, '_onecycle_pct_start', 0.3)
        if onecycle_steps is not None and max_lr is not None:
            # _onecycle_max_lr is always the DECODER (base) max_lr.
            # The encoder group (group 0 when multiple groups exist) uses max_lr * encoder_mult.
            # Use the last param group (decoder) as the reference for computing the schedule
            # position, since its LR range is [base_lr, max_lr] without the encoder scaling.
            n_pg = len(trainer.optimizer.param_groups)
            ref_group_idx = n_pg - 1  # last group = decoder params
            encoder_lr_mult = getattr(trainer, '_encoder_lr_multiplier', 1.0)
            last_lr = trainer.optimizer.state_dict()['param_groups'][ref_group_idx]['lr']

            _div_factor = getattr(trainer, '_onecycle_div_factor',
                float(getattr(config, 'max_lr_multiplier', 5.0))
                if getattr(trainer, 'use_warmup', False) else 25.0)
            base_lr  = max_lr / _div_factor
            final_lr = max_lr / 10000.0
            peak_step = int(pct_start * onecycle_steps)

            if last_lr >= max_lr:
                target_step = peak_step
                resume_lr   = max_lr
                logger.info(
                    f"Resume LR ({last_lr:.2e}) >= new max_lr ({max_lr:.2e}); "
                    f"positioning OneCycleLR at peak (step {peak_step})."
                )
            elif last_lr >= base_lr:
                ratio = (last_lr - base_lr) / (max_lr - base_lr)
                t = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * ratio))) / math.pi
                target_step = max(0, min(peak_step, int(round(t * peak_step))))
                resume_lr   = last_lr
            elif last_lr >= final_lr:
                decay_steps = onecycle_steps - peak_step
                ratio = (last_lr - final_lr) / (max_lr - final_lr)
                t = math.acos(max(-1.0, min(1.0, 2.0 * ratio - 1.0))) / math.pi
                target_step = max(peak_step, min(onecycle_steps - 1, peak_step + int(round(t * decay_steps))))
                resume_lr   = last_lr
            else:
                target_step = onecycle_steps - 1
                resume_lr   = final_lr

            # Build per-group max_lr: encoder group (group 0 when n_pg > 1) gets encoder_lr_mult × max_lr
            if n_pg > 1:
                max_lr_arg = [max_lr * encoder_lr_mult] + [max_lr] * (n_pg - 1)
            else:
                max_lr_arg = max_lr

            saved_lr = [g['lr'] for g in trainer.optimizer.param_groups]
            trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                trainer.optimizer,
                max_lr=max_lr_arg,
                total_steps=onecycle_steps,
                pct_start=pct_start,
                anneal_strategy='cos',
                cycle_momentum=False,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=_div_factor,
                final_div_factor=10000.0,
                last_epoch=-1,
            )
            trainer.scheduler.last_epoch = target_step
            resume_lr = min(resume_lr, max_lr)
            for i, g in enumerate(trainer.optimizer.param_groups):
                # Encoder group (group 0 in multi-group setup) keeps its higher LR
                mult = encoder_lr_mult if (n_pg > 1 and i == 0) else 1.0
                g['lr'] = resume_lr * mult
            trainer.scheduler._last_lr = [
                resume_lr * (encoder_lr_mult if (n_pg > 1 and i == 0) else 1.0)
                for i in range(n_pg)
            ]

            logger.info(
                f"OneCycleLR reconstructed from current config after resume: "
                f"decoder max_lr={max_lr:.2e}, encoder max_lr={max_lr * encoder_lr_mult:.2e}, "
                f"total_steps={onecycle_steps}, pct_start={pct_start}, "
                f"last_lr={last_lr:.2e} → positioned at step {target_step}/{onecycle_steps}, "
                f"resume_lr={resume_lr:.2e}"
            )
            del saved_lr
        else:
            logger.warning("Could not reconstruct OneCycleLR after resume: missing _onecycle_steps or _onecycle_max_lr")


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
