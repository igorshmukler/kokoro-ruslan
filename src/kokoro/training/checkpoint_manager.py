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


def _purge_tb_events_after_step(
    log_dir: str,
    keep_up_to_step: int,
    _SummaryWriter=None,
) -> object:
    """Rewrite TensorBoard event files keeping only events at step <= keep_up_to_step.

    Reads every .tfevents file in *log_dir*, retains events whose global_step is
    <= keep_up_to_step (plus metadata events that carry no step data), writes them
    all into a single new event file, deletes the old files, and returns a fresh
    SummaryWriter pointing at the same directory.

    Preserved event types:
      * Scalars  — replayed via ``add_scalar`` (float round-trip).
      * Images   — replayed via ``file_writer.add_summary`` (lossless proto copy).
      * Histograms — replayed via ``file_writer.add_summary`` (lossless proto copy).
      * Custom scalars layout — replayed via ``file_writer.add_summary`` (lossless proto copy).

    Falls back to a plain delete-and-reopen if the tensorboard event reader is
    unavailable.
    """
    _SW = _SummaryWriter if _SummaryWriter is not None else SummaryWriter
    log_dir_path = Path(log_dir)
    event_files = sorted(log_dir_path.glob("events.out.tfevents.*"))

    # kept_scalars: list of (tag, step, value) tuples to replay via add_scalar.
    # Storing (tag, step, value) rather than raw proto bytes avoids all
    # EventFileWriter API differences across PyTorch / TensorBoard versions.
    kept_scalars: list = []
    # kept_images / kept_histograms: list of (step, Summary.Value proto) tuples.
    # We preserve the raw proto so the encoded bytes (PNG for images, bucket
    # arrays for histograms) are copied verbatim without re-encoding.
    kept_images: list = []
    kept_histograms: list = []
    # kept_custom_scalars: list of (step, Summary.Value proto) for the
    # custom_scalars layout plugin (written by add_custom_scalars).
    kept_custom_scalars: list = []
    read_ok = False
    if event_files:
        try:
            from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

            for ef in event_files:
                try:
                    loader = EventFileLoader(str(ef))
                    for raw in loader.Load():
                        if not raw.HasField('summary'):
                            continue
                        if raw.step > keep_up_to_step:
                            continue
                        for value in raw.summary.value:
                            tag = value.tag
                            # simple_value covers add_scalar calls;
                            # tensor covers scalars, images and histograms in
                            # newer TensorBoard versions — distinguish by plugin.
                            if value.HasField('simple_value'):
                                kept_scalars.append((tag, raw.step, value.simple_value))
                            elif value.HasField('tensor'):
                                plugin = ""
                                if (value.HasField('metadata')
                                        and value.metadata.HasField('plugin_data')):
                                    plugin = value.metadata.plugin_data.plugin_name
                                if plugin == 'images':
                                    kept_images.append((raw.step, value))
                                elif plugin == 'histograms':
                                    kept_histograms.append((raw.step, value))
                                elif plugin == 'custom_scalars':
                                    kept_custom_scalars.append((raw.step, value))
                                else:
                                    # Attempt scalar conversion for plain tensor scalars.
                                    try:
                                        import numpy as np
                                        from tensorboard.util.tensor_util import make_ndarray
                                        arr = make_ndarray(value.tensor)
                                        kept_scalars.append((tag, raw.step, float(arr)))
                                    except Exception:
                                        pass  # skip non-scalar tensors
                            elif value.HasField('image'):
                                # Legacy proto format (older TensorBoard builds).
                                kept_images.append((raw.step, value))
                            elif value.HasField('histo'):
                                # Legacy proto format (older TensorBoard builds).
                                kept_histograms.append((raw.step, value))
                except Exception as _ef_err:
                    logger.warning(f"Could not read TensorBoard event file {ef}: {_ef_err}")
            read_ok = True
            logger.info(
                f"Read {len(kept_scalars)} scalars, {len(kept_images)} images, "
                f"{len(kept_histograms)} histograms, {len(kept_custom_scalars)} custom scalar layouts "
                f"(step <= {keep_up_to_step}) from {len(event_files)} event file(s) in {log_dir}"
            )
        except ImportError:
            logger.warning(
                "tensorboard EventFileLoader not available — "
                "falling back to delete-and-reopen (history will be lost)"
            )

    # Delete all existing event files
    for ef in event_files:
        try:
            ef.unlink()
        except OSError as _del_err:
            logger.warning(f"Could not delete TensorBoard event file {ef}: {_del_err}")

    # Open fresh writer (creates new event file)
    new_writer = _SW(log_dir=log_dir)

    if read_ok:
        # --- Scalars: replay via add_scalar (float, version-independent) ---
        if kept_scalars:
            try:
                for tag, step, value in kept_scalars:
                    new_writer.add_scalar(tag, value, global_step=step)
                new_writer.flush()
                logger.info(
                    f"Replayed {len(kept_scalars)} scalars into new TensorBoard file "
                    f"(purged everything after step {keep_up_to_step})"
                )
            except Exception as _write_err:
                logger.warning(
                    f"Could not replay historical scalars into TensorBoard writer: {_write_err}. "
                    "Chart history will start from the resume point."
                )

        # --- Images, histograms & custom scalars layout: replay via file_writer.add_summary (lossless) ---
        if kept_images or kept_histograms or kept_custom_scalars:
            try:
                from tensorboard.compat.proto.summary_pb2 import Summary

                for step, val in kept_images:
                    s = Summary(value=[val])
                    new_writer.file_writer.add_summary(s, global_step=step)

                for step, val in kept_histograms:
                    s = Summary(value=[val])
                    new_writer.file_writer.add_summary(s, global_step=step)

                for step, val in kept_custom_scalars:
                    s = Summary(value=[val])
                    new_writer.file_writer.add_summary(s, global_step=step)

                new_writer.flush()
                logger.info(
                    f"Replayed {len(kept_images)} images, {len(kept_histograms)} histograms, "
                    f"and {len(kept_custom_scalars)} custom scalar layouts "
                    f"into new TensorBoard file (purged everything after step {keep_up_to_step})"
                )
            except Exception as _write_err:
                logger.warning(
                    f"Could not replay historical images/histograms/custom scalars into TensorBoard writer: {_write_err}. "
                    "Image/histogram/custom scalar history will start from the resume point."
                )

    return new_writer


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
            'qk_norm': bool(getattr(config, 'qk_norm', False)),
            'ffn_output_norm': bool(getattr(config, 'ffn_output_norm', True)),
            'num_speakers': int(getattr(config, 'num_speakers', 1)),
            'speaker_embed_dim': int(getattr(config, 'speaker_embed_dim', 256)),
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
            # GLUFeedForward is at enc_layers[i].ff; linear1 is enc_layers[i].ff.linear1
            ff_block = getattr(enc_layers[0], 'ff', None)
            linear1 = getattr(ff_block, 'linear1', None) if ff_block is not None else None
            if linear1 is not None and hasattr(linear1, 'out_features'):
                metadata['architecture']['encoder_ff_dim'] = int(linear1.out_features // 2)

        decoder = getattr(model, 'decoder', None)
        dec_layers = getattr(decoder, 'layers', None) if decoder is not None else None
        if dec_layers and len(dec_layers) > 0:
            # GLUFeedForward is at dec_layers[i].ff; linear1 is dec_layers[i].ff.linear1
            ff_block = getattr(dec_layers[0], 'ff', None)
            linear1 = getattr(ff_block, 'linear1', None) if ff_block is not None else None
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

        _architecture_migrated = False
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

            # Collect missing keys — use findall to correctly parse quoted key names
            # (avoids the trailing '."' suffix on the last key when splitting by comma)
            missing_keys: list = []
            if missing_match:
                missing_keys = _re.findall(r'"([^"]+)"', missing_match.group(1))

            # Collect unexpected keys (keys in ckpt but not in model)
            unexpected_keys: list = []
            if unexpected_match:
                unexpected_keys = _re.findall(r'"([^"]+)"', unexpected_match.group(1))

            # Define the namespace prefixes that represent newly-added sub-modules
            _NEW_VARIANCE_PREFIX = 'duration_adaptor.variance_adaptor.'

            # Positional encoding migration: checkpoints trained with ALiBi decoder
            # self-attention store 'alibi_slopes' buffers.  After the switch to RoPE
            # (which uses persistent=False buffers — not in state_dict), these become
            # unexpected keys.  They are deterministic constants (fixed function of
            # num_heads); discarding them is safe.
            _ALIBI_SUFFIX = '.alibi_slopes'

            # FFN output norm migration: when ffn_output_norm is enabled on a
            # checkpoint trained without it, the output_norm.weight keys will be
            # missing.  RMSNorm weight initialises to ones — safe to re-init.
            _FFN_OUTPUT_NORM_SUFFIX = '.ff.output_norm.weight'

            # Speaker embedding migration: single-speaker → multi-speaker.
            # speaker_embedding.weight inits to zeros for speaker 0 (backward compat),
            # speaker_projection.weight/bias are freshly initialised.
            _SPEAKER_EMBEDDING_PREFIXES = ('speaker_embedding.', 'speaker_projection.')

            all_missing_are_known = (
                not missing_keys
                or all(
                    k.startswith(_NEW_VARIANCE_PREFIX)
                    or k.endswith(_FFN_OUTPUT_NORM_SUFFIX)
                    or k.startswith(_SPEAKER_EMBEDDING_PREFIXES)
                    for k in missing_keys
                )
            )
            all_unexpected_are_known = (
                not unexpected_keys
                or all(k.endswith(_ALIBI_SUFFIX) for k in unexpected_keys)
            )

            if all_missing_are_known and all_unexpected_are_known:
                if missing_keys:
                    logger.warning(
                        f"Checkpoint is missing {len(missing_keys)} key(s) from expected "
                        "architecture migrations (variance_adaptor / ffn_output_norm / speaker_embedding). "
                        "Loading shared weights and initialising missing keys from scratch."
                    )
                if unexpected_keys:
                    logger.warning(
                        f"Checkpoint contains {len(unexpected_keys)} legacy ALiBi positional "
                        "encoding buffer(s) (alibi_slopes) that are not used by the current "
                        "RoPE-based model.  These will be discarded."
                    )
                # Load the compatible weights; missing keys keep their randomly-initialised values
                incompatible = model.load_state_dict(model_state_dict, strict=False)
                residual_unexpected = [
                    k for k in incompatible.unexpected_keys
                    if not k.endswith(_ALIBI_SUFFIX)
                ]
                if residual_unexpected:
                    raise RuntimeError(
                        "Partial checkpoint load produced unexpected non-migration keys, aborting. "
                        f"Unexpected: {residual_unexpected}"
                    )
                logger.info(
                    f"Partial load succeeded. "
                    f"{len(incompatible.missing_keys)} keys initialised from scratch, "
                    f"{len(incompatible.unexpected_keys)} legacy keys discarded."
                )
                # Architecture changed — optimizer/scheduler states are incompatible.
                # Skip their restoration entirely (scheduler load may "succeed" but
                # produce wrong LR curve when total_steps or param layout changed).
                _architecture_migrated = True
            else:
                raise RuntimeError(
                    "Strict checkpoint model load failed due to architecture/state mismatch. "
                    f"Original error: {e}"
                ) from e

        if 'optimizer_state_dict' not in checkpoint_dict or 'scheduler_state_dict' not in checkpoint_dict:
            raise RuntimeError(
                "Checkpoint is missing optimizer/scheduler state required for training resume."
            )

        if _architecture_migrated:
            logger.warning(
                "Architecture migration detected (additive params). "
                "Attempting optimizer/scheduler state restore — will fall back gracefully if "
                "param group layout changed (e.g. new speaker_embed group added)."
            )
        # Always attempt optimizer restore. If param groups changed (e.g. new speaker_embed
        # group added), load_state_dict will fail and we fall through to the mismatch warning.
        try:
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        except (ValueError, RuntimeError) as _opt_err:
            _saved_n_groups = len(checkpoint_dict['optimizer_state_dict'].get('param_groups', []))
            _curr_n_groups = len(optimizer.param_groups)
            _saved_n_params = len(checkpoint_dict['optimizer_state_dict'].get('state', {}))
            _curr_n_params = sum(len(g['params']) for g in optimizer.param_groups)
            if _saved_n_groups != _curr_n_groups or _saved_n_params != _curr_n_params:
                logger.warning(
                    f"Optimizer state mismatch ({_saved_n_groups} groups/{_saved_n_params} params "
                    f"→ {_curr_n_groups} groups/{_curr_n_params} params); "
                    "skipping optimizer state restore — Adam moments will be re-initialized. "
                    "This is expected when architecture changes between runs "
                    "(e.g. adding speaker embedding layers)."
                )
            else:
                raise
        try:
            scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
        except (ValueError, RuntimeError, KeyError) as _sched_err:
            logger.warning(
                f"Scheduler state restore failed ({_sched_err}); "
                "scheduler will restart from step 0. "
                "This is expected after architecture changes."
            )

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
        # weights_only=False is required: checkpoints contain TrainingConfig and
        # RussianPhonemeProcessor objects that weights_only=True (PyTorch >=2.6
        # default) cannot deserialize.  These are our own trusted artifacts.
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device,
                                weights_only=False)
    except Exception as e:
        logger.warning(f"Could not load raw checkpoint metadata: {e}")

    # Load scaler state if available
    if trainer.use_mixed_precision and trainer.scaler:
        try:
            if checkpoint is None:
                checkpoint = torch.load(checkpoint_path, map_location=trainer.device,
                                        weights_only=False)
            if 'scaler' in checkpoint:
                trainer.scaler.load_state_dict(checkpoint['scaler'])
                logger.info(f"Loaded {trainer.device_type.upper()} scaler state from checkpoint")
            else:
                logger.info(f"No scaler state found in checkpoint, using default for {trainer.device_type}")
        except Exception as e:
            logger.warning(f"Could not load scaler state: {e}")

    # Load EMA model state if available
    if trainer.use_ema and trainer.ema_model is not None:
        _ema_loaded = False
        try:
            if checkpoint is None:
                checkpoint = torch.load(checkpoint_path, map_location=trainer.device,
                                        weights_only=False)
            if 'ema_model_state_dict' in checkpoint:
                try:
                    trainer.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                    _ema_loaded = True
                except RuntimeError as _ema_err:
                    # Architecture migration: allow known migration key sets in the EMA
                    # state dict, mirroring the permissive logic in _strict_resume.
                    _NEW_VP = 'duration_adaptor.variance_adaptor.'
                    _ALIBI_SUFFIX = '.alibi_slopes'
                    _FFN_NORM_SUFFIX = '.ff.output_norm.weight'
                    _SPEAKER_PREFIXES = ('speaker_embedding.', 'speaker_projection.')
                    _incompat = trainer.ema_model.load_state_dict(
                        checkpoint['ema_model_state_dict'], strict=False
                    )
                    _unexpected_ok = all(
                        k.endswith(_ALIBI_SUFFIX) for k in _incompat.unexpected_keys
                    )
                    _missing_ok = all(
                        k.startswith(_NEW_VP)
                        or k.endswith(_FFN_NORM_SUFFIX)
                        or k.startswith(_SPEAKER_PREFIXES)
                        for k in _incompat.missing_keys
                    )
                    if not (_unexpected_ok and _missing_ok):
                        raise RuntimeError(
                            "EMA state dict has unrecognised incompatible keys, aborting. "
                            f"Unexpected: {_incompat.unexpected_keys} "
                            f"Missing: {_incompat.missing_keys}"
                        ) from _ema_err
                    # Fill missing keys from the freshly-initialised live model
                    _current_sd = trainer.model.state_dict()
                    _ema_sd = trainer.ema_model.state_dict()
                    for k in _incompat.missing_keys:
                        if k in _current_sd:
                            _ema_sd[k] = _current_sd[k].clone()
                    trainer.ema_model.load_state_dict(_ema_sd)
                    logger.warning(
                        f"EMA partial load: {len(_incompat.missing_keys)} key(s) initialised "
                        f"from live model, {len(_incompat.unexpected_keys)} legacy key(s) discarded "
                        "(architecture migration)."
                    )
                    _ema_loaded = True

                if _ema_loaded:
                    if 'ema_updates' in checkpoint:
                        trainer.ema_updates = checkpoint['ema_updates']
                    logger.info(f"Loaded EMA model state from checkpoint (updates: {trainer.ema_updates})")
            else:
                logger.info("No EMA state found in checkpoint, initializing EMA from current model")
                trainer.ema_model.load_state_dict(trainer.model.state_dict())
        except Exception as e:
            logger.warning(f"Could not load EMA state: {e}. Re-initializing EMA from current model weights.")
            try:
                trainer.ema_model.load_state_dict(trainer.model.state_dict())
            except Exception as _reinit_err:
                logger.error(f"EMA re-initialization also failed: {_reinit_err}")

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
        # Prefer the explicit 'best_val_loss' key (saved since the epoch-4 fix);
        # fall back to 'val_loss' for older checkpoints that only stored the
        # current-epoch validation loss.
        if 'best_val_loss' in checkpoint and checkpoint['best_val_loss'] is not None:
            trainer.best_val_loss = float(checkpoint['best_val_loss'])
            logger.info(f"Restored best_val_loss={trainer.best_val_loss:.4f} from checkpoint (explicit key)")
        elif 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
            trainer.best_val_loss = float(checkpoint['val_loss'])
            logger.info(f"Restored best_val_loss={trainer.best_val_loss:.4f} from checkpoint (val_loss fallback)")
        else:
            logger.info("No val_loss in checkpoint; best_val_loss remains inf (first epoch will always save)")

        if 'best_val_epoch' in checkpoint and checkpoint['best_val_epoch'] is not None:
            trainer.best_val_epoch = int(checkpoint['best_val_epoch'])
            logger.info(f"Restored best_val_epoch={trainer.best_val_epoch} from checkpoint")
        else:
            logger.info("No best_val_epoch in checkpoint; best_val_epoch remains -1")

    trainer.dataset.phoneme_processor = phoneme_processor

    logger.info(f"Resumed from epoch {trainer.start_epoch}, best loss {trainer.best_loss:.4f}")

    # Rewrite TensorBoard event files: keep all events up to current_optimizer_step,
    # discard everything after.  This preserves the full history visible in charts
    # while removing any data written during the run that was interrupted/discarded.
    trainer.writer.close()
    keep_up_to = trainer.current_optimizer_step
    trainer.writer = _purge_tb_events_after_step(
        log_dir=trainer.log_dir,
        keep_up_to_step=keep_up_to,
        _SummaryWriter=_SW,
    )
    logger.info(
        f"TensorBoard events rewritten: preserved step <= {keep_up_to}, "
        f"purged everything after (optimizer_step={keep_up_to})"
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
            # Use the first decoder group (index 2 = decoder_decay, or index 1 in
            # smaller setups) as the LR reference for schedule positioning.
            # Must NOT use the stop head group (last group in the 4-group setup)
            # because it has a scaled LR that would produce a wrong resume position.
            ref_group_idx = min(2, n_pg - 1)
            if n_pg >= 4 and ref_group_idx == n_pg - 1:
                ref_group_idx = n_pg - 2  # fall back one group to stay off stop head
            encoder_lr_mult = getattr(trainer, '_encoder_lr_multiplier', 1.0)
            last_lr = trainer.optimizer.state_dict()['param_groups'][ref_group_idx]['lr']

            _div_factor = getattr(trainer, '_onecycle_div_factor',
                max(1.0, float(getattr(config, 'max_lr_multiplier', 5.0)))
                if getattr(trainer, 'use_warmup', False) else 25.0)
            base_lr  = max_lr / _div_factor
            final_lr = max_lr / 10000.0
            peak_step = int(pct_start * onecycle_steps)

            # Detect and log scheduler param changes since the saved checkpoint.
            _saved_sched_cfg = checkpoint.get('scheduler_config') if checkpoint is not None else None
            if _saved_sched_cfg is not None:
                _changed = []
                if _saved_sched_cfg.get('onecycle_steps') != onecycle_steps:
                    _changed.append(f"onecycle_steps: {_saved_sched_cfg.get('onecycle_steps')} → {onecycle_steps}")
                _saved_max_lr = _saved_sched_cfg.get('max_lr')
                if _saved_max_lr is not None and abs(_saved_max_lr - max_lr) > 1e-10:
                    _changed.append(f"max_lr: {_saved_max_lr:.2e} → {max_lr:.2e}")
                if _saved_sched_cfg.get('pct_start') != pct_start:
                    _changed.append(f"pct_start: {_saved_sched_cfg.get('pct_start')} → {pct_start}")
                _saved_div = _saved_sched_cfg.get('div_factor')
                if _saved_div is not None and abs(_saved_div - _div_factor) > 1e-6:
                    _changed.append(f"div_factor: {_saved_div:.4g} → {_div_factor:.4g}")
                if _changed:
                    logger.warning(
                        "Scheduler params changed since checkpoint — step-based positioning "
                        "will re-anchor the new schedule. Changes:\n  " + "\n  ".join(_changed)
                    )
                else:
                    logger.info("Scheduler params unchanged from checkpoint.")

            # Cycle-position-based positioning:
            #
            # Priority 1 — use scheduler_state_dict['last_epoch'] (the exact cycle
            # position within the OneCycleLR at save time) combined with the saved
            # scheduler_config['onecycle_steps'] for proportional mapping.
            #
            # This avoids the previous bug where `global_step - new_warmup_done` was
            # used as a proxy:  global_step is *cumulative across all runs*, but
            # warmup_done used the *new* run's warmup config.  When the new cycle is
            # shorter than global_step - warmup_done the result was capped at
            # onecycle_steps-1 (≈ final_lr ≈ 0), causing subsequent crash-resumes to
            # load near-zero-LR checkpoints whose LR-value fallback then fired and
            # snapped the scheduler back to peak_step (≈ 100% LR), producing the
            # catastrophic mel-loss regressions seen in the training logs.
            #
            # Priority 2 — fall back to global_step minus the *saved* warmup_steps
            # (from scheduler_config) for checkpoints that pre-date the last_epoch fix.
            _saved_global_step = checkpoint.get('global_step') if checkpoint is not None else None

            # Extract saved cycle position from the scheduler state dict.
            _saved_last_epoch: int | None = None
            if checkpoint is not None:
                _saved_sched_state = checkpoint.get('scheduler_state_dict')
                if isinstance(_saved_sched_state, dict):
                    _raw_le = _saved_sched_state.get('last_epoch')
                    if _raw_le is not None:
                        _saved_last_epoch = int(_raw_le)

            # Saved cycle length (for proportional mapping when total_steps changed).
            _old_onecycle_steps: int = (
                int(_saved_sched_cfg['onecycle_steps'])
                if _saved_sched_cfg is not None
                   and _saved_sched_cfg.get('onecycle_steps') is not None
                else onecycle_steps
            )
            # Saved warmup (to correctly strip it when falling back to global_step).
            _old_warmup_steps: int = (
                int(_saved_sched_cfg['warmup_steps'])
                if _saved_sched_cfg is not None
                   and _saved_sched_cfg.get('warmup_steps') is not None
                else (int(getattr(trainer, 'warmup_steps', 0))
                      if getattr(trainer, 'use_warmup', False) else 0)
            )

            def _lr_at_step(step: int) -> float:
                """Compute decoder resume_lr for a given target_step under the new schedule."""
                if step <= peak_step:
                    _pct = step / max(1, peak_step)
                    lr = base_lr + (max_lr - base_lr) * (1.0 - math.cos(math.pi * _pct)) / 2.0
                    return float(max(base_lr, min(max_lr, lr)))
                else:
                    _decay_steps = onecycle_steps - peak_step
                    _pct = (step - peak_step) / max(1, _decay_steps)
                    lr = final_lr + (max_lr - final_lr) * (1.0 + math.cos(math.pi * _pct)) / 2.0
                    return float(max(final_lr, min(max_lr, lr)))

            if _saved_last_epoch is not None:
                # Clamp the saved position to its own cycle bounds before mapping.
                _old_pos = max(0, min(_old_onecycle_steps, _saved_last_epoch))
                if _old_onecycle_steps == onecycle_steps:
                    # Identical cycle length: use position directly.
                    target_step = max(0, min(onecycle_steps - 1, _old_pos))
                    logger.info(
                        f"Direct scheduler resume: last_epoch={_saved_last_epoch} "
                        f"→ target_step={target_step}/{onecycle_steps}"
                    )
                else:
                    # Cycle length changed: map the fractional position proportionally.
                    _fraction = _old_pos / max(1, _old_onecycle_steps)
                    target_step = max(0, min(onecycle_steps - 1,
                                             int(round(_fraction * onecycle_steps))))
                    logger.info(
                        f"Proportional scheduler resume: "
                        f"last_epoch={_old_pos}/{_old_onecycle_steps} "
                        f"({_fraction:.4f}) → target_step={target_step}/{onecycle_steps}"
                    )
                resume_lr = _lr_at_step(target_step)
            elif _saved_global_step is not None:
                # Fallback for checkpoints without a readable scheduler_state_dict.
                # Use global_step minus the *saved* warmup (not the new run's warmup)
                # to avoid the cumulative-step / new-warmup mismatch that caused the
                # previous regression.
                _old_scheduler_step = max(0, int(_saved_global_step) - _old_warmup_steps)
                _fraction = min(1.0, _old_scheduler_step / max(1, _old_onecycle_steps))
                target_step = max(0, min(onecycle_steps - 1,
                                         int(round(_fraction * onecycle_steps))))
                resume_lr = _lr_at_step(target_step)
                logger.info(
                    f"Global-step fallback scheduler resume: "
                    f"global_step={_saved_global_step} warmup={_old_warmup_steps} "
                    f"→ old_step={_old_scheduler_step}/{_old_onecycle_steps} "
                    f"({_fraction:.4f}) → target_step={target_step}/{onecycle_steps}, "
                    f"resume_lr={resume_lr:.2e}"
                )
            else:
                # Last-resort: no scheduler_state_dict, no global_step.
                # Match the saved optimizer LR value to a cycle position.
                # WARNING: this can snap to peak if last_lr happened to equal max_lr
                # (e.g. checkpoint saved during warmup).  Avoid by ensuring checkpoints
                # always include scheduler_state_dict and global_step.
                if last_lr >= max_lr:
                    target_step = peak_step
                    resume_lr   = max_lr
                    logger.warning(
                        f"LR-value fallback (last resort): last_lr ({last_lr:.2e}) >= max_lr "
                        f"({max_lr:.2e}); positioning at peak (step {peak_step}). "
                        "This may restart the LR near 100%% — ensure checkpoints include "
                        "scheduler_state_dict and global_step to avoid this."
                    )
                elif last_lr >= base_lr:
                    ratio = (last_lr - base_lr) / (max_lr - base_lr)
                    t = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * ratio))) / math.pi
                    target_step = max(0, min(peak_step, int(round(t * peak_step))))
                    resume_lr   = last_lr
                    logger.info(
                        f"LR-value fallback (last resort): last_lr={last_lr:.2e} "
                        f"→ warmup phase target_step={target_step}"
                    )
                elif last_lr >= final_lr:
                    decay_steps = onecycle_steps - peak_step
                    ratio = (last_lr - final_lr) / (max_lr - final_lr)
                    t = math.acos(max(-1.0, min(1.0, 2.0 * ratio - 1.0))) / math.pi
                    target_step = max(peak_step, min(onecycle_steps - 1,
                                                     peak_step + int(round(t * decay_steps))))
                    resume_lr   = last_lr
                    logger.info(
                        f"LR-value fallback (last resort): last_lr={last_lr:.2e} "
                        f"→ decay phase target_step={target_step}"
                    )
                else:
                    target_step = onecycle_steps - 1
                    resume_lr   = final_lr
                    logger.info(
                        f"LR-value fallback (last resort): last_lr={last_lr:.2e} below final_lr "
                        f"→ positioning at end of cycle (target_step={target_step})"
                    )

            # Build per-group max_lr mirroring trainer._setup_scheduler().
            # Uses 'group_type' tag on each param group when present;
            # falls back to positional heuristic for legacy (untagged) checkpoints.
            _stop_head_lr_mult = float(
                getattr(getattr(trainer, 'config', None), 'stop_head_lr_multiplier', 0.1)
            )
            _decoder_ffn_lr_mult = float(
                getattr(getattr(trainer, 'config', None), 'decoder_ffn_lr_multiplier', 1.0)
            )
            _decoder_attn_lr_mult = float(
                getattr(getattr(trainer, 'config', None), 'decoder_attn_lr_multiplier', 1.0)
            )
            _var_embed_lr_mult = float(
                getattr(getattr(trainer, 'config', None), 'variance_embedding_lr_multiplier', 0.30)
            )
            def _group_mult(pg, idx, n):
                gt = pg.get('group_type')
                if gt == 'encoder':
                    return encoder_lr_mult
                elif gt == 'decoder_ffn':
                    return _decoder_ffn_lr_mult
                elif gt == 'decoder_attn':
                    return _decoder_attn_lr_mult
                elif gt == 'stop_head':
                    return _stop_head_lr_mult
                elif gt == 'variance_embed':
                    return _var_embed_lr_mult
                elif gt is not None:
                    return 1.0
                # Legacy positional fallback
                if n > 1 and idx == 0:
                    return encoder_lr_mult
                if n >= 4 and idx == n - 1:
                    return _stop_head_lr_mult
                return 1.0

            max_lr_arg = [max_lr * _group_mult(g, i, n_pg)
                          for i, g in enumerate(trainer.optimizer.param_groups)]
            if n_pg == 1:
                max_lr_arg = max_lr_arg[0]

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
                g['lr'] = resume_lr * _group_mult(g, i, n_pg)
            trainer.scheduler._last_lr = [
                resume_lr * _group_mult(g, i, n_pg)
                for i, g in enumerate(trainer.optimizer.param_groups)
            ]

            logger.info(
                f"OneCycleLR reconstructed from current config after resume: "
                f"decoder max_lr={max_lr:.2e}, encoder max_lr={max_lr * encoder_lr_mult:.2e}, "
                f"decoder_attn max_lr={max_lr * _decoder_attn_lr_mult:.2e}, "
                f"decoder_ffn max_lr={max_lr * _decoder_ffn_lr_mult:.2e}, "
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
