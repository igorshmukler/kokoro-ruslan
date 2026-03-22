#!/usr/bin/env python3
"""
Training logic for Kokoro Language Model with Enhanced Profiling, Mixed Precision, and Adaptive Memory Management
Extended to support mixed precision training on both CUDA and MPS devices with intelligent memory cleanup
"""

import os
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import torch.profiler
import datetime
import gc
import copy
import faulthandler
from pathlib import Path

from typing import Tuple, Dict, Any, Optional, Union

from torch.utils.tensorboard import SummaryWriter

from kokoro.training.config import TrainingConfig
from kokoro.utils.device_type import DeviceType
from kokoro.data.dataset import RuslanDataset, collate_fn, LengthBasedBatchSampler, DynamicFrameBatchSampler
from kokoro.model.model import KokoroModel
from kokoro.training.checkpoint_manager import (
    save_phoneme_processor, load_checkpoint, find_latest_checkpoint,
    save_checkpoint, save_final_model, build_model_metadata,
    resume_from_checkpoint,
)
from kokoro.utils.interbatch_profiler import InterbatchProfiler
from kokoro.training.mps_grad_scaler import MPSGradScaler
from kokoro.training.losses import calculate_training_losses
from kokoro.training.runtime_policies import RuntimeStepPolicy, MemoryOOMPolicy

from kokoro.utils.adaptive_memory_manager import AdaptiveMemoryManager
from kokoro.utils.ema import recommended_ema_decay
from kokoro.utils.device_utils import check_mps_mixed_precision_support

import math

# This tells PyTorch it's safe to load our custom config class
torch.serialization.add_safe_globals([TrainingConfig])

logger = logging.getLogger(__name__)


@dataclass
class BatchOnDevice:
    """Device-ready batch tensors for train/val/profiling paths."""
    mel_specs: torch.Tensor
    phoneme_indices: torch.Tensor
    phoneme_durations: torch.Tensor
    stop_token_targets: torch.Tensor
    mel_lengths: torch.Tensor
    phoneme_lengths: torch.Tensor
    pitches: Optional[torch.Tensor] = None
    energies: Optional[torch.Tensor] = None
    stress_indices: Optional[torch.Tensor] = None


@dataclass
class StepResult:
    """Outputs and losses from one forward/backward execution step."""
    total_loss: torch.Tensor
    loss_mel: torch.Tensor
    loss_duration: torch.Tensor
    loss_stop_token: torch.Tensor
    loss_pitch: Optional[torch.Tensor]
    loss_energy: Optional[torch.Tensor]
    predicted_mel: torch.Tensor
    predicted_log_durations: torch.Tensor
    predicted_pitch: Optional[torch.Tensor]
    predicted_energy: Optional[torch.Tensor]


@dataclass
class EpochMetrics:
    """Averaged epoch-level losses returned by train/validation loops."""
    total_loss: float
    mel_loss: float
    dur_loss: float
    stop_loss: float
    pitch_loss: float = 0.0
    energy_loss: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.total_loss, self.mel_loss, self.dur_loss, self.stop_loss)

    def __iter__(self):
        return iter(self.as_tuple())


class KokoroTrainer:
    """Main trainer class for the model with adaptive memory management"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create the tensorboard log directory
        log_dir = os.path.join(config.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir  # stored for writer re-creation on resume
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"Tensorboard log directory created at: {log_dir}")

        # Custom layout: group train and val scalars onto the same chart so
        # convergence comparisons are visible without switching between panels.
        self.writer.add_custom_scalars({
            "Epoch Losses": {
                "Total Loss (train vs val)":     ["Multiline", ["loss/train_total_epoch",    "loss/val_total_epoch"]],
                "Mel Loss (train vs val)":        ["Multiline", ["loss/train_mel_epoch",      "loss/val_mel_epoch"]],
                "Stop Loss (train vs val)":       ["Multiline", ["loss/train_stop_epoch",     "loss/val_stop_epoch"]],
                "Duration Loss (train vs val)":   ["Multiline", ["loss/train_duration_epoch", "loss/val_duration_epoch"]],
            },
            "Spectral Metrics": {
                "Spectral Convergence (train vs val)": ["Multiline", ["metrics/train_spectral_convergence", "metrics/val_spectral_convergence"]],
            },
            "Learning Rate": {
                "LR (encoder vs decoder vs stop vs ffn)": ["Multiline", ["stats/lr_encoder", "stats/lr_decoder", "stats/lr_decoder_ffn", "stats/lr_stop_head"]],
            },
        })

        # Attempt to load checkpoint metadata early so pitch/energy bounds are
        # available before the dataset is constructed.
        try:
            resume_checkpoint = getattr(self.config, 'resume_checkpoint', None)
            if resume_checkpoint:
                checkpoint_path = None
                if isinstance(resume_checkpoint, str) and resume_checkpoint.lower() == 'auto':
                    checkpoint_path = find_latest_checkpoint(self.config.output_dir)
                else:
                    checkpoint_path = str(resume_checkpoint)

                if checkpoint_path and os.path.exists(checkpoint_path):
                    try:
                        # Use weights_only=True to avoid loading large tensors
                        meta = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                        if isinstance(meta, dict):
                            mm = meta.get('model_metadata')
                            if isinstance(mm, dict):
                                arch = mm.get('architecture', {})
                                for name in ('pitch_min', 'pitch_max', 'energy_min', 'energy_max'):
                                    if name in arch:
                                        try:
                                            setattr(self.config, name, arch[name])
                                        except Exception:
                                            pass
                    except Exception as e:
                        logger.warning(f"Could not pre-load checkpoint metadata: {e}")
        except Exception:
            logger.debug("Early checkpoint metadata load skipped or failed")

        # Ensure fatal signals dump Python stacks for crash triage
        try:
            if not faulthandler.is_enabled():
                faulthandler.enable(all_threads=True)
                logger.info("faulthandler enabled (all_threads=True)")
        except Exception as e:
            logger.warning(f"Could not enable faulthandler: {e}")

        # Initialize adaptive memory manager
        self.memory_manager = AdaptiveMemoryManager(self.device, config)

        self._setup_mixed_precision()
        self._setup_datasets_and_dataloaders()
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self.runtime_step_policy = RuntimeStepPolicy(logger=logger)
        self.memory_oom_policy = MemoryOOMPolicy(logger=logger)

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_val_epoch = -1
        self.epochs_without_improvement = 0
        self.validation_losses = []

        self._setup_ema()

        # Profiler setup
        self.profiler = None
        self.profiling_stats = {}
        self.memory_snapshots = []
        self.profiler_log_dir = os.path.join(config.output_dir, "profiler_logs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.profiler_log_dir, exist_ok=True)
        self.interbatch_profiler = InterbatchProfiler(config)

        # Adaptive memory management
        self.enable_adaptive_memory = getattr(config, 'enable_adaptive_memory', True)
        self.memory_report_interval = getattr(config, 'memory_report_interval', 50)

        self._setup_grad_explosion_tracker()
        self._setup_weight_norm_constraints()

        self.optimizer_steps_completed = 0

    # ------------------------------------------------------------------
    # Private setup helpers (called exclusively from __init__)
    # ------------------------------------------------------------------

    def _setup_mixed_precision(self):
        """Initialize mixed precision training components (scaler, device_type)."""
        config = self.config
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', True)
        self.mixed_precision_dtype = getattr(config, 'mixed_precision_dtype', torch.float16)

        if self.use_mixed_precision:
            if self.device.type == DeviceType.CUDA.value:
                self.scaler = torch.amp.GradScaler('cuda',
                    init_scale=getattr(config, 'amp_init_scale', 65536.0),
                    growth_factor=getattr(config, 'amp_growth_factor', 2.0),
                    backoff_factor=getattr(config, 'amp_backoff_factor', 0.5),
                    growth_interval=getattr(config, 'amp_growth_interval', 2000)
                )
                self.device_type = 'cuda'
                logger.info("Mixed precision training enabled with CUDA GradScaler")
            elif self.device.type == DeviceType.MPS.value:
                if check_mps_mixed_precision_support():
                    self.scaler = MPSGradScaler(
                        init_scale=getattr(config, 'amp_init_scale', 65536.0),
                        growth_factor=getattr(config, 'amp_growth_factor', 2.0),
                        backoff_factor=getattr(config, 'amp_backoff_factor', 0.5),
                        growth_interval=getattr(config, 'amp_growth_interval', 2000)
                    )
                    self.device_type = DeviceType.MPS.value
                    logger.info("Mixed precision training enabled with MPS custom scaler")
                else:
                    logger.warning("MPS mixed precision not supported, disabling mixed precision")
                    self.use_mixed_precision = False
                    self.scaler = None
                    self.device_type = DeviceType.MPS.value
            else:
                logger.info(f"Mixed precision not supported on {self.device.type}, disabling")
                self.use_mixed_precision = False
                self.scaler = None
                self.device_type = self.device.type
        else:
            self.scaler = None
            self.device_type = self.device.type
            logger.info("Mixed precision training disabled by configuration")

        self.mixed_precision_stats = {
            'scale_updates': 0,
            'scale_decreases': 0,
            'overflow_count': 0,
            'successful_steps': 0,
            'skipped_steps': 0
        }
        # Flag so the MPS mixed-precision deprecation warning is emitted only once
        self._mps_amp_warned = False

    def _setup_datasets_and_dataloaders(self):
        """Build train/validation datasets and their corresponding DataLoaders."""
        config = self.config

        # Precompute features if requested
        if getattr(config, 'precompute_features', False):
            logger.info("Pre-computing features for all samples...")
            from kokoro.cli.precompute_features import precompute_features
            if not config.feature_cache_dir:
                config.feature_cache_dir = str(Path(config.data_dir) / ".feature_cache")
            precompute_features(config.data_dir, config, force_recompute=False)
            logger.info("Feature pre-computation complete")

        # Initialize datasets with train/validation split
        full_dataset = RuslanDataset(config.data_dir, config)

        val_split = getattr(config, 'validation_split', 0.1)
        if val_split > 0:
            import random
            dataset_size = len(full_dataset.samples)
            indices = list(range(dataset_size))
            random.seed(42)
            random.shuffle(indices)
            split_idx = int(dataset_size * (1 - val_split))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            logger.info(f"Dataset split: {len(train_indices)} training, {len(val_indices)} validation samples")
            self.dataset = RuslanDataset(config.data_dir, config, indices=train_indices)
            self.val_dataset = RuslanDataset(config.data_dir, config, indices=val_indices)
        else:
            logger.info("No validation split - using all data for training")
            self.dataset = full_dataset
            self.val_dataset = None

        use_dynamic = getattr(config, 'use_dynamic_batching', True)
        if use_dynamic:
            logger.info("Using dynamic frame-based batching")
            self.batch_sampler = DynamicFrameBatchSampler(
                dataset=self.dataset,
                max_frames=config.max_frames_per_batch,
                min_batch_size=config.min_batch_size,
                max_batch_size=config.max_batch_size,
                drop_last=True,
                shuffle=True
            )
        else:
            logger.info("Using fixed batch size")
            self.batch_sampler = LengthBasedBatchSampler(
                dataset=self.dataset,
                batch_size=config.batch_size,
                drop_last=True,
                shuffle=True
            )

        num_workers = max(0, int(getattr(config, 'num_workers', 0)))
        # Store shared DataLoader kwargs so _build_dataloader can access them
        self._num_workers = num_workers
        self._prefetch_factor = 2 if num_workers > 0 else None
        self._persistent_workers = num_workers > 0
        self._pin_memory = config.pin_memory and self.device.type == DeviceType.CUDA.value

        self.dataloader = self._build_dataloader(self.dataset, self.batch_sampler)

        if self.val_dataset is not None:
            if use_dynamic:
                val_batch_sampler = DynamicFrameBatchSampler(
                    dataset=self.val_dataset,
                    max_frames=config.max_frames_per_batch,
                    min_batch_size=config.min_batch_size,
                    max_batch_size=config.max_batch_size,
                    drop_last=False,
                    shuffle=False
                )
            else:
                val_batch_sampler = LengthBasedBatchSampler(
                    dataset=self.val_dataset,
                    batch_size=config.batch_size,
                    drop_last=False,
                    shuffle=False
                )
            self.val_dataloader = self._build_dataloader(self.val_dataset, val_batch_sampler)
        else:
            self.val_dataloader = None

    def _setup_model(self):
        """Instantiate KokoroModel, move to device, optionally compile, and create loss criteria."""
        config = self.config
        vocab_size = self.dataset.phoneme_processor.get_vocab_size()
        self.model = KokoroModel(
            vocab_size,
            config.n_mels,
            config.hidden_dim,
            n_encoder_layers=getattr(config, 'n_encoder_layers', 6),
            n_heads=getattr(config, 'n_heads', 8),
            encoder_ff_dim=getattr(config, 'encoder_ff_dim', 3072),
            encoder_dropout=getattr(config, 'encoder_dropout', 0.1),
            n_decoder_layers=getattr(config, 'n_decoder_layers', 6),
            decoder_ff_dim=getattr(config, 'decoder_ff_dim', 3072),
            max_decoder_seq_len=getattr(config, 'max_decoder_seq_len', 4000),
            use_variance_predictor=getattr(config, 'use_variance_predictor', True),
            variance_filter_size=getattr(config, 'variance_filter_size', 256),
            variance_kernel_size=getattr(config, 'variance_kernel_size', 3),
            variance_dropout=getattr(config, 'variance_dropout', 0.1),
            n_variance_bins=getattr(config, 'n_variance_bins', 256),
            pitch_min=getattr(config, 'pitch_min', 0.0),
            pitch_max=getattr(config, 'pitch_max', 1.0),
            energy_min=getattr(config, 'energy_min', 0.0),
            energy_max=getattr(config, 'energy_max', 1.0),
            use_stochastic_depth=getattr(config, 'use_stochastic_depth', True),
            stochastic_depth_rate=getattr(config, 'stochastic_depth_rate', 0.1)
        )
        self.model.to(self.device)

        # Apply torch.compile for CUDA only (MPS has limited dtype support)
        use_compile = getattr(config, 'use_torch_compile', False)
        if use_compile and torch.__version__ >= '2.0':
            if self.device_type == 'cuda':
                try:
                    compile_mode = getattr(config, 'torch_compile_mode', 'reduce-overhead')
                    compile_dynamic = getattr(config, 'torch_compile_dynamic', True)
                    self.model = torch.compile(
                        self.model,
                        mode=compile_mode,
                        fullgraph=False,
                        dynamic=compile_dynamic
                    )
                    logger.info(f"Model compiled with torch.compile (mode='{compile_mode}', dynamic={compile_dynamic})")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}. Training will proceed without compilation.")
            else:
                logger.info(f"torch.compile disabled for {self.device_type} (limited dtype support)")
        elif use_compile:
            logger.warning(f"torch.compile requested but PyTorch version is {torch.__version__} (requires 2.0+)")

        model_info = self.model.get_model_info()
        logger.info(f"Model initialized with {model_info['total_parameters']:,} parameters ({model_info['model_size_mb']:.1f} MB)")

        # Loss criteria
        self.criterion_mel = nn.L1Loss(reduction='none')
        # Huber loss (SmoothL1) for duration: transitions to L1 for large errors,
        # giving a constant-magnitude gradient that does NOT shrink as predictions
        # improve — unlike MSE whose gradient is 2*(pred-target) → 0 near convergence.
        # delta=1.0 matches a 1 log-frame error boundary, appropriate for log-duration targets.
        _duration_huber_delta = float(getattr(config, 'duration_huber_delta', 1.0))
        self.criterion_duration = nn.HuberLoss(reduction='none', delta=_duration_huber_delta)
        logger.info(f"Duration loss: HuberLoss(delta={_duration_huber_delta}) "
                    f"(replaces MSE to prevent gradient shrinkage near convergence)")
        # pos_weight corrects the severe class imbalance in stop token targets:
        # only the final frame of each sequence is positive (1.0); all others
        # are negative (0.0).  Without this the model learns to always predict 0
        # and achieves near-zero BCE loss while never detecting end-of-utterance.
        _stop_pos_w = torch.tensor(
            [getattr(config, 'stop_token_pos_weight', 30.0)],
            device=self.device
        )
        self.criterion_stop_token = nn.BCEWithLogitsLoss(
            reduction='none', pos_weight=_stop_pos_w
        )
        if getattr(config, 'use_variance_predictor', True):
            # Huber loss for pitch/energy for the same reason: prevents gradient
            # vanishing as predictions converge; delta is tunable via config.
            _pitch_huber_delta  = float(getattr(config, 'pitch_huber_delta',  0.05))
            _energy_huber_delta = float(getattr(config, 'energy_huber_delta', 0.05))
            self.criterion_pitch  = nn.HuberLoss(reduction='none', delta=_pitch_huber_delta)
            self.criterion_energy = nn.HuberLoss(reduction='none', delta=_energy_huber_delta)
            logger.info(
                f"Variance predictor losses initialized: "
                f"HuberLoss pitch(delta={_pitch_huber_delta}), "
                f"HuberLoss energy(delta={_energy_huber_delta})"
            )
        else:
            self.criterion_pitch = None
            self.criterion_energy = None

    def _setup_optimizer(self):
        """Create AdamW optimizer with per-group LR for encoder vs decoder.

        The encoder (embedding + positional encoding + encoder transformer layers)
        historically receives near-zero gradients due to the long backprop path
        through the decoder.  Assigning it a separate param group with a higher
        base LR (encoder_lr_multiplier × learning_rate) gives it proportionally
        more gradient signal without destabilising the decoder.
        """
        config = self.config
        default_use_fused = self.device_type == 'cuda' and torch.__version__ >= '2.0'
        forced_use_fused = getattr(config, 'use_fused_adamw', None)

        if forced_use_fused is None:
            use_fused = default_use_fused
        else:
            use_fused = bool(forced_use_fused)

        if self.device_type == 'mps' and forced_use_fused is None and getattr(config, 'try_fused_adamw_on_mps', False):
            use_fused = True

        fused_source = (
            'mps-opt-in' if (self.device_type == 'mps' and getattr(config, 'try_fused_adamw_on_mps', False))
            else 'auto' if forced_use_fused is None
            else 'forced'
        )
        logger.info(f"AdamW fused setting: effective={use_fused} (source={fused_source}, device={self.device_type})")

        base_lr = config.learning_rate
        weight_decay = getattr(config, 'weight_decay', 0.01)
        adam_eps = getattr(config, 'adam_eps', 1e-8)
        adam_betas = getattr(config, 'adam_betas', (0.9, 0.999))
        encoder_lr_mult = float(getattr(config, 'encoder_lr_multiplier', 3.0))
        self._encoder_lr_multiplier = encoder_lr_mult

        # Partition parameters into four groups:
        #   1. encoder_params    — text/stress embeddings + positional encoding +
        #      transformer encoder layers, higher LR multiplier, weight_decay=0
        #      (L2 on embedding rows counteracts learned representations; encoder
        #      transformer layers benefit from reduced regularisation given the
        #      elevated LR)
        #   2. decoder_no_decay  — decoder biases and all norm affine params (scale +
        #      shift).  weight_decay=0: penalising these toward zero fights the
        #      learned statistics and hurts convergence.
        #   3. decoder_decay     — all other decoder/variance-predictor weights,
        #      regularised with the configured weight_decay.
        #   4. stop_head         — stop_token_predictor.weight + .bias at a reduced
        #      LR (stop_head_lr_multiplier × base_lr, default 0.1×).
        #      Rationale: the stop head is a single Linear(hidden_dim→1) that
        #      classifies a heavily imbalanced binary target (~137:1 neg/pos ratio).
        #      Under the ascending phase of OneCycleLR its gradient amplified by
        #      pos_weight spikes disproportionately and destabilises the run.
        #      Isolating it to a lower LR caps the effective step size without
        #      requiring changes to global gradient clipping.  Note: gradient flow
        #      from the stop loss back into decoder_outputs is already detached
        #      (model.py _project_decoder_outputs), so this group governs only
        #      the stop head's own parameter updates.
        encoder_prefixes = (
            'text_embedding.',
            'stress_embedding.',
            'encoder_positional_encoding.',
            'positional_encoding.',   # legacy attribute name
            'transformer_encoder_layers.',
        )
        # stop_token_predictor parameters are carved out before the
        # decoder_no_decay / decoder_decay split so they don't appear in both.
        _stop_head_names = {'stop_token_predictor.weight', 'stop_token_predictor.bias'}
        # Parameter names matching these patterns are excluded from weight decay.
        # ".bias" catches every bias vector; the norm patterns catch LayerNorm /
        # RMSNorm / BatchNorm affine weight and bias regardless of nesting depth.
        # "duration_adaptor." covers all variance/pitch/energy/duration predictor
        # weights: these are small regression heads (~L2 norm 1.4) with weak
        # gradient signals.  At weight_decay=0.01 the L2 penalty overwhelms the
        # gradient and drives the weights toward zero, leaving only the bias
        # (which learns the training-set mean F0/energy) — a degenerate solution
        # that causes F0 RMSE to plateau even as mel loss continues to improve.
        _no_decay_suffixes = ('.bias',)
        _no_decay_substrings = (
            'norm.weight', 'norm.bias',
            'layer_norm.weight', 'layer_norm.bias',
            'duration_adaptor.',
        )
        seen_ids: set = set()
        encoder_params = []
        stop_head_params: list = []
        decoder_named: list = []  # (name, param) for further splitting
        for name, param in self.model.named_parameters():
            if id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            if any(name.startswith(prefix) for prefix in encoder_prefixes):
                encoder_params.append(param)
            elif name in _stop_head_names:
                stop_head_params.append(param)
            else:
                decoder_named.append((name, param))

        # Split decoder params into no_decay vs decay, and isolate FFN params
        decoder_no_decay: list = []
        decoder_decay_tuples: list = []  # (name, param) for decay candidates
        for _name, _param in decoder_named:
            if (any(_name.endswith(s) for s in _no_decay_suffixes)
                    or any(s in _name for s in _no_decay_substrings)):
                decoder_no_decay.append(_param)
            else:
                decoder_decay_tuples.append((_name, _param))
        # From decay candidates, carve out FFN params (decoder.layers.*.ff.*)
        decoder_decay_ffn: list = []
        decoder_decay_other: list = []
        for _name, _param in decoder_decay_tuples:
            if ".ff." in _name or ".ff" in _name:
                decoder_decay_ffn.append(_param)
            else:
                decoder_decay_other.append(_param)
        decoder_params = [p for _, p in decoder_named]  # kept for logging only

        encoder_lr = base_lr * encoder_lr_mult
        stop_head_lr_mult = float(getattr(config, 'stop_head_lr_multiplier', 0.1))
        stop_head_lr = base_lr * stop_head_lr_mult
        decoder_ffn_lr_mult = float(getattr(config, 'decoder_ffn_lr_multiplier', 1.0))

        # Build param groups, tagging each with a group_type for scheduler mapping
        param_groups = []
        param_groups.append({'params': encoder_params, 'lr': encoder_lr, 'weight_decay': 0.0,
                             'eps': adam_eps, 'betas': adam_betas, 'group_type': 'encoder'})

        # Decoder no-decay (biases / norms)
        if decoder_no_decay:
            param_groups.append({'params': decoder_no_decay, 'lr': base_lr, 'weight_decay': 0.0,
                                 'eps': adam_eps, 'betas': adam_betas, 'group_type': 'decoder_other'})

        # Decoder decay: separate FFN vs other so FFN can use reduced LR
        if decoder_decay_other:
            param_groups.append({'params': decoder_decay_other, 'lr': base_lr, 'weight_decay': weight_decay,
                                 'eps': adam_eps, 'betas': adam_betas, 'group_type': 'decoder_other'})
        if decoder_decay_ffn:
            param_groups.append({'params': decoder_decay_ffn, 'lr': base_lr * decoder_ffn_lr_mult,
                                 'weight_decay': weight_decay, 'eps': adam_eps, 'betas': adam_betas,
                                 'group_type': 'decoder_ffn'})

        # Stop head group (always present as its own group when identified)
        if stop_head_params:
            param_groups.append({'params': stop_head_params, 'lr': stop_head_lr, 'weight_decay': 0.0,
                                 'eps': adam_eps, 'betas': adam_betas, 'group_type': 'stop_head'})
        if not encoder_params:
            # Fallback: single group (no identifiable encoder params)
            param_groups = [{'params': list(self.model.parameters()), 'lr': base_lr,
                             'weight_decay': weight_decay, 'eps': adam_eps, 'betas': adam_betas}]
            self._encoder_lr_multiplier = 1.0
            logger.warning("No encoder params identified for separate LR group — using single param group")
        else:
            # Count params per logical group for logging
            n_encoder = len(encoder_params)
            n_decoder_no_decay = len(decoder_no_decay)
            n_decoder_other = len(decoder_decay_other)
            n_decoder_ffn = len(decoder_decay_ffn)
            n_stop = len(stop_head_params)
            logger.info(
                f"Optimizer param groups: encoder={n_encoder} params (lr={encoder_lr:.2e}, wd=0.0), "
                f"decoder_no_decay={n_decoder_no_decay} params (lr={base_lr:.2e}, wd=0.0), "
                f"decoder_other_decay={n_decoder_other} params (lr={base_lr:.2e}, wd={weight_decay}), "
                f"decoder_ffn_decay={n_decoder_ffn} params (lr={base_lr*decoder_ffn_lr_mult:.2e}, wd={weight_decay}), "
                f"stop_head={n_stop} params (lr={stop_head_lr:.2e}, wd=0.0)"
            )

        try:
            self.optimizer = torch.optim.AdamW(param_groups, fused=use_fused)
            if use_fused:
                logger.info(f"Using fused AdamW optimizer on {self.device_type.upper()}")
            else:
                logger.info("Using standard AdamW optimizer")
        except (TypeError, ValueError, RuntimeError) as e:
            if use_fused:
                logger.warning(f"Fused AdamW not available on {self.device_type}: {e}. Falling back to standard AdamW.")
                self.optimizer = torch.optim.AdamW(param_groups, fused=False)
                logger.info("Using standard AdamW optimizer (fallback)")
            else:
                raise

    def _setup_scheduler(self):
        """Configure OneCycleLR (with optional linear warmup) or CosineAnnealingWarmRestarts."""
        config = self.config
        use_onecycle = getattr(config, 'use_onecycle_lr', True)
        if use_onecycle:
            steps_per_epoch = len(self.dataloader)
            gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
            optimizer_steps_per_epoch = (steps_per_epoch + gradient_accumulation_steps - 1) // gradient_accumulation_steps
            total_steps = config.num_epochs * optimizer_steps_per_epoch

            max_lr = config.learning_rate * getattr(config, 'max_lr_multiplier', 5.0)
            pct_start = getattr(config, 'pct_start', 0.3)

            self.use_warmup = getattr(config, 'use_warmup', True)
            self.warmup_steps = getattr(config, 'warmup_steps', 500)
            self.warmup_start_lr = config.learning_rate * getattr(config, 'warmup_start_lr_ratio', 0.01)
            # Clamp warmup target to max_lr: when max_lr_multiplier < 1 (max_lr < learning_rate),
            # ramping the warmup above max_lr creates a LR drop at the warmup→OneCycleLR boundary.
            self.warmup_target_lr = min(config.learning_rate, max_lr)
            self.current_optimizer_step = 0

            if self.use_warmup:
                # OneCycleLR starts after warmup, so adjust total_steps.
                self.warmup_steps, onecycle_steps = KokoroTrainer._apply_warmup_guard(
                    self.warmup_steps, total_steps
                )
                logger.info(f"Linear warmup enabled: {self.warmup_steps} steps ({self.warmup_start_lr:.2e} → {self.warmup_target_lr:.2e})")
            else:
                onecycle_steps = total_steps

            # When manual warmup is enabled, OneCycleLR must start exactly at
            # warmup_target_lr (= min(learning_rate, max_lr)).  OneCycleLR's initial LR is
            # max_lr / div_factor, so div_factor = max_lr / warmup_target_lr = max_lr_multiplier
            # (when max_lr_multiplier >= 1) ensures a seamless handoff.
            # Guard: when max_lr_multiplier < 1 (max_lr < learning_rate), the raw formula
            # gives div_factor < 1 → base_lr > max_lr, inverting the ascending cosine phase.
            # Clamping div_factor to ≥ 1.0 ensures base_lr ≤ max_lr at all times.
            # When warmup is disabled, use the classic div_factor=25.0 (ramp from max_lr/25).
            _max_lr_multiplier = getattr(config, 'max_lr_multiplier', 5.0)
            onecycle_div_factor = max(1.0, float(_max_lr_multiplier)) if self.use_warmup else 25.0

            # Build per-group max_lr:
            #   group 0 (encoder):          max_lr * encoder_lr_multiplier
            #   groups 1..n-2 (decoder):    max_lr
            #   group n-1 (stop head):      max_lr * stop_head_lr_multiplier
            # The stop head is always the last param group when present.
            _enc_mult = getattr(self, '_encoder_lr_multiplier', 1.0)
            _stop_head_lr_mult = float(getattr(config, 'stop_head_lr_multiplier', 0.1))
            decoder_ffn_mult = float(getattr(config, 'decoder_ffn_lr_multiplier', 1.0))

            # Construct per-param-group max_lr list by inspecting the custom 'group_type'
            max_lr_arg = []
            for pg in self.optimizer.param_groups:
                gt = pg.get('group_type', 'decoder_other')
                if gt == 'encoder':
                    max_lr_arg.append(max_lr * _enc_mult)
                elif gt == 'decoder_ffn':
                    max_lr_arg.append(max_lr * decoder_ffn_mult)
                elif gt == 'stop_head':
                    max_lr_arg.append(max_lr * _stop_head_lr_mult)
                else:
                    # decoder_other / decoder_no_decay etc.
                    max_lr_arg.append(max_lr)

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr_arg,
                total_steps=onecycle_steps,
                pct_start=pct_start,
                anneal_strategy='cos',
                cycle_momentum=False,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=onecycle_div_factor,
                final_div_factor=10000.0,
                last_epoch=-1
            )
            self.scheduler_per_batch = True
            # Store parameters needed to reconstruct the scheduler after checkpoint resume.
            # _onecycle_max_lr is always the DECODER (base) max_lr; encoder uses
            # _onecycle_max_lr * _encoder_lr_multiplier.
            self._onecycle_steps = onecycle_steps
            self._onecycle_max_lr = max_lr          # decoder reference max_lr
            self._onecycle_pct_start = pct_start
            self._onecycle_div_factor = onecycle_div_factor
            logger.info(
                f"OneCycleLR scheduler initialized: decoder max_lr={max_lr:.2e}, "
                f"encoder max_lr={max_lr * _enc_mult:.2e}, total_steps={onecycle_steps} "
                f"(steps_per_epoch={optimizer_steps_per_epoch}, gradient_accumulation={gradient_accumulation_steps})"
            )
        else:
            self.use_warmup = False
            self.current_optimizer_step = 0
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.lr_T_0,
                T_mult=self.config.lr_T_mult,
                eta_min=self.config.lr_eta_min
            )
            self.scheduler_per_batch = False
            logger.info("CosineAnnealingWarmRestarts scheduler initialized (legacy mode)")

    def _setup_ema(self):
        """Initialize Exponential Moving Average (EMA) of model weights."""
        config = self.config
        self.use_ema = getattr(config, 'use_ema', True)
        cfg_ema = getattr(config, 'ema_decay', None)
        self.ema_update_every = getattr(config, 'ema_update_every', 1)

        if cfg_ema is None and self.use_ema:
            try:
                n_train = len(self.dataset)
            except Exception:
                n_train = 0
            effective_batch = int(getattr(config, 'batch_size', 1) * getattr(config, 'gradient_accumulation_steps', 1))
            k = float(getattr(config, 'ema_half_life_epochs', 1.0))
            self.ema_decay = recommended_ema_decay(n_train=n_train, batch_size=effective_batch, k=k)
            try:
                config.ema_decay = self.ema_decay
            except Exception:
                pass
            logger.info(f"Computed EMA decay: {self.ema_decay:.6f} (n_train={n_train}, eff_batch={effective_batch}, k={k})")
        else:
            self.ema_decay = float(cfg_ema) if cfg_ema is not None else 0.9999
            logger.info(f"EMA decay: {self.ema_decay:.6f} (source={'config' if cfg_ema is not None else 'default'})")

        self.ema_updates = 0

        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model).to(self.device)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            logger.info(f"EMA initialized: decay={self.ema_decay:.6f}, update_every={self.ema_update_every}")
        else:
            self.ema_model = None
            logger.info("EMA disabled")

    def _setup_weight_norm_constraints(self) -> None:
        """Cache weight references for all decoder FF layers used by post-step norm clamping."""
        named_modules = dict(self.model.named_modules())
        n_decoder_layers = getattr(self.config, 'n_decoder_layers', 6)
        self._dec_ff_weights: list = []
        for i in range(n_decoder_layers):
            for linear_name in ('linear1', 'linear2'):
                m = named_modules.get(f'decoder.layers.{i}.ff.{linear_name}')
                if m is not None:
                    self._dec_ff_weights.append(m.weight)
                else:
                    logger.warning(
                        f"dec_ffn_max_weight_norm: module 'decoder.layers.{i}.ff.{linear_name}' "
                        "not found in model — weight-norm clamp will be skipped for this layer."
                    )
        # Legacy single-layer reference kept for backward compat with any external code.
        self._dec_ff0_linear1_weight = self._dec_ff_weights[0] if self._dec_ff_weights else None
        logger.info(
            f"Weight-norm constraints registered for {len(self._dec_ff_weights)} "
            "decoder FF weight matrices."
        )

    @torch.no_grad()
    def _apply_weight_norm_constraints(self) -> None:
        """Project all decoder FF weights back onto their max-norm ball.

        Called after every successful optimizer step.  The L2 norm of each
        weight matrix is compared against *dec_ffn_max_weight_norm* and rescaled
        when it exceeds the ceiling.  All decoder FF linear1 and linear2 weights
        across all layers share the same cap.

        Config key: dec_ffn_max_weight_norm (default 60.0).
        Falls back to the legacy dec_ff0_linear1_max_weight_norm key if present.
        Set to 0 or negative to disable.
        """
        max_norm: float = getattr(
            self.config, 'dec_ffn_max_weight_norm',
            getattr(self.config, 'dec_ff0_linear1_max_weight_norm', 60.0)
        )
        if max_norm <= 0.0 or not self._dec_ff_weights:
            return
        for w in self._dec_ff_weights:
            current_norm = w.norm(2).item()
            if current_norm > max_norm:
                w.mul_(max_norm / current_norm)

    def _setup_grad_explosion_tracker(self):
        """Initialize gradient explosion detection and localized spike clipping state."""
        config = self.config
        self.grad_explosion_norm_ema = None
        self.grad_explosion_ema_alpha = getattr(config, 'grad_explosion_ema_alpha', 0.95)
        self.grad_explosion_abs_floor = getattr(config, 'grad_explosion_abs_floor', 1000.0)
        self.grad_explosion_multiplier = getattr(config, 'grad_explosion_multiplier', 3.0)
        self.grad_explosion_warmup_steps = getattr(config, 'grad_explosion_warmup_steps', 250)
        self.grad_explosion_warmup_floor = getattr(config, 'grad_explosion_warmup_floor', 5000.0)
        self.grad_explosion_min_ema_steps = getattr(config, 'grad_explosion_min_ema_steps', 50)
        self.grad_explosion_ema_steps = 0
        self.grad_explosion_streak = 0

        # Localized gradient spike clipping
        self.projection_spike_clip_norm = getattr(config, 'projection_spike_clip_norm', 50.0)
        self.attention_spike_clip_norm = getattr(config, 'attention_spike_clip_norm', 20.0)
        self.ffn_spike_clip_norm = getattr(config, 'ffn_spike_clip_norm', 15.0)
        # Stop-head isolation clip: applied only to stop_token_predictor.weight/.bias
        # before the global clip, so stop-gradient spikes cannot consume the mel budget.
        self.stop_head_spike_clip_norm = getattr(config, 'stop_head_spike_clip_norm', 1.0)

    def get_autocast_context(self):
        """Get the appropriate autocast context for the device"""
        from contextlib import nullcontext

        if not self.use_mixed_precision:
            return nullcontext()

        if self.device_type == DeviceType.CUDA.value:
            return torch.amp.autocast('cuda', dtype=self.mixed_precision_dtype)
        elif self.device_type == DeviceType.MPS.value:
            # CRITICAL: MPS has a bug with fp16 matrix multiplication that causes:
            # "Destination NDArray and Accumulator NDArray cannot have different datatype"
            # This affects ALiBi, attention, and all linear layers with long sequences
            # Solution: Always use fp32 on MPS (slower but stable)
            self.use_mixed_precision = False
            if not self._mps_amp_warned:
                logger.warning("Mixed precision disabled on MPS due to backend bugs with fp16")
                logger.warning("Training will use fp32 on MPS (slower but stable)")
                self._mps_amp_warned = True
            return nullcontext()
        else:
            return nullcontext()

    def adaptive_memory_cleanup(self, batch_idx: int, force: bool = False) -> Dict[str, Any]:
        """Perform adaptive memory cleanup"""
        memory_policy = getattr(self, 'memory_oom_policy', None)
        if memory_policy is None:
            memory_policy = MemoryOOMPolicy(logger=logger)
            self.memory_oom_policy = memory_policy

        return memory_policy.adaptive_cleanup(
            enable_adaptive_memory=self.enable_adaptive_memory,
            memory_manager=self.memory_manager,
            batch_idx=batch_idx,
            force=force,
            clear_device_cache_fn=self.clear_device_cache,
        )

    def handle_oom_with_adaptive_cleanup(self, batch_idx: int, error: Exception) -> bool:
        """
        Handle OOM error with adaptive cleanup
        Returns True if training should continue, False if unrecoverable
        """
        memory_policy = getattr(self, 'memory_oom_policy', None)
        if memory_policy is None:
            memory_policy = MemoryOOMPolicy(logger=logger)
            self.memory_oom_policy = memory_policy

        return memory_policy.handle_oom(
            enable_adaptive_memory=self.enable_adaptive_memory,
            memory_manager=self.memory_manager,
            batch_idx=batch_idx,
            error=error,
            device_type=self.device_type,
            clear_device_cache_fn=self.clear_device_cache,
            gc_collect_fn=gc.collect,
        )

    def reset_profiling_stats(self):
        """Reset profiling statistics"""
        self.profiling_stats = {
            'stage_stats': {},
            'memory_snapshots': [],
            'device_info': {
                'device_name': self._get_device_name(),
                'device_available': self._is_device_available(),
                'device_type': self.device.type,
                'mixed_precision_enabled': self.use_mixed_precision,
                'mixed_precision_dtype': str(self.mixed_precision_dtype) if self.use_mixed_precision else None
            }
        }
        self.memory_snapshots = []
        self.interbatch_profiler.reset()

    def _get_device_name(self):
        """Get device name for different device types"""
        if self.device.type == DeviceType.CUDA.value:
            return torch.cuda.get_device_name()
        elif self.device.type == DeviceType.MPS.value:
            return 'Apple Silicon GPU'
        else:
            return 'CPU'

    def _is_device_available(self):
        """Check if device is available"""
        if self.device.type == DeviceType.CUDA.value:
            return torch.cuda.is_available()
        elif self.device.type == DeviceType.MPS.value:
            return torch.backends.mps.is_available()
        else:
            return True

    def start_torch_profiler(self, output_dir: str = None):
        """Start PyTorch profiler with comprehensive settings"""
        if output_dir is None:
            output_dir = self.profiler_log_dir

        os.makedirs(output_dir, exist_ok=True)

        profiler_kwargs = {
            'schedule': torch.profiler.schedule(
                wait=self.config.profile_wait_steps,
                warmup=self.config.profile_warmup_steps,
                active=self.config.profile_steps,
                repeat=1
            ),
            'on_trace_ready': torch.profiler.tensorboard_trace_handler(output_dir),
            'with_stack': True,
            'record_shapes': True,
        }

        # Add device-specific profiling options
        if self.device.type == DeviceType.CUDA.value:
            profiler_kwargs.update({
                'profile_memory': True,
                'with_flops': True
            })
        elif self.device.type == DeviceType.MPS.value:
            # MPS profiling capabilities are more limited
            profiler_kwargs.update({
                'profile_memory': False,  # Not supported on MPS
                'with_flops': False       # Not supported on MPS
            })

        self.profiler = torch.profiler.profile(**profiler_kwargs)
        logger.info(f"Started PyTorch profiler for {self.device.type}, output dir: {output_dir}")
        return self.profiler

    def stop_torch_profiler(self):
        """Stop PyTorch profiler"""
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
            logger.info("PyTorch profiler stopped")

    def _query_device_memory_mb(self) -> Tuple[float, float, float, float]:
        """Return (current, peak, reserved, total) memory in MB for the active device.

        For CUDA all four values come from the CUDA memory API.  For MPS only
        ``current_allocated_memory`` is available, so the remaining three fields
        are approximated from that value.  Returns zeros on CPU or when the
        device memory API is unavailable.
        """
        if self.device.type == DeviceType.CUDA.value:
            current  = torch.cuda.memory_allocated() / 1024**2
            peak     = torch.cuda.max_memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total    = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
            return current, peak, reserved, total
        elif self.device.type == DeviceType.MPS.value:
            try:
                current = torch.mps.current_allocated_memory() / 1024**2
                # MPS doesn't track peak/reserved separately; total is approximate
                return current, current, current, 8192.0
            except Exception:
                return 0.0, 0.0, 0.0, 0.0
        return 0.0, 0.0, 0.0, 0.0

    def profile_step(self):
        """Step the profiler and append a memory snapshot."""
        if self.profiler:
            self.profiler.step()

        current_memory, peak_memory, reserved_memory, total_memory = self._query_device_memory_mb()

        self.memory_snapshots.append({
            'timestamp': time.time(),
            'current_memory_mb': current_memory,
            'peak_memory_mb': peak_memory,
            'reserved_memory_mb': reserved_memory,
            'total_memory_mb': total_memory,
            'scaler_scale': self.scaler.get_scale() if self.scaler else None
        })

    def log_memory_stats(self, stage_name: str):
        """Log memory statistics for a specific stage."""
        current_memory, peak_memory, _, _ = self._query_device_memory_mb()

        if stage_name not in self.profiling_stats.get('stage_stats', {}):
            self.profiling_stats.setdefault('stage_stats', {})[stage_name] = {
                'memory_used_mb': current_memory,
                'peak_memory_mb': peak_memory,
                'call_count': 1,
                'total_time_ms': 0
            }
        else:
            stats = self.profiling_stats['stage_stats'][stage_name]
            stats['memory_used_mb'] = max(stats['memory_used_mb'], current_memory)
            stats['peak_memory_mb'] = max(stats['peak_memory_mb'], peak_memory)
            stats['call_count'] += 1

    def get_profiling_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiling report including mixed precision stats"""
        report = {
            'device_info': self.profiling_stats.get('device_info', {}),
            'stage_stats': self.profiling_stats.get('stage_stats', {}),
            'memory_snapshots': self.memory_snapshots,
            'interbatch_stats': self.interbatch_profiler.get_statistics(),
            'mixed_precision_stats': self.mixed_precision_stats.copy() if self.use_mixed_precision else None
        }

        # Memory summary
        if self.memory_snapshots:
            latest_snapshot = self.memory_snapshots[-1]
            report['memory_summary'] = {
                'current_memory_mb': latest_snapshot['current_memory_mb'],
                'peak_memory_mb': latest_snapshot['peak_memory_mb'],
                'reserved_memory_mb': latest_snapshot['reserved_memory_mb'],
                'total_memory_mb': latest_snapshot['total_memory_mb'],
                'stage_stats': self.profiling_stats.get('stage_stats', {}),
                'current_scaler_scale': latest_snapshot.get('scaler_scale')
            }

        # Memory analysis
        stage_stats = self.profiling_stats.get('stage_stats', {})
        if stage_stats:
            most_memory_intensive = max(stage_stats.keys(),
                                      key=lambda x: stage_stats[x]['memory_used_mb'])
            total_memory_used = sum(stats['memory_used_mb'] for stats in stage_stats.values())

            report['memory_analysis'] = {
                'most_memory_intensive_stage': most_memory_intensive,
                'total_memory_used_mb': total_memory_used
            }

        # Model info
        if hasattr(self.model, 'get_model_info'):
            report['model_info'] = self.model.get_model_info()

        return report

    def analyze_profiling_results(self, profiling_report: Dict[str, Any]):
        """Analyze and print profiling results in a readable format"""
        print("\n" + "="*60)
        print("GPU/MPS PROFILING ANALYSIS REPORT")
        print("="*60)

        # Device information
        device_info = profiling_report.get('device_info', {})
        print(f"\nDevice: {device_info.get('device_name', 'Unknown')}")
        print(f"Device Type: {device_info.get('device_type', 'Unknown')}")
        print(f"Device Available: {device_info.get('device_available', False)}")
        print(f"Mixed Precision: {device_info.get('mixed_precision_enabled', False)}")
        if device_info.get('mixed_precision_dtype'):
            print(f"Mixed Precision Dtype: {device_info.get('mixed_precision_dtype')}")

        # Mixed precision statistics
        mp_stats = profiling_report.get('mixed_precision_stats')
        if mp_stats:
            print(f"\nMixed Precision Statistics:")
            print(f"  Successful Steps: {mp_stats.get('successful_steps', 0)}")
            print(f"  Skipped Steps: {mp_stats.get('skipped_steps', 0)}")
            print(f"  Scale Updates: {mp_stats.get('scale_updates', 0)}")
            print(f"  Scale Decreases: {mp_stats.get('scale_decreases', 0)}")
            print(f"  Overflow Count: {mp_stats.get('overflow_count', 0)}")

            total_steps = mp_stats.get('successful_steps', 0) + mp_stats.get('skipped_steps', 0)
            if total_steps > 0:
                success_rate = (mp_stats.get('successful_steps', 0) / total_steps) * 100
                print(f"  Success Rate: {success_rate:.1f}%")

        # Memory analysis
        memory_summary = profiling_report.get('memory_summary', {})
        if memory_summary:
            print(f"\nMemory Usage:")
            print(f"  Current: {memory_summary.get('current_memory_mb', 0):.1f} MB")
            print(f"  Peak: {memory_summary.get('peak_memory_mb', 0):.1f} MB")
            print(f"  Reserved: {memory_summary.get('reserved_memory_mb', 0):.1f} MB")

            device_type = device_info.get('device_type', 'unknown')
            if device_type == DeviceType.CUDA.value:
                print(f"  Total GPU: {memory_summary.get('total_memory_mb', 0):.1f} MB")
            elif device_type == DeviceType.MPS.value:
                print(f"  Estimated Total: {memory_summary.get('total_memory_mb', 0):.1f} MB")

            # Memory efficiency
            total_memory = memory_summary.get('total_memory_mb', 1)
            peak_memory = memory_summary.get('peak_memory_mb', 0)
            if total_memory > 0:
                memory_efficiency = (peak_memory / total_memory) * 100
                print(f"  Memory Efficiency: {memory_efficiency:.1f}%")

            if memory_summary.get('current_scaler_scale'):
                print(f"  Current Scaler Scale: {memory_summary.get('current_scaler_scale'):.0f}")

        # Print interbatch profiling report
        self.interbatch_profiler.print_report()

    def clear_device_cache(self):
        """Clear device cache based on device type"""
        if self.device.type == DeviceType.CUDA.value:
            torch.cuda.empty_cache()
        elif self.device.type == DeviceType.MPS.value:
            torch.mps.empty_cache()

    def _validate_transferred_batch(self, transferred: BatchOnDevice) -> None:
        """Validate required batch structure after device transfer."""
        base_batch_size = transferred.mel_specs.size(0)
        required_tensors = {
            'phoneme_indices': transferred.phoneme_indices,
            'phoneme_durations': transferred.phoneme_durations,
            'stop_token_targets': transferred.stop_token_targets,
            'mel_lengths': transferred.mel_lengths,
            'phoneme_lengths': transferred.phoneme_lengths,
        }

        for name, tensor in required_tensors.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"Batch field '{name}' is not a tensor")
            if tensor.size(0) != base_batch_size:
                raise ValueError(
                    f"Batch field '{name}' has inconsistent batch dimension "
                    f"{tensor.size(0)} != {base_batch_size}"
                )

        optional_tensors = {
            'pitches': transferred.pitches,
            'energies': transferred.energies,
            'stress_indices': transferred.stress_indices,
        }
        for name, tensor in optional_tensors.items():
            if tensor is not None and tensor.size(0) != base_batch_size:
                raise ValueError(
                    f"Optional batch field '{name}' has inconsistent batch dimension "
                    f"{tensor.size(0)} != {base_batch_size}"
                )

    def _transfer_batch_to_device(self, batch: Dict[str, Any]) -> BatchOnDevice:
        """Transfer and validate a batch in one centralized path."""
        required_keys = (
            'mel_specs', 'phoneme_indices', 'phoneme_durations',
            'stop_token_targets', 'mel_lengths', 'phoneme_lengths'
        )
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            raise KeyError(f"Batch is missing required keys: {missing_keys}")

        non_blocking = self.device.type == 'cuda'

        def _to_device(key: str, required: bool = True) -> Optional[torch.Tensor]:
            value = batch.get(key)
            if value is None:
                if required:
                    raise ValueError(f"Batch field '{key}' is None")
                return None
            if not torch.is_tensor(value):
                raise TypeError(f"Batch field '{key}' must be a tensor, got {type(value).__name__}")
            return value.to(self.device, non_blocking=non_blocking)

        transferred = BatchOnDevice(
            mel_specs=_to_device('mel_specs'),
            phoneme_indices=_to_device('phoneme_indices'),
            phoneme_durations=_to_device('phoneme_durations'),
            stop_token_targets=_to_device('stop_token_targets'),
            mel_lengths=_to_device('mel_lengths'),
            phoneme_lengths=_to_device('phoneme_lengths'),
            pitches=_to_device('pitches', required=False),
            energies=_to_device('energies', required=False),
            stress_indices=_to_device('stress_indices', required=False),
        )

        self._validate_transferred_batch(transferred)
        return transferred

    def _profiler_record(self, name: str, active: bool):
        """Return a torch.profiler.record_function context when *active*, else a no-op.

        This lets callers write a single ``with self._profiler_record(name, flag):``
        block instead of duplicating the enclosed code inside an if/else.
        """
        from contextlib import nullcontext
        return torch.profiler.record_function(name) if active else nullcontext()

    def _has_nonfinite_gradients(self) -> bool:
        """Check whether any model gradient contains NaN/Inf values."""
        for param in self.model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return True
        return False

    def _compute_grad_explosion_threshold(self) -> Tuple[float, float, bool]:
        """Compute dynamic gradient explosion threshold with early-step warmup handling."""
        warmup_steps = max(0, self.grad_explosion_warmup_steps)

        if warmup_steps > 0 and self.optimizer_steps_completed < warmup_steps:
            progress = self.optimizer_steps_completed / float(warmup_steps)
            warmup_floor = self.grad_explosion_warmup_floor
            dynamic_abs_floor = warmup_floor - (warmup_floor - self.grad_explosion_abs_floor) * progress
        else:
            dynamic_abs_floor = self.grad_explosion_abs_floor

        ema_ready = self.grad_explosion_ema_steps >= self.grad_explosion_min_ema_steps
        ema_threshold = 0.0 if self.grad_explosion_norm_ema is None else self.grad_explosion_norm_ema * self.grad_explosion_multiplier

        threshold = dynamic_abs_floor if not ema_ready else max(dynamic_abs_floor, ema_threshold)
        return threshold, dynamic_abs_floor, ema_ready

    def _preclip_projection_spikes(self) -> Dict[str, Tuple[float, float]]:
        """Clip localized spikes in mel projection/attention/FFN gradients before global norm checks."""
        if (self.projection_spike_clip_norm <= 0
                and self.attention_spike_clip_norm <= 0
                and self.ffn_spike_clip_norm <= 0
                and self.stop_head_spike_clip_norm <= 0):
            return {}

        projection_params = {
            'mel_projection_in.weight',
            'mel_projection_in.bias',
            'mel_projection_out.weight',
            'mel_projection_out.bias',
        }

        # Stop-head parameters get their own tighter per-parameter ceiling so that
        # a stop-gradient spike cannot bleed into mel/encoder gradient budget.
        stop_head_params = {
            'stop_token_predictor.weight',
            'stop_token_predictor.bias',
        }

        attention_name_fragments = (
            '.self_attn.w_q.weight',
            '.self_attn.w_k.weight',
            '.self_attn.w_v.weight',
            '.self_attn.w_o.weight',
            '.cross_attn.w_q.weight',
            '.cross_attn.w_k.weight',
            '.cross_attn.w_v.weight',
            '.cross_attn.w_o.weight',
        )

        # linear1 and linear2 in decoder and encoder FFN layers are the consistent
        # high-delta culprits identified via checkpoint regression analysis.
        ffn_name_fragments = ('.linear1.weight', '.linear2.weight')

        clipped: Dict[str, Tuple[float, float]] = {}
        projection_max_norm = float(self.projection_spike_clip_norm)
        attention_max_norm = float(self.attention_spike_clip_norm)
        ffn_max_norm = float(self.ffn_spike_clip_norm)
        encoder_ffn_max_norm = float(getattr(self.config, 'encoder_ffn_spike_clip_norm', 10.0))

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            max_norm = None
            if name in projection_params and projection_max_norm > 0:
                max_norm = projection_max_norm
            elif name in stop_head_params and float(self.stop_head_spike_clip_norm) > 0:
                max_norm = float(self.stop_head_spike_clip_norm)
            elif attention_max_norm > 0 and (name.startswith('decoder.layers.') or name.startswith('transformer_encoder_layers.')) and any(fragment in name for fragment in attention_name_fragments):
                max_norm = attention_max_norm
            elif encoder_ffn_max_norm > 0 and name.startswith('transformer_encoder_layers.') and any(fragment in name for fragment in ffn_name_fragments):
                # Encoder FFN layers are the primary spike source — use tighter clip
                max_norm = encoder_ffn_max_norm
            elif ffn_max_norm > 0 and any(fragment in name for fragment in ffn_name_fragments):
                max_norm = ffn_max_norm

            if max_norm is None:
                continue

            grad = param.grad.data
            if not torch.isfinite(grad).all():
                continue

            grad_norm = grad.norm(2).item()
            if grad_norm > max_norm:
                scale = max_norm / (grad_norm + 1e-12)
                grad.mul_(scale)
                clipped[name] = (grad_norm, max_norm)

        return clipped

    def _build_dataloader(self, dataset, batch_sampler) -> DataLoader:
        """Construct a DataLoader with the shared worker/memory kwargs set during __init__."""
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            prefetch_factor=self._prefetch_factor,
            persistent_workers=self._persistent_workers,
        )

    def _average_pitch_energy_by_duration(self,
                                          values: torch.Tensor,
                                          durations: torch.Tensor,
                                          phoneme_lengths: torch.Tensor) -> torch.Tensor:
        """
        Average frame-level values (pitch/energy) to phoneme-level using durations.
        Fully vectorized via torch.repeat_interleave and scatter_add_ — no Python loops.

        Args:
            values: Frame-level values (batch, mel_frames)
            durations: Phoneme durations (batch, phonemes)
            phoneme_lengths: Actual phoneme lengths (batch,)

        Returns:
            Phoneme-level averaged values (batch, phonemes)
        """
        batch_size, num_phonemes = durations.shape
        num_frames = values.shape[1]
        device = durations.device

        # Zero out durations for padding positions (beyond each item's phoneme_lengths)
        ph_range = torch.arange(num_phonemes, device=device).unsqueeze(0)   # (1, num_phonemes)
        valid_ph_mask = ph_range < phoneme_lengths.unsqueeze(1)              # (batch, num_phonemes)
        durations_masked = durations.long() * valid_ph_mask.long()           # (batch, num_phonemes)

        # Total frames covered across the whole batch
        total_dur_per_b = durations_masked.sum(dim=1)                        # (batch,)
        total_dur = int(total_dur_per_b.sum().item())

        if total_dur == 0:
            return torch.zeros(batch_size, num_phonemes, device=device, dtype=values.dtype)

        # Build flat per-frame phoneme and batch indices with repeat_interleave
        # flat_dur[b * num_phonemes + p] = durations_masked[b, p]
        flat_dur = durations_masked.reshape(-1)                              # (batch * num_phonemes,)

        ph_idx_flat = torch.arange(num_phonemes, device=device).repeat(batch_size)          # (batch * num_phonemes,)
        b_idx_flat  = torch.arange(batch_size,   device=device).repeat_interleave(num_phonemes)

        # Each (b, p) entry is repeated dur[b, p] times → gives per-frame assignments
        ph_per_frame = torch.repeat_interleave(ph_idx_flat, flat_dur)        # (total_dur,)
        b_per_frame  = torch.repeat_interleave(b_idx_flat,  flat_dur)        # (total_dur,)

        # Convert global position in the flat frame sequence to a local frame index
        # within each batch item so we can index into `values`
        cum_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            total_dur_per_b.cumsum(0)[:-1]
        ])                                                                    # (batch,)
        global_idx      = torch.arange(total_dur, dtype=torch.long, device=device)
        local_frame_idx = global_idx - torch.repeat_interleave(cum_offsets, total_dur_per_b)
        local_frame_idx = local_frame_idx.clamp(0, num_frames - 1)

        # Gather the value for each (batch, local_frame) pair
        frame_values = values[b_per_frame, local_frame_idx]                  # (total_dur,)

        # Scatter-sum into a flat (batch * num_phonemes,) buffer, then reshape
        global_ph_idx = b_per_frame * num_phonemes + ph_per_frame            # (total_dur,)
        phoneme_sum = torch.zeros(batch_size * num_phonemes, device=device, dtype=values.dtype)
        phoneme_sum.scatter_add_(0, global_ph_idx, frame_values)
        phoneme_sum = phoneme_sum.reshape(batch_size, num_phonemes)

        # Divide by duration count (clamp to 1 avoids divide-by-zero for zero-dur phonemes)
        phoneme_values = phoneme_sum / durations_masked.float().clamp(min=1)

        # Ensure padding positions are exactly zero
        phoneme_values = phoneme_values * valid_ph_mask.float()

        return phoneme_values

    def _update_ema(self):
        """
        Update EMA model weights using exponential moving average.
        EMA weights = decay * EMA weights + (1 - decay) * current weights
        """
        if not self.use_ema or self.ema_model is None:
            return

        # Only update every N steps if configured
        if self.ema_updates % self.ema_update_every != 0:
            self.ema_updates += 1
            return

        with torch.no_grad():
            # Get state dicts
            model_params = self.model.state_dict()
            ema_params = self.ema_model.state_dict()

            # Update each parameter
            for key in model_params.keys():
                if model_params[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    # Apply EMA update: ema = decay * ema + (1 - decay) * current
                    ema_params[key].mul_(self.ema_decay).add_(
                        model_params[key], alpha=(1 - self.ema_decay)
                    )

            self.ema_updates += 1

    def _step_scheduler_with_warmup(self):
        """
        Step the learning rate scheduler with optional linear warmup.
        During warmup, manually set LR to increase linearly.
        After warmup, use the configured scheduler (e.g., OneCycleLR).

        Encoder param group (group 0 when multiple groups exist) receives
        _encoder_lr_multiplier × the base warmup LR so it warms up to the
        same higher base LR that OneCycleLR will continue from.
        """
        if self.use_warmup and self.current_optimizer_step < self.warmup_steps:
            # Linear warmup phase
            warmup_progress = self.current_optimizer_step / self.warmup_steps
            base_lr = self.warmup_start_lr + (self.warmup_target_lr - self.warmup_start_lr) * warmup_progress

            encoder_lr_mult   = getattr(self, '_encoder_lr_multiplier', 1.0)
            stop_head_lr_mult = float(getattr(getattr(self, 'config', None),
                                              'stop_head_lr_multiplier', 0.1))
            decoder_ffn_lr_mult = float(getattr(getattr(self, 'config', None),
                                                'decoder_ffn_lr_multiplier', 1.0))
            for param_group in self.optimizer.param_groups:
                gt = param_group.get('group_type')
                if gt == 'encoder':
                    mult = encoder_lr_mult
                elif gt == 'stop_head':
                    mult = stop_head_lr_mult
                elif gt == 'decoder_ffn':
                    mult = decoder_ffn_lr_mult
                else:
                    mult = 1.0
                param_group['lr'] = base_lr * mult

            self.current_optimizer_step += 1
        else:
            # After warmup, use OneCycleLR or other scheduler
            self.scheduler.step()
            if self.use_warmup:
                self.current_optimizer_step += 1

    @staticmethod
    def _apply_spec_augment(
        mel_specs: torch.Tensor,
        time_mask_max: int = 30,
        freq_mask_max: int = 10,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ) -> torch.Tensor:
        """Apply SpecAugment (Park et al. 2019): random time & frequency masking.

        mel_specs: (B, T, mel_dim) — teacher-forced decoder input.
        Returns a masked clone; the original tensor is NOT modified.
        Only the teacher-forced input is masked; the loss target remains the
        unmasked ground-truth mel, so gradients on unmasked frames are exact.
        """
        B, T, mel_dim = mel_specs.shape
        masked = mel_specs.clone()
        for _ in range(num_time_masks):
            t = torch.randint(0, max(1, min(time_mask_max, T // 4)), (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            masked[:, t0:t0 + t, :] = 0.0
        for _ in range(num_freq_masks):
            f = torch.randint(0, max(1, freq_mask_max), (1,)).item()
            f0 = torch.randint(0, max(1, mel_dim - f), (1,)).item()
            masked[:, :, f0:f0 + f] = 0.0
        return masked

    def _calculate_losses(self, predicted_mel, predicted_log_durations, predicted_stop_logits,
                         mel_specs, phoneme_durations, stop_token_targets,
                         mel_lengths, phoneme_lengths,
                         predicted_pitch=None, predicted_energy=None,
                         pitch_targets=None, energy_targets=None):
        """Calculate losses with optimized masking."""
        return calculate_training_losses(
            device=self.device,
            config=self.config,
            model=getattr(self, 'model', None),
            criterion_mel=self.criterion_mel,
            criterion_duration=self.criterion_duration,
            criterion_stop_token=self.criterion_stop_token,
            criterion_pitch=self.criterion_pitch,
            criterion_energy=self.criterion_energy,
            average_by_duration=self._average_pitch_energy_by_duration,
            logger=logger,
            predicted_mel=predicted_mel,
            predicted_log_durations=predicted_log_durations,
            predicted_stop_logits=predicted_stop_logits,
            mel_specs=mel_specs,
            phoneme_durations=phoneme_durations,
            stop_token_targets=stop_token_targets,
            mel_lengths=mel_lengths,
            phoneme_lengths=phoneme_lengths,
            predicted_pitch=predicted_pitch,
            predicted_energy=predicted_energy,
            pitch_targets=pitch_targets,
            energy_targets=energy_targets,
        )

    @staticmethod
    def _apply_warmup_guard(warmup_steps: int, total_steps: int) -> Tuple[int, int]:
        """
        Guard warmup_steps so that OneCycleLR always receives a positive total_steps.

        Returns:
            (clamped_warmup_steps, onecycle_steps) where onecycle_steps >= 1.
        """
        if warmup_steps >= total_steps:
            logger.warning(
                f"warmup_steps ({warmup_steps}) >= total_steps ({total_steps}). "
                "Clamping warmup to total_steps - 1 to avoid OneCycleLR crash."
            )
            warmup_steps = max(0, total_steps - 1)
        onecycle_steps = max(1, total_steps - warmup_steps)
        return warmup_steps, onecycle_steps

    def setup_checkpoint_resumption(self):
        """Delegate checkpoint resumption to checkpoint_manager.resume_from_checkpoint.

        Passes the trainer-module-scoped ``load_checkpoint`` and
        ``SummaryWriter`` so that test monkeypatches on those names propagate
        correctly into the implementation.
        """
        resume_from_checkpoint(
            self,
            _load_checkpoint_fn=load_checkpoint,
            _SummaryWriter=SummaryWriter,
        )

    def _log_lr_scalars(self, step: int) -> None:
        """Write per-group learning-rate scalars to TensorBoard.

        Resolves group indices by 'group_type' tag so the mapping is stable
        regardless of how many groups are present.  Recognised group types:
          encoder        → stats/lr_encoder
          decoder_other  → stats/lr_decoder   (canonical decoder reference)
          decoder_ffn    → stats/lr_decoder_ffn
          stop_head      → stats/lr_stop_head
        Falls back to positional heuristics for legacy (untagged) checkpoints.
        """
        tagged: dict[str, float] = {}
        untagged: list[float] = []
        for pg in self.optimizer.param_groups:
            gt = pg.get('group_type')
            if gt:
                tagged[gt] = pg['lr']
            else:
                untagged.append(pg['lr'])

        if tagged:
            if 'encoder' in tagged:
                self.writer.add_scalar('stats/lr_encoder', tagged['encoder'], step)
            if 'decoder_other' in tagged:
                self.writer.add_scalar('stats/lr_decoder', tagged['decoder_other'], step)
            if 'decoder_ffn' in tagged:
                self.writer.add_scalar('stats/lr_decoder_ffn', tagged['decoder_ffn'], step)
            if 'stop_head' in tagged:
                self.writer.add_scalar('stats/lr_stop_head', tagged['stop_head'], step)
        else:
            # Legacy fallback: positional heuristics
            n_pg = len(self.optimizer.param_groups)
            if n_pg >= 4:
                self.writer.add_scalar('stats/lr_encoder',   self.optimizer.param_groups[0]['lr'], step)
                self.writer.add_scalar('stats/lr_decoder',   self.optimizer.param_groups[2]['lr'], step)
                self.writer.add_scalar('stats/lr_stop_head', self.optimizer.param_groups[-1]['lr'], step)
            elif n_pg > 1:
                self.writer.add_scalar('stats/lr_encoder', self.optimizer.param_groups[0]['lr'], step)
                dec_idx = min(2, n_pg - 1)
                self.writer.add_scalar('stats/lr_decoder', self.optimizer.param_groups[dec_idx]['lr'], step)
            else:
                self.writer.add_scalar('stats/learning_rate', self.optimizer.param_groups[0]['lr'], step)

    def _build_lr_postfix(self) -> dict:
        """Return a dict of LR key(s) for the tqdm progress-bar postfix.

        Resolves groups by 'group_type' tag when available, else falls back to
        positional heuristics for legacy (untagged) checkpoints.
        """
        tagged: dict[str, float] = {}
        for pg in self.optimizer.param_groups:
            gt = pg.get('group_type')
            if gt:
                tagged[gt] = pg['lr']

        if tagged:
            result = {}
            if 'encoder' in tagged:
                result['lr_enc'] = tagged['encoder']
            if 'decoder_other' in tagged:
                result['lr_dec'] = tagged['decoder_other']
            if 'decoder_ffn' in tagged:
                result['lr_ffn'] = tagged['decoder_ffn']
            if 'stop_head' in tagged:
                result['lr_stop'] = tagged['stop_head']
            if result:
                return result

        # Legacy fallback
        n_pg = len(self.optimizer.param_groups)
        if n_pg >= 4:
            return {
                'lr_enc':  self.optimizer.param_groups[0]['lr'],
                'lr_dec':  self.optimizer.param_groups[2]['lr'],
                'lr_stop': self.optimizer.param_groups[-1]['lr'],
            }
        elif n_pg > 1:
            dec_idx = min(2, n_pg - 1)
            return {
                'lr_enc': self.optimizer.param_groups[0]['lr'],
                'lr_dec': self.optimizer.param_groups[dec_idx]['lr'],
            }
        return {'lr': self.optimizer.param_groups[0]['lr']}

    def _log_histograms_epoch(self, epoch: int) -> None:
        """Log weight histograms for all named model parameters to TensorBoard (once per epoch)."""
        try:
            for name, param in self.model.named_parameters():
                tag_name = name.replace('.', '/')
                if param.data is not None:
                    self.writer.add_histogram(
                        f'weights/{tag_name}', param.data.cpu().float(), self.optimizer_steps_completed
                    )
            self.writer.flush()
        except Exception as e:
            logger.debug(f"Failed to log weight histograms: {e}")

    def validate_epoch(self, epoch: int) -> EpochMetrics:
        """
        Run validation loop to monitor overfitting.
        Uses EMA model if available for better validation metrics.
        Returns: (avg_total_loss, avg_mel_loss, avg_dur_loss, avg_stop_loss)
        """
        if self.val_dataloader is None:
            return EpochMetrics(0.0, 0.0, 0.0, 0.0)

        # Use EMA model for validation if available, otherwise use regular model
        model_to_validate = self.ema_model if (self.use_ema and self.ema_model is not None) else self.model
        model_to_validate.eval()  # Set to evaluation mode

        if self.use_ema and self.ema_model is not None:
            logger.info("Validation using EMA model")

        total_loss_epoch = 0.0
        mel_loss_epoch = 0.0
        dur_loss_epoch = 0.0
        stop_loss_epoch = 0.0
        pitch_loss_epoch = 0.0
        energy_loss_epoch = 0.0
        # Metrics: spectral convergence and F0 RMSE
        spectral_conv_sum = 0.0
        spectral_conv_count = 0
        f0_rmse_sum = 0.0
        f0_rmse_count = 0
        num_batches = 0
        # Histogram accumulators for predicted distributions
        hist_pred_log_durations: list = []
        hist_pred_pitch: list = []
        hist_pred_energy: list = []

        with torch.no_grad():  # Disable gradient computation
            progress_bar = tqdm(self.val_dataloader, desc=f"Validation Epoch {epoch+1}")
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Data loading - batch transfer for efficiency
                    transferred = self._transfer_batch_to_device(batch)
                    mel_specs = transferred.mel_specs
                    phoneme_indices = transferred.phoneme_indices
                    phoneme_durations = transferred.phoneme_durations
                    stop_token_targets = transferred.stop_token_targets
                    mel_lengths = transferred.mel_lengths
                    phoneme_lengths = transferred.phoneme_lengths
                    pitches = transferred.pitches
                    energies = transferred.energies
                    stress_indices = transferred.stress_indices

                    # Forward pass (without mixed precision for validation consistency)
                    predicted_mel, predicted_log_durations, predicted_stop_logits, predicted_pitch, predicted_energy = \
                        model_to_validate(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets,
                                 pitch_targets=pitches, energy_targets=energies,
                                 stress_indices=stress_indices)

                    # Loss calculation — pass frame-level pitch/energy targets directly.
                    # losses.py detects they are already frame-level and aligns lengths
                    # without phoneme expansion (fixes double-averaging that froze f0_RMSE).
                    total_loss, loss_mel, loss_duration, loss_stop_token, loss_pitch, loss_energy = self._calculate_losses(
                        predicted_mel, predicted_log_durations, predicted_stop_logits,
                        mel_specs, phoneme_durations, stop_token_targets,
                        mel_lengths, phoneme_lengths,
                        predicted_pitch, predicted_energy, pitches, energies
                    )

                    # Log spectrograms for first validation batch only
                    if batch_idx == 0:
                        try:
                            pred_mel_img = predicted_mel[0].detach().cpu().T.unsqueeze(0)
                            gt_mel_img   = mel_specs[0].detach().cpu().T.unsqueeze(0)
                            self.writer.add_image('spectrogram/val_predicted',     pred_mel_img, self.optimizer_steps_completed, dataformats='CHW')
                            self.writer.add_image('spectrogram/val_ground_truth',  gt_mel_img,   self.optimizer_steps_completed, dataformats='CHW')
                            self.writer.flush()
                        except Exception as e:
                            logger.debug(f"Failed to log val spectrogram: {e}")

                    # Accumulate losses
                    total_loss_epoch += total_loss.item()
                    mel_loss_epoch += loss_mel.item()
                    dur_loss_epoch += loss_duration.item()
                    stop_loss_epoch += loss_stop_token.item()
                    if loss_pitch is not None:
                        pitch_loss_epoch += loss_pitch.item()
                    if loss_energy is not None:
                        energy_loss_epoch += loss_energy.item()
                    num_batches += 1

                    # Accumulate predictions for histogram logging
                    hist_pred_log_durations.append(predicted_log_durations.detach().cpu().float())
                    if predicted_pitch is not None:
                        hist_pred_pitch.append(predicted_pitch.detach().cpu().float())
                    if predicted_energy is not None:
                        hist_pred_energy.append(predicted_energy.detach().cpu().float())

                    # --- Spectral convergence (on mel spectrograms) ---
                    try:
                        # mel_specs & predicted_mel are (batch, time, n_mels)
                        batch_sc = 0.0
                        batch_sc_count = 0
                        for b in range(mel_specs.size(0)):
                            L = int(mel_lengths[b].item())
                            if L <= 0:
                                continue
                            ref = mel_specs[b, :L, :]
                            pred = predicted_mel[b, :L, :]
                            # Frobenius norm per sample
                            num = torch.norm(ref - pred, p='fro')
                            den = torch.norm(ref, p='fro')
                            if den.item() > 0:
                                sc = (num / den).item()
                                batch_sc += sc
                                batch_sc_count += 1
                        if batch_sc_count > 0:
                            spectral_conv_sum += (batch_sc / batch_sc_count)
                            spectral_conv_count += 1
                    except Exception:
                        # Non-critical metric; continue on failure
                        pass

                    # --- F0 RMSE (frame-level) ---
                    try:
                        if (pitches is not None) and (predicted_pitch is not None):
                            batch_f0 = 0.0
                            batch_f0_count = 0
                            for b in range(pitches.size(0)):
                                L = int(mel_lengths[b].item())
                                if L <= 0:
                                    continue
                                tgt = pitches[b, :L]
                                pred_f0 = predicted_pitch[b, :L]
                                # RMSE for this sample
                                mse = torch.mean((tgt - pred_f0) ** 2).item()
                                rmse = float(math.sqrt(mse))
                                batch_f0 += rmse
                                batch_f0_count += 1
                            if batch_f0_count > 0:
                                f0_rmse_sum += (batch_f0 / batch_f0_count)
                                f0_rmse_count += 1
                    except Exception:
                        pass

                    # Update progress bar
                    progress_bar.set_postfix({
                        'val_loss': f"{total_loss.item():.4f}",
                        'mel': f"{loss_mel.item():.4f}",
                        'dur': f"{loss_duration.item():.4f}",
                        'stop': f"{loss_stop_token.item():.4f}"
                    })

                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue

        # Restore both the validated model and the regular model to train mode.
        # When EMA is active, model_to_validate is the EMA model (a separate object
        # from self.model); restoring only self.model would leave the EMA model in
        # eval mode for the rest of the epoch.
        model_to_validate.train()
        if model_to_validate is not self.model:
            self.model.train()

        # Compute averages
        if num_batches > 0:
            avg_total_loss = total_loss_epoch / num_batches
            avg_mel_loss = mel_loss_epoch / num_batches
            avg_dur_loss = dur_loss_epoch / num_batches
            avg_stop_loss = stop_loss_epoch / num_batches
            avg_pitch_loss = pitch_loss_epoch / num_batches if pitch_loss_epoch > 0 else 0.0
            avg_energy_loss = energy_loss_epoch / num_batches if energy_loss_epoch > 0 else 0.0
            # Finalize metric averages
            avg_spectral_conv = (spectral_conv_sum / spectral_conv_count) if spectral_conv_count > 0 else None
            avg_f0_rmse = (f0_rmse_sum / f0_rmse_count) if f0_rmse_count > 0 else None

            logger.info(f"Validation Epoch {epoch+1} - "
                       f"Loss: {avg_total_loss:.4f}, "
                       f"Mel: {avg_mel_loss:.4f}, "
                       f"Dur: {avg_dur_loss:.4f}, "
                       f"Stop: {avg_stop_loss:.4f}")
            if avg_pitch_loss > 0:
                logger.info(f"  Pitch: {avg_pitch_loss:.4f}, Energy: {avg_energy_loss:.4f}")
                self.writer.add_scalar('loss/val_pitch', avg_pitch_loss, self.optimizer_steps_completed)
                self.writer.add_scalar('loss/val_energy', avg_energy_loss, self.optimizer_steps_completed)
            if avg_spectral_conv is not None:
                logger.info(f"  SpectralConv: {avg_spectral_conv:.6f}")
                self.writer.add_scalar('metrics/val_spectral_convergence', avg_spectral_conv, self.optimizer_steps_completed)
            if avg_f0_rmse is not None:
                logger.info(f"  f0_RMSE: {avg_f0_rmse:.6f}")
                self.writer.add_scalar('metrics/val_f0_rmse', avg_f0_rmse, self.optimizer_steps_completed)

            # Log histograms of predicted distributions over the full validation set
            try:
                if hist_pred_log_durations:
                    all_log_dur = torch.cat([t.reshape(-1) for t in hist_pred_log_durations])
                    self.writer.add_histogram('val_predictions/log_durations', all_log_dur, self.optimizer_steps_completed)
                if hist_pred_pitch:
                    all_pitch = torch.cat([t.reshape(-1) for t in hist_pred_pitch])
                    self.writer.add_histogram('val_predictions/pitch', all_pitch, self.optimizer_steps_completed)
                if hist_pred_energy:
                    all_energy = torch.cat([t.reshape(-1) for t in hist_pred_energy])
                    self.writer.add_histogram('val_predictions/energy', all_energy, self.optimizer_steps_completed)
            except Exception as e:
                logger.debug(f"Failed to log val prediction histograms: {e}")

            self.writer.flush()

            return EpochMetrics(
                total_loss=avg_total_loss,
                mel_loss=avg_mel_loss,
                dur_loss=avg_dur_loss,
                stop_loss=avg_stop_loss,
                pitch_loss=avg_pitch_loss,
                energy_loss=avg_energy_loss,
            )
        else:
            return EpochMetrics(0.0, 0.0, 0.0, 0.0)

    def save_checkpoint_with_scaler(
        self, epoch: int, loss: float, val_loss: float = None,
        val_mel_loss: float = None, val_stop_loss: float = None, val_dur_loss: float = None,
        best_val_loss: float = None, best_val_epoch: int = -1,
    ):
        """Save checkpoint including scaler state and EMA weights"""
        model_metadata = build_model_metadata(self.config, self.model)
        checkpoint = {
            'epoch': epoch,
            'global_step': self.current_optimizer_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_optimizer_step': self.current_optimizer_step,
            'optimizer_steps_completed': self.optimizer_steps_completed,
            'loss': loss,
            'train_loss': loss,
            'val_loss': val_loss,
            'val_mel_loss': val_mel_loss,
            'val_stop_loss': val_stop_loss,
            'val_dur_loss': val_dur_loss,
            # Explicit best-model tracking so resume always restores the correct baseline
            'best_val_loss': best_val_loss if best_val_loss is not None else val_loss,
            'best_val_epoch': best_val_epoch,
            'config': self.config,
            'model_metadata': model_metadata,
            # Snapshot of OneCycleLR params at save time so resume_from_checkpoint
            # can detect when scheduler config has changed between runs and log a
            # clear warning before step-based re-anchoring takes effect.
            'scheduler_config': {
                'onecycle_steps': getattr(self, '_onecycle_steps', None),
                'max_lr': getattr(self, '_onecycle_max_lr', None),
                'pct_start': getattr(self, '_onecycle_pct_start', None),
                'div_factor': getattr(self, '_onecycle_div_factor', None),
                'warmup_steps': self.warmup_steps if getattr(self, 'use_warmup', False) else 0,
            },
        }

        if self.use_mixed_precision and self.scaler:
            checkpoint['scaler'] = self.scaler.state_dict()
            checkpoint['device_type'] = self.device_type

        if self.use_ema and self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
            checkpoint['ema_updates'] = self.ema_updates
            logger.info(f"Saving EMA weights (updates: {self.ema_updates})")

        checkpoint_path = os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def train_epoch(self, epoch: int) -> EpochMetrics:
        """Train for one epoch with enhanced profiling, mixed precision, and adaptive memory management"""
        self.model.train()

        # Use tensor accumulation to reduce .item() calls (CPU-GPU sync overhead)
        total_loss_epoch = 0.0
        mel_loss_epoch = 0.0
        dur_loss_epoch = 0.0
        stop_loss_epoch = 0.0

        # Spectral convergence tracking for training
        train_sc_sum = 0.0
        train_sc_count = 0
        # Cache last batch spectrograms for end-of-epoch image logging
        _last_train_gt_mel = None
        _last_train_pred_mel = None

        # Track losses as tensors for batched accumulation (sync every N batches)
        loss_accumulation_interval = 10
        accumulated_losses = []

        # Initial estimate only; for dynamic batching this may change when the
        # sampler rebuilds batches at iterator start.
        num_batches = len(self.dataloader)
        processed_loss_batches = 0

        # Gradient accumulation for larger effective batch sizes
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        accumulated_step = 0  # Track steps within accumulation cycle

        # Determine if profiling for this epoch
        is_profiling_epoch = (epoch == self.config.profile_epoch_start) and self.config.enable_profiling
        enable_interbatch_profiling = getattr(self.config, 'enable_interbatch_profiling', False)

        if is_profiling_epoch:
            logger.info(f"Starting profiler for epoch {epoch+1} for {self.config.profile_steps} steps on {self.device_type}.")
            self.reset_profiling_stats()
            self.profiler = self.start_torch_profiler()
            self.profiler.__enter__()
        elif enable_interbatch_profiling:
            logger.info(f"Starting interbatch profiling for epoch {epoch+1}")
            self.interbatch_profiler.reset()

        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # DynamicFrameBatchSampler rebuilds self.batches inside __iter__.
                # Once iteration has started, read the rebuilt count so
                # is_last_batch/accumulation logic uses the current epoch's true size.
                if batch_idx == 0:
                    try:
                        sampler_batches = getattr(self.batch_sampler, 'batches', None)
                        if sampler_batches is not None:
                            rebuilt_num_batches = len(sampler_batches)
                            if rebuilt_num_batches > 0 and rebuilt_num_batches != num_batches:
                                logger.info(
                                    "Epoch %d batch count updated after sampler rebuild: %d -> %d",
                                    epoch + 1,
                                    num_batches,
                                    rebuilt_num_batches,
                                )
                                num_batches = rebuilt_num_batches
                    except Exception:
                        # Keep the initial estimate if sampler internals are unavailable.
                        pass

                # Start interbatch profiling for this batch
                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.start_batch()

                # Adaptive memory cleanup check
                cleanup_result = self.adaptive_memory_cleanup(batch_idx)

                # If profiling, advance the profiler schedule
                if is_profiling_epoch and self.profiler:
                    self.profile_step()
                    # Exit profiler early if we've collected enough data
                    total_profile_steps = (self.config.profile_wait_steps +
                                         self.config.profile_warmup_steps +
                                         self.config.profile_steps)
                    if batch_idx >= total_profile_steps:
                        logger.info("Profiler collected enough steps. Generating report.")
                        self.profiler.__exit__(None, None, None)

                        # Generate and analyze profiling report
                        report = self.get_profiling_report()
                        self.analyze_profiling_results(report)

                        self.profiler = None
                        is_profiling_epoch = False

                # Data loading with interbatch profiling
                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.start_data_loading()

                with self._profiler_record("Data_Loading", is_profiling_epoch):
                    transferred = self._transfer_batch_to_device(batch)
                    mel_specs = transferred.mel_specs
                    phoneme_indices = transferred.phoneme_indices
                    phoneme_durations = transferred.phoneme_durations
                    stop_token_targets = transferred.stop_token_targets
                    mel_lengths = transferred.mel_lengths
                    phoneme_lengths = transferred.phoneme_lengths
                    pitches = transferred.pitches
                    energies = transferred.energies
                    stress_indices = transferred.stress_indices
                    # SpecAugment: mask teacher-forced mel input; loss target stays unmasked.
                    # Epoch gate: disabled until after the OneCycleLR peak so spec augment
                    # does not compound ramp-phase instability.  Default matches config.py.
                    _spec_aug_start = getattr(self.config, 'spec_augment_start_epoch', 18)
                    if getattr(self.config, 'use_spec_augment', False) and epoch >= _spec_aug_start:
                        mel_for_model = KokoroTrainer._apply_spec_augment(
                            mel_specs,
                            time_mask_max=getattr(self.config, 'spec_augment_time_mask_max', 30),
                            freq_mask_max=getattr(self.config, 'spec_augment_freq_mask_max', 10),
                            num_time_masks=getattr(self.config, 'spec_augment_num_time_masks', 2),
                            num_freq_masks=getattr(self.config, 'spec_augment_num_freq_masks', 2),
                        )
                    else:
                        mel_for_model = mel_specs

                # Adaptive long-sequence handling: cap/truncate oversized batches
                # instead of skipping them entirely so training still benefits from
                # difficult long-form samples.
                max_dim_cap = int(getattr(self.config, 'max_sequence_dim_cap', 2000))
                transferred, mel_for_model, truncation_info = KokoroTrainer._cap_batch_sequence_dimensions(
                    transferred=transferred,
                    mel_for_model=mel_for_model,
                    max_mel_len=max_dim_cap,
                    max_phoneme_len=max_dim_cap,
                )
                if truncation_info['truncated']:
                    logger.warning(
                        "⚠️ BATCH %d - Capped long sequence dimensions "
                        "(mel: %d→%d, phoneme: %d→%d) to keep training stable",
                        batch_idx,
                        truncation_info['orig_mel_len'],
                        truncation_info['new_mel_len'],
                        truncation_info['orig_phoneme_len'],
                        truncation_info['new_phoneme_len'],
                    )

                mel_specs = transferred.mel_specs
                phoneme_indices = transferred.phoneme_indices
                phoneme_durations = transferred.phoneme_durations
                stop_token_targets = transferred.stop_token_targets
                mel_lengths = transferred.mel_lengths
                phoneme_lengths = transferred.phoneme_lengths
                pitches = transferred.pitches
                energies = transferred.energies
                stress_indices = transferred.stress_indices

                # Log variance predictor ranges periodically only in verbose mode
                if getattr(self.config, 'verbose', False) and batch_idx % 50 == 0 and pitches is not None and energies is not None:
                    pitch_min, pitch_max = pitches.min().item(), pitches.max().item()
                    energy_min, energy_max = energies.min().item(), energies.max().item()
                    if pitch_max > 1.5 or energy_max > 1.5:
                        logger.error(f"⚠️ BATCH {batch_idx} - UNNORMALIZED VARIANCE INPUTS!")
                        logger.error(f"   Pitch: [{pitch_min:.3f}, {pitch_max:.3f}]")
                        logger.error(f"   Energy: [{energy_min:.3f}, {energy_max:.3f}]")
                    else:
                        logger.info(f"Batch {batch_idx} variance ranges OK - Pitch: [{pitch_min:.3f}, {pitch_max:.3f}], Energy: [{energy_min:.3f}, {energy_max:.3f}]")

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.end_data_loading()

                if is_profiling_epoch:
                    self.log_memory_stats("data_loading")


                # Adaptive stabilization for sequence/duration outliers (no hard skipping).
                # IMPORTANT: soft_mel_length is now aligned with the hard cap (1400) so that
                # *normal* Russian TTS sequences (700-1300 frames) no longer trigger aggressive
                # gradient clipping that was starving the encoder of training signal.
                max_mel_length = 1400  # Hard cap for sequence dimensions
                max_duration_value = 150  # Hard cap for extreme durations
                adaptive_loss_scale = 1.0
                adaptive_clip_norm = getattr(self.config, 'max_grad_norm', 5.0)  # configurable ceiling

                mel_length = mel_specs.shape[1]
                max_duration_in_batch = phoneme_durations.max().item()

                # Soft stabilization: only fires for sequences ABOVE the hard cap threshold.
                # Setting soft_mel_length == max_mel_length means sequences in the normal
                # training range (< 1400 frames) are never penalised by soft clipping.
                soft_mel_length = 1400  # Raised from 900: stop penalising normal sequences
                soft_duration_value = 150
                soft_risk_ratio = max(mel_length / soft_mel_length, max_duration_in_batch / soft_duration_value)
                if soft_risk_ratio > 1.0:
                    adaptive_loss_scale = min(adaptive_loss_scale, max(0.5, 1.0 / (soft_risk_ratio ** 0.65)))
                    adaptive_clip_norm = min(adaptive_clip_norm, max(0.3, 0.8 / (soft_risk_ratio ** 0.35)))

                mel_risk_ratio = mel_length / max_mel_length
                duration_risk_ratio = max_duration_in_batch / max_duration_value
                risk_ratio = max(mel_risk_ratio, duration_risk_ratio)

                if risk_ratio > 1.0:
                    adaptive_loss_scale = max(0.25, 1.0 / risk_ratio)
                    adaptive_clip_norm = max(0.05, 0.5 / (risk_ratio ** 0.5))
                    if getattr(self.config, 'verbose', False):
                        logger.warning(f"\n{'='*80}")
                        logger.warning(f"⚠️  BATCH {batch_idx} - HIGH-RISK BATCH (stabilizing, not skipping)")
                        logger.warning(f"{'='*80}")
                        logger.warning(
                            f"mel_len={mel_length} (threshold={max_mel_length}), "
                            f"max_duration={max_duration_in_batch:.0f} (threshold={max_duration_value})"
                        )
                        logger.warning(
                            f"Applying adaptive stabilization: "
                            f"loss_scale={adaptive_loss_scale:.3f}, clip_norm={adaptive_clip_norm:.3f}"
                        )
                        logger.warning(f"{'='*80}\n")

                # Zero gradients only at start of accumulation cycle
                if accumulated_step == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                # Forward pass with mixed precision and interbatch profiling
                crash_context = None
                if self.device.type == DeviceType.MPS.value:
                    crash_context = (
                        f"[CrashCorrelation] epoch={epoch+1} batch={batch_idx}/{num_batches} "
                        f"opt_step={self.current_optimizer_step} "
                        f"accum={accumulated_step+1}/{gradient_accumulation_steps} "
                        f"mel_len={mel_specs.shape[1]} phoneme_len={phoneme_indices.shape[1]} "
                        f"batch_size={mel_specs.shape[0]}"
                    )

                # Attach context to modules so attention-level logs can include same ID
                if crash_context is not None:
                    self.model._crash_context = crash_context
                    for module in self.model.modules():
                        module._crash_context = crash_context

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.start_forward_pass()

                with self._profiler_record("Training_Step", is_profiling_epoch):
                    # Use exact accumulation divisor so tail micro-batches at epoch end
                    # are not under-scaled when fewer than gradient_accumulation_steps remain.
                    accumulation_divisor = KokoroTrainer._effective_accumulation_divisor(
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        accumulated_step=accumulated_step,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                    )

                    step_result = self._execute_training_step(
                        transferred, mel_for_model,
                        loss_scale=(1.0 / accumulation_divisor) * adaptive_loss_scale,
                    )

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.end_forward_pass()
                    self.interbatch_profiler.start_backward_pass()
                    self.interbatch_profiler.end_backward_pass()

                if is_profiling_epoch:
                    self.log_memory_stats("training_step")

                if step_result is None:
                    # Non-finite outputs or losses — reset accumulation and skip
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_step = 0
                    if self.enable_adaptive_memory:
                        self.memory_manager.emergency_cleanup()
                    else:
                        self.clear_device_cache()
                    if enable_interbatch_profiling or is_profiling_epoch:
                        self.interbatch_profiler.end_batch(mel_specs.size(0))
                    continue

                total_loss              = step_result.total_loss
                loss_mel                = step_result.loss_mel
                loss_duration           = step_result.loss_duration
                loss_stop_token         = step_result.loss_stop_token
                loss_pitch              = step_result.loss_pitch
                loss_energy             = step_result.loss_energy
                predicted_mel           = step_result.predicted_mel
                predicted_log_durations = step_result.predicted_log_durations
                predicted_pitch         = step_result.predicted_pitch
                predicted_energy        = step_result.predicted_energy

                # Advance gradient accumulation cycle after successful backward
                accumulated_step += 1

                # MPS cache clearing - only when needed based on memory pressure
                if self.device_type == DeviceType.MPS.value:
                    # Only clear if memory pressure is moderate or higher
                    pressure = cleanup_result.get('pressure_level', 'low')
                    if pressure in ['moderate', 'high', 'critical']:
                        torch.mps.empty_cache()
                    # Run GC only on high/critical pressure or every 100 batches
                    if pressure in ['high', 'critical'] or batch_idx % 100 == 0:
                        import gc
                        gc.collect()

                # Determine if we should step the optimizer (accumulation complete or last batch)
                is_last_batch = (batch_idx == num_batches - 1)
                should_step = (accumulated_step >= gradient_accumulation_steps) or is_last_batch

                # Gradient explosion detection only at optimizer-step boundaries
                if should_step:
                    clipped_projection_grads = self._preclip_projection_spikes()
                    if clipped_projection_grads and getattr(self.config, 'verbose', False):
                        clipped_msg = ', '.join(
                            f"{name}: {before:.2f}→{after:.2f}"
                            for name, (before, after) in clipped_projection_grads.items()
                        )
                        logger.warning(f"Pre-clipped projection gradient spikes: {clipped_msg}")

                    total_grad_norm = 0.0
                    grad_norms_by_param = []
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            total_grad_norm += grad_norm ** 2
                            grad_norms_by_param.append((name, grad_norm))
                    total_grad_norm = total_grad_norm ** 0.5

                    # Log total gradient norm to TensorBoard at every optimizer step
                    self.writer.add_scalar('stats/grad_norm', total_grad_norm, self.optimizer_steps_completed)

                    grad_norm_ema_for_threshold = self.grad_explosion_norm_ema if self.grad_explosion_norm_ema is not None else 0.0
                    gradient_explosion_threshold, dynamic_abs_floor, ema_ready = self._compute_grad_explosion_threshold()

                    has_nonfinite_grads = self._has_nonfinite_gradients()
                    is_exploding = total_grad_norm > gradient_explosion_threshold

                    if is_exploding:
                        self.grad_explosion_streak += 1
                        log_fn = logger.error if has_nonfinite_grads else (logger.warning if self.grad_explosion_streak > 1 else logger.error)
                        log_fn(f"\n{'='*80}")
                        log_fn(f"❌ BATCH {batch_idx} - GRADIENT EXPLOSION DETECTED")
                        log_fn(f"{'='*80}")
                        log_fn(
                            f"Total gradient norm: {total_grad_norm:.2f} "
                            f"(threshold: {gradient_explosion_threshold:.2f}, "
                            f"ema: {grad_norm_ema_for_threshold:.2f}, "
                            f"dynamic_floor: {dynamic_abs_floor:.2f}, "
                            f"ema_ready: {ema_ready}, "
                            f"multiplier: {self.grad_explosion_multiplier:.2f})"
                        )
                        log_fn("Applying emergency clipping instead of skipping this batch")

                        top10 = sorted(grad_norms_by_param, key=lambda x: x[1], reverse=True)[:10]
                        log_fn("Top 10 parameter gradient norms (global):")
                        for param_name, param_norm in top10:
                            log_fn(f"  {param_name}: {param_norm:.2f}")

                        adaptive_clip_norm = min(adaptive_clip_norm, 0.3)  # Floor raised from 0.05
                        log_fn(f"Emergency clip norm set to {adaptive_clip_norm:.3f}")
                        log_fn(f"{'='*80}\n")
                    else:
                        self.grad_explosion_streak = 0

                    if self.grad_explosion_norm_ema is None:
                        self.grad_explosion_norm_ema = total_grad_norm
                    else:
                        alpha = self.grad_explosion_ema_alpha
                        self.grad_explosion_norm_ema = alpha * self.grad_explosion_norm_ema + (1 - alpha) * total_grad_norm
                    self.grad_explosion_ema_steps += 1

                if should_step and self._has_nonfinite_gradients():
                    logger.error(f"\n{'='*80}")
                    logger.error(f"❌ BATCH {batch_idx} - NON-FINITE GRADIENTS DETECTED")
                    logger.error(f"{'='*80}")
                    logger.error(f"Skipping optimizer step at batch {batch_idx}")
                    logger.error(f"Accumulated step: {accumulated_step}/{gradient_accumulation_steps}")
                    logger.error(f"Optimizer step number: {self.current_optimizer_step}")

                    # Save problematic batch data for debugging
                    try:
                        debug_path = Path(self.config.output_dir) / f"debug_batch_{batch_idx}_epoch_{epoch}.pt"
                        debug_data = {
                            'batch_idx': batch_idx,
                            'epoch': epoch,
                            'optimizer_step': self.current_optimizer_step,
                            'accumulated_step': accumulated_step,
                            'batch_data': {
                                'phoneme_indices': phoneme_indices.cpu(),
                                'mel_specs': mel_specs.cpu(),
                                'phoneme_durations': phoneme_durations.cpu(),
                                'stop_token_targets': stop_token_targets.cpu(),
                                'mel_lengths': mel_lengths.cpu(),
                                'phoneme_lengths': phoneme_lengths.cpu(),
                            },
                            'outputs': {
                                'predicted_mel': predicted_mel.detach().cpu() if predicted_mel is not None else None,
                                'predicted_log_durations': predicted_log_durations.detach().cpu() if predicted_log_durations is not None else None,
                                'predicted_pitch': predicted_pitch.detach().cpu() if predicted_pitch is not None else None,
                                'predicted_energy': predicted_energy.detach().cpu() if predicted_energy is not None else None,
                            },
                            'losses': {
                                'total_loss': total_loss.item(),
                                'mel_loss': loss_mel.item(),
                                'duration_loss': loss_duration.item(),
                                'stop_loss': loss_stop_token.item(),
                                'pitch_loss': loss_pitch.item() if loss_pitch is not None else None,
                                'energy_loss': loss_energy.item() if loss_energy is not None else None,
                            }
                        }
                        if pitches is not None:
                            debug_data['batch_data']['pitches'] = pitches.cpu()
                        if energies is not None:
                            debug_data['batch_data']['energies'] = energies.cpu()

                        torch.save(debug_data, debug_path)
                        logger.error(f"💾 Saved problematic batch data to: {debug_path}")
                    except Exception as e:
                        logger.error(f"Failed to save debug data: {e}")

                    logger.error(f"{'='*80}\n")
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_step = 0
                    if self.enable_adaptive_memory:
                        self.memory_manager.emergency_cleanup()
                    else:
                        self.clear_device_cache()
                    continue

                # Optimizer step with mixed precision - only when accumulation is complete
                if should_step:
                    step_successful, clipped_norm = self._optimizer_step_with_clipping(
                        clip_norm=adaptive_clip_norm,
                        step_scheduler=True,
                        update_ema=True,
                    )
                    # Log post-global-clip norm at the same step as stats/grad_norm
                    self.writer.add_scalar('stats/grad_norm_clipped', clipped_norm, self.optimizer_steps_completed)

                    if step_successful:
                        self.optimizer_steps_completed += 1
                        self._apply_weight_norm_constraints()
                    else:
                        logger.warning(
                            "Optimizer step skipped (AMP overflow/non-finite grads); "
                            "scheduler and EMA were not advanced for this step"
                        )

                    # Log every 10 steps
                    if self.optimizer_steps_completed % 10 == 0:
                        self.writer.add_scalar('loss/total', total_loss.item(), self.optimizer_steps_completed)
                        self.writer.add_scalar('loss/mel', loss_mel.item(), self.optimizer_steps_completed)
                        self.writer.add_scalar('loss/duration', loss_duration.item(), self.optimizer_steps_completed)
                        self.writer.add_scalar('loss/stop', loss_stop_token.item(), self.optimizer_steps_completed)
                        if loss_pitch is not None:
                            self.writer.add_scalar('loss/pitch', loss_pitch.item(), self.optimizer_steps_completed)
                        if loss_energy is not None:
                            self.writer.add_scalar('loss/energy', loss_energy.item(), self.optimizer_steps_completed)
                        # Learning Rate to track the scheduler
                        self._log_lr_scalars(self.optimizer_steps_completed)

                        # Force TensorBoard to update the file
                        self.writer.flush()

                    # Log spectrogram every 200 optimizer steps
                    if self.optimizer_steps_completed % 200 == 0:
                        with torch.no_grad():
                            try:
                                gt_mel = mel_specs[0].detach().cpu()        # (T, n_mels)
                                gt_mel_img = gt_mel.T.unsqueeze(0)          # (1, n_mels, T) for TensorBoard
                                self.writer.add_image(
                                    'spectrogram/train_ground_truth',
                                    gt_mel_img,
                                    self.optimizer_steps_completed,
                                    dataformats='CHW'
                                )
                                # predicted_mel is still available here (deleted later in the batch loop)
                                if _last_train_pred_mel is not None:
                                    pred_mel_img = _last_train_pred_mel.T.unsqueeze(0)
                                    self.writer.add_image(
                                        'spectrogram/train_predicted',
                                        pred_mel_img,
                                        self.optimizer_steps_completed,
                                        dataformats='CHW'
                                    )
                            except Exception as e:
                                logger.debug(f"Failed to log train spectrogram: {e}")

                        # Log gradient histograms every 200 steps (gradients are live here,
                        # before the next batch's zero_grad)
                        try:
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    tag_name = name.replace('.', '/')
                                    self.writer.add_histogram(
                                        f'gradients/{tag_name}',
                                        param.grad.data.cpu().float(),
                                        self.optimizer_steps_completed
                                    )
                            self.writer.flush()
                        except Exception as e:
                            logger.debug(f"Failed to log gradient histograms: {e}")

                    # Reset accumulation counter after stepping
                    accumulated_step = 0

                    # Clear MPS cache after optimizer step only if needed
                    if self.device_type == DeviceType.MPS.value:
                        pressure = cleanup_result.get('pressure_level', 'low')
                        if pressure in ['high', 'critical']:
                            torch.mps.empty_cache()

                # End batch profiling
                if enable_interbatch_profiling or is_profiling_epoch:
                    batch_size = mel_specs.size(0)
                    self.interbatch_profiler.end_batch(batch_size)

                # Accumulate losses as tensors to reduce CPU-GPU sync
                accumulated_losses.append({
                    'total': total_loss.detach(),
                    'mel': loss_mel.detach(),
                    'dur': loss_duration.detach(),
                    'stop': loss_stop_token.detach(),
                })
                processed_loss_batches += 1

                # Sync accumulated losses periodically to reduce overhead
                if len(accumulated_losses) >= loss_accumulation_interval or batch_idx == num_batches - 1:
                    for acc_loss in accumulated_losses:
                        total_loss_epoch += acc_loss['total'].item()
                        mel_loss_epoch += acc_loss['mel'].item()
                        dur_loss_epoch += acc_loss['dur'].item()
                        stop_loss_epoch += acc_loss['stop'].item()
                    accumulated_losses.clear()

                # Capture progress bar values BEFORE deleting tensors
                current_total_loss = total_loss.item()
                current_mel_loss = loss_mel.item()
                current_dur_loss = loss_duration.item()
                current_stop_loss = loss_stop_token.item()
                pitch_loss_value = loss_pitch.item() if loss_pitch is not None else 0.0
                energy_loss_value = loss_energy.item() if loss_energy is not None else 0.0
                has_pitch_loss = loss_pitch is not None and loss_pitch.item() > 0
                has_energy_loss = loss_energy is not None and loss_energy.item() > 0

                # Compute spectral convergence and cache spectrograms before freeing tensors
                try:
                    with torch.no_grad():
                        for b in range(mel_specs.size(0)):
                            L = int(mel_lengths[b].item())
                            if L <= 0:
                                continue
                            ref = mel_specs[b, :L, :]
                            pred_sc = predicted_mel[b, :L, :]
                            num = torch.norm(ref - pred_sc, p='fro')
                            den = torch.norm(ref, p='fro')
                            if den.item() > 0:
                                train_sc_sum += (num / den).item()
                                train_sc_count += 1
                    # Cache first sample for end-of-epoch image logging
                    _last_train_gt_mel = mel_specs[0].detach().cpu()
                    _last_train_pred_mel = predicted_mel[0].detach().cpu()
                except Exception:
                    pass

                # Free output tensors every batch — large, not needed after loss/backward
                del predicted_mel, predicted_log_durations
                if predicted_pitch is not None:
                    del predicted_pitch
                    predicted_pitch = None
                if predicted_energy is not None:
                    del predicted_energy
                    predicted_energy = None

                # MPS cache clear: every 3 batches normally, every batch under pressure
                if self.device_type == DeviceType.MPS.value:
                    pressure = cleanup_result.get('pressure_level', 'low')
                    if pressure in ['high', 'critical'] or batch_idx % 3 == 0:
                        torch.mps.empty_cache()
                    if pressure in ['high', 'critical'] or batch_idx % 10 == 0:
                        gc.collect()

                # Enhanced progress bar with mixed precision info and memory pressure
                postfix_dict = {
                    'total_loss': current_total_loss,
                    'mel_loss': current_mel_loss,
                    'dur_loss': current_dur_loss,
                    'stop_loss': current_stop_loss,
                }
                postfix_dict.update(self._build_lr_postfix())

                # Add variance losses if they exist
                if has_pitch_loss:
                    postfix_dict['pitch_loss'] = pitch_loss_value
                if has_energy_loss:
                    postfix_dict['energy_loss'] = energy_loss_value

                if self.use_mixed_precision:
                    postfix_dict['scale'] = f"{self.scaler.get_scale():.0f}"
                    if self.device_type == DeviceType.MPS.value:
                        postfix_dict['device'] = 'MPS'

                # Add memory pressure info if adaptive memory is enabled
                if self.enable_adaptive_memory:
                    postfix_dict['mem'] = cleanup_result.get('pressure_level', 'unknown')[:3]  # First 3 chars
                    if cleanup_result.get('cleaned', False):
                        postfix_dict['mem'] += '*'  # Indicate cleanup occurred

                progress_bar.set_postfix(postfix_dict)

                # Print memory management report periodically (only when verbose)
                if self.enable_adaptive_memory and getattr(self.config, 'verbose', False) and batch_idx % self.memory_report_interval == 0 and batch_idx > 0:
                    logger.info(f"Memory management stats at batch {batch_idx}:")
                    report = self.memory_manager.get_memory_report()
                    logger.info(f"  Current pressure: {report['current_pressure']}")
                    logger.info(f"  Memory usage: {report.get('current_memory_usage_percent', 0):.1f}%")
                    logger.info(f"  Cleanups performed: {report['cleanup_count']}")
                    logger.info(f"  Cleanup overhead: {report['cleanup_overhead_percent']:.2f}%")

                # Print interbatch profiling stats periodically
                if enable_interbatch_profiling and batch_idx % getattr(self.config, 'interbatch_report_interval', 100) == 0 and batch_idx > 0:
                    logger.info(f"Interbatch profiling stats at batch {batch_idx}:")
                    stats = self.interbatch_profiler.get_statistics()
                    logger.info(f"  Avg interbatch time: {stats.get('interbatch_mean_ms', 0):.1f}ms")
                    logger.info(f"  Avg data loading time: {stats.get('data_loading_mean_ms', 0):.1f}ms")
                    logger.info(f"  Throughput: {stats.get('throughput_samples_per_sec', 0):.2f} samples/sec")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM error at batch {batch_idx} on {self.device_type}: {e}")

                    # Try adaptive memory cleanup to recover
                    can_continue = self.handle_oom_with_adaptive_cleanup(batch_idx, e)

                    if can_continue:
                        logger.info("Attempting to continue after OOM recovery")
                        continue
                    else:
                        logger.error("Unable to recover from OOM error")
                        raise e
                else:
                    logger.error(f"Runtime error in batch {batch_idx}: {e}")
                    raise e
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                # Try emergency cleanup and continue
                if self.enable_adaptive_memory:
                    self.memory_manager.emergency_cleanup()
                else:
                    self.clear_device_cache()
                continue

        # Ensure profiler is exited if it was started but didn't complete its schedule
        if self.profiler:
            logger.info("Profiler exiting at end of epoch.")
            self.profiler.__exit__(None, None, None)
            # Generate final report
            report = self.get_profiling_report()
            self.analyze_profiling_results(report)
            self.profiler = None

        # Print final interbatch profiling report for this epoch
        if enable_interbatch_profiling:
            logger.info(f"Final interbatch profiling report for epoch {epoch+1}:")
            self.interbatch_profiler.print_report()

        # Log mixed precision statistics for this epoch
        if self.use_mixed_precision:
            mp_stats = self.mixed_precision_stats
            total_steps = mp_stats['successful_steps'] + mp_stats.get('skipped_steps', 0) + mp_stats['overflow_count']
            if total_steps > 0:
                success_rate = (mp_stats['successful_steps'] / total_steps) * 100
                device_info = f"({self.device_type.upper()})"
                logger.info(f"Mixed Precision Stats {device_info} - Success: {mp_stats['successful_steps']}, "
                           f"Skipped: {mp_stats.get('skipped_steps', 0)}, "
                           f"Overflows: {mp_stats['overflow_count']}, "
                           f"Success Rate: {success_rate:.1f}%, "
                           f"Current Scale: {self.scaler.get_scale():.0f}")

        # Log epoch-level train spectral convergence.
        # Use optimizer_steps_completed (same axis as val_spectral_convergence)
        # so the Custom Scalar "Multiline" chart aligns both series correctly.
        if train_sc_count > 0:
            avg_train_sc = train_sc_sum / train_sc_count
            self.writer.add_scalar('metrics/train_spectral_convergence', avg_train_sc, self.optimizer_steps_completed)
            logger.info(f"  Train SpectralConv (epoch {epoch+1}): {avg_train_sc:.6f}")

        # Log end-of-epoch train spectrograms (ground truth + predicted)
        try:
            if _last_train_gt_mel is not None:
                self.writer.add_image(
                    'spectrogram/train_ground_truth_epoch',
                    _last_train_gt_mel.T.unsqueeze(0), epoch, dataformats='CHW'
                )
            if _last_train_pred_mel is not None:
                self.writer.add_image(
                    'spectrogram/train_predicted_epoch',
                    _last_train_pred_mel.T.unsqueeze(0), epoch, dataformats='CHW'
                )
            self.writer.flush()
        except Exception as e:
            logger.debug(f"Failed to log end-of-epoch train spectrograms: {e}")

        if processed_loss_batches == 0:
            logger.warning("No successful training batches were processed in this epoch")
            return EpochMetrics(0.0, 0.0, 0.0, 0.0)

        return EpochMetrics(
            total_loss=(total_loss_epoch / processed_loss_batches),
            mel_loss=(mel_loss_epoch / processed_loss_batches),
            dur_loss=(dur_loss_epoch / processed_loss_batches),
            stop_loss=(stop_loss_epoch / processed_loss_batches),
        )

    def _snapshot_cache_stats(self, dataset) -> Optional[Dict[str, float]]:
        """Safely snapshot dataset feature-cache stats for epoch delta reporting."""
        if dataset is None or not hasattr(dataset, 'get_feature_cache_stats'):
            return None

        try:
            stats = dataset.get_feature_cache_stats()
        except Exception as e:
            logger.debug(f"Could not snapshot feature cache stats: {e}")
            return None

        if not stats.get('enabled', False):
            return None
        return stats

    def _log_epoch_cache_delta(self, epoch: int, split_name: str, dataset, start_stats: Optional[Dict[str, float]]):
        """Log feature-cache counters for one epoch/split."""
        if start_stats is None:
            return

        end_stats = self._snapshot_cache_stats(dataset)
        if end_stats is None:
            return

        requests = max(0, int(end_stats['requests'] - start_stats['requests']))
        mem_hits = max(0, int(end_stats['mem_hits'] - start_stats['mem_hits']))
        disk_hits = max(0, int(end_stats['disk_hits'] - start_stats['disk_hits']))
        hits = mem_hits + disk_hits
        misses = max(0, int(end_stats['misses'] - start_stats['misses']))

        if requests <= 0:
            return

        hit_rate = (hits / requests) * 100.0
        # Compute epoch-average latencies (ms) when raw counters are available
        try:
            mem_ns_start = int(start_stats.get('mem_latency_ns_total', 0))
            mem_count_start = int(start_stats.get('mem_latency_count', 0))
            mem_ns_end = int(end_stats.get('mem_latency_ns_total', 0))
            mem_count_end = int(end_stats.get('mem_latency_count', 0))

            disk_ns_start = int(start_stats.get('disk_latency_ns_total', 0))
            disk_count_start = int(start_stats.get('disk_latency_count', 0))
            disk_ns_end = int(end_stats.get('disk_latency_ns_total', 0))
            disk_count_end = int(end_stats.get('disk_latency_count', 0))

            mem_count_delta = max(0, mem_count_end - mem_count_start)
            disk_count_delta = max(0, disk_count_end - disk_count_start)

            mem_epoch_ms = ( (mem_ns_end - mem_ns_start) / mem_count_delta / 1e6 ) if mem_count_delta > 0 else 0.0
            disk_epoch_ms = ( (disk_ns_end - disk_ns_start) / disk_count_delta / 1e6 ) if disk_count_delta > 0 else 0.0
        except Exception:
            mem_epoch_ms = 0.0
            disk_epoch_ms = 0.0

        # Show 'N/A' for mem/disk latency when there were no in-memory/disk hits
        mem_lat_display = "N/A" if mem_count_delta == 0 or mem_hits == 0 else f"{mem_epoch_ms:.3f}"
        disk_lat_display = "N/A" if disk_count_delta == 0 or disk_hits == 0 else f"{disk_epoch_ms:.3f}"

        logger.info(
            "Epoch %d %s cache: requests=%d hits=%d (mem=%d,disk=%d) misses=%d hit_rate=%.1f%% mem_lat_ms=%s disk_lat_ms=%s",
            epoch + 1,
            split_name,
            requests,
            hits,
            mem_hits,
            disk_hits,
            misses,
            hit_rate,
            mem_lat_display,
            disk_lat_display,
        )

    def train(self):
        """Main training function with mixed precision support and adaptive memory management"""
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.setup_checkpoint_resumption()
        save_phoneme_processor(self.dataset.phoneme_processor, self.config.output_dir)

        logger.info(f"Starting training on device: {self.device} ({self.device_type})")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Mixed precision training: {'Enabled' if self.use_mixed_precision else 'Disabled'}")
        if self.use_mixed_precision:
            logger.info(f"Mixed precision dtype: {self.mixed_precision_dtype}")
            if self.device_type == DeviceType.MPS.value:
                logger.info("Using custom MPS gradient scaler (experimental)")
        logger.info(f"Adaptive memory management: {'Enabled' if self.enable_adaptive_memory else 'Disabled'}")
        logger.info(f"Total epochs: {self.config.num_epochs}, Starting from epoch: {self.start_epoch + 1}")
        logger.info(f"Model vocabulary size: {self.dataset.phoneme_processor.get_vocab_size()}")
        logger.info(f"Initial learning rate: {self.config.learning_rate}")
        if self.scheduler_per_batch:
            logger.info(f"Scheduler: OneCycleLR (steps per batch, max_lr={self.config.learning_rate * getattr(self.config, 'max_lr_multiplier', 5.0):.2e})")
        else:
            logger.info(f"Scheduler: CosineAnnealingWarmRestarts (steps per epoch, T_0={self.config.lr_T_0}, T_mult={self.config.lr_T_mult})")
        logger.info(f"Loss weights: Mel={1.0}, Duration={self.config.duration_loss_weight}, StopToken={self.config.stop_token_loss_weight}")

        enable_profiling = getattr(self.config, 'enable_profiling', False)
        if enable_profiling:
            logger.info(f"Profiler logs will be saved to: {self.profiler_log_dir}")

        # Log interbatch profiling settings
        enable_interbatch_profiling = getattr(self.config, 'enable_interbatch_profiling', False)
        if enable_interbatch_profiling and enable_profiling:
            logger.info(f"Interbatch profiling enabled with report interval: {getattr(self.config, 'interbatch_report_interval', 100)}")

        # Log adaptive memory settings
        if self.enable_adaptive_memory:
            logger.info(f"Adaptive memory management enabled:")
            logger.info(f"  Memory report interval: {self.memory_report_interval} batches")
            thresholds = self.memory_manager.thresholds
            logger.info(f"  Memory thresholds: Low={thresholds.low_threshold*100:.0f}%, "
                       f"Moderate={thresholds.moderate_threshold*100:.0f}%, "
                       f"High={thresholds.high_threshold*100:.0f}%, "
                       f"Critical={thresholds.critical_threshold*100:.0f}%")

        # Run standalone profiling if requested
        if hasattr(self.config, 'run_standalone_profiling') and self.config.run_standalone_profiling:
            logger.info(f"Running standalone profiling before training on {self.device_type}...")
            self.profile_training_steps(self.config.profile_steps)

        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            train_cache_start = self._snapshot_cache_stats(self.dataset)
            if self.enable_adaptive_memory:
                epoch_cleanup_count_start = self.memory_manager.cleanup_count
                epoch_cleanup_time_start = self.memory_manager.total_cleanup_time
                epoch_memory_history_start = len(self.memory_manager.memory_history)

            train_metrics = self.train_epoch(epoch)
            avg_total_loss = train_metrics.total_loss
            avg_mel_loss = train_metrics.mel_loss
            avg_dur_loss = train_metrics.dur_loss
            avg_stop_loss = train_metrics.stop_loss

            # Step scheduler per epoch only if NOT using OneCycleLR (which steps per batch)
            if not self.scheduler_per_batch:
                self.scheduler.step()

            _n_pg_epoch = len(self.optimizer.param_groups)
            if _n_pg_epoch > 1:
                _lr_str = (f"LR enc: {self.optimizer.param_groups[0]['lr']:.8f}, "
                           f"dec: {self.optimizer.param_groups[-1]['lr']:.8f}")
            else:
                _lr_str = f"LR: {self.optimizer.param_groups[0]['lr']:.8f}"
            logger.info(f"Epoch {epoch+1} completed. "
                        f"Avg Total Loss: {avg_total_loss:.4f}, "
                        f"Avg Mel Loss: {avg_mel_loss:.4f}, "
                        f"Avg Dur Loss: {avg_dur_loss:.4f}, "
                        f"Avg Stop Loss: {avg_stop_loss:.4f}, "
                        f"{_lr_str}")

            # Log epoch-level train losses to TensorBoard.
            # Use optimizer_steps_completed (not epoch index) so these points
            # land on the same x-axis as all per-step training scalars and
            # can be visually compared with validation epoch scalars.
            _epoch_step = self.optimizer_steps_completed
            self.writer.add_scalar('loss/train_total_epoch', avg_total_loss, _epoch_step)
            self.writer.add_scalar('loss/train_mel_epoch', avg_mel_loss, _epoch_step)
            self.writer.add_scalar('loss/train_duration_epoch', avg_dur_loss, _epoch_step)
            self.writer.add_scalar('loss/train_stop_epoch', avg_stop_loss, _epoch_step)
            self.writer.flush()

            # Log weight histograms once per epoch
            self._log_histograms_epoch(epoch)

            self._log_epoch_cache_delta(epoch, "train", self.dataset, train_cache_start)

            # Run validation if enabled
            current_val_loss = None
            val_mel_loss = val_dur_loss = val_stop_loss = None
            validation_interval = getattr(self.config, 'validation_interval', 1)
            if self.val_dataloader is not None and (epoch + 1) % validation_interval == 0:
                val_cache_start = self._snapshot_cache_stats(self.val_dataset)
                val_metrics = self.validate_epoch(epoch)
                val_total_loss = val_metrics.total_loss
                val_mel_loss = val_metrics.mel_loss
                val_dur_loss = val_metrics.dur_loss
                val_stop_loss = val_metrics.stop_loss
                current_val_loss = val_total_loss
                self._log_epoch_cache_delta(epoch, "val", self.val_dataset, val_cache_start)
                self.validation_losses.append(val_total_loss)

                # Log epoch-level val losses to TensorBoard.
                # Use optimizer_steps_completed so val points align with the
                # per-step training scalars on the same x-axis.
                self.writer.add_scalar('loss/val_total_epoch', val_total_loss, self.optimizer_steps_completed)
                self.writer.add_scalar('loss/val_mel_epoch', val_mel_loss, self.optimizer_steps_completed)
                self.writer.add_scalar('loss/val_duration_epoch', val_dur_loss, self.optimizer_steps_completed)
                self.writer.add_scalar('loss/val_stop_epoch', val_stop_loss, self.optimizer_steps_completed)
                self.writer.flush()

                # Check for improvement
                min_delta = getattr(self.config, 'early_stopping_min_delta', 0.001)
                if val_total_loss < (self.best_val_loss - min_delta):
                    improvement = self.best_val_loss - val_total_loss
                    self.best_val_loss = val_total_loss
                    self.best_val_epoch = epoch
                    self.epochs_without_improvement = 0
                    logger.info(f"✓ Validation loss improved by {improvement:.4f} - saving best model")

                    # Save best model
                    self.save_checkpoint_with_scaler(
                        epoch, avg_total_loss, val_loss=val_total_loss,
                        val_mel_loss=val_mel_loss, val_stop_loss=val_stop_loss, val_dur_loss=val_dur_loss,
                        best_val_loss=self.best_val_loss, best_val_epoch=self.best_val_epoch,
                    )
                else:
                    self.epochs_without_improvement += 1
                    logger.info(f"⚠ No validation improvement for {self.epochs_without_improvement} epoch(s) "
                               f"(best: {self.best_val_loss:.4f})")

                # Early stopping check
                patience = getattr(self.config, 'early_stopping_patience', 10)
                if self.epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                    break

            # Log memory management stats for this epoch
            if self.enable_adaptive_memory:
                memory_report = self.memory_manager.get_memory_report()
                epoch_cleanup_count = self.memory_manager.cleanup_count - epoch_cleanup_count_start
                epoch_cleanup_time = self.memory_manager.total_cleanup_time - epoch_cleanup_time_start
                epoch_duration = max(1e-6, time.time() - epoch_start_time)
                epoch_cleanup_overhead_percent = (epoch_cleanup_time / epoch_duration) * 100

                epoch_memory_history = self.memory_manager.memory_history[epoch_memory_history_start:]
                epoch_memory_trend = 0.0
                if len(epoch_memory_history) >= 10:
                    recent = epoch_memory_history[-10:]
                    old_avg = sum(m['usage_percent'] for m in recent[:5]) / 5
                    new_avg = sum(m['usage_percent'] for m in recent[5:]) / 5
                    epoch_memory_trend = new_avg - old_avg
                elif len(epoch_memory_history) >= 2:
                    epoch_memory_trend = epoch_memory_history[-1]['usage_percent'] - epoch_memory_history[0]['usage_percent']

                logger.info(f"Memory Management Summary - Epoch {epoch+1}:")
                logger.info(f"  Current Pressure: {memory_report['current_pressure']}")
                logger.info(f"  Cleanups This Epoch: {epoch_cleanup_count}")
                logger.info(f"  Epoch Memory Trend: {epoch_memory_trend:+.2f}%")
                logger.info(f"  Cleanup Overhead: {epoch_cleanup_overhead_percent:.2f}%")

            # Save periodic checkpoints (only if not using validation or if validation isn't better)
            if (epoch + 1) % self.config.save_every == 0:
                # Only save if we're not doing validation or if this is just a periodic save
                should_save_periodic = (self.val_dataloader is None or
                                       self.epochs_without_improvement > 0)
                if should_save_periodic:
                    self.save_checkpoint_with_scaler(
                        epoch, avg_total_loss, val_loss=current_val_loss,
                        val_mel_loss=val_mel_loss, val_stop_loss=val_stop_loss, val_dur_loss=val_dur_loss,
                        best_val_loss=self.best_val_loss, best_val_epoch=self.best_val_epoch,
                    )
                    logger.info(f"Periodic checkpoint saved for epoch {epoch+1}")

            # Strategic memory cleanup at epoch end
            if self.enable_adaptive_memory:
                self.memory_manager.adaptive_cleanup(epoch * len(self.dataloader), force=True)
            else:
                self.clear_device_cache()

        logger.info("Training finished. Saving final model.")
        save_final_model(self.model, self.config, self.config.output_dir)

        # Print validation summary if validation was enabled
        if self.val_dataloader is not None and len(self.validation_losses) > 0:
            logger.info("\n" + "="*60)
            logger.info("VALIDATION SUMMARY")
            logger.info("="*60)
            logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
            logger.info(f"Final Validation Loss: {self.validation_losses[-1]:.4f}")
            logger.info(f"Total Validation Runs: {len(self.validation_losses)}")

            # Check for overfitting
            if len(self.validation_losses) >= 2:
                # Compare final vs best
                if self.validation_losses[-1] > self.best_val_loss:
                    difference = self.validation_losses[-1] - self.best_val_loss
                    pct_increase = (difference / self.best_val_loss) * 100
                    logger.info(f"⚠ Potential overfitting detected:")
                    logger.info(f"  Final validation loss is {pct_increase:.1f}% higher than best")
                    logger.info(f"  Consider using the checkpoint from epoch with best validation loss")
                else:
                    logger.info("✓ No significant overfitting detected")

            logger.info("="*60 + "\n")

        # Print cumulative feature cache summary (if enabled)
        cache_splits = [
            ("train", self.dataset),
            ("val", self.val_dataset),
        ]
        cache_logged = False
        for split_name, dataset in cache_splits:
            stats = self._snapshot_cache_stats(dataset)
            if stats is None:
                continue

            if not cache_logged:
                logger.info("=" * 60)
                logger.info("FEATURE CACHE SUMMARY")
                logger.info("=" * 60)
                cache_logged = True

            # Display 'N/A' for mem latency when there are no in-memory entries/hits
            mem_hits = int(stats.get('mem_hits', 0))
            mem_latency = stats.get('mem_latency_ms_avg', 0.0)
            mem_lat_display = "N/A" if mem_hits == 0 or mem_latency == 0.0 else f"{mem_latency:.3f}"
            disk_hits = int(stats.get('disk_hits', 0))
            disk_latency = stats.get('disk_latency_ms_avg', 0.0)
            disk_lat_display = "N/A" if disk_hits == 0 or disk_latency == 0.0 else f"{disk_latency:.3f}"

            logger.info(
                "%s cache cumulative: requests=%d hits=%d (mem=%d,disk=%d) misses=%d hit_rate=%.1f%% in_mem_entries=%d in_mem_size=%.1fMB mem_lat_ms=%s disk_lat_ms=%s",
                split_name,
                stats['requests'],
                stats['hits'],
                stats['mem_hits'],
                stats['disk_hits'],
                stats['misses'],
                stats['hit_rate'],
                stats['in_mem_entries'],
                stats['in_mem_mb'],
                mem_lat_display,
                disk_lat_display,
            )

        if cache_logged:
            logger.info("=" * 60)

        # Print final mixed precision statistics
        if self.use_mixed_precision:
            mp_stats = self.mixed_precision_stats
            total_steps = mp_stats['successful_steps'] + mp_stats.get('skipped_steps', 0) + mp_stats['overflow_count']
            if total_steps > 0:
                success_rate = (mp_stats['successful_steps'] / total_steps) * 100
                logger.info(f"Final Mixed Precision Statistics ({self.device_type.upper()}):")
                logger.info(f"  Total Steps: {total_steps}")
                logger.info(f"  Successful Steps: {mp_stats['successful_steps']}")
                logger.info(f"  Skipped Steps: {mp_stats.get('skipped_steps', 0)}")
                logger.info(f"  Overflow Count: {mp_stats['overflow_count']}")
                logger.info(f"  Success Rate: {success_rate:.1f}%")
                logger.info(f"  Scale Updates: {mp_stats['scale_updates']}")
                logger.info(f"  Scale Decreases: {mp_stats['scale_decreases']}")

        # Print final memory management report
        if self.enable_adaptive_memory:
            logger.info("Final Memory Management Report:")
            self.memory_manager.print_report()

        # TODO: Add support for SIGKILL
        self.writer.close()
        logger.info("TensorBoard writer closed.")

    def _time_training_loop(self, use_amp: bool, num_batches: int = 10) -> float:
        """
        Time a training loop with or without AMP

        Args:
            use_amp: Whether to use automatic mixed precision
            num_batches: Number of batches to time

        Returns:
            Total time in seconds for the specified batches
        """
        self.model.train()
        total_time = 0.0
        batches_processed = 0

        # Save current AMP state to restore later
        original_amp_state = self.use_mixed_precision
        original_scaler = self.scaler

        # Set AMP state for this timing run
        self.use_mixed_precision = use_amp
        if use_amp and original_scaler is None:
            # Create temporary scaler if needed
            if self.device.type == DeviceType.CUDA.value:
                self.scaler = torch.amp.GradScaler('cuda')
            elif self.device.type == DeviceType.MPS.value:
                self.scaler = MPSGradScaler()
        elif not use_amp:
            self.scaler = None

        logger.info(f"Timing training loop {'WITH' if use_amp else 'WITHOUT'} AMP for {num_batches} batches...")

        # Warmup: run 2 batches to avoid cold start effects
        warmup_batches = min(2, num_batches // 2)
        batch_iter = iter(self.dataloader)

        for _ in range(warmup_batches):
            try:
                batch = next(batch_iter)
                self._run_single_batch(batch, measure_time=False)
            except StopIteration:
                batch_iter = iter(self.dataloader)
                batch = next(batch_iter)
                self._run_single_batch(batch, measure_time=False)

        # Actual timing
        start_time = time.time()

        for _ in range(num_batches):
            try:
                batch = next(batch_iter)
                self._run_single_batch(batch, measure_time=False)
                batches_processed += 1
            except StopIteration:
                batch_iter = iter(self.dataloader)
                batch = next(batch_iter)
                self._run_single_batch(batch, measure_time=False)
                batches_processed += 1

        # Ensure all GPU operations are complete
        if self.device.type == DeviceType.CUDA.value:
            torch.cuda.synchronize()
        elif self.device.type == DeviceType.MPS.value:
            torch.mps.synchronize()

        total_time = time.time() - start_time

        # Restore original AMP state
        self.use_mixed_precision = original_amp_state
        self.scaler = original_scaler

        avg_time_per_batch = total_time / batches_processed if batches_processed > 0 else 0
        logger.info(f"  Processed {batches_processed} batches in {total_time:.2f}s ({avg_time_per_batch:.3f}s per batch)")

        return total_time

    def _execute_training_step(
        self,
        transferred: BatchOnDevice,
        mel_for_model: torch.Tensor,
        loss_scale: float = 1.0,
    ) -> Optional[StepResult]:
        """Core forward + pitch/energy averaging + loss + backward step.

        Handles finite-output and finite-loss guards internally, logging errors
        and returning ``None`` when a batch must be skipped.  Does **not**
        perform ``zero_grad()``, ``optimizer.step()``, gradient-accumulation
        bookkeeping, or profiling annotations — those remain with the caller so
        that ``train_epoch`` can wrap them with its extra instrumentation while
        ``_run_single_batch`` uses the simpler path.

        Args:
            transferred:   Pre-transferred batch tensors (output of
                           ``_transfer_batch_to_device``).
            mel_for_model: Mel spectrogram fed to the model; may differ from
                           ``transferred['mel_specs']`` when SpecAugment is
                           active.
            loss_scale:    Scalar applied to ``total_loss`` before
                           ``.backward()`` — combine ``1/grad_accum_steps``
                           and ``adaptive_loss_scale`` in the caller.

        Returns:
            A dict with keys ``total_loss``, ``loss_mel``, ``loss_duration``,
            ``loss_stop_token``, ``loss_pitch``, ``loss_energy``,
            ``predicted_mel``, ``predicted_log_durations``,
            ``predicted_pitch``, ``predicted_energy``; or ``None`` if any
            output or loss is non-finite (caller should skip the batch).
        """
        mel_specs          = transferred.mel_specs
        phoneme_indices    = transferred.phoneme_indices
        phoneme_durations  = transferred.phoneme_durations
        stop_token_targets = transferred.stop_token_targets
        mel_lengths        = transferred.mel_lengths
        phoneme_lengths    = transferred.phoneme_lengths
        pitches            = transferred.pitches
        energies           = transferred.energies
        stress_indices     = transferred.stress_indices

        # --- Forward pass -------------------------------------------------
        with self.get_autocast_context():
            predicted_mel, predicted_log_durations, predicted_stop_logits, \
                predicted_pitch, predicted_energy = self.model(
                    phoneme_indices, mel_for_model, phoneme_durations,
                    stop_token_targets, pitch_targets=pitches,
                    energy_targets=energies, stress_indices=stress_indices,
                )

        # --- Finite-output guard ------------------------------------------
        if not (
            torch.isfinite(predicted_mel).all()
            and torch.isfinite(predicted_log_durations).all()
            and torch.isfinite(predicted_stop_logits).all()
            and (predicted_pitch is None or torch.isfinite(predicted_pitch).all())
            and (predicted_energy is None or torch.isfinite(predicted_energy).all())
        ):
            logger.error(
                f"Non-finite model outputs — skipping batch "
                f"(mel={torch.isfinite(predicted_mel).all().item()}, "
                f"dur={torch.isfinite(predicted_log_durations).all().item()}, "
                f"stop={torch.isfinite(predicted_stop_logits).all().item()}, "
                f"pitch={True if predicted_pitch is None else torch.isfinite(predicted_pitch).all().item()}, "
                f"energy={True if predicted_energy is None else torch.isfinite(predicted_energy).all().item()})"
            )
            if not torch.isfinite(predicted_mel).all():
                logger.error(f"predicted_mel: {torch.isnan(predicted_mel).sum()} NaNs, {torch.isinf(predicted_mel).sum()} Infs")
            if not torch.isfinite(predicted_log_durations).all():
                logger.error(f"predicted_log_durations: {torch.isnan(predicted_log_durations).sum()} NaNs, {torch.isinf(predicted_log_durations).sum()} Infs")
            if predicted_pitch is not None and not torch.isfinite(predicted_pitch).all():
                logger.error(f"predicted_pitch: {torch.isnan(predicted_pitch).sum()} NaNs, {torch.isinf(predicted_pitch).sum()} Infs")
            if predicted_energy is not None and not torch.isfinite(predicted_energy).all():
                logger.error(f"predicted_energy: {torch.isnan(predicted_energy).sum()} NaNs, {torch.isinf(predicted_energy).sum()} Infs")
            return None

        # --- Loss calculation ---------------------------------------------
        # Pass frame-level pitch/energy targets directly to losses.py, which
        # detects they are already frame-level and skips phoneme expansion.
        # Previously this block averaged targets to phoneme-level first, then
        # losses.py expanded them back — creating phoneme-constant targets that
        # prevented the pitch predictor from learning intra-phoneme variation.
        with self.get_autocast_context():
            total_loss, loss_mel, loss_duration, loss_stop_token, \
                loss_pitch, loss_energy = self._calculate_losses(
                    predicted_mel, predicted_log_durations, predicted_stop_logits,
                    mel_specs, phoneme_durations, stop_token_targets,
                    mel_lengths, phoneme_lengths,
                    predicted_pitch, predicted_energy, pitches, energies,
                )

        # --- Finite-loss guard -------------------------------------------
        if not (
            torch.isfinite(total_loss)
            and torch.isfinite(loss_mel)
            and torch.isfinite(loss_duration)
            and torch.isfinite(loss_stop_token)
            and (loss_pitch is None or torch.isfinite(loss_pitch))
            and (loss_energy is None or torch.isfinite(loss_energy))
        ):
            logger.error(
                f"Non-finite loss — skipping batch "
                f"(total={torch.isfinite(total_loss).item()}, "
                f"mel={torch.isfinite(loss_mel).item()}, "
                f"dur={torch.isfinite(loss_duration).item()}, "
                f"stop={torch.isfinite(loss_stop_token).item()}, "
                f"pitch={True if loss_pitch is None else torch.isfinite(loss_pitch).item()}, "
                f"energy={True if loss_energy is None else torch.isfinite(loss_energy).item()})"
            )
            logger.error(
                f"Loss values: total={total_loss}, mel={loss_mel}, "
                f"dur={loss_duration}, stop={loss_stop_token}, "
                f"pitch={loss_pitch}, energy={loss_energy}"
            )
            return None

        # --- Backward pass -----------------------------------------------
        if self.scaler is not None:
            self.scaler.scale(total_loss * loss_scale).backward()
        else:
            (total_loss * loss_scale).backward()

        return StepResult(
            total_loss=total_loss,
            loss_mel=loss_mel,
            loss_duration=loss_duration,
            loss_stop_token=loss_stop_token,
            loss_pitch=loss_pitch,
            loss_energy=loss_energy,
            predicted_mel=predicted_mel,
            predicted_log_durations=predicted_log_durations,
            predicted_pitch=predicted_pitch,
            predicted_energy=predicted_energy,
        )

    def _optimizer_step_with_clipping(
        self,
        clip_norm: float,
        *,
        step_scheduler: bool,
        update_ema: bool,
    ) -> Tuple[bool, float]:
        runtime_policy = getattr(self, 'runtime_step_policy', None)
        if runtime_policy is None:
            runtime_policy = RuntimeStepPolicy(logger=logger)
            self.runtime_step_policy = runtime_policy

        return runtime_policy.optimizer_step_with_clipping(
            model=self.model,
            optimizer=self.optimizer,
            use_mixed_precision=self.use_mixed_precision,
            device_type=self.device_type,
            scaler=self.scaler,
            mixed_precision_stats=self.mixed_precision_stats,
            clip_norm=clip_norm,
            step_scheduler=step_scheduler,
            scheduler_per_batch=self.scheduler_per_batch,
            step_scheduler_fn=self._step_scheduler_with_warmup,
            update_ema=update_ema,
            update_ema_fn=self._update_ema,
        )

    @staticmethod
    def _effective_accumulation_divisor(
        *,
        gradient_accumulation_steps: int,
        accumulated_step: int,
        batch_idx: int,
        num_batches: int,
    ) -> int:
        """Return the correct loss divisor for gradient accumulation.

        For normal full accumulation windows this equals
        ``gradient_accumulation_steps``. For the final partial window at the end
        of an epoch, this returns the exact number of micro-batches that will
        contribute to the upcoming optimizer step.
        """
        total_target = max(1, int(gradient_accumulation_steps))
        remaining_including_current = max(1, int(num_batches) - int(batch_idx))
        already_accumulated = max(0, int(accumulated_step))
        return max(1, min(total_target, already_accumulated + remaining_including_current))

    @staticmethod
    def _cap_batch_sequence_dimensions(
        *,
        transferred: BatchOnDevice,
        mel_for_model: torch.Tensor,
        max_mel_len: int,
        max_phoneme_len: int,
    ) -> Tuple[BatchOnDevice, torch.Tensor, Dict[str, Union[int, bool]]]:
        """Cap oversized mel/phoneme dimensions in a batch and keep training.

        Returns updated ``(transferred, mel_for_model, info)`` where ``info``
        records original/new lengths and whether truncation occurred.
        """
        info: Dict[str, Union[int, bool]] = {
            'truncated': False,
            'orig_mel_len': int(transferred.mel_specs.size(1)),
            'new_mel_len': int(transferred.mel_specs.size(1)),
            'orig_phoneme_len': int(transferred.phoneme_indices.size(1)),
            'new_phoneme_len': int(transferred.phoneme_indices.size(1)),
        }

        mel_cap = max(1, int(max_mel_len))
        phoneme_cap = max(1, int(max_phoneme_len))

        if transferred.mel_specs.size(1) > mel_cap:
            info['truncated'] = True
            info['new_mel_len'] = mel_cap

            transferred.mel_specs = transferred.mel_specs[:, :mel_cap, :]
            mel_for_model = mel_for_model[:, :mel_cap, :]
            transferred.stop_token_targets = transferred.stop_token_targets[:, :mel_cap]
            transferred.mel_lengths = transferred.mel_lengths.clamp(max=mel_cap)
            if transferred.pitches is not None:
                transferred.pitches = transferred.pitches[:, :mel_cap]
            if transferred.energies is not None:
                transferred.energies = transferred.energies[:, :mel_cap]

        if transferred.phoneme_indices.size(1) > phoneme_cap:
            info['truncated'] = True
            info['new_phoneme_len'] = phoneme_cap

            transferred.phoneme_indices = transferred.phoneme_indices[:, :phoneme_cap]
            transferred.phoneme_durations = transferred.phoneme_durations[:, :phoneme_cap]
            transferred.phoneme_lengths = transferred.phoneme_lengths.clamp(max=phoneme_cap)
            if transferred.stress_indices is not None:
                transferred.stress_indices = transferred.stress_indices[:, :phoneme_cap]

        return transferred, mel_for_model, info

    def _run_single_batch(self, batch: Dict, measure_time: bool = False) -> Optional[float]:
        """
        Run a single training batch

        Args:
            batch: Batch data dictionary
            measure_time: Whether to return timing information

        Returns:
            Time taken if measure_time is True, else None
        """
        start_time = time.time() if measure_time else None

        transferred = self._transfer_batch_to_device(batch)
        self.optimizer.zero_grad()

        step_result = self._execute_training_step(transferred, transferred.mel_specs)

        if step_result is not None:
            self._optimizer_step_with_clipping(
                clip_norm=0.5,
                step_scheduler=False,
                update_ema=False,
            )

        if measure_time:
            return time.time() - start_time
        return None

    def profile_amp_benefits(self, num_batches: int = 10) -> Dict[str, float]:
        """
        Compare training speed with and without AMP

        Args:
            num_batches: Number of batches to test (default: 10)

        Returns:
            Dictionary with timing results and speedup
        """
        logger.info("\n" + "="*60)
        logger.info("AUTOMATIC MIXED PRECISION (AMP) PROFILING")
        logger.info("="*60)
        logger.info(f"Device: {self.device.type.upper()}")
        logger.info(f"Testing with {num_batches} batches")
        logger.info(f"Mixed precision dtype: {self.mixed_precision_dtype}")

        # Check if AMP is supported
        if self.device.type == DeviceType.CPU.value:
            logger.warning("AMP is not supported on CPU. Skipping profiling.")
            return {'with_amp': 0.0, 'without_amp': 0.0, 'speedup': 1.0, 'supported': False}

        # Run without AMP first (baseline)
        logger.info("\n[1/2] Running WITHOUT AMP (baseline)...")
        without_amp_time = self._time_training_loop(use_amp=False, num_batches=num_batches)

        # Clear cache between runs
        self.clear_device_cache()

        # Run with AMP
        logger.info("\n[2/2] Running WITH AMP...")
        with_amp_time = self._time_training_loop(use_amp=True, num_batches=num_batches)

        # Calculate speedup
        speedup = without_amp_time / with_amp_time if with_amp_time > 0 else 0.0

        # Print results
        logger.info("\n" + "="*60)
        logger.info("AMP PROFILING RESULTS")
        logger.info("="*60)
        logger.info(f"Without AMP: {without_amp_time:.2f}s ({without_amp_time/num_batches:.3f}s per batch)")
        logger.info(f"With AMP:    {with_amp_time:.2f}s ({with_amp_time/num_batches:.3f}s per batch)")
        logger.info(f"Speedup:     {speedup:.2f}x")

        if speedup > 1.2:
            logger.info(f"✓ AMP provides significant speedup ({speedup:.2f}x faster)")
            logger.info("  Recommendation: Keep AMP enabled (use_mixed_precision=True)")
        elif speedup > 1.05:
            logger.info(f"✓ AMP provides modest speedup ({speedup:.2f}x faster)")
            logger.info("  Recommendation: AMP is beneficial, keep it enabled")
        elif speedup > 0.95:
            logger.info(f"≈ AMP has minimal impact ({speedup:.2f}x)")
            logger.info("  Recommendation: AMP overhead is negligible, can keep enabled")
        else:
            logger.info(f"⚠ AMP is slower ({speedup:.2f}x)")
            logger.info("  Recommendation: Consider disabling AMP (use_mixed_precision=False)")
            logger.info("  Note: This can happen on some architectures or with small models")

        logger.info("="*60 + "\n")

        return {
            'with_amp': with_amp_time,
            'without_amp': without_amp_time,
            'speedup': speedup,
            'supported': True
        }

    def profile_training_steps(self, num_steps: int = 10):
        """Profile a specific number of training steps with mixed precision support and adaptive memory management"""
        logger.info(f"Starting profiling for {num_steps} training steps on {self.device.type}")

        self.reset_profiling_stats()
        self.start_torch_profiler()

        self.model.train()
        total_time = 0
        step_count = 0

        for batch_idx, batch in enumerate(self.dataloader):
            if step_count >= num_steps:
                break

            start_time = time.time()

            try:
                self.interbatch_profiler.start_batch()
                cleanup_result = self.adaptive_memory_cleanup(batch_idx)
                self.profile_step()

                self.interbatch_profiler.start_data_loading()
                with torch.profiler.record_function("Data_Loading"):
                    transferred = self._transfer_batch_to_device(batch)
                    mel_specs = transferred.mel_specs
                self.interbatch_profiler.end_data_loading()
                self.log_memory_stats("data_loading")

                with torch.profiler.record_function("Zero_Grad"):
                    self.optimizer.zero_grad()

                self.interbatch_profiler.start_forward_pass()
                self.interbatch_profiler.start_backward_pass()
                with torch.profiler.record_function("Training_Step"):
                    step_result = self._execute_training_step(
                        transferred,
                        mel_specs,
                        loss_scale=1.0,
                    )
                self.interbatch_profiler.end_forward_pass()
                self.interbatch_profiler.end_backward_pass()

                if step_result is None:
                    if self.enable_adaptive_memory:
                        self.memory_manager.emergency_cleanup()
                    else:
                        self.clear_device_cache()
                    self.interbatch_profiler.end_batch(mel_specs.size(0))
                    continue

                self.log_memory_stats("training_step")

                with torch.profiler.record_function("Optimizer_Step"):
                    self._optimizer_step_with_clipping(
                        clip_norm=0.5,
                        step_scheduler=False,
                        update_ema=False,
                    )
                self.log_memory_stats("optimizer_step")

                self.interbatch_profiler.end_batch(mel_specs.size(0))

                step_time = time.time() - start_time
                total_time += step_time
                step_count += 1

                if step_count % 2 == 0:
                    memory_info = f", Mem: {cleanup_result.get('pressure_level', 'unknown')}" if self.enable_adaptive_memory else ""
                    logger.info(f"Profiling Step {step_count}, Time: {step_time:.3f}s{memory_info}")

            except Exception as e:
                logger.error(f"Error in profiling step {step_count}: {e}")
                if self.enable_adaptive_memory:
                    self.memory_manager.emergency_cleanup()
                else:
                    self.clear_device_cache()
                continue

        self.stop_torch_profiler()

        report = self.get_profiling_report()
        logger.info(f"Training profiling completed. Total time: {total_time:.2f}s, "
                   f"Avg time per step: {total_time/step_count:.3f}s")
        self.analyze_profiling_results(report)

        if self.enable_adaptive_memory:
            logger.info("Memory Management Report during profiling:")
            self.memory_manager.print_report()

        return report

def train_model(config: TrainingConfig):
    """Main training function - backward compatibility wrapper"""
    trainer = KokoroTrainer(config)
    trainer.train()


# Example usage (if running train.py directly)
if __name__ == "__main__":
    class TrainingConfig:
        def __init__(self):
            self.data_dir = "data/processed_data"
            self.output_dir = "output_models"
            self.num_epochs = 100
            self.batch_size = 16
            self.learning_rate = 1e-4
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.lr_T_0 = 20
            self.lr_T_mult = 2
            self.lr_eta_min = 1e-6
            self.save_every = 5
            self.resume_checkpoint = 'auto'
            self.n_mels = 80
            self.hidden_dim = 768
            self.duration_loss_weight = 0.1
            self.stop_token_loss_weight = 1.0
            self.max_seq_length = 2500
            self.sample_rate = 22050
            self.hop_length = 256
            self.win_length = 1024
            self.n_fft = 1024
            self.f_min = 0.0
            self.f_max = 8000.0
            self.num_workers = 1
            self.pin_memory = False # Pin memory only for CUDA, automatically disabled for MPS

            # Enhanced profiler configurations
            self.enable_profiling = False
            self.profile_epoch_start = 1  # Start profiling from this epoch (0-indexed)
            self.profile_wait_steps = 1  # Number of steps to wait before starting warmup
            self.profile_warmup_steps = 1 # Number of steps to warm up the profiler
            self.profile_steps = 5       # Number of active steps to profile
            self.run_standalone_profiling = False  # Run standalone profiling before training

            # Interbatch profiling configurations
            self.enable_interbatch_profiling = False  # Enable interbatch profiling
            self.interbatch_report_interval = 100    # Report interbatch stats every N batches

            # Mixed precision training configurations
            self.use_mixed_precision = True  # Enable mixed precision training (CUDA and MPS)
            self.mixed_precision_dtype = torch.float16  # Mixed precision dtype (float16 or bfloat16)
            self.amp_init_scale = 65536.0    # Initial scale for GradScaler
            self.amp_growth_factor = 2.0     # Scale growth factor
            self.amp_backoff_factor = 0.5    # Scale backoff factor when overflow detected
            self.amp_growth_interval = 2000  # Steps between scale growth attempts

            # Optimizer configurations
            self.weight_decay = 0.02
            self.adam_eps = 1e-8
            self.adam_betas = (0.9, 0.999)

            # Adaptive memory management configurations
            self.enable_adaptive_memory = True   # Enable adaptive memory management
            self.memory_report_interval = 50    # Report memory stats every N batches

    temp_config = TrainingConfig()

    # Device-specific adjustments
    if temp_config.device == DeviceType.MPS.value:
        print("Configuring for MPS (Apple Silicon) training:")
        print("  - Pin memory disabled for MPS")
        print("  - Mixed precision with custom MPS scaler")
        print("  - Adaptive memory management optimized for unified memory")
        print("  - Reduced batch size recommended for MPS")
        # Optionally reduce batch size for MPS
        temp_config.batch_size = max(8, temp_config.batch_size // 2)
        print(f"  - Adjusted batch size to {temp_config.batch_size}")
    elif temp_config.device == DeviceType.CUDA.value:
        print("Configuring for CUDA training:")
        print("  - Pin memory enabled for CUDA")
        print("  - Mixed precision with CUDA native GradScaler")
        print("  - Adaptive memory management optimized for dedicated GPU memory")
        temp_config.pin_memory = True
    else:
        print("Configuring for CPU training:")
        print("  - Mixed precision disabled for CPU")
        print("  - Adaptive memory management optimized for system RAM")
        temp_config.use_mixed_precision = False

    print(f"Adaptive memory management: {'Enabled' if temp_config.enable_adaptive_memory else 'Disabled'}")

    train_model(temp_config)
