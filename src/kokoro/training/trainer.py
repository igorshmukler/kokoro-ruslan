#!/usr/bin/env python3
"""
Training logic for Kokoro Language Model with Enhanced Profiling, Mixed Precision, and Adaptive Memory Management
Extended to support mixed precision training on both CUDA and MPS devices with intelligent memory cleanup
"""

import os
import time
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

from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from kokoro.training.config import TrainingConfig
from kokoro.utils.device_type import DeviceType
from kokoro.data.dataset import RuslanDataset, collate_fn, LengthBasedBatchSampler
from kokoro.model.model import KokoroModel
from kokoro.training.checkpoint_manager import (
    save_phoneme_processor, load_checkpoint, find_latest_checkpoint,
    save_checkpoint, save_final_model
)
from kokoro.utils.interbatch_profiler import InterbatchProfiler
from kokoro.training.mps_grad_scaler import MPSGradScaler

from kokoro.utils.adaptive_memory_manager import AdaptiveMemoryManager


logger = logging.getLogger(__name__)


def check_mps_mixed_precision_support():
    """Check if MPS supports mixed precision training"""
    if not torch.backends.mps.is_available():
        return False

    # Check PyTorch version - MPS mixed precision was added in PyTorch 2.0+
    torch_version = torch.__version__.split('.')
    major, minor = int(torch_version[0]), int(torch_version[1])

    if major < 2:
        logger.info(f"MPS autocast requires PyTorch 2.0+, found {torch.__version__}")
        return False

    # Test if autocast works on MPS
    try:
        device = torch.device('mps')
        x = torch.randn(2, 2, device=device)
        # Try using the autocast API
        with torch.autocast(device_type='mps', dtype=torch.float16):
            y = torch.mm(x, x)
        logger.info("MPS autocast support verified")
        return True
    except (RuntimeError, AttributeError) as e:
        logger.warning(f"MPS autocast not supported: {e}")
        return False


class KokoroTrainer:
    """Main trainer class for the model with adaptive memory management"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Ensure fatal signals (including SIGABRT) dump Python stacks for crash triage
        try:
            if not faulthandler.is_enabled():
                faulthandler.enable(all_threads=True)
                logger.info("faulthandler enabled (all_threads=True)")
        except Exception as e:
            logger.warning(f"Could not enable faulthandler: {e}")

        # Initialize adaptive memory manager
        self.memory_manager = AdaptiveMemoryManager(self.device, config)

        # Initialize mixed precision training components
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', True)
        self.mixed_precision_dtype = getattr(config, 'mixed_precision_dtype', torch.float16)

        # Check device support for mixed precision
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

        # Precompute features if requested
        if getattr(config, 'precompute_features', False):
            logger.info("Pre-computing features for all samples...")
            from kokoro.cli.precompute_features import precompute_features
            # Set cache directory if not already set
            if not config.feature_cache_dir:
                from pathlib import Path
                config.feature_cache_dir = str(Path(config.data_dir) / ".feature_cache")
            precompute_features(config.data_dir, config, force_recompute=False)
            logger.info("Feature pre-computation complete")

        # Initialize datasets with train/validation split
        full_dataset = RuslanDataset(config.data_dir, config)

        # Create train/validation split
        val_split = getattr(config, 'validation_split', 0.1)
        if val_split > 0:
            dataset_size = len(full_dataset.samples)  # Total before filtering
            indices = list(range(dataset_size))

            # Shuffle indices for random split
            import random
            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(indices)

            split_idx = int(dataset_size * (1 - val_split))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            logger.info(f"Dataset split: {len(train_indices)} training, {len(val_indices)} validation samples")

            # Create separate datasets
            self.dataset = RuslanDataset(config.data_dir, config, indices=train_indices)
            self.val_dataset = RuslanDataset(config.data_dir, config, indices=val_indices)
        else:
            logger.info("No validation split - using all data for training")
            self.dataset = full_dataset
            self.val_dataset = None

        # Initialize batch sampler for training (dynamic or fixed)
        use_dynamic = getattr(config, 'use_dynamic_batching', True)

        if use_dynamic:
            from kokoro.data.dataset import DynamicFrameBatchSampler

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
            from kokoro.data.dataset import LengthBasedBatchSampler

            logger.info("Using fixed batch size")
            self.batch_sampler = LengthBasedBatchSampler(
                dataset=self.dataset,
                batch_size=config.batch_size,
                drop_last=True,
                shuffle=True
            )

        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,  # Avoid multiprocessing issues with audio loading
            pin_memory=config.pin_memory and self.device.type == DeviceType.CUDA.value,
            prefetch_factor=None,  # Not used with num_workers=0
            persistent_workers=False  # Not used with num_workers=0
        )

        # Initialize validation dataloader if validation set exists
        if self.val_dataset is not None:
            # Use dynamic batching for validation too
            if use_dynamic:
                from kokoro.data.dataset import DynamicFrameBatchSampler

                val_batch_sampler = DynamicFrameBatchSampler(
                    dataset=self.val_dataset,
                    max_frames=config.max_frames_per_batch,
                    min_batch_size=config.min_batch_size,
                    max_batch_size=config.max_batch_size,
                    drop_last=False,  # Use all validation data
                    shuffle=False  # Don't shuffle validation
                )
            else:
                from kokoro.data.dataset import LengthBasedBatchSampler

                val_batch_sampler = LengthBasedBatchSampler(
                    dataset=self.val_dataset,
                    batch_size=config.batch_size,
                    drop_last=False,  # Use all validation data
                    shuffle=False  # Don't shuffle validation
                )

            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_sampler=val_batch_sampler,
                collate_fn=collate_fn,
                num_workers=0,  # Avoid multiprocessing issues with audio loading
                pin_memory=config.pin_memory and self.device.type == DeviceType.CUDA.value,
                prefetch_factor=None,  # Not used with num_workers=0
                persistent_workers=False  # Not used with num_workers=0
            )
        else:
            self.val_dataloader = None

        # Initialize model
        vocab_size = self.dataset.phoneme_processor.get_vocab_size()
        self.model = KokoroModel(
            vocab_size,
            config.n_mels,
            config.hidden_dim,
            use_variance_predictor=getattr(config, 'use_variance_predictor', True),
            variance_filter_size=getattr(config, 'variance_filter_size', 256),
            variance_kernel_size=getattr(config, 'variance_kernel_size', 3),
            variance_dropout=getattr(config, 'variance_dropout', 0.1),
            n_variance_bins=getattr(config, 'n_variance_bins', 256),
            pitch_min=getattr(config, 'pitch_min', 50.0),
            pitch_max=getattr(config, 'pitch_max', 800.0),
            energy_min=getattr(config, 'energy_min', 0.0),
            energy_max=getattr(config, 'energy_max', 100.0),
            use_stochastic_depth=getattr(config, 'use_stochastic_depth', True),
            stochastic_depth_rate=getattr(config, 'stochastic_depth_rate', 0.1)
        )
        self.model.to(self.device)

        # Apply torch.compile for optimized training (PyTorch 2.0+)
        # Note: Disabled for MPS due to float64 support limitations
        use_compile = getattr(config, 'use_torch_compile', False)
        if use_compile and torch.__version__ >= '2.0':
            # Only enable torch.compile for CUDA - MPS has limited dtype support
            if self.device_type == 'cuda':
                try:
                    compile_mode = getattr(config, 'torch_compile_mode', 'reduce-overhead')
                    compile_dynamic = getattr(config, 'torch_compile_dynamic', True)

                    self.model = torch.compile(
                        self.model,
                        mode=compile_mode,
                        fullgraph=False,  # Allow graph breaks for complex model architecture
                        dynamic=compile_dynamic  # Better handling of dynamic shapes
                    )
                    logger.info(f"Model compiled with torch.compile (mode='{compile_mode}', dynamic={compile_dynamic})")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}. Training will proceed without compilation.")
            else:
                logger.info(f"torch.compile disabled for {self.device_type} (limited dtype support)")
        elif use_compile:
            logger.warning(f"torch.compile requested but PyTorch version is {torch.__version__} (requires 2.0+)")

        # Log model information
        model_info = self.model.get_model_info()
        logger.info(f"Model initialized with {model_info['total_parameters']:,} parameters ({model_info['model_size_mb']:.1f} MB)")

        # Initialize optimizer and loss functions
        default_use_fused = self.device_type == 'cuda' and torch.__version__ >= '2.0'
        forced_use_fused = getattr(config, 'use_fused_adamw', None)

        if forced_use_fused is None:
            use_fused = default_use_fused
        else:
            use_fused = bool(forced_use_fused)

        if self.device_type == 'mps' and getattr(config, 'try_fused_adamw_on_mps', False):
            use_fused = True

        fused_source = (
            'mps-opt-in' if (self.device_type == 'mps' and getattr(config, 'try_fused_adamw_on_mps', False))
            else 'auto' if forced_use_fused is None
            else 'forced'
        )
        logger.info(f"AdamW fused setting: effective={use_fused} (source={fused_source}, device={self.device_type})")

        optimizer_kwargs = {
            'lr': config.learning_rate,
            'weight_decay': getattr(config, 'weight_decay', 0.01),
            'eps': getattr(config, 'adam_eps', 1e-8),
            'betas': getattr(config, 'adam_betas', (0.9, 0.999)),
            'fused': use_fused,
        }

        try:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
            if use_fused:
                logger.info(f"Using fused AdamW optimizer on {self.device_type.upper()} (experimental on non-CUDA backends)")
            else:
                logger.info("Using standard AdamW optimizer")
        except (TypeError, ValueError, RuntimeError) as e:
            if use_fused:
                logger.warning(
                    f"Fused AdamW not available/supported on {self.device_type}: {e}. Falling back to standard AdamW."
                )
                optimizer_kwargs['fused'] = False
                self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
                logger.info("Using standard AdamW optimizer (fallback)")
            else:
                raise

        self.criterion_mel = nn.L1Loss(reduction='none')
        self.criterion_duration = nn.MSELoss(reduction='none')
        self.criterion_stop_token = nn.BCEWithLogitsLoss(reduction='none')

        # Variance losses (pitch and energy)
        if getattr(config, 'use_variance_predictor', True):
            self.criterion_pitch = nn.MSELoss(reduction='none')
            self.criterion_energy = nn.MSELoss(reduction='none')
            logger.info("Variance predictor losses initialized (pitch/energy)")
        else:
            self.criterion_pitch = None
            self.criterion_energy = None

        # Learning rate scheduler with warmup
        use_onecycle = getattr(config, 'use_onecycle_lr', True)
        if use_onecycle:
            # Calculate total training steps for OneCycleLR
            # With gradient accumulation, optimizer steps = batches / accumulation_steps
            steps_per_epoch = len(self.dataloader)
            gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
            optimizer_steps_per_epoch = (steps_per_epoch + gradient_accumulation_steps - 1) // gradient_accumulation_steps
            total_steps = config.num_epochs * optimizer_steps_per_epoch

            max_lr = config.learning_rate * getattr(config, 'max_lr_multiplier', 5.0)
            pct_start = getattr(config, 'pct_start', 0.3)

            # Warmup configuration
            self.use_warmup = getattr(config, 'use_warmup', True)
            self.warmup_steps = getattr(config, 'warmup_steps', 500)
            self.warmup_start_lr = config.learning_rate * getattr(config, 'warmup_start_lr_ratio', 0.01)
            self.warmup_target_lr = config.learning_rate  # Warmup ends at base learning_rate
            self.current_optimizer_step = 0  # Track optimizer steps (not batches)

            if self.use_warmup:
                # OneCycleLR starts after warmup, so adjust total_steps
                onecycle_steps = total_steps - self.warmup_steps
                logger.info(f"Linear warmup enabled: {self.warmup_steps} steps ({self.warmup_start_lr:.2e} → {self.warmup_target_lr:.2e})")
            else:
                onecycle_steps = total_steps

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                total_steps=onecycle_steps,
                pct_start=pct_start,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,  # initial_lr = max_lr / div_factor
                final_div_factor=10000.0,  # min_lr = initial_lr / final_div_factor
                last_epoch=-1  # Start from beginning
            )
            self.scheduler_per_batch = True
            logger.info(f"OneCycleLR scheduler initialized: max_lr={max_lr:.2e}, total_steps={onecycle_steps} "
                       f"(steps_per_epoch={optimizer_steps_per_epoch}, gradient_accumulation={gradient_accumulation_steps})")
        else:
            # Legacy CosineAnnealingWarmRestarts (no warmup)
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

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.validation_losses = []  # Track validation loss history

        # EMA (Exponential Moving Average) of model weights
        self.use_ema = getattr(config, 'use_ema', True)
        self.ema_decay = getattr(config, 'ema_decay', 0.9999)
        self.ema_update_every = getattr(config, 'ema_update_every', 1)
        self.ema_updates = 0  # Track number of EMA updates

        if self.use_ema:
            # Initialize EMA model as a deep copy of the current model
            # This ensures all model parameters and configuration are correctly copied
            self.ema_model = copy.deepcopy(self.model).to(self.device)
            self.ema_model.eval()  # EMA model is always in eval mode

            # Freeze EMA model (no gradients needed)
            for param in self.ema_model.parameters():
                param.requires_grad = False

            logger.info(f"EMA initialized: decay={self.ema_decay}, update_every={self.ema_update_every}")
        else:
            self.ema_model = None
            logger.info("EMA disabled")

        # Mixed precision training stats
        self.mixed_precision_stats = {
            'scale_updates': 0,
            'scale_decreases': 0,
            'overflow_count': 0,
            'successful_steps': 0,
            'skipped_steps': 0
        }

        # Enhanced profiler setup
        self.profiler = None
        self.profiling_stats = {}
        self.memory_snapshots = []
        self.log_dir = os.path.join(config.output_dir, "profiler_logs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize interbatch profiler
        self.interbatch_profiler = InterbatchProfiler(config)

        # Adaptive memory management configuration
        self.enable_adaptive_memory = getattr(config, 'enable_adaptive_memory', True)
        self.memory_report_interval = getattr(config, 'memory_report_interval', 500)

        # Gradient explosion tracking (adaptive threshold)
        self.grad_explosion_norm_ema = None
        self.grad_explosion_ema_alpha = getattr(config, 'grad_explosion_ema_alpha', 0.95)
        self.grad_explosion_abs_floor = getattr(config, 'grad_explosion_abs_floor', 1000.0)
        self.grad_explosion_multiplier = getattr(config, 'grad_explosion_multiplier', 3.0)
        self.grad_explosion_warmup_steps = getattr(config, 'grad_explosion_warmup_steps', 250)
        self.grad_explosion_warmup_floor = getattr(config, 'grad_explosion_warmup_floor', 5000.0)
        self.grad_explosion_min_ema_steps = getattr(config, 'grad_explosion_min_ema_steps', 50)
        self.grad_explosion_ema_steps = 0
        self.grad_explosion_streak = 0

        # Localized gradient spike mitigation for mel projection layers
        self.projection_spike_clip_norm = getattr(config, 'projection_spike_clip_norm', 50.0)
        self.attention_spike_clip_norm = getattr(config, 'attention_spike_clip_norm', 20.0)

        # Optimizer-step counter independent of scheduler mode
        self.optimizer_steps_completed = 0

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

            # Disable mixed precision to avoid repeated warnings
            self.use_mixed_precision = False

            if not self.use_mixed_precision:
                logger.warning("Mixed precision disabled on MPS due to backend bugs with fp16")
                logger.warning("Training will use fp32 on MPS (slower but stable)")

            return nullcontext()
        else:
            return nullcontext()

    def adaptive_memory_cleanup(self, batch_idx: int, force: bool = False) -> Dict[str, Any]:
        """Perform adaptive memory cleanup"""
        if self.enable_adaptive_memory:
            return self.memory_manager.adaptive_cleanup(batch_idx, force)
        else:
            # Fallback to original cleanup behavior
            if batch_idx % 200 == 0 and batch_idx > 0:
                self.clear_device_cache()
            return {'cleaned': False, 'pressure_level': 'disabled'}

    def handle_oom_with_adaptive_cleanup(self, batch_idx: int, error: Exception) -> bool:
        """
        Handle OOM error with adaptive cleanup
        Returns True if training should continue, False if unrecoverable
        """
        logger.error(f"OOM error at batch {batch_idx} on {self.device_type}: {error}")

        if self.enable_adaptive_memory:
            # Emergency cleanup
            cleanup_result = self.memory_manager.emergency_cleanup()

            # Log results
            if cleanup_result['success']:
                logger.info(f"Emergency cleanup freed {cleanup_result['memory_freed_mb']:.1f}MB")
                return True  # Try to continue
            else:
                logger.error("Emergency cleanup failed to free significant memory")
                return False  # Unrecoverable
        else:
            # Fallback emergency cleanup
            self.clear_device_cache()
            gc.collect()
            return True

    def print_memory_management_report(self):
        """Print comprehensive memory management report"""
        if self.enable_adaptive_memory:
            report = self.memory_manager.get_memory_report()

            print("\n" + "="*60)
            print("ADAPTIVE MEMORY MANAGEMENT REPORT")
            print("="*60)

            print(f"\nDevice: {report['device_type'].upper()}")
            print(f"Total Batches Processed: {report['total_batches']}")
            print(f"Total Cleanups Performed: {report['cleanup_count']}")
            print(f"Cleanup Frequency: {report['cleanup_frequency']:.4f} cleanups/batch")

            print(f"\nPerformance Impact:")
            print(f"  Total Cleanup Time: {report['total_cleanup_time_ms']:.1f}ms")
            print(f"  Average Cleanup Time: {report['avg_cleanup_time_ms']:.1f}ms")
            print(f"  Cleanup Overhead: {report['cleanup_overhead_percent']:.2f}%")

            print(f"\nMemory Status:")
            print(f"  Current Pressure Level: {report['current_pressure'].upper()}")
            print(f"  Current Usage: {report.get('current_memory_usage_percent', 0):.1f}%")
            print(f"  Average Usage: {report.get('avg_memory_usage_percent', 0):.1f}%")
            print(f"  Peak Usage: {report.get('max_memory_usage_percent', 0):.1f}%")
            print(f"  Memory Trend: {report['memory_trend']:+.2f}% (positive = increasing)")
            print(f"  Consecutive High Pressure Batches: {report['consecutive_high_pressure']}")

            print(f"\nRecommendations:")
            recommendations = []

            # Performance recommendations
            if report['cleanup_overhead_percent'] > 5.0:
                recommendations.append("• High cleanup overhead detected - consider optimizing cleanup frequency")

            if report['cleanup_frequency'] > 0.1:
                recommendations.append("• Very frequent cleanups - consider increasing batch size or reducing model size")

            # Memory recommendations
            if report.get('avg_memory_usage_percent', 0) > 85:
                recommendations.append("• High average memory usage - consider reducing batch size")
                if report['device_type'] == 'mps':
                    recommendations.append("• For MPS: Unified memory architecture may benefit from smaller batches")

            if report['memory_trend'] > 5.0:
                recommendations.append("• Memory usage increasing - potential memory leak or insufficient cleanup")

            if report['consecutive_high_pressure'] > 50:
                recommendations.append("• Sustained high memory pressure - consider model architecture optimization")

            # Device-specific recommendations
            if report['device_type'] == 'mps':
                recommendations.append("• MPS detected: Monitor for memory fragmentation in unified memory")
                if report.get('avg_memory_usage_percent', 0) > 70:
                    recommendations.append("• Consider using smaller batch sizes for MPS vs equivalent CUDA setup")
            elif report['device_type'] == 'cuda':
                if report['cleanup_frequency'] < 0.01:
                    recommendations.append("• CUDA: Low cleanup frequency may indicate room for batch size increase")

            if not recommendations:
                recommendations.append("• Memory management appears optimal for current configuration")

            for rec in recommendations:
                print(rec)

            print("="*60)
        else:
            logger.info("Adaptive memory management disabled")

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
            output_dir = self.log_dir

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

    def profile_step(self):
        """Step the profiler and log memory stats"""
        if self.profiler:
            self.profiler.step()

        # Log memory statistics based on device type
        current_memory = 0
        peak_memory = 0
        reserved_memory = 0
        total_memory = 0

        if self.device.type == DeviceType.CUDA.value:
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            reserved_memory = torch.cuda.memory_reserved() / 1024**2  # MB
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**2  # MB
        elif self.device.type == DeviceType.MPS.value:
            # MPS doesn't have detailed memory stats, use approximations
            try:
                current_memory = torch.mps.current_allocated_memory() / 1024**2  # MB
                peak_memory = current_memory  # MPS doesn't track peak separately
                reserved_memory = current_memory
                # Estimate total memory (this is approximate for Apple Silicon)
                total_memory = 8192  # Default estimate, could be made configurable
            except:
                # Fallback if MPS memory functions aren't available
                current_memory = peak_memory = reserved_memory = total_memory = 0

        self.memory_snapshots.append({
            'timestamp': time.time(),
            'current_memory_mb': current_memory,
            'peak_memory_mb': peak_memory,
            'reserved_memory_mb': reserved_memory,
            'total_memory_mb': total_memory,
            'scaler_scale': self.scaler.get_scale() if self.scaler else None
        })

    def log_memory_stats(self, stage_name: str):
        """Log memory statistics for a specific stage"""
        current_memory = 0
        peak_memory = 0

        if self.device.type == DeviceType.CUDA.value:
            current_memory = torch.cuda.memory_allocated() / 1024**2
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        elif self.device.type == DeviceType.MPS.value:
            try:
                current_memory = torch.mps.current_allocated_memory() / 1024**2
                peak_memory = current_memory
            except:
                current_memory = peak_memory = 0

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
        """Clip localized spikes in mel projection/attention gradients before global norm checks."""
        if self.projection_spike_clip_norm <= 0 and self.attention_spike_clip_norm <= 0:
            return {}

        projection_params = {
            'mel_projection_in.weight',
            'mel_projection_in.bias',
            'mel_projection_out.weight',
            'mel_projection_out.bias',
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

        clipped: Dict[str, Tuple[float, float]] = {}
        projection_max_norm = float(self.projection_spike_clip_norm)
        attention_max_norm = float(self.attention_spike_clip_norm)

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            max_norm = None
            if name in projection_params and projection_max_norm > 0:
                max_norm = projection_max_norm
            elif attention_max_norm > 0 and name.startswith('decoder.layers.') and any(fragment in name for fragment in attention_name_fragments):
                max_norm = attention_max_norm

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

    def _average_pitch_energy_by_duration(self,
                                          values: torch.Tensor,
                                          durations: torch.Tensor,
                                          phoneme_lengths: torch.Tensor) -> torch.Tensor:
        """
        Average frame-level values (pitch/energy) to phoneme-level using durations

        Args:
            values: Frame-level values (batch, mel_frames)
            durations: Phoneme durations (batch, phonemes)
            phoneme_lengths: Actual phoneme lengths (batch,)

        Returns:
            Phoneme-level averaged values (batch, phonemes)
        """
        batch_size = durations.shape[0]
        num_phonemes = durations.shape[1]
        device = durations.device

        # Create output tensor
        phoneme_values = torch.zeros(batch_size, num_phonemes, device=device)

        for b in range(batch_size):
            curr_durations = durations[b]  # (phonemes,)
            curr_values = values[b]  # (frames,)
            actual_phoneme_len = int(phoneme_lengths[b].item())

            frame_idx = 0

            for p in range(actual_phoneme_len):
                dur = int(curr_durations[p].item())
                if dur > 0 and frame_idx < len(curr_values):
                    # Average the values for this phoneme's frames
                    end_idx = min(frame_idx + dur, len(curr_values))
                    phoneme_values[b, p] = curr_values[frame_idx:end_idx].mean()
                    frame_idx = end_idx

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
        """
        if self.use_warmup and self.current_optimizer_step < self.warmup_steps:
            # Linear warmup phase
            warmup_progress = self.current_optimizer_step / self.warmup_steps
            current_lr = self.warmup_start_lr + (self.warmup_target_lr - self.warmup_start_lr) * warmup_progress

            # Set LR for all parameter groups
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            self.current_optimizer_step += 1
        else:
            # After warmup, use OneCycleLR or other scheduler
            self.scheduler.step()
            if self.use_warmup:
                self.current_optimizer_step += 1

    def _calculate_losses(self, predicted_mel, predicted_log_durations, predicted_stop_logits,
                         mel_specs, phoneme_durations, stop_token_targets,
                         mel_lengths, phoneme_lengths,
                         predicted_pitch=None, predicted_energy=None,
                         pitch_targets=None, energy_targets=None):
        """Calculate losses with optimized masking"""
        batch_size = mel_specs.size(0)
        max_mel_len = mel_specs.size(1)
        max_phoneme_len = phoneme_durations.size(1)
        n_mels = mel_specs.size(2)

        # Create base masks efficiently (2D only, no feature dimension expansion yet)
        # Use direct comparison without intermediate .expand() to save memory
        mel_mask_2d = torch.arange(max_mel_len, device=self.device).unsqueeze(0) < mel_lengths.unsqueeze(1)
        phoneme_mask_2d = torch.arange(max_phoneme_len, device=self.device).unsqueeze(0) < phoneme_lengths.unsqueeze(1)

        # Pre-compute mask normalization factors (reused across losses)
        mel_mask_count = mel_mask_2d.sum().clamp(min=1)
        phoneme_mask_count = phoneme_mask_2d.sum().clamp(min=1)

        # Mel Spectrogram Loss
        loss_mel_unreduced = self.criterion_mel(predicted_mel, mel_specs)
        mel_mask_3d = mel_mask_2d.unsqueeze(-1).expand_as(loss_mel_unreduced)
        mel_valid = mel_mask_3d & torch.isfinite(loss_mel_unreduced)
        if mel_valid.any():
            loss_mel = loss_mel_unreduced[mel_valid].mean()
        else:
            loss_mel = torch.tensor(0.0, device=self.device)

        # Duration Loss - convert to float once
        phoneme_mask_float = phoneme_mask_2d.float()
        target_log_durations = torch.log(phoneme_durations.float() + 1e-5)
        loss_duration_unreduced = self.criterion_duration(predicted_log_durations, target_log_durations)
        duration_valid = phoneme_mask_2d & torch.isfinite(loss_duration_unreduced)
        if duration_valid.any():
            loss_duration = loss_duration_unreduced[duration_valid].mean()
        else:
            loss_duration = torch.tensor(0.0, device=self.device)

        # Stop Token Loss - reuse mel_mask_2d directly (no need to slice from 3D)
        loss_stop_token_unreduced = self.criterion_stop_token(predicted_stop_logits, stop_token_targets)
        stop_valid = mel_mask_2d & torch.isfinite(loss_stop_token_unreduced)
        if stop_valid.any():
            loss_stop_token = loss_stop_token_unreduced[stop_valid].mean()
        else:
            loss_stop_token = torch.tensor(0.0, device=self.device)

        # Pitch Loss (if variance predictor enabled) - reuse phoneme mask
        loss_pitch = torch.tensor(0.0, device=self.device)
        if predicted_pitch is not None and pitch_targets is not None and self.criterion_pitch is not None:
            loss_pitch_unreduced = self.criterion_pitch(predicted_pitch, pitch_targets)
            pitch_valid = phoneme_mask_2d & torch.isfinite(loss_pitch_unreduced)
            if pitch_valid.any():
                loss_pitch = loss_pitch_unreduced[pitch_valid].mean()

        # Energy Loss (if variance predictor enabled) - reuse phoneme mask
        loss_energy = torch.tensor(0.0, device=self.device)
        if predicted_energy is not None and energy_targets is not None and self.criterion_energy is not None:
            loss_energy_unreduced = self.criterion_energy(predicted_energy, energy_targets)
            energy_valid = phoneme_mask_2d & torch.isfinite(loss_energy_unreduced)
            if energy_valid.any():
                loss_energy = loss_energy_unreduced[energy_valid].mean()

        # Monitor and auto-recover from variance predictor divergence
        variance_diverged = False
        if loss_pitch > 10.0 or loss_energy > 10.0:
            logger.warning(f"⚠️  Variance predictor divergence detected - pitch: {loss_pitch:.2f}, energy: {loss_energy:.2f}")

            # Log prediction vs target statistics for debugging
            if predicted_pitch is not None and pitch_targets is not None:
                pred_min, pred_max = predicted_pitch.min().item(), predicted_pitch.max().item()
                targ_min, targ_max = pitch_targets.min().item(), pitch_targets.max().item()
                logger.warning(f"   Pitch predictions: [{pred_min:.3f}, {pred_max:.3f}]")
                logger.warning(f"   Pitch targets: [{targ_min:.3f}, {targ_max:.3f}]")

            if predicted_energy is not None and energy_targets is not None:
                pred_min, pred_max = predicted_energy.min().item(), predicted_energy.max().item()
                targ_min, targ_max = energy_targets.min().item(), energy_targets.max().item()
                logger.warning(f"   Energy predictions: [{pred_min:.3f}, {pred_max:.3f}]")
                logger.warning(f"   Energy targets: [{targ_min:.3f}, {targ_max:.3f}]")

            logger.warning(f"   Auto-recovery: Resetting variance predictor weights and reducing loss contribution")

            # Reset variance predictor weights to initial state
            if hasattr(self.model, 'pitch_predictor') and self.model.pitch_predictor is not None:
                self.model.pitch_predictor._init_weights()
            if hasattr(self.model, 'energy_predictor') and self.model.energy_predictor is not None:
                self.model.energy_predictor._init_weights()

            # Zero out these losses for this batch to prevent gradient explosion
            loss_pitch = torch.tensor(0.0, device=self.device)
            loss_energy = torch.tensor(0.0, device=self.device)
            variance_diverged = True

        # Clamp individual losses to prevent numerical explosion (critical for MPS stability)
        loss_mel = torch.clamp(loss_mel, max=100.0)
        loss_duration = torch.clamp(loss_duration, max=100.0)
        loss_stop_token = torch.clamp(loss_stop_token, max=100.0)
        if not variance_diverged:
            loss_pitch = torch.clamp(loss_pitch, max=10.0)
            loss_energy = torch.clamp(loss_energy, max=10.0)

        # Combine all losses (pitch/energy normalized to [0,1], safe to use)
        total_loss = (loss_mel +
                     loss_duration * self.config.duration_loss_weight +
                     loss_stop_token * self.config.stop_token_loss_weight +
                     loss_pitch * getattr(self.config, 'pitch_loss_weight', 0.1) +
                     loss_energy * getattr(self.config, 'energy_loss_weight', 0.1))

        # Final NaN/Inf check (caller decides whether to skip the batch)
        if not torch.isfinite(total_loss):
            logger.warning(f"Non-finite total_loss detected! "
                          f"mel={loss_mel:.2f}, dur={loss_duration:.2f}, stop={loss_stop_token:.2f}, "
                          f"pitch={loss_pitch:.2f}, energy={loss_energy:.2f}")

        return total_loss, loss_mel, loss_duration, loss_stop_token, loss_pitch, loss_energy

    def _reset_variance_predictors(self):
        """Reset variance predictor weights - critical when changing normalization"""
        logger.warning("🔄 Resetting variance predictor weights - extractors now return normalized [0,1] values")

        reset_count = 0
        # Check all possible attribute names
        for attr_name in ['pitch_predictor', 'variance_adaptor', 'pitch_adaptor']:
            if hasattr(self.model, attr_name):
                predictor = getattr(self.model, attr_name)
                if predictor is not None and hasattr(predictor, 'pitch_predictor'):
                    predictor.pitch_predictor._init_weights()
                    logger.info(f"  ✓ Pitch predictor reinitialized (via {attr_name})")
                    reset_count += 1
                elif predictor is not None and hasattr(predictor, '_init_weights'):
                    predictor._init_weights()
                    logger.info(f"  ✓ {attr_name} reinitialized")
                    reset_count += 1

        for attr_name in ['energy_predictor', 'variance_adaptor', 'energy_adaptor']:
            if hasattr(self.model, attr_name):
                predictor = getattr(self.model, attr_name)
                if predictor is not None and hasattr(predictor, 'energy_predictor'):
                    predictor.energy_predictor._init_weights()
                    logger.info(f"  ✓ Energy predictor reinitialized (via {attr_name})")
                    reset_count += 1
                elif predictor is not None and hasattr(predictor, '_init_weights') and 'energy' in attr_name:
                    predictor._init_weights()
                    logger.info(f"  ✓ {attr_name} reinitialized")
                    reset_count += 1

        if reset_count == 0:
            logger.warning("  ⚠️  No variance predictors found to reset - checking model structure")
            logger.warning(f"  Model attributes: {[attr for attr in dir(self.model) if 'predict' in attr.lower() or 'variance' in attr.lower()]}")

    def setup_checkpoint_resumption(self):
        """Handle checkpoint resumption with mixed precision state"""
        if not self.config.resume_checkpoint:
            logger.info("No resume checkpoint specified, starting from scratch.")
            # Still reset variance predictors for normalized features
            self._reset_variance_predictors()
            return

        checkpoint_path = None
        if self.config.resume_checkpoint.lower() == 'auto':
            checkpoint_path = find_latest_checkpoint(self.config.output_dir)
            if not checkpoint_path:
                logger.info("No checkpoint found for auto-resume, starting from scratch.")
                # Still reset variance predictors for normalized features
                self._reset_variance_predictors()
                return
        else:
            checkpoint_path = self.config.resume_checkpoint
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        self.start_epoch, self.best_loss, phoneme_processor = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler, self.config.output_dir
        )

        # Load scaler state if available
        if self.use_mixed_precision and self.scaler:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'scaler' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler'])
                    logger.info(f"Loaded {self.device_type.upper()} scaler state from checkpoint")
                else:
                    logger.info(f"No scaler state found in checkpoint, using default for {self.device_type}")
            except Exception as e:
                logger.warning(f"Could not load scaler state: {e}")

        # Load EMA model state if available
        if self.use_ema and self.ema_model is not None:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'ema_model_state_dict' in checkpoint:
                    self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                    if 'ema_updates' in checkpoint:
                        self.ema_updates = checkpoint['ema_updates']
                    logger.info(f"Loaded EMA model state from checkpoint (updates: {self.ema_updates})")
                else:
                    logger.info("No EMA state found in checkpoint, initializing EMA from current model")
                    self.ema_model.load_state_dict(self.model.state_dict())
            except Exception as e:
                logger.warning(f"Could not load EMA state: {e}")

        self.dataset.phoneme_processor = phoneme_processor
        logger.info(f"Resumed from epoch {self.start_epoch}, best loss {self.best_loss:.4f}")

        # CRITICAL: Reset variance predictors after loading checkpoint
        # Must happen AFTER checkpoint load to override old weights
        self._reset_variance_predictors()

    def validate_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
        """
        Run validation loop to monitor overfitting.
        Uses EMA model if available for better validation metrics.
        Returns: (avg_total_loss, avg_mel_loss, avg_dur_loss, avg_stop_loss)
        """
        if self.val_dataloader is None:
            return 0.0, 0.0, 0.0, 0.0

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
        num_batches = 0

        with torch.no_grad():  # Disable gradient computation
            progress_bar = tqdm(self.val_dataloader, desc=f"Validation Epoch {epoch+1}")
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Data loading - batch transfer for efficiency
                    non_blocking = self.device.type == 'cuda'

                    # Batch transfer all required tensors at once
                    tensor_keys = ['mel_specs', 'phoneme_indices', 'phoneme_durations',
                                   'stop_token_targets', 'mel_lengths', 'phoneme_lengths']
                    transferred = {k: batch[k].to(self.device, non_blocking=non_blocking)
                                   for k in tensor_keys}

                    mel_specs = transferred['mel_specs']
                    phoneme_indices = transferred['phoneme_indices']
                    phoneme_durations = transferred['phoneme_durations']
                    stop_token_targets = transferred['stop_token_targets']
                    mel_lengths = transferred['mel_lengths']
                    phoneme_lengths = transferred['phoneme_lengths']

                    # Optional tensors (pitch and energy)
                    pitches = batch.get('pitches', None)
                    energies = batch.get('energies', None)
                    if pitches is not None:
                        pitches = pitches.to(self.device, non_blocking=non_blocking)
                    if energies is not None:
                        energies = energies.to(self.device, non_blocking=non_blocking)

                    # Forward pass (without mixed precision for validation consistency)
                    predicted_mel, predicted_log_durations, predicted_stop_logits, predicted_pitch, predicted_energy = \
                        model_to_validate(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets,
                                 pitch_targets=pitches, energy_targets=energies)

                    # Convert mel-frame level pitch/energy to phoneme level for loss calculation
                    phoneme_pitches = None
                    phoneme_energies = None
                    if pitches is not None and predicted_pitch is not None:
                        phoneme_pitches = self._average_pitch_energy_by_duration(
                            pitches, phoneme_durations, phoneme_lengths
                        )
                    if energies is not None and predicted_energy is not None:
                        phoneme_energies = self._average_pitch_energy_by_duration(
                            energies, phoneme_durations, phoneme_lengths
                        )

                    # Loss calculation
                    total_loss, loss_mel, loss_duration, loss_stop_token, loss_pitch, loss_energy = self._calculate_losses(
                        predicted_mel, predicted_log_durations, predicted_stop_logits,
                        mel_specs, phoneme_durations, stop_token_targets,
                        mel_lengths, phoneme_lengths,
                        predicted_pitch, predicted_energy, phoneme_pitches, phoneme_energies
                    )

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

        self.model.train()  # Set back to training mode

        # Compute averages
        if num_batches > 0:
            avg_total_loss = total_loss_epoch / num_batches
            avg_mel_loss = mel_loss_epoch / num_batches
            avg_dur_loss = dur_loss_epoch / num_batches
            avg_stop_loss = stop_loss_epoch / num_batches
            avg_pitch_loss = pitch_loss_epoch / num_batches if pitch_loss_epoch > 0 else 0.0
            avg_energy_loss = energy_loss_epoch / num_batches if energy_loss_epoch > 0 else 0.0

            logger.info(f"Validation Epoch {epoch+1} - "
                       f"Loss: {avg_total_loss:.4f}, "
                       f"Mel: {avg_mel_loss:.4f}, "
                       f"Dur: {avg_dur_loss:.4f}, "
                       f"Stop: {avg_stop_loss:.4f}")
            if avg_pitch_loss > 0:
                logger.info(f"  Pitch: {avg_pitch_loss:.4f}, Energy: {avg_energy_loss:.4f}")

            return (avg_total_loss, avg_mel_loss, avg_dur_loss, avg_stop_loss)
        else:
            return (0.0, 0.0, 0.0, 0.0)

    def save_checkpoint_with_scaler(self, epoch: int, loss: float):
        """Save checkpoint including scaler state and EMA weights"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
        }

        if self.use_mixed_precision and self.scaler:
            checkpoint['scaler'] = self.scaler.state_dict()
            checkpoint['device_type'] = self.device_type  # Store device type for proper restoration

        # Save EMA model weights if enabled
        if self.use_ema and self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
            checkpoint['ema_updates'] = self.ema_updates
            logger.info(f"Saving EMA weights (updates: {self.ema_updates})")

        checkpoint_path = os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def train_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
        """Train for one epoch with enhanced profiling, mixed precision, and adaptive memory management"""
        self.model.train()

        # Use tensor accumulation to reduce .item() calls (CPU-GPU sync overhead)
        total_loss_epoch = 0.0
        mel_loss_epoch = 0.0
        dur_loss_epoch = 0.0
        stop_loss_epoch = 0.0

        # Track losses as tensors for batched accumulation (sync every N batches)
        loss_accumulation_interval = 10
        accumulated_losses = []

        num_batches = len(self.dataloader)

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

                # Only add profiler overhead when actually profiling
                if is_profiling_epoch:
                    with torch.profiler.record_function("Data_Loading"):
                        # Use non_blocking only for CUDA
                        non_blocking = self.device.type == 'cuda'

                        # Batch transfer all required tensors at once for better performance
                        tensor_keys = ['mel_specs', 'phoneme_indices', 'phoneme_durations',
                                       'stop_token_targets', 'mel_lengths', 'phoneme_lengths']
                        transferred = {k: batch[k].to(self.device, non_blocking=non_blocking)
                                       for k in tensor_keys}

                        mel_specs = transferred['mel_specs']
                        phoneme_indices = transferred['phoneme_indices']
                        phoneme_durations = transferred['phoneme_durations']
                        stop_token_targets = transferred['stop_token_targets']
                        mel_lengths = transferred['mel_lengths']
                        phoneme_lengths = transferred['phoneme_lengths']

                        # Optional tensors (pitch and energy)
                        pitches = batch.get('pitches', None)
                        energies = batch.get('energies', None)
                        if pitches is not None:
                            pitches = pitches.to(self.device, non_blocking=non_blocking)
                        if energies is not None:
                            energies = energies.to(self.device, non_blocking=non_blocking)
                else:
                    # No profiler overhead
                    non_blocking = self.device.type == 'cuda'

                    tensor_keys = ['mel_specs', 'phoneme_indices', 'phoneme_durations',
                                   'stop_token_targets', 'mel_lengths', 'phoneme_lengths']
                    transferred = {k: batch[k].to(self.device, non_blocking=non_blocking)
                                   for k in tensor_keys}

                    mel_specs = transferred['mel_specs']
                    phoneme_indices = transferred['phoneme_indices']
                    phoneme_durations = transferred['phoneme_durations']
                    stop_token_targets = transferred['stop_token_targets']
                    mel_lengths = transferred['mel_lengths']
                    phoneme_lengths = transferred['phoneme_lengths']

                    # Optional tensors (pitch and energy) - get them first
                    pitches = batch.get('pitches', None)
                    energies = batch.get('energies', None)
                    if pitches is not None:
                        pitches = pitches.to(self.device, non_blocking=non_blocking)
                    if energies is not None:
                        energies = energies.to(self.device, non_blocking=non_blocking)

                    # Validate tensor dimensions to prevent MPS overflow
                    max_dim = max(mel_specs.shape[1], phoneme_indices.shape[1])
                    if max_dim > 2000:
                        logger.warning(f"⚠️ Skipping batch {batch_idx}: excessive dimensions (max_dim={max_dim})")
                        continue

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


                # Adaptive stabilization for sequence/duration outliers (no hard skipping)
                max_mel_length = 1400  # Earlier guard for long batches
                max_duration_value = 150  # Earlier guard for extreme durations
                adaptive_loss_scale = 1.0
                adaptive_clip_norm = 0.5

                mel_length = mel_specs.shape[1]
                max_duration_in_batch = phoneme_durations.max().item()

                # Soft stabilization starts earlier to reduce projection-layer gradient spikes
                soft_mel_length = 900
                soft_duration_value = 80
                soft_risk_ratio = max(mel_length / soft_mel_length, max_duration_in_batch / soft_duration_value)
                if soft_risk_ratio > 1.0:
                    adaptive_loss_scale = min(adaptive_loss_scale, max(0.5, 1.0 / (soft_risk_ratio ** 0.65)))
                    adaptive_clip_norm = min(adaptive_clip_norm, max(0.1, 0.4 / (soft_risk_ratio ** 0.35)))

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

                if is_profiling_epoch:
                    with torch.profiler.record_function("Model_Forward"):
                        if self.use_mixed_precision:
                            with self.get_autocast_context():
                                predicted_mel, predicted_log_durations, predicted_stop_logits, predicted_pitch, predicted_energy = \
                                    self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets,
                                             pitch_targets=pitches, energy_targets=energies)
                        else:
                            predicted_mel, predicted_log_durations, predicted_stop_logits, predicted_pitch, predicted_energy = \
                                self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets,
                                         pitch_targets=pitches, energy_targets=energies)
                else:
                    if self.use_mixed_precision:
                        with self.get_autocast_context():
                            predicted_mel, predicted_log_durations, predicted_stop_logits, predicted_pitch, predicted_energy = \
                                self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets,
                                         pitch_targets=pitches, energy_targets=energies)
                    else:
                        predicted_mel, predicted_log_durations, predicted_stop_logits, predicted_pitch, predicted_energy = \
                            self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets,
                                     pitch_targets=pitches, energy_targets=energies)

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.end_forward_pass()

                if is_profiling_epoch:
                    self.log_memory_stats("forward_pass")

                outputs_are_finite = torch.isfinite(predicted_mel).all() and \
                    torch.isfinite(predicted_log_durations).all() and \
                    torch.isfinite(predicted_stop_logits).all() and \
                    (predicted_pitch is None or torch.isfinite(predicted_pitch).all()) and \
                    (predicted_energy is None or torch.isfinite(predicted_energy).all())

                if not outputs_are_finite:
                    logger.error(f"\n{'='*80}")
                    logger.error(f"❌ BATCH {batch_idx} - NON-FINITE MODEL OUTPUTS DETECTED")
                    logger.error(f"{'='*80}")
                    logger.error(
                        f"Output Finiteness: "
                        f"mel={torch.isfinite(predicted_mel).all().item()}, "
                        f"dur={torch.isfinite(predicted_log_durations).all().item()}, "
                        f"stop={torch.isfinite(predicted_stop_logits).all().item()}, "
                        f"pitch={True if predicted_pitch is None else torch.isfinite(predicted_pitch).all().item()}, "
                        f"energy={True if predicted_energy is None else torch.isfinite(predicted_energy).all().item()}"
                    )

                    # Detailed output statistics
                    if not torch.isfinite(predicted_mel).all():
                        logger.error(f"predicted_mel: {torch.isnan(predicted_mel).sum()} NaNs, {torch.isinf(predicted_mel).sum()} Infs")
                    if not torch.isfinite(predicted_log_durations).all():
                        logger.error(f"predicted_log_durations: {torch.isnan(predicted_log_durations).sum()} NaNs, {torch.isinf(predicted_log_durations).sum()} Infs")
                    if predicted_pitch is not None and not torch.isfinite(predicted_pitch).all():
                        logger.error(f"predicted_pitch: {torch.isnan(predicted_pitch).sum()} NaNs, {torch.isinf(predicted_pitch).sum()} Infs")
                    if predicted_energy is not None and not torch.isfinite(predicted_energy).all():
                        logger.error(f"predicted_energy: {torch.isnan(predicted_energy).sum()} NaNs, {torch.isinf(predicted_energy).sum()} Infs")
                    logger.error(f"{'='*80}\n")

                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_step = 0
                    if self.enable_adaptive_memory:
                        self.memory_manager.emergency_cleanup()
                    else:
                        self.clear_device_cache()
                    if enable_interbatch_profiling or is_profiling_epoch:
                        self.interbatch_profiler.end_batch(mel_specs.size(0))
                    continue


                # Convert mel-frame level pitch/energy to phoneme level for loss calculation
                phoneme_pitches = None
                phoneme_energies = None
                if pitches is not None and predicted_pitch is not None:
                    # Average pitch from mel-frame level to phoneme level using durations
                    phoneme_pitches = self._average_pitch_energy_by_duration(
                        pitches, phoneme_durations, phoneme_lengths
                    )
                if energies is not None and predicted_energy is not None:
                    # Average energy from mel-frame level to phoneme level using durations
                    phoneme_energies = self._average_pitch_energy_by_duration(
                        energies, phoneme_durations, phoneme_lengths
                    )

                # Loss calculation with mixed precision
                if is_profiling_epoch:
                    with torch.profiler.record_function("Loss_Calculation"):
                        if self.use_mixed_precision:
                            with self.get_autocast_context():
                                total_loss, loss_mel, loss_duration, loss_stop_token, loss_pitch, loss_energy = self._calculate_losses(
                                    predicted_mel, predicted_log_durations, predicted_stop_logits,
                                    mel_specs, phoneme_durations, stop_token_targets,
                                    mel_lengths, phoneme_lengths,
                                    predicted_pitch, predicted_energy, phoneme_pitches, phoneme_energies
                                )
                        else:
                            total_loss, loss_mel, loss_duration, loss_stop_token, loss_pitch, loss_energy = self._calculate_losses(
                                predicted_mel, predicted_log_durations, predicted_stop_logits,
                                mel_specs, phoneme_durations, stop_token_targets,
                                mel_lengths, phoneme_lengths,
                                predicted_pitch, predicted_energy, phoneme_pitches, phoneme_energies
                            )
                else:
                    if self.use_mixed_precision:
                        with self.get_autocast_context():
                            total_loss, loss_mel, loss_duration, loss_stop_token, loss_pitch, loss_energy = self._calculate_losses(
                                predicted_mel, predicted_log_durations, predicted_stop_logits,
                                mel_specs, phoneme_durations, stop_token_targets,
                                mel_lengths, phoneme_lengths,
                                predicted_pitch, predicted_energy, phoneme_pitches, phoneme_energies
                            )
                    else:
                        total_loss, loss_mel, loss_duration, loss_stop_token, loss_pitch, loss_energy = self._calculate_losses(
                            predicted_mel, predicted_log_durations, predicted_stop_logits,
                            mel_specs, phoneme_durations, stop_token_targets,
                            mel_lengths, phoneme_lengths,
                            predicted_pitch, predicted_energy, phoneme_pitches, phoneme_energies
                        )

                if is_profiling_epoch:
                    self.log_memory_stats("loss_calculation")


                finite_losses = (
                    torch.isfinite(total_loss) and
                    torch.isfinite(loss_mel) and
                    torch.isfinite(loss_duration) and
                    torch.isfinite(loss_stop_token) and
                    (loss_pitch is None or torch.isfinite(loss_pitch)) and
                    (loss_energy is None or torch.isfinite(loss_energy))
                )

                if not finite_losses:
                    logger.error(f"\n{'='*80}")
                    logger.error(f"❌ BATCH {batch_idx} - NON-FINITE LOSS DETECTED")
                    logger.error(f"{'='*80}")
                    logger.error(
                        f"Loss finiteness: "
                        f"total={torch.isfinite(total_loss).item()}, "
                        f"mel={torch.isfinite(loss_mel).item()}, "
                        f"dur={torch.isfinite(loss_duration).item()}, "
                        f"stop={torch.isfinite(loss_stop_token).item()}, "
                        f"pitch={True if loss_pitch is None else torch.isfinite(loss_pitch).item()}, "
                        f"energy={True if loss_energy is None else torch.isfinite(loss_energy).item()}"
                    )
                    logger.error(f"Loss values: total={total_loss}, mel={loss_mel}, dur={loss_duration}, "
                                f"stop={loss_stop_token}, pitch={loss_pitch}, energy={loss_energy}")
                    logger.error(f"{'='*80}\n")
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_step = 0
                    if self.enable_adaptive_memory:
                        self.memory_manager.emergency_cleanup()
                    else:
                        self.clear_device_cache()
                    continue


                # Scale loss by gradient accumulation steps for proper gradient averaging
                # Also apply adaptive scaling for high-risk batches
                scaled_total_loss = (total_loss / gradient_accumulation_steps) * adaptive_loss_scale

                # Backward pass with mixed precision and interbatch profiling

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.start_backward_pass()

                if is_profiling_epoch:
                    with torch.profiler.record_function("Backward_Pass"):
                        if self.use_mixed_precision:
                            if self.device_type == 'cuda':
                                self.scaler.scale(scaled_total_loss).backward()
                            else:  # MPS
                                scaled_loss = self.scaler.scale(scaled_total_loss)
                                scaled_loss.backward()
                        else:
                            scaled_total_loss.backward()
                else:
                    if self.use_mixed_precision:
                        if self.device_type == 'cuda':
                            self.scaler.scale(scaled_total_loss).backward()
                        else:  # MPS
                            scaled_loss = self.scaler.scale(scaled_total_loss)
                            scaled_loss.backward()
                    else:
                        scaled_total_loss.backward()

                # Advance gradient accumulation cycle after successful backward
                accumulated_step += 1

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.end_backward_pass()

                if is_profiling_epoch:
                    self.log_memory_stats("backward_pass")

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

                        adaptive_clip_norm = min(adaptive_clip_norm, 0.05)
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
                    if self.use_mixed_precision:
                        if self.device_type == 'cuda':
                            # CUDA path with built-in GradScaler
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=adaptive_clip_norm)

                            old_scale = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            new_scale = self.scaler.get_scale()

                            # Step scheduler with warmup immediately after optimizer step
                            if self.scheduler_per_batch:
                                self._step_scheduler_with_warmup()

                            # Update EMA model weights
                            self._update_ema()

                            # Update mixed precision stats
                            if new_scale != old_scale:
                                self.mixed_precision_stats['scale_updates'] += 1
                                if new_scale < old_scale:
                                    self.mixed_precision_stats['scale_decreases'] += 1
                                    self.mixed_precision_stats['overflow_count'] += 1
                                else:
                                    self.mixed_precision_stats['successful_steps'] += 1
                            else:
                                self.mixed_precision_stats['successful_steps'] += 1

                        else:  # MPS path with custom scaler
                            # For MPS, clip gradients before unscaling
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=adaptive_clip_norm)

                            old_scale = self.scaler.get_scale()
                            step_successful = self.scaler.step(self.optimizer)
                            self.scaler.update()
                            new_scale = self.scaler.get_scale()

                            # Step scheduler with warmup immediately after optimizer step
                            if self.scheduler_per_batch:
                                self._step_scheduler_with_warmup()

                            # Update EMA model weights
                            self._update_ema()

                            # Update mixed precision stats
                            if step_successful:
                                self.mixed_precision_stats['successful_steps'] += 1
                            else:
                                self.mixed_precision_stats['skipped_steps'] += 1
                                self.mixed_precision_stats['overflow_count'] += 1

                            if new_scale != old_scale:
                                self.mixed_precision_stats['scale_updates'] += 1
                                if new_scale < old_scale:
                                    self.mixed_precision_stats['scale_decreases'] += 1
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=adaptive_clip_norm)
                        self.optimizer.step()

                        # Step scheduler with warmup immediately after optimizer step
                        if self.scheduler_per_batch:
                            self._step_scheduler_with_warmup()

                        # Update EMA model weights
                        self._update_ema()

                    self.optimizer_steps_completed += 1

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
                    'pitch': loss_pitch.detach() if loss_pitch is not None else None,
                    'energy': loss_energy.detach() if loss_energy is not None else None
                })

                # Sync accumulated losses periodically to reduce overhead
                if len(accumulated_losses) >= loss_accumulation_interval or batch_idx == num_batches - 1:
                    for acc_loss in accumulated_losses:
                        total_loss_epoch += acc_loss['total'].item()
                        mel_loss_epoch += acc_loss['mel'].item()
                        dur_loss_epoch += acc_loss['dur'].item()
                        stop_loss_epoch += acc_loss['stop'].item()
                    accumulated_losses.clear()

                    # Delete tensors and clear MPS cache only when memory pressure is high
                    if self.device_type == DeviceType.MPS.value:
                        pressure = cleanup_result.get('pressure_level', 'low')
                        if pressure in ['high', 'critical']:
                            # Explicitly delete large tensors
                            del predicted_mel, predicted_log_durations, predicted_stop_logits
                            if 'predicted_pitch' in locals():
                                del predicted_pitch
                            if 'predicted_energy' in locals():
                                del predicted_energy
                            # Clear cache
                            torch.mps.empty_cache()
                            import gc
                            gc.collect()

                # Get current loss values for progress bar (still need .item() for display)
                current_total_loss = total_loss.item()
                current_mel_loss = loss_mel.item()
                current_dur_loss = loss_duration.item()
                current_stop_loss = loss_stop_token.item()
                pitch_loss_value = loss_pitch.item() if loss_pitch is not None else 0.0
                energy_loss_value = loss_energy.item() if loss_energy is not None else 0.0

                # Enhanced progress bar with mixed precision info and memory pressure
                postfix_dict = {
                    'total_loss': current_total_loss,
                    'mel_loss': current_mel_loss,
                    'dur_loss': current_dur_loss,
                    'stop_loss': current_stop_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                }

                # Add variance losses if they exist
                if loss_pitch is not None and loss_pitch.item() > 0:
                    postfix_dict['pitch_loss'] = pitch_loss_value
                if loss_energy is not None and loss_energy.item() > 0:
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

                # Print memory management report periodically
                if self.enable_adaptive_memory and batch_idx % self.memory_report_interval == 0 and batch_idx > 0:
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

        return (total_loss_epoch / num_batches,
                mel_loss_epoch / num_batches,
                dur_loss_epoch / num_batches,
                stop_loss_epoch / num_batches)

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
            logger.info(f"Profiler logs will be saved to: {self.log_dir}")

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
            avg_total_loss, avg_mel_loss, avg_dur_loss, avg_stop_loss = self.train_epoch(epoch)

            # Step scheduler per epoch only if NOT using OneCycleLR (which steps per batch)
            if not self.scheduler_per_batch:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1} completed. "
                        f"Avg Total Loss: {avg_total_loss:.4f}, "
                        f"Avg Mel Loss: {avg_mel_loss:.4f}, "
                        f"Avg Dur Loss: {avg_dur_loss:.4f}, "
                        f"Avg Stop Loss: {avg_stop_loss:.4f}, "
                        f"Current LR: {current_lr:.8f}")

            # Run validation if enabled
            validation_interval = getattr(self.config, 'validation_interval', 1)
            if self.val_dataloader is not None and (epoch + 1) % validation_interval == 0:
                val_total_loss, val_mel_loss, val_dur_loss, val_stop_loss = self.validate_epoch(epoch)
                self.validation_losses.append(val_total_loss)

                # Check for improvement
                min_delta = getattr(self.config, 'early_stopping_min_delta', 0.001)
                if val_total_loss < (self.best_val_loss - min_delta):
                    improvement = self.best_val_loss - val_total_loss
                    self.best_val_loss = val_total_loss
                    self.epochs_without_improvement = 0
                    logger.info(f"✓ Validation loss improved by {improvement:.4f} - saving best model")

                    # Save best model
                    if self.use_mixed_precision:
                        self.save_checkpoint_with_scaler(epoch, val_total_loss)
                    else:
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            epoch, val_total_loss, self.config, self.config.output_dir
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
                logger.info(f"Memory Management Summary - Epoch {epoch+1}:")
                logger.info(f"  Current Pressure: {memory_report['current_pressure']}")
                logger.info(f"  Cleanups This Epoch: {memory_report['cleanup_count']}")
                logger.info(f"  Memory Trend: {memory_report['memory_trend']:+.2f}%")
                logger.info(f"  Cleanup Overhead: {memory_report['cleanup_overhead_percent']:.2f}%")

            # Save periodic checkpoints (only if not using validation or if validation isn't better)
            if (epoch + 1) % self.config.save_every == 0:
                # Only save if we're not doing validation or if this is just a periodic save
                should_save_periodic = (self.val_dataloader is None or
                                       self.epochs_without_improvement > 0)
                if should_save_periodic:
                    if self.use_mixed_precision:
                        self.save_checkpoint_with_scaler(epoch, avg_total_loss)
                    else:
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            epoch, avg_total_loss, self.config, self.config.output_dir
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
            self.print_memory_management_report()

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

        # Data loading - batch transfer for efficiency
        non_blocking = self.device.type == 'cuda'

        # Batch transfer all required tensors at once
        tensor_keys = ['mel_specs', 'phoneme_indices', 'phoneme_durations',
                       'stop_token_targets', 'mel_lengths', 'phoneme_lengths']
        transferred = {k: batch[k].to(self.device, non_blocking=non_blocking)
                       for k in tensor_keys}

        mel_specs = transferred['mel_specs']
        phoneme_indices = transferred['phoneme_indices']
        phoneme_durations = transferred['phoneme_durations']
        stop_token_targets = transferred['stop_token_targets']
        mel_lengths = transferred['mel_lengths']
        phoneme_lengths = transferred['phoneme_lengths']

        # Optional tensors (pitch and energy)
        pitches = batch.get('pitches', None)
        energies = batch.get('energies', None)
        if pitches is not None:
            pitches = pitches.to(self.device, non_blocking=non_blocking)
        if energies is not None:
            energies = energies.to(self.device, non_blocking=non_blocking)

        self.optimizer.zero_grad()

        # Forward pass with optional AMP
        if self.use_mixed_precision:
            with self.get_autocast_context():
                predicted_mel, predicted_log_durations, predicted_stop_logits, predicted_pitch, predicted_energy = \
                    self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets,
                             pitch_targets=pitches, energy_targets=energies)

                # Convert pitch/energy to phoneme level
                phoneme_pitches = None
                phoneme_energies = None
                if pitches is not None and predicted_pitch is not None:
                    phoneme_pitches = self._average_pitch_energy_by_duration(
                        pitches, phoneme_durations, phoneme_lengths
                    )
                if energies is not None and predicted_energy is not None:
                    phoneme_energies = self._average_pitch_energy_by_duration(
                        energies, phoneme_durations, phoneme_lengths
                    )

                # Loss calculation
                total_loss, _, _, _, _, _ = self._calculate_losses(
                    predicted_mel, predicted_log_durations, predicted_stop_logits,
                    mel_specs, phoneme_durations, stop_token_targets,
                    mel_lengths, phoneme_lengths,
                    predicted_pitch, predicted_energy, phoneme_pitches, phoneme_energies
                )

            # Backward pass with scaler
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # No AMP
            predicted_mel, predicted_log_durations, predicted_stop_logits, predicted_pitch, predicted_energy = \
                self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets,
                         pitch_targets=pitches, energy_targets=energies)

            # Convert pitch/energy to phoneme level
            phoneme_pitches = None
            phoneme_energies = None
            if pitches is not None and predicted_pitch is not None:
                phoneme_pitches = self._average_pitch_energy_by_duration(
                    pitches, phoneme_durations, phoneme_lengths
                )
            if energies is not None and predicted_energy is not None:
                phoneme_energies = self._average_pitch_energy_by_duration(
                    energies, phoneme_durations, phoneme_lengths
                )

            # Loss calculation
            total_loss, _, _, _, _, _ = self._calculate_losses(
                predicted_mel, predicted_log_durations, predicted_stop_logits,
                mel_specs, phoneme_durations, stop_token_targets,
                mel_lengths, phoneme_lengths,
                predicted_pitch, predicted_energy, phoneme_pitches, phoneme_energies
            )

            # Regular backward pass
            total_loss.backward()
            self.optimizer.step()

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
                # Start interbatch profiling
                self.interbatch_profiler.start_batch()

                # Adaptive memory cleanup check during profiling
                cleanup_result = self.adaptive_memory_cleanup(batch_idx)

                # Profile step
                self.profile_step()

                # Data loading profiling
                self.interbatch_profiler.start_data_loading()
                with torch.profiler.record_function("Data_Loading"):
                    mel_specs = batch['mel_specs'].to(self.device, non_blocking=self.device.type=='cuda')
                    phoneme_indices = batch['phoneme_indices'].to(self.device, non_blocking=self.device.type=='cuda')
                    phoneme_durations = batch['phoneme_durations'].to(self.device, non_blocking=self.device.type=='cuda')
                    stop_token_targets = batch['stop_token_targets'].to(self.device, non_blocking=self.device.type=='cuda')
                    mel_lengths = batch['mel_lengths'].to(self.device, non_blocking=self.device.type=='cuda')
                    phoneme_lengths = batch['phoneme_lengths'].to(self.device, non_blocking=self.device.type=='cuda')

                    # Validate tensor dimensions to prevent MPS overflow (INT_MAX limit)
                    max_dim = max(mel_specs.shape[1], phoneme_indices.shape[1])
                    if max_dim > 2000:  # Safety threshold
                        logger.warning(f"⚠️ Skipping batch {batch_idx}: excessive dimensions (max_dim={max_dim})")
                        continue

                self.interbatch_profiler.end_data_loading()

                self.log_memory_stats("data_loading")

                with torch.profiler.record_function("Zero_Grad"):
                    self.optimizer.zero_grad()

                # Forward pass with mixed precision
                self.interbatch_profiler.start_forward_pass()
                with torch.profiler.record_function("Model_Forward"):
                    if self.use_mixed_precision:
                        with self.get_autocast_context():
                            predicted_mel, predicted_log_durations, predicted_stop_logits = \
                                self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets)
                    else:
                        predicted_mel, predicted_log_durations, predicted_stop_logits = \
                            self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets)
                self.interbatch_profiler.end_forward_pass()

                self.log_memory_stats("forward_pass")

                # Loss calculation with mixed precision
                with torch.profiler.record_function("Loss_Calculation"):
                    if self.use_mixed_precision:
                        with self.get_autocast_context():
                            total_loss, loss_mel, loss_duration, loss_stop_token = self._calculate_losses(
                                predicted_mel, predicted_log_durations, predicted_stop_logits,
                                mel_specs, phoneme_durations, stop_token_targets,
                                mel_lengths, phoneme_lengths
                            )
                    else:
                        total_loss, loss_mel, loss_duration, loss_stop_token = self._calculate_losses(
                            predicted_mel, predicted_log_durations, predicted_stop_logits,
                            mel_specs, phoneme_durations, stop_token_targets,
                            mel_lengths, phoneme_lengths
                        )

                self.log_memory_stats("loss_calculation")

                # Backward pass with mixed precision
                self.interbatch_profiler.start_backward_pass()
                with torch.profiler.record_function("Backward_Pass"):
                    if self.use_mixed_precision:
                        self.scaler.scale(total_loss).backward()
                    else:
                        total_loss.backward()
                self.interbatch_profiler.end_backward_pass()

                self.log_memory_stats("backward_pass")

                # Optimizer step with mixed precision
                with torch.profiler.record_function("Optimizer_Step"):
                    if self.use_mixed_precision:
                        if self.device_type == 'cuda':
                            # CUDA path with built-in GradScaler
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                            old_scale = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            new_scale = self.scaler.get_scale()

                            # Update mixed precision stats
                            if new_scale != old_scale:
                                self.mixed_precision_stats['scale_updates'] += 1
                                if new_scale < old_scale:
                                    self.mixed_precision_stats['scale_decreases'] += 1
                                    self.mixed_precision_stats['overflow_count'] += 1
                                else:
                                    self.mixed_precision_stats['successful_steps'] += 1
                            else:
                                self.mixed_precision_stats['successful_steps'] += 1

                        else:  # MPS path with custom scaler
                            # Clip gradients before unscaling for MPS
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                            old_scale = self.scaler.get_scale()
                            step_successful = self.scaler.step(self.optimizer)
                            self.scaler.update()
                            new_scale = self.scaler.get_scale()

                            # Update mixed precision stats
                            if step_successful:
                                self.mixed_precision_stats['successful_steps'] += 1
                            else:
                                self.mixed_precision_stats['skipped_steps'] += 1
                                self.mixed_precision_stats['overflow_count'] += 1

                            if new_scale != old_scale:
                                self.mixed_precision_stats['scale_updates'] += 1
                                if new_scale < old_scale:
                                    self.mixed_precision_stats['scale_decreases'] += 1
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        self.optimizer.step()

                self.log_memory_stats("optimizer_step")

                # End batch profiling
                batch_size = mel_specs.size(0)
                self.interbatch_profiler.end_batch(batch_size)

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

        # Generate and analyze report
        report = self.get_profiling_report()
        logger.info(f"Training profiling completed. Total time: {total_time:.2f}s, "
                   f"Avg time per step: {total_time/step_count:.3f}s")

        # Print analysis
        self.analyze_profiling_results(report)

        # Print memory management report if enabled
        if self.enable_adaptive_memory:
            logger.info("Memory Management Report during profiling:")
            self.print_memory_management_report()

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
            self.hidden_dim = 512
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
            self.weight_decay = 0.01
            self.adam_eps = 1e-8
            self.adam_betas = (0.9, 0.999)

            # Adaptive memory management configurations
            self.enable_adaptive_memory = True   # Enable adaptive memory management
            self.memory_report_interval = 500    # Report memory stats every N batches

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
