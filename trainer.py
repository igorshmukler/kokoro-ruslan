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

from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from config import TrainingConfig
from device_type import DeviceType
from dataset import RuslanDataset, collate_fn, LengthBasedBatchSampler
from model import KokoroModel
from checkpoint_manager import (
    save_phoneme_processor, load_checkpoint, find_latest_checkpoint,
    save_checkpoint, save_final_model
)
from interbatch_profiler import InterbatchProfiler
from mps_grad_scaler import MPSGradScaler

from adaptive_memory_manager import AdaptiveMemoryManager


logger = logging.getLogger(__name__)


def check_mps_mixed_precision_support():
    """Check if MPS supports mixed precision training"""
    if not torch.backends.mps.is_available():
        return False

    # Check PyTorch version - MPS mixed precision was added in PyTorch 2.0+
    torch_version = torch.__version__.split('.')
    major, minor = int(torch_version[0]), int(torch_version[1])

    if major < 2:
        return False

    # Test if autocast works on MPS
    try:
        device = torch.device('mps')
        x = torch.randn(2, 2, device=device)
        with torch.autocast(device_type='mps', dtype=torch.float16):
            y = torch.mm(x, x)
        return True
    except:
        return False


class KokoroTrainer:
    """Main trainer class for the model with adaptive memory management"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

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

        # Initialize dataset
        self.dataset = RuslanDataset(config.data_dir, config)

        # Initialize the custom LengthBasedBatchSampler
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
            num_workers=2,
            pin_memory=config.pin_memory and self.device.type == DeviceType.CUDA.value,
            prefetch_factor=3,
            persistent_workers=True
        )

        # Initialize model
        vocab_size = self.dataset.phoneme_processor.get_vocab_size()
        self.model = KokoroModel(vocab_size, config.n_mels, config.hidden_dim)
        self.model.to(self.device)

        # Log model information
        model_info = self.model.get_model_info()
        logger.info(f"Model initialized with {model_info['total_parameters']:,} parameters ({model_info['model_size_mb']:.1f} MB)")

        # Initialize optimizer and loss functions
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=getattr(config, 'weight_decay', 0.01),
            eps=getattr(config, 'adam_eps', 1e-8),
            betas=getattr(config, 'adam_betas', (0.9, 0.999))
        )

        self.criterion_mel = nn.L1Loss(reduction='none')
        self.criterion_duration = nn.MSELoss(reduction='none')
        self.criterion_stop_token = nn.BCEWithLogitsLoss(reduction='none')

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.lr_T_0,
            T_mult=self.config.lr_T_mult,
            eta_min=self.config.lr_eta_min
        )

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')

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

    def get_autocast_context(self):
        """Get the appropriate autocast context for the device"""
        if not self.use_mixed_precision:
            return torch.no_grad().__enter__()  # No-op context

        if self.device_type == DeviceType.CUDA.value:
            return torch.amp.autocast('cuda')
        elif self.device_type == DeviceType.MPS.value:
            return torch.autocast(device_type='mps', dtype=self.mixed_precision_dtype)
        else:
            return torch.no_grad().__enter__()  # No-op context

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

    def _calculate_losses(self, predicted_mel, predicted_log_durations, predicted_stop_logits,
                         mel_specs, phoneme_durations, stop_token_targets,
                         mel_lengths, phoneme_lengths):
        """Calculate losses with masking (extracted for reuse)"""
        # Mel Spectrogram Loss
        max_mel_len_batch = mel_specs.size(1)
        mel_mask = torch.arange(max_mel_len_batch, device=self.device).expand(
            len(mel_lengths), max_mel_len_batch) < mel_lengths.unsqueeze(1)
        mel_mask = mel_mask.unsqueeze(-1).expand_as(predicted_mel).float()

        loss_mel_unreduced = self.criterion_mel(predicted_mel, mel_specs)
        loss_mel = (loss_mel_unreduced * mel_mask).sum() / mel_mask.sum()

        # Duration Loss
        max_phoneme_len_batch = phoneme_durations.size(1)
        phoneme_mask = torch.arange(max_phoneme_len_batch, device=self.device).expand(
            len(phoneme_lengths), max_phoneme_len_batch) < phoneme_lengths.unsqueeze(1)
        phoneme_mask = phoneme_mask.float()

        target_log_durations = torch.log(phoneme_durations.float() + 1e-5)
        loss_duration_unreduced = self.criterion_duration(predicted_log_durations, target_log_durations)
        loss_duration = (loss_duration_unreduced * phoneme_mask).sum() / phoneme_mask.sum()

        # Stop Token Loss
        stop_token_mask = mel_mask[:, :, 0]
        loss_stop_token_unreduced = self.criterion_stop_token(predicted_stop_logits, stop_token_targets)
        loss_stop_token = (loss_stop_token_unreduced * stop_token_mask).sum() / stop_token_mask.sum()

        # Combine all losses
        total_loss = (loss_mel +
                     loss_duration * self.config.duration_loss_weight +
                     loss_stop_token * self.config.stop_token_loss_weight)

        return total_loss, loss_mel, loss_duration, loss_stop_token

    def setup_checkpoint_resumption(self):
        """Handle checkpoint resumption with mixed precision state"""
        if not self.config.resume_checkpoint:
            logger.info("No resume checkpoint specified, starting from scratch.")
            return

        checkpoint_path = None
        if self.config.resume_checkpoint.lower() == 'auto':
            checkpoint_path = find_latest_checkpoint(self.config.output_dir)
            if not checkpoint_path:
                logger.info("No checkpoint found for auto-resume, starting from scratch.")
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

        self.dataset.phoneme_processor = phoneme_processor
        logger.info(f"Resumed from epoch {self.start_epoch}, best loss {self.best_loss:.4f}")

    def save_checkpoint_with_scaler(self, epoch: int, loss: float):
        """Save checkpoint including scaler state"""
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

        checkpoint_path = os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def train_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
        """Train for one epoch with enhanced profiling, mixed precision, and adaptive memory management"""
        self.model.train()
        total_loss_epoch = 0.0
        mel_loss_epoch = 0.0
        dur_loss_epoch = 0.0
        stop_loss_epoch = 0.0

        num_batches = len(self.dataloader)

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

                with torch.profiler.record_function("Data_Loading"):
                    # Use non_blocking only for CUDA
                    non_blocking = self.device.type == 'cuda'
                    mel_specs = batch['mel_specs'].to(self.device, non_blocking=non_blocking)
                    phoneme_indices = batch['phoneme_indices'].to(self.device, non_blocking=non_blocking)
                    phoneme_durations = batch['phoneme_durations'].to(self.device, non_blocking=non_blocking)
                    stop_token_targets = batch['stop_token_targets'].to(self.device, non_blocking=non_blocking)
                    mel_lengths = batch['mel_lengths'].to(self.device, non_blocking=non_blocking)
                    phoneme_lengths = batch['phoneme_lengths'].to(self.device, non_blocking=non_blocking)

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.end_data_loading()

                if is_profiling_epoch:
                    self.log_memory_stats("data_loading")

                with torch.profiler.record_function("Zero_Grad"):
                    self.optimizer.zero_grad()

                # Forward pass with mixed precision and interbatch profiling
                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.start_forward_pass()

                with torch.profiler.record_function("Model_Forward"):
                    if self.use_mixed_precision:
                        with self.get_autocast_context():
                            predicted_mel, predicted_log_durations, predicted_stop_logits = \
                                self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets)
                    else:
                        predicted_mel, predicted_log_durations, predicted_stop_logits = \
                            self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets)

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.end_forward_pass()

                if is_profiling_epoch:
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

                if is_profiling_epoch:
                    self.log_memory_stats("loss_calculation")

                # Backward pass with mixed precision and interbatch profiling
                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.start_backward_pass()

                with torch.profiler.record_function("Backward_Pass"):
                    if self.use_mixed_precision:
                        if self.device_type == 'cuda':
                            self.scaler.scale(total_loss).backward()
                        else:  # MPS
                            scaled_loss = self.scaler.scale(total_loss)
                            scaled_loss.backward()
                    else:
                        total_loss.backward()

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.end_backward_pass()

                if is_profiling_epoch:
                    self.log_memory_stats("backward_pass")

                # Optimizer step with mixed precision
                with torch.profiler.record_function("Optimizer_Step"):
                    if self.use_mixed_precision:
                        if self.device_type == 'cuda':
                            # CUDA path with built-in GradScaler
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
                            # For MPS, clip gradients before unscaling
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                if is_profiling_epoch:
                    self.log_memory_stats("optimizer_step")

                # End batch profiling
                if enable_interbatch_profiling or is_profiling_epoch:
                    batch_size = mel_specs.size(0)
                    self.interbatch_profiler.end_batch(batch_size)

                total_loss_epoch += total_loss.item()
                mel_loss_epoch += loss_mel.item()
                dur_loss_epoch += loss_duration.item()
                stop_loss_epoch += loss_stop_token.item()

                # Enhanced progress bar with mixed precision info and memory pressure
                postfix_dict = {
                    'total_loss': total_loss.item(),
                    'mel_loss': loss_mel.item(),
                    'dur_loss': loss_duration.item(),
                    'stop_loss': loss_stop_token.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                }

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
        logger.info(f"Mixed precision training: {'Enabled' if self.use_mixed_precision else 'Disabled'}")
        if self.use_mixed_precision:
            logger.info(f"Mixed precision dtype: {self.mixed_precision_dtype}")
            if self.device_type == DeviceType.MPS.value:
                logger.info("Using custom MPS gradient scaler (experimental)")
        logger.info(f"Adaptive memory management: {'Enabled' if self.enable_adaptive_memory else 'Disabled'}")
        logger.info(f"Total epochs: {self.config.num_epochs}, Starting from epoch: {self.start_epoch + 1}")
        logger.info(f"Model vocabulary size: {self.dataset.phoneme_processor.get_vocab_size()}")
        logger.info(f"Initial learning rate: {self.config.learning_rate}")
        logger.info(f"Scheduler: CosineAnnealingWarmRestarts (T_0={self.config.lr_T_0}, T_mult={self.config.lr_T_mult}, eta_min={self.config.lr_eta_min})")
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

            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1} completed. "
                        f"Avg Total Loss: {avg_total_loss:.4f}, "
                        f"Avg Mel Loss: {avg_mel_loss:.4f}, "
                        f"Avg Dur Loss: {avg_dur_loss:.4f}, "
                        f"Avg Stop Loss: {avg_stop_loss:.4f}, "
                        f"Current LR: {current_lr:.8f}")

            # Log memory management stats for this epoch
            if self.enable_adaptive_memory:
                memory_report = self.memory_manager.get_memory_report()
                logger.info(f"Memory Management Summary - Epoch {epoch+1}:")
                logger.info(f"  Current Pressure: {memory_report['current_pressure']}")
                logger.info(f"  Cleanups This Epoch: {memory_report['cleanup_count']}")
                logger.info(f"  Memory Trend: {memory_report['memory_trend']:+.2f}%")
                logger.info(f"  Cleanup Overhead: {memory_report['cleanup_overhead_percent']:.2f}%")

            if (epoch + 1) % self.config.save_every == 0:
                if self.use_mixed_precision:
                    self.save_checkpoint_with_scaler(epoch, avg_total_loss)
                else:
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, avg_total_loss, self.config, self.config.output_dir
                    )
                logger.info(f"Checkpoint saved for epoch {epoch+1}")

            # Strategic memory cleanup at epoch end
            if self.enable_adaptive_memory:
                self.memory_manager.adaptive_cleanup(epoch * len(self.dataloader), force=True)
            else:
                self.clear_device_cache()

        logger.info("Training finished. Saving final model.")
        save_final_model(self.model, self.config, self.config.output_dir)

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
                        if self.device_type == DeviceType.CUDA.value:
                            self.scaler.scale(total_loss).backward()
                        else:  # MPS
                            scaled_loss = self.scaler.scale(total_loss)
                            scaled_loss.backward()
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
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
