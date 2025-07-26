#!/usr/bin/env python3
"""
Training logic for Kokoro Language Model with Enhanced Profiling
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

from typing import Tuple, Dict, Any

from config import TrainingConfig
from dataset import RuslanDataset, collate_fn, LengthBasedBatchSampler
from model import KokoroModel
from checkpoint_manager import (
    save_phoneme_processor, load_checkpoint, find_latest_checkpoint,
    save_checkpoint, save_final_model
)
from interbatch_profiler import InterbatchProfiler


logger = logging.getLogger(__name__)

class KokoroTrainer:
    """Main trainer class for Kokoro model with enhanced profiling capabilities"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

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
            num_workers=2, # config.num_workers,
            pin_memory=config.pin_memory,
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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=0.01)

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

        # Enhanced profiler setup
        self.profiler = None
        self.profiling_stats = {}
        self.memory_snapshots = []
        self.log_dir = os.path.join(config.output_dir, "profiler_logs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize interbatch profiler
        self.interbatch_profiler = InterbatchProfiler(config)

    def reset_profiling_stats(self):
        """Reset profiling statistics"""
        self.profiling_stats = {
            'stage_stats': {},
            'memory_snapshots': [],
            'device_info': {
                'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'cuda_available': torch.cuda.is_available(),
                'device_type': self.device.type
            }
        }
        self.memory_snapshots = []
        self.interbatch_profiler.reset()

    def start_torch_profiler(self, output_dir: str = None):
        """Start PyTorch profiler with comprehensive settings"""
        if output_dir is None:
            output_dir = self.log_dir

        os.makedirs(output_dir, exist_ok=True)

        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self.config.profile_wait_steps,
                warmup=self.config.profile_warmup_steps,
                active=self.config.profile_steps,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            with_stack=True,
            profile_memory=True,
            record_shapes=True,
            with_flops=True  # Track FLOPs if supported
        )

        logger.info(f"Started PyTorch profiler, output dir: {output_dir}")
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

        # Log memory statistics
        if self.device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            reserved_memory = torch.cuda.memory_reserved() / 1024**2  # MB
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**2  # MB

            self.memory_snapshots.append({
                'timestamp': time.time(),
                'current_memory_mb': current_memory,
                'peak_memory_mb': peak_memory,
                'reserved_memory_mb': reserved_memory,
                'total_memory_mb': total_memory
            })

    def log_memory_stats(self, stage_name: str):
        """Log memory statistics for a specific stage"""
        if self.device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024**2
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2

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
        """Generate comprehensive profiling report"""
        report = {
            'device_info': self.profiling_stats.get('device_info', {}),
            'stage_stats': self.profiling_stats.get('stage_stats', {}),
            'memory_snapshots': self.memory_snapshots,
            'interbatch_stats': self.interbatch_profiler.get_statistics()
        }

        # Memory summary
        if self.memory_snapshots:
            latest_snapshot = self.memory_snapshots[-1]
            report['memory_summary'] = {
                'current_memory_mb': latest_snapshot['current_memory_mb'],
                'peak_memory_mb': latest_snapshot['peak_memory_mb'],
                'reserved_memory_mb': latest_snapshot['reserved_memory_mb'],
                'total_memory_mb': latest_snapshot['total_memory_mb'],
                'stage_stats': self.profiling_stats.get('stage_stats', {})
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
        print("GPU PROFILING ANALYSIS REPORT")
        print("="*60)

        # Device information
        device_info = profiling_report.get('device_info', {})
        print(f"\nDevice: {device_info.get('device_name', 'Unknown')}")
        print(f"CUDA Available: {device_info.get('cuda_available', False)}")

        # Memory analysis
        memory_summary = profiling_report.get('memory_summary', {})
        if memory_summary:
            print(f"\nMemory Usage:")
            print(f"  Current: {memory_summary.get('current_memory_mb', 0):.1f} MB")
            print(f"  Peak: {memory_summary.get('peak_memory_mb', 0):.1f} MB")
            print(f"  Reserved: {memory_summary.get('reserved_memory_mb', 0):.1f} MB")
            print(f"  Total GPU: {memory_summary.get('total_memory_mb', 0):.1f} MB")

            # Memory efficiency
            total_memory = memory_summary.get('total_memory_mb', 1)
            peak_memory = memory_summary.get('peak_memory_mb', 0)
            memory_efficiency = (peak_memory / total_memory) * 100
            print(f"  Memory Efficiency: {memory_efficiency:.1f}%")

        # Stage-wise analysis
        stage_stats = memory_summary.get('stage_stats', {})
        if stage_stats:
            print(f"\nStage-wise Performance:")
            print(f"{'Stage':<30} {'Memory (MB)':<12} {'Calls':<8}")
            print("-" * 55)

            sorted_stages = sorted(stage_stats.items(),
                                 key=lambda x: x[1]['memory_used_mb'], reverse=True)

            for stage_name, stats in sorted_stages[:10]:  # Top 10 memory users
                memory_mb = stats.get('memory_used_mb', 0)
                call_count = stats.get('call_count', 0)
                print(f"{stage_name:<30} {memory_mb:>8.1f} {call_count:>8}")

        # Memory analysis summary
        memory_analysis = profiling_report.get('memory_analysis', {})
        if memory_analysis:
            print(f"\nMemory Analysis:")
            print(f"  Most Memory Intensive: {memory_analysis.get('most_memory_intensive_stage', 'N/A')}")
            print(f"  Total Memory Used: {memory_analysis.get('total_memory_used_mb', 0):.1f} MB")

        # Model information
        model_info = profiling_report.get('model_info', {})
        if model_info:
            print(f"\nModel Information:")
            print(f"  Parameters: {model_info.get('total_parameters', 0):,}")
            print(f"  Model Size: {model_info.get('model_size_mb', 0):.1f} MB")
            if 'hidden_dim' in model_info:
                print(f"  Hidden Dim: {model_info.get('hidden_dim', 0)}")
            if 'n_encoder_layers' in model_info:
                print(f"  Encoder Layers: {model_info.get('n_encoder_layers', 0)}")
            if 'n_decoder_layers' in model_info:
                print(f"  Decoder Layers: {model_info.get('n_decoder_layers', 0)}")

        print("\n" + "="*60)
        print("Recommendations:")

        # Generate recommendations based on profiling data
        recommendations = []

        peak_memory = memory_summary.get('peak_memory_mb', 0)
        if peak_memory > 8000:  # > 8GB
            recommendations.append("• Consider reducing batch size or sequence length")
            recommendations.append("• Enable gradient checkpointing for training")

        total_memory = memory_summary.get('total_memory_mb', 1)
        memory_efficiency = (peak_memory / total_memory) * 100 if total_memory > 0 else 0

        if memory_efficiency < 50:
            recommendations.append("• GPU memory utilization is low, consider increasing batch size")
        elif memory_efficiency > 90:
            recommendations.append("• High memory usage detected, monitor for OOM errors")

        most_intensive = memory_analysis.get('most_memory_intensive_stage', '')
        if most_intensive and 'decoder' in most_intensive.lower():
            recommendations.append("• Decoder is memory intensive, consider optimizing attention computation")
        elif most_intensive and 'encoder' in most_intensive.lower():
            recommendations.append("• Encoder layers using significant memory, consider layer-wise gradient checkpointing")

        # Add interbatch-specific recommendations
        interbatch_stats = profiling_report.get('interbatch_stats', {})
        data_efficiency = interbatch_stats.get('data_loading_efficiency_pct', 0)
        interbatch_overhead = interbatch_stats.get('interbatch_overhead_pct', 0)

        if data_efficiency > 20:
            recommendations.append("• Data loading taking >20% of time - increase num_workers or enable pin_memory")
        if interbatch_overhead > 10:
            recommendations.append("• High interbatch overhead - check for synchronization bottlenecks")

        if not recommendations:
            recommendations.append("• Profiling results look good, no immediate optimizations needed")

        for rec in recommendations:
            print(rec)

        print("="*60)

        # Print interbatch profiling report
        self.interbatch_profiler.print_report()

    def profile_training_steps(self, num_steps: int = 10):
        """Profile a specific number of training steps"""
        logger.info(f"Starting profiling for {num_steps} training steps")

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

                # Profile step
                self.profile_step()

                # Data loading profiling
                self.interbatch_profiler.start_data_loading()
                with torch.profiler.record_function("Data_Loading"):
                    mel_specs = batch['mel_specs'].to(self.device, non_blocking=True)
                    phoneme_indices = batch['phoneme_indices'].to(self.device, non_blocking=True)
                    phoneme_durations = batch['phoneme_durations'].to(self.device, non_blocking=True)
                    stop_token_targets = batch['stop_token_targets'].to(self.device, non_blocking=True)
                    mel_lengths = batch['mel_lengths'].to(self.device, non_blocking=True)
                    phoneme_lengths = batch['phoneme_lengths'].to(self.device, non_blocking=True)
                self.interbatch_profiler.end_data_loading()

                self.log_memory_stats("data_loading")

                # Forward pass profiling
                self.interbatch_profiler.start_forward_pass()
                with torch.profiler.record_function("Model_Forward"):
                    predicted_mel, predicted_log_durations, predicted_stop_logits = \
                        self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets)
                self.interbatch_profiler.end_forward_pass()

                self.log_memory_stats("forward_pass")

                # Loss calculation profiling
                with torch.profiler.record_function("Loss_Calculation"):
                    total_loss, loss_mel, loss_duration, loss_stop_token = self._calculate_losses(
                        predicted_mel, predicted_log_durations, predicted_stop_logits,
                        mel_specs, phoneme_durations, stop_token_targets,
                        mel_lengths, phoneme_lengths
                    )

                self.log_memory_stats("loss_calculation")

                # Backward pass profiling
                self.interbatch_profiler.start_backward_pass()
                with torch.profiler.record_function("Backward_Pass"):
                    total_loss.backward()
                self.interbatch_profiler.end_backward_pass()

                self.log_memory_stats("backward_pass")

                # End batch profiling
                batch_size = mel_specs.size(0)
                self.interbatch_profiler.end_batch(batch_size)

                step_time = time.time() - start_time
                total_time += step_time
                step_count += 1

                if step_count % 2 == 0:
                    logger.info(f"Profiling Step {step_count}, Time: {step_time:.3f}s")

            except Exception as e:
                logger.error(f"Error in profiling step {step_count}: {e}")
                continue

        self.stop_torch_profiler()

        # Generate and analyze report
        report = self.get_profiling_report()
        logger.info(f"Training profiling completed. Total time: {total_time:.2f}s, "
                   f"Avg time per step: {total_time/step_count:.3f}s")

        # Print analysis
        self.analyze_profiling_results(report)

        return report

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
        """Handle checkpoint resumption"""
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
        self.dataset.phoneme_processor = phoneme_processor
        logger.info(f"Resumed from epoch {self.start_epoch}, best loss {self.best_loss:.4f}")

    def train_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
        """Train for one epoch with enhanced profiling"""
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
            logger.info(f"Starting profiler for epoch {epoch+1} for {self.config.profile_steps} steps.")
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
                    mel_specs = batch['mel_specs'].to(self.device, non_blocking=True)
                    phoneme_indices = batch['phoneme_indices'].to(self.device, non_blocking=True)
                    phoneme_durations = batch['phoneme_durations'].to(self.device, non_blocking=True)
                    stop_token_targets = batch['stop_token_targets'].to(self.device, non_blocking=True)
                    mel_lengths = batch['mel_lengths'].to(self.device, non_blocking=True)
                    phoneme_lengths = batch['phoneme_lengths'].to(self.device, non_blocking=True)

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.end_data_loading()

                if is_profiling_epoch:
                    self.log_memory_stats("data_loading")

                with torch.profiler.record_function("Zero_Grad"):
                    self.optimizer.zero_grad()

                # Forward pass with interbatch profiling
                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.start_forward_pass()

                with torch.profiler.record_function("Model_Forward"):
                    predicted_mel, predicted_log_durations, predicted_stop_logits = \
                        self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets)

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.end_forward_pass()

                if is_profiling_epoch:
                    self.log_memory_stats("forward_pass")

                with torch.profiler.record_function("Loss_Calculation"):
                    total_loss, loss_mel, loss_duration, loss_stop_token = self._calculate_losses(
                        predicted_mel, predicted_log_durations, predicted_stop_logits,
                        mel_specs, phoneme_durations, stop_token_targets,
                        mel_lengths, phoneme_lengths
                    )

                if is_profiling_epoch:
                    self.log_memory_stats("loss_calculation")

                # Backward pass with interbatch profiling
                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.start_backward_pass()

                with torch.profiler.record_function("Backward_Pass"):
                    total_loss.backward()

                if enable_interbatch_profiling or is_profiling_epoch:
                    self.interbatch_profiler.end_backward_pass()

                if is_profiling_epoch:
                    self.log_memory_stats("backward_pass")

                with torch.profiler.record_function("Optimizer_Step"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                if is_profiling_epoch:
                    self.log_memory_stats("optimizer_step")

                # End batch profiling
                if enable_interbatch_profiling or is_profiling_epoch:
                    batch_size = mel_specs.size(0)
                    self.interbatch_profiler.end_batch(batch_size)

                # Memory cleanup timing
                cleanup_start = time.time()
                if batch_idx % 100 == 0 and batch_idx > 0:
                    if self.device.type == 'mps':
                        torch.mps.empty_cache()
                    elif self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                cleanup_time = time.time() - cleanup_start

                if cleanup_time > 1.0:
                    logger.warning(f"Slow memory cleanup: {cleanup_time:.2f}s at batch {batch_idx}")

                total_loss_epoch += total_loss.item()
                mel_loss_epoch += loss_mel.item()
                dur_loss_epoch += loss_duration.item()
                stop_loss_epoch += loss_stop_token.item()

                progress_bar.set_postfix({
                    'total_loss': total_loss.item(),
                    'mel_loss': loss_mel.item(),
                    'dur_loss': loss_duration.item(),
                    'stop_loss': loss_stop_token.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })

                # Print interbatch profiling stats periodically
                if enable_interbatch_profiling and batch_idx % getattr(self.config, 'interbatch_report_interval', 100) == 0 and batch_idx > 0:
                    logger.info(f"Interbatch profiling stats at batch {batch_idx}:")
                    stats = self.interbatch_profiler.get_statistics()
                    logger.info(f"  Avg interbatch time: {stats.get('interbatch_mean_ms', 0):.1f}ms")
                    logger.info(f"  Avg data loading time: {stats.get('data_loading_mean_ms', 0):.1f}ms")
                    logger.info(f"  Throughput: {stats.get('throughput_samples_per_sec', 0):.2f} samples/sec")

                if batch_idx % 50 == 0:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    elif self.device.type == 'mps':
                        torch.mps.empty_cache()

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps':
                    torch.mps.empty_cache()
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

        return (total_loss_epoch / num_batches,
                mel_loss_epoch / num_batches,
                dur_loss_epoch / num_batches,
                stop_loss_epoch / num_batches)

    def train(self):
        """Main training function"""
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.setup_checkpoint_resumption()
        save_phoneme_processor(self.dataset.phoneme_processor, self.config.output_dir)

        logger.info(f"Starting training on device: {self.device}")
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

        # Run standalone profiling if requested
        if hasattr(self.config, 'run_standalone_profiling') and self.config.run_standalone_profiling:
            logger.info("Running standalone profiling before training...")
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

            if (epoch + 1) % self.config.save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, avg_total_loss, self.config, self.config.output_dir
                )
                logger.info(f"Checkpoint saved for epoch {epoch+1}")

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()

        logger.info("Training finished. Saving final model.")
        save_final_model(self.model, self.config, self.config.output_dir)


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
            self.num_workers = 0
            self.pin_memory = False # Currently, not supported on MPS

            # Enhanced profiler configurations
            self.enable_profiling = False
            self.profile_epoch_start = 1  # Start profiling from this epoch (0-indexed)
            self.profile_wait_steps = 1  # Number of steps to wait before starting warmup
            self.profile_warmup_steps = 1 # Number of steps to warm up the profiler
            self.profile_steps = 5       # Number of active steps to profile
            self.run_standalone_profiling = False  # Run standalone profiling before training

            # Interbatch profiling configurations
            self.enable_interbatch_profiling = True  # Enable interbatch profiling
            self.interbatch_report_interval = 100    # Report interbatch stats every N batches

    temp_config = TrainingConfig()
    train_model(temp_config)
