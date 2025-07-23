#!/usr/bin/env python3
"""
Training logic for Kokoro Language Model
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from typing import Tuple
from config import TrainingConfig
from dataset import RuslanDataset, collate_fn, LengthBasedBatchSampler
from model import KokoroModel
from checkpoint_manager import (
    save_phoneme_processor, load_checkpoint, find_latest_checkpoint,
    save_checkpoint, save_final_model
)

logger = logging.getLogger(__name__)

class KokoroTrainer:
    """Main trainer class for Kokoro model"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize dataset
        self.dataset = RuslanDataset(config.data_dir, config)

        # Initialize the custom LengthBasedBatchSampler
        # Set shuffle=True in the sampler to shuffle batches and potentially within windows
        self.batch_sampler = LengthBasedBatchSampler(
            dataset=self.dataset,
            batch_size=config.batch_size,
            drop_last=True, # Drop last partial batch for consistency
            shuffle=True    # Shuffle the order of batches
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler, # Use batch_sampler instead of batch_size and shuffle
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
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

        self.criterion_mel = nn.L1Loss(reduction='none') # Use reduction='none' for masking
        self.criterion_duration = nn.MSELoss(reduction='none') # Use reduction='none' for masking
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
        """Train for one epoch"""
        self.model.train()
        total_loss_epoch = 0.0
        mel_loss_epoch = 0.0
        dur_loss_epoch = 0.0
        stop_loss_epoch = 0.0

        # Ensure the dataloader is iterated properly with the batch_sampler
        # The length of the dataloader will now be the number of batches from the sampler
        num_batches = len(self.dataloader)

        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                mel_specs = batch['mel_specs'].to(self.device, non_blocking=True)
                phoneme_indices = batch['phoneme_indices'].to(self.device, non_blocking=True)
                phoneme_durations = batch['phoneme_durations'].to(self.device, non_blocking=True)
                stop_token_targets = batch['stop_token_targets'].to(self.device, non_blocking=True)
                mel_lengths = batch['mel_lengths'].to(self.device, non_blocking=True)
                phoneme_lengths = batch['phoneme_lengths'].to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                predicted_mel, predicted_log_durations, predicted_stop_logits = \
                    self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets)

                # --- Calculate Losses with Masking ---

                # 1. Mel Spectrogram Loss
                # Create a mask for valid mel frames
                # mel_specs_padded has shape (batch_size, max_mel_len_in_batch, n_mels)
                # predicted_mel has shape (batch_size, max_mel_len_in_batch, n_mels)
                max_mel_len_batch = mel_specs.size(1)
                mel_mask = torch.arange(max_mel_len_batch, device=self.device).expand(len(mel_lengths), max_mel_len_batch) < mel_lengths.unsqueeze(1)
                mel_mask = mel_mask.unsqueeze(-1).expand_as(predicted_mel).float() # Expand to n_mels dimension

                # Apply mask to loss
                loss_mel_unreduced = self.criterion_mel(
                    predicted_mel,
                    mel_specs
                )
                loss_mel = (loss_mel_unreduced * mel_mask).sum() / mel_mask.sum()


                # 2. Duration Loss
                # Create a mask for valid phoneme durations
                # phoneme_indices_padded has shape (batch_size, max_phoneme_len_in_batch)
                # predicted_log_durations has shape (batch_size, max_phoneme_len_in_batch)
                max_phoneme_len_batch = phoneme_indices.size(1)
                phoneme_mask = torch.arange(max_phoneme_len_batch, device=self.device).expand(len(phoneme_lengths), max_phoneme_len_batch) < phoneme_lengths.unsqueeze(1)
                phoneme_mask = phoneme_mask.float() # Convert to float for multiplication

                target_log_durations = torch.log(phoneme_durations.float() + 1e-5)

                loss_duration_unreduced = self.criterion_duration(
                    predicted_log_durations,
                    target_log_durations
                )
                loss_duration = (loss_duration_unreduced * phoneme_mask).sum() / phoneme_mask.sum()

                # 3. Stop Token Loss
                # The stop token loss is also on the mel sequence, so use the same mel_mask
                loss_stop_token_unreduced = self.criterion_stop_token(
                    predicted_stop_logits,
                    stop_token_targets
                )
                # For stop token loss, the mask doesn't need to expand to n_mels, just sequence length
                stop_token_mask = mel_mask[:, :, 0] # Take the 0th channel of mel_mask
                loss_stop_token = (loss_stop_token_unreduced * stop_token_mask).sum() / stop_token_mask.sum()

                # Combine all losses
                total_loss = loss_mel + \
                             loss_duration * self.config.duration_loss_weight + \
                             loss_stop_token * self.config.stop_token_loss_weight

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

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
    # You would typically load config from a file or command-line arguments
    # For demonstration, creating a dummy config
    # MAKE SURE YOUR ACTUAL TrainingConfig CLASS HAS THESE NEW ATTRIBUTES
    class TrainingConfig:
        def __init__(self):
            self.data_dir = "data/processed_data" # Adjust to your actual data path
            self.output_dir = "output_models"
            self.num_epochs = 100
            self.batch_size = 16
            self.learning_rate = 1e-4 # Often reduced for stability
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.lr_T_0 = 20
            self.lr_T_mult = 2
            self.lr_eta_min = 1e-6
            self.save_every = 5
            self.resume_checkpoint = 'auto' # or specific path, or ""
            self.n_mels = 80
            self.hidden_dim = 512
            self.duration_loss_weight = 0.1
            self.stop_token_loss_weight = 1.0
            self.max_seq_length = 1024 # Add this to your config
            self.sample_rate = 22050
            self.hop_length = 256
            self.win_length = 1024
            self.n_fft = 1024
            self.f_min = 0.0
            self.f_max = 8000.0
            self.num_workers = 0 # For MPS, keep at 0
            self.pin_memory = False # For MPS, keep at False

    temp_config = TrainingConfig()
    train_model(temp_config)
