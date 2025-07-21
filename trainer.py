#!/usr/bin/env python3
"""
Training logic for Kokoro Language Model
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from config import TrainingConfig
from dataset import RuslanDataset, collate_fn
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
        self.device = config.device

        # Initialize dataset
        self.dataset = RuslanDataset(config.data_dir, config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )

        # Initialize model
        vocab_size = self.dataset.phoneme_processor.get_vocab_size()
        self.model = KokoroModel(vocab_size, config.n_mels)
        self.model.to(self.device)

        # Log model information
        model_info = self.model.get_model_info()
        logger.info(f"Model initialized with {model_info['total_parameters']:,} parameters ({model_info['model_size_mb']:.1f} MB)")

        # Initialize optimizer and loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.criterion = torch.nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')

    def setup_checkpoint_resumption(self):
        """Handle checkpoint resumption"""
        if not self.config.resume_checkpoint:
            return

        if self.config.resume_checkpoint.lower() == 'auto':
            # Automatically find latest checkpoint
            latest_checkpoint = find_latest_checkpoint(self.config.output_dir)
            if latest_checkpoint:
                self.start_epoch, self.best_loss, phoneme_processor = load_checkpoint(
                    latest_checkpoint, self.model, self.optimizer, self.scheduler, self.config.output_dir
                )
                # Update dataset's phoneme processor
                self.dataset.phoneme_processor = phoneme_processor
            else:
                logger.info("No checkpoint found for auto-resume, starting from scratch")
        else:
            # Load specific checkpoint
            if os.path.exists(self.config.resume_checkpoint):
                self.start_epoch, self.best_loss, phoneme_processor = load_checkpoint(
                    self.config.resume_checkpoint, self.model, self.optimizer, 
                    self.scheduler, self.config.output_dir
                )
                self.dataset.phoneme_processor = phoneme_processor
            else:
                logger.error(f"Checkpoint not found: {self.config.resume_checkpoint}")
                raise FileNotFoundError(f"Checkpoint not found: {self.config.resume_checkpoint}")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                mel_specs = batch['mel_specs'].to(self.device, non_blocking=True)
                phoneme_indices = batch['phoneme_indices'].to(self.device, non_blocking=True)

                # Forward pass
                self.optimizer.zero_grad()

                predictions = self.model(phoneme_indices, mel_specs)

                # Calculate loss
                loss = self.criterion(predictions, mel_specs)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

                # Clear cache periodically to prevent memory buildup
                if batch_idx % 50 == 0:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                # Clear cache on error
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                continue

        return total_loss / len(self.dataloader)

    def train(self):
        """Main training function"""
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup checkpoint resumption
        self.setup_checkpoint_resumption()

        # Save phoneme processor separately at the start
        save_phoneme_processor(self.dataset.phoneme_processor, self.config.output_dir)

        # Training info
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Training from epoch {self.start_epoch + 1} to {self.config.num_epochs}")
        logger.info(f"Model vocabulary size: {self.dataset.phoneme_processor.get_vocab_size()}")

        # Training loop
        for epoch in range(self.start_epoch, self.config.num_epochs):
            avg_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

            # Step scheduler
            self.scheduler.step(avg_loss)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, 
                    epoch, avg_loss, self.config, self.config.output_dir
                )

            # Clear cache after each epoch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Save final model
        save_final_model(self.model, self.config, self.config.output_dir)


def train_model(config: TrainingConfig):
    """Main training function - backward compatibility wrapper"""
    trainer = KokoroTrainer(config)
    trainer.train()
