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
        self.device = torch.device(config.device)

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

        # Learning rate scheduler: Cosine Annealing with Warm Restarts
        # This replaces ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.lr_T_0,         # Number of epochs for the first restart cycle
            T_mult=self.config.lr_T_mult,   # Multiplier for subsequent cycle lengths
            eta_min=self.config.lr_eta_min  # Minimum learning rate
        )

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf') # Still useful for checkpointing best model, if desired

    def setup_checkpoint_resumption(self):
        """Handle checkpoint resumption"""
        if not self.config.resume_checkpoint:
            logger.info("No resume checkpoint specified, starting from scratch.")
            return

        checkpoint_path = None
        if self.config.resume_checkpoint.lower() == 'auto':
            # Automatically find latest checkpoint
            checkpoint_path = find_latest_checkpoint(self.config.output_dir)
            if not checkpoint_path:
                logger.info("No checkpoint found for auto-resume, starting from scratch.")
                return
        else:
            # Use specific checkpoint path
            checkpoint_path = self.config.resume_checkpoint
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        self.start_epoch, self.best_loss, phoneme_processor = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler, self.config.output_dir
        )
        # Update dataset's phoneme processor
        self.dataset.phoneme_processor = phoneme_processor
        logger.info(f"Resumed from epoch {self.start_epoch}, best loss {self.best_loss:.4f}")


    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.dataloader)

        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                mel_specs = batch['mel_specs'].to(self.device, non_blocking=True)
                phoneme_indices = batch['phoneme_indices'].to(self.device, non_blocking=True)
                # You might also want to pass mel_lengths here if your model uses them
                # mel_lengths = batch['mel_lengths'].to(self.device, non_blocking=True)

                # Forward pass
                self.optimizer.zero_grad()

                # Call your model. Assuming it takes phoneme_indices.
                # If your model needs teacher forcing (e.g., autoregressive decoder),
                # it should also take `mel_specs` as an input to guide the decoding.
                # If `KokoroModel` expects mel_specs for teacher forcing:
                predictions = self.model(phoneme_indices, mel_specs)
                # If KokoroModel only takes phoneme_indices:
                # predictions = self.model(phoneme_indices)


                # Determine the minimum sequence length in the batch for both predictions and targets
                min_len = min(predictions.size(1), mel_specs.size(1))

                # Crop both predictions and mel_specs to this minimum length
                predictions = predictions[:, :min_len, :]
                mel_specs_cropped = mel_specs[:, :min_len, :]

                # Calculate loss
                loss = self.criterion(predictions, mel_specs_cropped) # Use the cropped mel_specs
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()

                progress_bar.set_postfix({'loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr']})

                # Clear cache periodically to prevent memory buildup
                if batch_idx % 50 == 0:
                    if self.device.type == 'cuda': # Check for CUDA
                        torch.cuda.empty_cache()
                    elif self.device.type == 'mps': # Check for MPS
                        torch.mps.empty_cache()

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                # Clear cache on error
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps':
                    torch.mps.empty_cache()
                continue # Continue to next batch despite error

        return total_loss / num_batches

    def train(self):
        """Main training function"""
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup checkpoint resumption
        self.setup_checkpoint_resumption()

        # Save phoneme processor separately at the start (or if resuming, it's already loaded)
        # This ensures the processor consistent with data used for training/inference
        save_phoneme_processor(self.dataset.phoneme_processor, self.config.output_dir)

        # Training info
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Total epochs: {self.config.num_epochs}, Starting from epoch: {self.start_epoch + 1}")
        logger.info(f"Model vocabulary size: {self.dataset.phoneme_processor.get_vocab_size()}")
        logger.info(f"Initial learning rate: {self.config.learning_rate}")
        logger.info(f"Scheduler: CosineAnnealingWarmRestarts (T_0={self.config.lr_T_0}, T_mult={self.config.lr_T_mult}, eta_min={self.config.lr_eta_min})")


        # Training loop
        for epoch in range(self.start_epoch, self.config.num_epochs):
            avg_loss = self.train_epoch(epoch)

            # Step scheduler AFTER each epoch.
            # For CosineAnnealingWarmRestarts, you typically don't pass a metric
            # unless you're using it in a more complex way. It's time-based.
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}, Current LR: {current_lr:.8f}")

            # Save checkpoint (consider saving best model based on avg_loss as well)
            if (epoch + 1) % self.config.save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, avg_loss, self.config, self.config.output_dir
                )
                logger.info(f"Checkpoint saved for epoch {epoch+1}")

            # Clear cache after each epoch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()

        # Save final model
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
    temp_config = TrainingConfig(
        data_dir="data/processed_data", # Adjust to your actual data path
        output_dir="output_models",
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        lr_T_0=10,
        lr_T_mult=2,
        lr_eta_min=1e-6,
        save_every=5,
        resume_checkpoint='auto' # or specific path, or ""
    )
    train_model(temp_config)
