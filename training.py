#!/usr/bin/env python3
"""
Kokoro Language Model Training Script for Russian (Ruslan Corpus)
Training script optimized for Mac MPS acceleration
"""

import os
import json
import torch
import torchaudio
import numpy as np
import argparse
import signal
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import logging
from tqdm import tqdm
import re
import pickle

# Import our model and phoneme processor
from model import KokoroModel
from russian_phoneme_processor import RussianPhonemeProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
interrupted = False
current_model = None
current_optimizer = None
current_scheduler = None
current_config = None
current_epoch = 0
current_phoneme_processor = None

def signal_handler(signum, frame):
    """Handle keyboard interrupt (Ctrl+C) gracefully"""
    global interrupted
    interrupted = True
    logger.warning("\n" + "="*50)
    logger.warning("Keyboard interrupt received (Ctrl+C)")
    logger.warning("Training will stop after current batch...")
    logger.warning("="*50)

def save_interrupt_checkpoint(epoch: int, loss: float):
    """Save checkpoint when interrupted"""
    global current_model, current_optimizer, current_scheduler, current_config, current_phoneme_processor

    if current_model is None or current_config is None:
        logger.error("Cannot save interrupt checkpoint - model or config not available")
        return

    try:
        # Create output directory if it doesn't exist
        os.makedirs(current_config.output_dir, exist_ok=True)

        # Save interrupt checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': current_model.state_dict(),
            'optimizer_state_dict': current_optimizer.state_dict() if current_optimizer else None,
            'scheduler_state_dict': current_scheduler.state_dict() if current_scheduler else None,
            'loss': loss,
            'config': current_config,
            'interrupted': True,  # Mark as interrupted checkpoint
            'phoneme_processor': current_phoneme_processor
        }

        interrupt_path = os.path.join(current_config.output_dir, f"interrupt_checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, interrupt_path)
        logger.info(f"✅ Interrupt checkpoint saved: {interrupt_path}")

        # Also save as latest checkpoint for easy resuming
        latest_path = os.path.join(current_config.output_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)
        logger.info(f"✅ Latest checkpoint saved: {latest_path}")

        # Save phoneme processor separately
        if current_phoneme_processor:
            save_phoneme_processor(current_phoneme_processor, current_config.output_dir)

        logger.info("💾 All checkpoint data saved successfully!")
        logger.info(f"📝 To resume training, use: --resume {interrupt_path}")
        logger.info("   or use: --resume auto (to auto-detect latest)")

    except Exception as e:
        logger.error(f"❌ Error saving interrupt checkpoint: {e}")
        logger.error("⚠️  Training progress may be lost!")

@dataclass
class TrainingConfig:
    """Training configuration for Kokoro model"""
    data_dir: str = "./ruslan_corpus"
    output_dir: str = "./kokoro_russian_model"
    batch_size: int = 12  # Increased for MPS
    learning_rate: float = 1e-4
    num_epochs: int = 100
    max_seq_length: int = 1024
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    save_every: int = 2
    use_mixed_precision: bool = True  # Enable for MPS
    num_workers: int = 0  # Set to 0 for MPS compatibility
    pin_memory: bool = False  # Disable for MPS
    resume_checkpoint: Optional[str] = None  # Path to checkpoint to resume from


class RuslanDataset(Dataset):
    """Dataset class for Ruslan corpus - optimized for MPS"""

    def __init__(self, data_dir: str, config: TrainingConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self.phoneme_processor = RussianPhonemeProcessor()

        # Pre-create mel transform for efficiency
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=2.0,
            normalized=False
        )

        # Load metadata
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples from corpus at {data_dir}")
        logger.info(f"Using phoneme processor: {self.phoneme_processor}")

    def _load_samples(self) -> List[Dict]:
        """Load samples from corpus directory"""
        samples = []

        # Look for metadata file (adjust path as needed)
        metadata_file = self.data_dir / "metadata_RUSLAN_22200.csv"
        if metadata_file.exists():
            logger.info(f"Loading metadata from {metadata_file}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        audio_file = parts[0]
                        text = parts[1]

                        audio_path = self.data_dir / "wavs" / f"{audio_file}.wav"
                        if audio_path.exists():
                            samples.append({
                                'audio_path': str(audio_path),
                                'text': text,
                                'audio_file': audio_file
                            })
        else:
            # Fallback: scan for wav files and corresponding text files
            logger.warning(f"Metadata file not found: {metadata_file}. Falling back to directory scan.")
            wav_dir = self.data_dir / "wavs"
            txt_dir = self.data_dir / "texts"
    
            if wav_dir.exists():
                for wav_file in wav_dir.glob("*.wav"):
                    txt_file = txt_dir / f"{wav_file.stem}.txt"
                    if txt_file.exists():
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        samples.append({
                            'audio_path': str(wav_file),
                            'text': text,
                            'audio_file': wav_file.stem
                        })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load audio
        audio, sr = torchaudio.load(sample['audio_path'])

        # Resample if necessary
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            audio = resampler(audio)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Normalize audio to prevent numerical issues
        audio = audio / (torch.max(torch.abs(audio)) + 1e-9)

        # Extract mel spectrogram using pre-created transform
        mel_spec = self.mel_transform(audio).squeeze(0)  # Remove channel dimension

        # Convert to log scale and normalize
        mel_spec = torch.log(mel_spec + 1e-9)  # Add small epsilon to avoid log(0)

        # Clip extremely long sequences to prevent memory issues
        max_frames = 800  # Reduced for better MPS memory management
        if mel_spec.shape[1] > max_frames:
            mel_spec = mel_spec[:, :max_frames]

        # Process text to phonemes using the dedicated processor
        phoneme_indices = self.phoneme_processor.text_to_indices(sample['text'])

        return {
            'mel_spec': mel_spec,
            'phoneme_indices': torch.tensor(phoneme_indices, dtype=torch.long),
            'text': sample['text'],
            'audio_file': sample['audio_file']
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader - optimized for MPS"""
    mel_specs = [item['mel_spec'].transpose(0, 1) for item in batch]  # (time, mel_dim)
    phoneme_indices = [item['phoneme_indices'] for item in batch]
    texts = [item['text'] for item in batch]
    audio_files = [item['audio_file'] for item in batch]

    # Pad sequences
    mel_specs_padded = pad_sequence(mel_specs, batch_first=True, padding_value=0)
    phoneme_indices_padded = pad_sequence(phoneme_indices, batch_first=True, padding_value=0)

    return {
        'mel_specs': mel_specs_padded,
        'phoneme_indices': phoneme_indices_padded,
        'texts': texts,
        'audio_files': audio_files
    }


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


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler, output_dir: str) -> Tuple[int, float, RussianPhonemeProcessor]:
    """Load checkpoint and return starting epoch, best loss, and phoneme processor"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Add safe globals for our custom classes
    torch.serialization.add_safe_globals([TrainingConfig, RussianPhonemeProcessor])

    try:
        # Try loading with weights_only=True first (new default)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))

        # Check if this was an interrupted checkpoint
        if checkpoint.get('interrupted', False):
            logger.info("📝 Resuming from interrupted training session")

        # Try to load phoneme processor from checkpoint first
        if 'phoneme_processor' in checkpoint:
            phoneme_processor = checkpoint['phoneme_processor']
        else:
            # Fall back to loading from separate file
            phoneme_processor = load_phoneme_processor(output_dir)

        logger.info(f"✅ Resumed from epoch {start_epoch} with loss {best_loss:.4f}")
        return start_epoch, best_loss, phoneme_processor

    except Exception as e:
        logger.warning(f"Loading with weights_only=True failed: {e}")
        logger.info("Trying to load with weights_only=False for compatibility...")

        try:
            # Try loading with weights_only=False for older checkpoints
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('loss', float('inf'))

            if checkpoint.get('interrupted', False):
                logger.info("📝 Resuming from interrupted training session")

            if 'phoneme_processor' in checkpoint:
                phoneme_processor = checkpoint['phoneme_processor']
            else:
                phoneme_processor = load_phoneme_processor(output_dir)

            logger.info(f"✅ Resumed from epoch {start_epoch} with loss {best_loss:.4f}")
            return start_epoch, best_loss, phoneme_processor

        except Exception as e2:
            logger.error(f"Error loading checkpoint even with weights_only=False: {e2}")
            raise e2

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the output directory"""
    checkpoint_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        return None

    # Look for latest checkpoint first
    latest_checkpoint = checkpoint_dir / "latest_checkpoint.pth"
    if latest_checkpoint.exists():
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

    # Fall back to searching for numbered checkpoints
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    interrupt_files = list(checkpoint_dir.glob("interrupt_checkpoint_epoch_*.pth"))

    all_checkpoints = checkpoint_files + interrupt_files

    if not all_checkpoints:
        return None

    # Sort by epoch number
    all_checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    latest_checkpoint = all_checkpoints[-1]

    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)


def train_model(config: TrainingConfig):
    """Main training function - optimized for MPS with interrupt support"""
    global current_model, current_optimizer, current_scheduler, current_config, current_epoch, current_phoneme_processor

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Store config globally for interrupt handling
    current_config = config

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize dataset
    dataset = RuslanDataset(config.data_dir, config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # Initialize model
    vocab_size = dataset.phoneme_processor.get_vocab_size()
    model = KokoroModel(vocab_size, config.n_mels)
    model.to(config.device)
    current_model = model

    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model initialized with {model_info['total_parameters']:,} parameters ({model_info['model_size_mb']:.1f} MB)")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    current_optimizer = optimizer

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    current_scheduler = scheduler

    # Handle checkpoint resumption
    start_epoch = 0
    best_loss = float('inf')
    phoneme_processor = dataset.phoneme_processor
    current_phoneme_processor = phoneme_processor

    if config.resume_checkpoint:
        if config.resume_checkpoint.lower() == 'auto':
            # Automatically find latest checkpoint
            latest_checkpoint = find_latest_checkpoint(config.output_dir)
            if latest_checkpoint:
                start_epoch, best_loss, phoneme_processor = load_checkpoint(
                    latest_checkpoint, model, optimizer, scheduler, config.output_dir
                )
                # Update dataset's phoneme processor
                dataset.phoneme_processor = phoneme_processor
                current_phoneme_processor = phoneme_processor
            else:
                logger.info("No checkpoint found for auto-resume, starting from scratch")
        else:
            # Load specific checkpoint
            if os.path.exists(config.resume_checkpoint):
                start_epoch, best_loss, phoneme_processor = load_checkpoint(
                    config.resume_checkpoint, model, optimizer, scheduler, config.output_dir
                )
                dataset.phoneme_processor = phoneme_processor
                current_phoneme_processor = phoneme_processor
            else:
                logger.error(f"Checkpoint not found: {config.resume_checkpoint}")
                return

    # Save phoneme processor separately at the start
    save_phoneme_processor(dataset.phoneme_processor, config.output_dir)

    # Training loop
    logger.info(f"🚀 Starting training on device: {config.device}")
    logger.info(f"📊 Training from epoch {start_epoch + 1} to {config.num_epochs}")
    logger.info(f"📚 Model vocabulary size: {vocab_size}")
    logger.info("💡 Press Ctrl+C to gracefully stop training and save checkpoint")

    try:
        for epoch in range(start_epoch, config.num_epochs):
            if interrupted:
                logger.info("🛑 Training interrupted by user")
                break

            current_epoch = epoch
            model.train()
            total_loss = 0.0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                # Check for interrupt at the beginning of each batch
                if interrupted:
                    logger.info("🛑 Stopping training due to keyboard interrupt...")
                    avg_loss = total_loss / max(batch_idx, 1)
                    save_interrupt_checkpoint(epoch, avg_loss)
                    return

                try:
                    # Move to device
                    mel_specs = batch['mel_specs'].to(config.device, non_blocking=True)
                    phoneme_indices = batch['phoneme_indices'].to(config.device, non_blocking=True)

                    # Forward pass
                    optimizer.zero_grad()

                    predictions = model(phoneme_indices, mel_specs)

                    # Calculate loss
                    loss = criterion(predictions, mel_specs)
                    loss.backward()

                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

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

            # Check for interrupt after epoch
            if interrupted:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"🛑 Training interrupted at end of epoch {epoch+1}")
                save_interrupt_checkpoint(epoch, avg_loss)
                return

            avg_loss = total_loss / len(dataloader)
            logger.info(f"✅ Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

            # Step scheduler
            scheduler.step(avg_loss)

            # Save checkpoint
            if (epoch + 1) % config.save_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'config': config,
                    'phoneme_processor': current_phoneme_processor
                }
                checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

            # Clear cache after each epoch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Training completed normally
        if not interrupted:
            logger.info("🎉 Training completed successfully!")

            # Save final model
            final_model_path = os.path.join(config.output_dir, "kokoro_russian_final.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'phoneme_processor': current_phoneme_processor
            }, final_model_path)
            logger.info(f"💾 Final model saved: {final_model_path}")

    except KeyboardInterrupt:
        # This catches any remaining KeyboardInterrupt that wasn't handled by the signal handler
        logger.info("🛑 Training interrupted by keyboard interrupt")
        avg_loss = total_loss / max(len(dataloader), 1)
        save_interrupt_checkpoint(current_epoch, avg_loss)

    except Exception as e:
        logger.error(f"❌ Unexpected error during training: {e}")
        # Try to save emergency checkpoint
        try:
            avg_loss = total_loss / max(len(dataloader), 1) if 'total_loss' in locals() else float('inf')
            save_interrupt_checkpoint(current_epoch, avg_loss)
        except:
            logger.error("⚠️  Could not save emergency checkpoint")
        raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Kokoro Language Model Training Script for Russian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with custom directories
  python train_kokoro.py --corpus /path/to/corpus --output ./my_model

  # Resume training from latest checkpoint (including interrupted ones)
  python train_kokoro.py --corpus /path/to/corpus --output ./my_model --resume auto

  # Resume from specific checkpoint
  python train_kokoro.py --corpus /path/to/corpus --output ./my_model --resume /path/to/checkpoint.pth

  # Training with all custom options
  python train_kokoro.py --corpus /path/to/corpus --output ./my_model --batch-size 16 --epochs 50

Keyboard Interrupt Support:
  - Press Ctrl+C to gracefully stop training and save a checkpoint
  - Use --resume auto to automatically resume from the latest checkpoint (including interrupted ones)
  - Interrupted checkpoints are saved as both 'interrupt_checkpoint_epoch_X.pth' and 'latest_checkpoint.pth'
        """
    )

    parser.add_argument(
        '--corpus', '-c',
        type=str,
        default='./ruslan_corpus',
        help='Path to the corpus directory (default: ./ruslan_corpus)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./kokoro_russian_model',
        help='Path to the output model directory (default: ./kokoro_russian_model)'
    )

    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Resume training from checkpoint. Use "auto" to auto-detect latest checkpoint, or provide path to specific checkpoint file'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Batch size for training (default: 8)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )

    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )

    parser.add_argument(
        '--save-every',
        type=int,
        default=2,
        help='Save checkpoint every N epochs (default: 2)'
    )

    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()

    # Check MPS availability
    if torch.backends.mps.is_available():
        logger.info("🚀 MPS (Metal Performance Shaders) is available and will be used for acceleration")
    else:
        logger.warning("⚠️  MPS is not available, falling back to CPU")

    # Configuration with CLI arguments
    config = TrainingConfig(
        data_dir=args.corpus,
        output_dir=args.output,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        n_fft=1024,
        n_mels=80,
        f_min=0.0,
        f_max=8000.0,
        save_every=args.save_every,
        use_mixed_precision=True,
        num_workers=0,  # Important for MPS
        pin_memory=False,  # Important for MPS
        resume_checkpoint=args.resume
    )

    # Validate directories
    if not os.path.exists(config.data_dir):
        logger.error(f"❌ Corpus directory not found: {config.data_dir}")
        logger.info("Please ensure the corpus is available in the specified directory")
        return

    logger.info(f"📂 Using corpus directory: {config.data_dir}")
    logger.info(f"💾 Using output directory: {config.output_dir}")

    if config.resume_checkpoint:
        logger.info(f"🔄 Resume mode: {config.resume_checkpoint}")

    # Start training
    try:
        train_model(config)
        logger.info("🏁 Training session ended")
    except KeyboardInterrupt:
        logger.info("🛑 Training session terminated by user")
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
