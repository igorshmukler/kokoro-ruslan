#!/usr/bin/env python3
"""
Kokoro Language Model Training Script for Russian (Ruslan Corpus)
Training script without espeak dependency - uses direct phoneme processing
"""

import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import logging
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration for Kokoro model"""
    data_dir: str = "./ruslan_corpus"
    output_dir: str = "./kokoro_russian_model"
    batch_size: int = 16
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_every: int = 10

class RussianPhonemeProcessor:
    """
    Russian phoneme processor without espeak dependency
    Uses rule-based grapheme-to-phoneme conversion for Russian
    """
    
    def __init__(self):
        # Russian phoneme mapping (simplified)
        self.vowels = {
            'а': 'a', 'о': 'o', 'у': 'u', 'ы': 'i', 'э': 'e',
            'я': 'ja', 'ё': 'jo', 'ю': 'ju', 'и': 'i', 'е': 'je'
        }
        
        self.consonants = {
            'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ж': 'zh',
            'з': 'z', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
            'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'ф': 'f',
            'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch'
        }
        
        self.special_chars = {
            'ь': '', 'ъ': '', ' ': ' ', '.': '.', ',': ',',
            '!': '!', '?': '?', '-': '-'
        }
        
        # Combined phoneme vocabulary
        self.phonemes = list(self.vowels.values()) + list(self.consonants.values()) + \
                       list(self.special_chars.values())
        self.phonemes = list(set(self.phonemes))  # Remove duplicates
        
        # Create phoneme to index mapping
        self.phoneme_to_idx = {p: i for i, p in enumerate(self.phonemes)}
        self.idx_to_phoneme = {i: p for i, p in enumerate(self.phonemes)}
        
    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert Russian text to phonemes"""
        text = text.lower().strip()
        phonemes = []
        
        for char in text:
            if char in self.vowels:
                phonemes.append(self.vowels[char])
            elif char in self.consonants:
                phonemes.append(self.consonants[char])
            elif char in self.special_chars:
                if self.special_chars[char]:  # Skip empty mappings
                    phonemes.append(self.special_chars[char])
            else:
                # Unknown character, skip or replace with space
                phonemes.append(' ')
                
        return phonemes
    
    def phonemes_to_indices(self, phonemes: List[str]) -> List[int]:
        """Convert phonemes to indices"""
        return [self.phoneme_to_idx.get(p, 0) for p in phonemes]

class RuslanDataset(Dataset):
    """Dataset class for Ruslan corpus"""
    
    def __init__(self, data_dir: str, config: TrainingConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self.phoneme_processor = RussianPhonemeProcessor()
        
        # Load metadata
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples from Ruslan corpus")
        
    def _load_samples(self) -> List[Dict]:
        """Load samples from Ruslan corpus directory"""
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
        
        # Extract mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
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
        mel_spec = mel_transform(audio).squeeze(0)  # Remove channel dimension
        
        # Convert to log scale and normalize
        mel_spec = torch.log(mel_spec + 1e-9)  # Add small epsilon to avoid log(0)
        
        # Process text to phonemes
        phonemes = self.phoneme_processor.text_to_phonemes(sample['text'])
        phoneme_indices = self.phoneme_processor.phonemes_to_indices(phonemes)
        
        return {
            'mel_spec': mel_spec,
            'phoneme_indices': torch.tensor(phoneme_indices, dtype=torch.long),
            'text': sample['text'],
            'audio_file': sample['audio_file']
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader"""
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

class KokoroModel(torch.nn.Module):
    """
    Simplified Kokoro-style model architecture
    Text-to-Speech model with attention mechanism
    """
    
    def __init__(self, vocab_size: int, mel_dim: int = 80, hidden_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        
        # Text encoder
        self.text_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = torch.nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        
        # Project bidirectional LSTM output back to hidden_dim
        self.text_projection = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Mel feature projection to match hidden dimension
        self.mel_projection_in = torch.nn.Linear(mel_dim, hidden_dim)
        
        # Decoder
        self.decoder = torch.nn.LSTM(
            hidden_dim + hidden_dim, hidden_dim, batch_first=True
        )
        
        # Attention mechanism
        self.attention = torch.nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # Output projection
        self.mel_projection_out = torch.nn.Linear(hidden_dim, mel_dim)
        
    def forward(self, phoneme_indices: torch.Tensor, mel_specs: torch.Tensor = None) -> torch.Tensor:
        batch_size = phoneme_indices.size(0)
        
        # Text encoding
        text_emb = self.text_embedding(phoneme_indices)
        text_encoded, _ = self.text_encoder(text_emb)
        
        # Project bidirectional output to hidden_dim
        text_encoded = self.text_projection(text_encoded)
        
        if mel_specs is not None:
            # Training mode
            seq_len = mel_specs.size(1)
            outputs = []
            
            hidden = None
            for t in range(seq_len):
                # Current mel frame
                if t == 0:
                    mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=mel_specs.device)
                else:
                    mel_input = mel_specs[:, t-1:t, :]
                
                # Project mel to hidden dimension
                mel_projected = self.mel_projection_in(mel_input)
                
                # Attention over text
                attended, _ = self.attention(
                    mel_projected,
                    text_encoded,
                    text_encoded
                )
                
                # Decoder step
                decoder_input = torch.cat([mel_projected, attended], dim=2)
                decoder_out, hidden = self.decoder(decoder_input, hidden)
                
                # Project back to mel
                mel_pred = self.mel_projection_out(decoder_out)
                outputs.append(mel_pred)
            
            return torch.cat(outputs, dim=1)
        else:
            # Inference mode (simplified)
            max_len = 1000
            outputs = []
            hidden = None
            mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=phoneme_indices.device)
            
            for t in range(max_len):
                # Project mel to hidden dimension
                mel_projected = self.mel_projection_in(mel_input)
                
                # Attention over text
                attended, _ = self.attention(
                    mel_projected,
                    text_encoded,
                    text_encoded
                )
                
                # Decoder step
                decoder_input = torch.cat([mel_projected, attended], dim=2)
                decoder_out, hidden = self.decoder(decoder_input, hidden)
                
                # Project back to mel
                mel_pred = self.mel_projection_out(decoder_out)
                outputs.append(mel_pred)
                
                # Use prediction as next input
                mel_input = mel_pred
            
            return torch.cat(outputs, dim=1)

def train_model(config: TrainingConfig):
    """Main training function"""
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = RuslanDataset(config.data_dir, config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize model
    vocab_size = len(dataset.phoneme_processor.phonemes)
    model = KokoroModel(vocab_size, config.n_mels)
    model.to(config.device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            mel_specs = batch['mel_specs'].to(config.device)
            phoneme_indices = batch['phoneme_indices'].to(config.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(phoneme_indices, mel_specs)
            
            # Calculate loss
            loss = criterion(predictions, mel_specs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'phoneme_processor': dataset.phoneme_processor
            }
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "kokoro_russian_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'phoneme_processor': dataset.phoneme_processor,
        'config': config
    }, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

def main():
    """Main function"""
    # Configuration
    config = TrainingConfig(
        data_dir="./ruslan_corpus",
        output_dir="./kokoro_russian_model",
        batch_size=8,  # Adjust based on GPU memory
        learning_rate=1e-4,
        num_epochs=100,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        n_fft=1024,
        n_mels=80,
        f_min=0.0,
        f_max=8000.0,  # Set to sr/2 or lower to avoid issues
        save_every=10
    )
    
    # Check if data directory exists
    if not os.path.exists(config.data_dir):
        logger.error(f"Data directory not found: {config.data_dir}")
        logger.info("Please ensure the Ruslan corpus is available in the specified directory")
        return
    
    # Start training
    train_model(config)

if __name__ == "__main__":
    main()