#!/usr/bin/env python3
"""
Dataset implementation for Ruslan corpus
"""

import torch
import torchaudio
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging

from config import TrainingConfig
from russian_phoneme_processor import RussianPhonemeProcessor

logger = logging.getLogger(__name__)


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

        # Add a warning about dummy durations if using the placeholder method
        logger.warning(
            "Phoneme durations are being generated as a placeholder (uniform distribution). "
            "For high-quality TTS, you MUST replace this with actual phoneme durations "
            "obtained from a forced aligner (e.g., Montreal Forced Aligner)."
        )

    def _load_samples(self) -> List[Dict]:
        """
        Load samples from corpus directory.

        NOTE: For proper duration modeling, this method would ideally also load
        or point to pre-computed phoneme durations (e.g., from forced alignment files).
        This current implementation only loads audio path and text.
        """
        samples = []

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
                                # Potentially add 'duration_path': self.data_dir / "durations" / f"{audio_file}.json"
                            })
        else:
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
        # This max_frames should ideally be aligned with your model's capacity
        max_frames = self.config.max_seq_length
        if mel_spec.shape[1] > max_frames:
            mel_spec = mel_spec[:, :max_frames]

        # Process text to phonemes using the dedicated processor
        phoneme_indices = self.phoneme_processor.text_to_indices(sample['text'])
        phoneme_indices_tensor = torch.tensor(phoneme_indices, dtype=torch.long)

        # --- Generate Phoneme Durations (PLACEHOLDER/DUMMY) ---
        # IMPORTANT: For actual training, these MUST come from forced alignment data.
        # This is a highly simplified approximation.
        num_mel_frames = mel_spec.shape[1]
        num_phonemes = phoneme_indices_tensor.shape[0]

        if num_phonemes == 0:
            phoneme_durations = torch.zeros_like(phoneme_indices_tensor, dtype=torch.long)
        else:
            # Assign an average duration and distribute remainder
            avg_duration = num_mel_frames / num_phonemes
            phoneme_durations = torch.full((num_phonemes,), int(avg_duration), dtype=torch.long)

            # Distribute any remaining frames
            remainder = num_mel_frames - torch.sum(phoneme_durations).item()
            for i in range(remainder):
                if i < num_phonemes: # Ensure we don't go out of bounds
                    phoneme_durations[i] += 1

            # Ensure no phoneme has zero or negative duration (should be at least 1)
            phoneme_durations = torch.clamp(phoneme_durations, min=1)


        # --- Generate Stop Token Targets ---
        # A tensor of zeros, with a 1.0 at the last actual mel frame.
        stop_token_targets = torch.zeros(mel_spec.shape[1], dtype=torch.float32)
        if mel_spec.shape[1] > 0:
            stop_token_targets[-1] = 1.0 # Mark the last frame as the stop token

        return {
            'mel_spec': mel_spec,
            'phoneme_indices': phoneme_indices_tensor,
            'phoneme_durations': phoneme_durations,    # NEW
            'stop_token_targets': stop_token_targets,  # NEW
            'text': sample['text'],
            'audio_file': sample['audio_file']
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader - optimized for MPS"""
    # Transpose mel_spec from (n_mels, time) to (time, n_mels) for batch_first=True padding
    mel_specs = [item['mel_spec'].transpose(0, 1) for item in batch]
    phoneme_indices = [item['phoneme_indices'] for item in batch]
    phoneme_durations = [item['phoneme_durations'] for item in batch] # NEW
    stop_token_targets = [item['stop_token_targets'] for item in batch] # NEW (float32)

    texts = [item['text'] for item in batch]
    audio_files = [item['audio_file'] for item in batch]

    # Pad sequences
    mel_specs_padded = pad_sequence(mel_specs, batch_first=True, padding_value=0.0)
    phoneme_indices_padded = pad_sequence(phoneme_indices, batch_first=True, padding_value=0)
    # Durations are integers, padding with 0 is fine
    phoneme_durations_padded = pad_sequence(phoneme_durations, batch_first=True, padding_value=0)
    # Stop token targets are floats, padding with 0.0 is crucial for BCEWithLogitsLoss
    stop_token_targets_padded = pad_sequence(stop_token_targets, batch_first=True, padding_value=0.0)


    return {
        'mel_specs': mel_specs_padded,
        'phoneme_indices': phoneme_indices_padded,
        'phoneme_durations': phoneme_durations_padded, # NEW
        'stop_token_targets': stop_token_targets_padded, # NEW
        'texts': texts,
        'audio_files': audio_files
    }
