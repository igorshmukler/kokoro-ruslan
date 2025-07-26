#!/usr/bin/env python3
"""
Dataset implementation for Ruslan corpus
"""

import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import logging
import random
import numpy as np
from tqdm import tqdm

from config import TrainingConfig
from russian_phoneme_processor import RussianPhonemeProcessor

logger = logging.getLogger(__name__)


class RuslanDataset(Dataset):
    """Dataset class for Ruslan corpus - optimized for MPS"""

    def __init__(self, data_dir: str, config: TrainingConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self.phoneme_processor = RussianPhonemeProcessor()

        # Validate MelSpectrogram parameters
        if self.config.win_length > self.config.n_fft:
            raise ValueError(
                f"win_length ({self.config.win_length}) cannot be greater than n_fft ({self.config.n_fft}). "
                "Please check your TrainingConfig."
            )
        if self.config.hop_length <= 0:
            raise ValueError("hop_length must be a positive integer.")

        # Pre-create mel transform for efficiency
        # Explicitly setting window_fn and using `return_fast=False` for robustness
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=2.0,
            normalized=False,
            # Explicitly define window function
            window_fn=torch.hann_window,
            # return_fast=False can sometimes help with backend specific issues,
            # although it might be slightly slower. Keep as default if not needed.
            # You can uncomment this if issues persist:
            # return_fast=False
        )

        # Load metadata and pre-calculate lengths for batching
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
        Load samples from corpus directory and pre-calculate lengths.
        """
        samples = []

        metadata_file = self.data_dir / "metadata_RUSLAN_22200.csv"
        if metadata_file.exists():
            logger.info(f"Loading metadata from {metadata_file}")
            # Get total number of lines for accurate progress bar
            with open(metadata_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)

            with open(metadata_file, 'r', encoding='utf-8') as f:
                # Wrap the file iterator with tqdm
                for line in tqdm(f, total=total_lines, desc="Loading metadata"):
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        audio_file_stem = parts[0]
                        text = parts[1]

                        audio_path = self.data_dir / "wavs" / f"{audio_file_stem}.wav"
                        if audio_path.exists():
                            # Pre-calculate audio length (in mel frames)
                            try:
                                waveform, sr = torchaudio.load(audio_path)
                                if sr != self.config.sample_rate:
                                    resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                                    waveform = resampler(waveform)

                                # Ensure waveform is at least win_length for STFT
                                if waveform.shape[1] < self.config.win_length:
                                    # Pad short audio with zeros
                                    padding_needed = self.config.win_length - waveform.shape[1]
                                    waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
                                    # logger.debug(f"Padded short audio {audio_file_stem}. New length: {waveform.shape[1]}")

                                # Estimate mel frames (audio_length // hop_length)
                                # Add 1 because the number of frames is usually (waveform_length - win_length) / hop_length + 1
                                audio_length_frames = (waveform.shape[1] - self.config.n_fft) // self.config.hop_length + 1
                                if audio_length_frames < 1: # Handle cases where padding wasn't enough for even one frame
                                    audio_length_frames = 1

                            except Exception as e:
                                logger.warning(f"Could not load or process audio {audio_path}: {e}. Skipping.")
                                continue

                            # Pre-calculate phoneme length
                            phoneme_indices = self.phoneme_processor.text_to_indices(text)
                            phoneme_length = len(phoneme_indices)

                            # Clip extremely long sequences to prevent memory issues during training
                            if audio_length_frames > self.config.max_seq_length:
                                logger.warning(f"Clipping {audio_file_stem}. Audio frames: {audio_length_frames} > max_seq_length: {self.config.max_seq_length}")
                                audio_length_frames = self.config.max_seq_length
                                # Also adjust phoneme length if it's too long, proportionally
                                if phoneme_length > 0:
                                    # Estimate a new phoneme length based on the clipped audio length
                                    original_audio_len_samples = waveform.shape[1]
                                    original_audio_len_frames = (original_audio_len_samples - self.config.n_fft) // self.config.hop_length + 1

                                    if original_audio_len_frames > 0: # Avoid division by zero
                                        # Proportionally scale phoneme length
                                        phoneme_length = int(phoneme_length * (self.config.max_seq_length / original_audio_len_frames))
                                    phoneme_length = max(1, phoneme_length) # Ensure at least 1 phoneme if original was > 0

                            samples.append({
                                'audio_path': str(audio_path),
                                'text': text,
                                'audio_file': audio_file_stem,
                                'audio_length': audio_length_frames, # in mel frames
                                'phoneme_length': phoneme_length
                            })
        else:
            logger.warning(f"Metadata file not found: {metadata_file}. Falling back to directory scan. "
                           "Note: Lengths will be estimated on the fly for scanned files, which might be slower.")
            wav_dir = self.data_dir / "wavs"
            txt_dir = self.data_dir / "texts"

            if wav_dir.exists():
                # For glob, we can convert to list first to get total count for tqdm
                wav_files = list(wav_dir.glob("*.wav"))
                for wav_file in tqdm(wav_files, desc="Scanning audio files"):
                    txt_file = txt_dir / f"{wav_file.stem}.txt"
                    if txt_file.exists():
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()

                        # Pre-calculate audio length (in mel frames)
                        try:
                            waveform, sr = torchaudio.load(wav_file)
                            if sr != self.config.sample_rate:
                                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                                waveform = resampler(waveform)

                            # Ensure waveform is at least win_length for STFT
                            if waveform.shape[1] < self.config.win_length:
                                padding_needed = self.config.win_length - waveform.shape[1]
                                waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
                                # logger.debug(f"Padded short audio {wav_file.stem}. New length: {waveform.shape[1]}")

                            audio_length_frames = (waveform.shape[1] - self.config.n_fft) // self.config.hop_length + 1
                            if audio_length_frames < 1:
                                audio_length_frames = 1

                        except Exception as e:
                            logger.warning(f"Could not load or process audio {wav_file}: {e}. Skipping.")
                            continue

                        # Pre-calculate phoneme length
                        phoneme_indices = self.phoneme_processor.text_to_indices(text)
                        phoneme_length = len(phoneme_indices)

                        # Clip extremely long sequences
                        if audio_length_frames > self.config.max_seq_length:
                             logger.warning(f"Clipping {wav_file.stem}. Audio frames: {audio_length_frames} > max_seq_length: {self.config.max_seq_length}")
                             audio_length_frames = self.config.max_seq_length
                             if phoneme_length > 0:
                                original_audio_len_samples = waveform.shape[1]
                                original_audio_len_frames = (original_audio_len_samples - self.config.n_fft) // self.config.hop_length + 1
                                if original_audio_len_frames > 0:
                                    phoneme_length = int(phoneme_length * (self.config.max_seq_length / original_audio_len_frames))
                                phoneme_length = max(1, phoneme_length)


                        samples.append({
                            'audio_path': str(wav_file),
                            'text': text,
                            'audio_file': wav_file.stem,
                            'audio_length': audio_length_frames,
                            'phoneme_length': phoneme_length
                        })

        # Sort samples by their combined length (or just audio_length) for efficient batching
        # Sorting by audio length is generally most impactful for Mel-spectrograms
        samples.sort(key=lambda x: x['audio_length'])
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

        # Pad audio to ensure it's at least `win_length` for STFT ---
        # This is critical for torch.stft not to fail on very short samples.
        if audio.shape[1] < self.config.win_length:
            padding_needed = self.config.win_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding_needed))
            # logger.debug(f"Padded audio in __getitem__. New length: {audio.shape[1]}")

        # Extract mel spectrogram using pre-created transform
        mel_spec = self.mel_transform(audio).squeeze(0)  # Remove channel dimension

        # Convert to log scale and normalize
        mel_spec = torch.log(mel_spec + 1e-9)  # Add small epsilon to avoid log(0)

        # Clip extremely long sequences to prevent memory issues
        max_frames = self.config.max_seq_length
        if mel_spec.shape[1] > max_frames:
            mel_spec = mel_spec[:, :max_frames]

        # Process text to phonemes using the dedicated processor
        phoneme_indices = self.phoneme_processor.text_to_indices(sample['text'])
        phoneme_indices_tensor = torch.tensor(phoneme_indices, dtype=torch.long)

        # --- Generate Phoneme Durations (PLACEHOLDER/DUMMY) ---
        num_mel_frames = mel_spec.shape[1]
        num_phonemes = phoneme_indices_tensor.shape[0]

        if num_phonemes == 0:
            phoneme_durations = torch.zeros_like(phoneme_indices_tensor, dtype=torch.long)
        else:
            avg_duration = num_mel_frames / num_phonemes
            phoneme_durations = torch.full((num_phonemes,), int(avg_duration), dtype=torch.long)

            remainder = num_mel_frames - torch.sum(phoneme_durations).item()
            # Distribute remainder frames to early phonemes
            for i in range(remainder):
                if i < num_phonemes: # Ensure we don't go out of bounds
                    phoneme_durations[i] += 1
            phoneme_durations = torch.clamp(phoneme_durations, min=1)

        # --- Generate Stop Token Targets ---
        stop_token_targets = torch.zeros(mel_spec.shape[1], dtype=torch.float32)
        if mel_spec.shape[1] > 0:
            stop_token_targets[-1] = 1.0

        return {
            'mel_spec': mel_spec,
            'phoneme_indices': phoneme_indices_tensor,
            'phoneme_durations': phoneme_durations,
            'stop_token_targets': stop_token_targets,
            'text': sample['text'],
            'audio_file': sample['audio_file'],
            'mel_length': mel_spec.shape[1], # Actual length after potential clipping
            'phoneme_length': phoneme_indices_tensor.shape[0] # Actual length
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader - optimized for MPS"""
    # Transpose mel_spec from (n_mels, time) to (time, n_mels) for batch_first=True padding
    mel_specs = [item['mel_spec'].transpose(0, 1) for item in batch]
    phoneme_indices = [item['phoneme_indices'] for item in batch]
    phoneme_durations = [item['phoneme_durations'] for item in batch]
    stop_token_targets = [item['stop_token_targets'] for item in batch] # (float32)

    # Extract original lengths for loss masking if needed later
    mel_lengths = torch.tensor([item['mel_length'] for item in batch], dtype=torch.long)
    phoneme_lengths = torch.tensor([item['phoneme_length'] for item in batch], dtype=torch.long)

    texts = [item['text'] for item in batch]
    audio_files = [item['audio_file'] for item in batch]

    # Pad sequences
    mel_specs_padded = pad_sequence(mel_specs, batch_first=True, padding_value=0.0)
    phoneme_indices_padded = pad_sequence(phoneme_indices, batch_first=True, padding_value=0)
    phoneme_durations_padded = pad_sequence(phoneme_durations, batch_first=True, padding_value=0)
    stop_token_targets_padded = pad_sequence(stop_token_targets, batch_first=True, padding_value=0.0)

    return {
        'mel_specs': mel_specs_padded,
        'phoneme_indices': phoneme_indices_padded,
        'phoneme_durations': phoneme_durations_padded,
        'stop_token_targets': stop_token_targets_padded,
        'mel_lengths': mel_lengths,        # Add mel lengths to the batch
        'phoneme_lengths': phoneme_lengths, # Add phoneme lengths to the batch
        'texts': texts,
        'audio_files': audio_files
    }


class LengthBasedBatchSampler(Sampler):
    """
    Samples mini-batches of indices for training.
    The samples are grouped by lengths to minimize padding.
    Assumes the dataset is already sorted by length.
    """
    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool = False, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Create buckets of indices based on length
        # Since the dataset is pre-sorted by audio_length, we can just group
        self.batches = self._create_batches()

    def _create_batches(self) -> List[List[int]]:
        batches = []
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            # Shuffle within "length-similar" windows to maintain some randomness
            # without completely destroying the length ordering.
            # A common strategy is to shuffle fixed-size chunks of the sorted data.
            window_size = 1000 # Example window size, tune as needed
            num_windows = len(indices) // window_size
            shuffled_indices = []
            for i in range(num_windows):
                window = indices[i * window_size : (i + 1) * window_size]
                random.shuffle(window)
                shuffled_indices.extend(window)
            # Add remaining indices
            remaining_indices = indices[num_windows * window_size:]
            random.shuffle(remaining_indices)
            shuffled_indices.extend(remaining_indices)
            indices = shuffled_indices

        # Group into batches
        current_batch = []
        for idx in indices:
            current_batch.append(idx)
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []

        if len(current_batch) > 0 and not self.drop_last:
            batches.append(current_batch)

        # Shuffle the order of batches
        if self.shuffle:
            random.shuffle(batches)

        return batches

    def __iter__(self):
        # Iterate over the prepared batches
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)
