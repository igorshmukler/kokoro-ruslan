#!/usr/bin/env python3
"""
Dataset implementation for Ruslan corpus
"""

import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import logging
import random
import numpy as np
from tqdm import tqdm
import pickle
import os

from config import TrainingConfig
from russian_phoneme_processor import RussianPhonemeProcessor
from mfa_integration import MFAIntegration

try:
    from variance_predictor import PitchExtractor, EnergyExtractor
    VARIANCE_AVAILABLE = True
except ImportError:
    VARIANCE_AVAILABLE = False
    logger.warning("Variance predictor module not available, pitch/energy extraction disabled")

logger = logging.getLogger(__name__)


class RuslanDataset(Dataset):
    """Dataset class for Ruslan corpus - optimized for MPS"""

    def __init__(self, data_dir: str, config: TrainingConfig, use_mfa: bool = True,
                 indices: Optional[List[int]] = None):
        self.data_dir = Path(data_dir)
        self.config = config
        self.phoneme_processor = RussianPhonemeProcessor()
        self.use_mfa = use_mfa
        self.indices = indices  # Subset of indices for train/val split

        # Check if variance prediction is enabled
        self.use_variance = getattr(config, 'use_variance_predictor', True) and VARIANCE_AVAILABLE
        if self.use_variance:
            logger.info("Variance prediction enabled (pitch/energy extraction)")
        else:
            logger.info("Variance prediction disabled")

        # Initialize MFA integration if enabled
        self.mfa = None
        if self.use_mfa and hasattr(config, 'mfa_alignment_dir'):
            alignment_dir = Path(config.mfa_alignment_dir)
            if alignment_dir.exists():
                self.mfa = MFAIntegration(
                    corpus_dir=str(self.data_dir),
                    output_dir=str(alignment_dir.parent),
                    hop_length=config.hop_length,
                    sample_rate=config.sample_rate
                )
                logger.info(f"MFA integration enabled, using alignments from: {alignment_dir}")
            else:
                logger.warning(f"MFA alignment directory not found: {alignment_dir}")
                logger.warning("Falling back to estimated durations")
                self.use_mfa = False
        else:
            logger.warning("MFA not enabled in config. Using estimated durations.")

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

        # Setup cache for audio metadata
        self.cache_dir = self.data_dir / ".cache"
        self.cache_file = self.cache_dir / "audio_metadata.pkl"

        # Load metadata and pre-calculate lengths for batching
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples from corpus at {data_dir}")
        logger.info(f"Using phoneme processor: {self.phoneme_processor}")

        # Warn about duration source
        if self.mfa:
            logger.info("Using MFA forced alignment for phoneme durations (high quality)")
        else:
            logger.warning(
                "Phoneme durations are being estimated (lower quality). "
                "For high-quality TTS, run MFA alignment first using: "
                "python mfa_integration.py --corpus ./ruslan_corpus --output ./mfa_output"
            )

    def _get_audio_info_cached(self, audio_path: Path, cache: dict) -> Optional[tuple]:
        """
        Get audio info (sample_rate, num_frames) without loading full waveform.
        Uses cache to avoid repeated torchaudio.info() calls.
        """
        audio_path_str = str(audio_path)

        # Check cache first
        if audio_path_str in cache:
            return cache[audio_path_str]

        try:
            # Use torchaudio.info() which only reads metadata, not audio data
            info = torchaudio.info(audio_path_str)
            sample_rate = info.sample_rate
            num_frames = info.num_frames

            # Calculate actual frames after potential resampling
            if sample_rate != self.config.sample_rate:
                # Estimate frames after resampling
                num_frames = int(num_frames * self.config.sample_rate / sample_rate)

            # Ensure at least win_length frames
            if num_frames < self.config.win_length:
                num_frames = self.config.win_length

            # Cache the result
            result = (sample_rate, num_frames)
            cache[audio_path_str] = result
            return result

        except Exception as e:
            logger.warning(f"Could not get info for {audio_path}: {e}")
            return None

    def _load_cache(self) -> dict:
        """Load cached audio metadata if available"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded audio metadata cache ({len(cache)} entries)")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self, cache: dict):
        """Save audio metadata cache"""
        try:
            self.cache_dir.mkdir(exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache, f)
            logger.info(f"Saved audio metadata cache ({len(cache)} entries)")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_samples(self) -> List[Dict]:
        """
        Load samples from corpus directory and pre-calculate lengths.
        Uses caching to avoid repeated audio loading.
        """
        samples = []

        # Load cache
        audio_info_cache = self._load_cache()
        cache_updated = False

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
                            # Get audio info efficiently (cached)
                            audio_info = self._get_audio_info_cached(audio_path, audio_info_cache)
                            if audio_info is None:
                                continue

                            cache_updated = True
                            sample_rate, num_frames = audio_info

                            # Calculate mel frames
                            audio_length_frames = (num_frames - self.config.n_fft) // self.config.hop_length + 1
                            if audio_length_frames < 1:
                                audio_length_frames = 1

                            # Pre-calculate phoneme length
                            phoneme_indices = self.phoneme_processor.text_to_indices(text)
                            phoneme_length = len(phoneme_indices)

                            # Clip extremely long sequences to prevent memory issues during training
                            original_frames = audio_length_frames
                            if audio_length_frames > self.config.max_seq_length:
                                logger.warning(f"Clipping {audio_file_stem}. Audio frames: {audio_length_frames} > max_seq_length: {self.config.max_seq_length}")
                                audio_length_frames = self.config.max_seq_length
                                # Also adjust phoneme length if it's too long, proportionally
                                if phoneme_length > 0 and original_frames > 0:
                                    # Proportionally scale phoneme length
                                    phoneme_length = int(phoneme_length * (self.config.max_seq_length / original_frames))
                                    phoneme_length = max(1, phoneme_length)

                            samples.append({
                                'audio_path': str(audio_path),
                                'text': text,
                                'audio_file': audio_file_stem,
                                'audio_length': audio_length_frames,
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

                        # Get audio info efficiently (cached)
                        audio_info = self._get_audio_info_cached(wav_file, audio_info_cache)
                        if audio_info is None:
                            continue

                        cache_updated = True
                        sample_rate, num_frames = audio_info

                        # Calculate mel frames
                        audio_length_frames = (num_frames - self.config.n_fft) // self.config.hop_length + 1
                        if audio_length_frames < 1:
                            audio_length_frames = 1

                        # Pre-calculate phoneme length
                        phoneme_indices = self.phoneme_processor.text_to_indices(text)
                        phoneme_length = len(phoneme_indices)

                        # Clip extremely long sequences
                        original_frames = audio_length_frames
                        if audio_length_frames > self.config.max_seq_length:
                            logger.warning(f"Clipping {wav_file.stem}. Audio frames: {audio_length_frames} > max_seq_length: {self.config.max_seq_length}")
                            audio_length_frames = self.config.max_seq_length
                            if phoneme_length > 0 and original_frames > 0:
                                phoneme_length = int(phoneme_length * (self.config.max_seq_length / original_frames))
                                phoneme_length = max(1, phoneme_length)

                        samples.append({
                            'audio_path': str(wav_file),
                            'text': text,
                            'audio_file': wav_file.stem,
                            'audio_length': audio_length_frames,
                            'phoneme_length': phoneme_length
                        })

        # Save updated cache
        if cache_updated:
            self._save_cache(audio_info_cache)

        # Sort samples by their combined length (or just audio_length) for efficient batching
        # Sorting by audio length is generally most impactful for Mel-spectrograms
        samples.sort(key=lambda x: x['audio_length'])

        # If indices are provided, filter to subset (for train/val split)
        if self.indices is not None:
            samples = [samples[i] for i in self.indices if i < len(samples)]
            logger.info(f"Using subset of {len(samples)} samples (from indices)")

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

        # --- Get Phoneme Durations (MFA or Estimated) ---
        num_mel_frames = mel_spec.shape[1]
        num_phonemes = phoneme_indices_tensor.shape[0]

        # Try to get MFA alignments first
        mfa_durations = None
        if self.mfa is not None:
            mfa_durations = self.mfa.get_phoneme_durations(sample['audio_file'])

        if mfa_durations is not None and len(mfa_durations) > 0:
            # Use MFA durations
            phoneme_durations = torch.tensor(mfa_durations, dtype=torch.long)

            # Handle length mismatch between MFA phonemes and our phonemes
            if len(phoneme_durations) != num_phonemes:
                # Log warning only occasionally to avoid spam
                if idx % 1000 == 0:
                    logger.debug(f"MFA duration length ({len(phoneme_durations)}) != "
                               f"phoneme length ({num_phonemes}) for {sample['audio_file']}")

                # Adjust durations to match phoneme count
                if len(phoneme_durations) > num_phonemes:
                    # Truncate
                    phoneme_durations = phoneme_durations[:num_phonemes]
                else:
                    # Pad with estimated durations for missing phonemes
                    missing = num_phonemes - len(phoneme_durations)
                    avg_dur = int(phoneme_durations.float().mean().item()) if len(phoneme_durations) > 0 else 5
                    padding = torch.full((missing,), avg_dur, dtype=torch.long)
                    phoneme_durations = torch.cat([phoneme_durations, padding])

            # Ensure durations are at least 1 frame
            phoneme_durations = torch.clamp(phoneme_durations, min=1)

        else:
            # Fall back to estimated durations
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

        # --- Extract Pitch and Energy (if variance prediction enabled) ---
        pitch = None
        energy = None

        if self.use_variance:
            try:
                # Extract pitch from audio
                pitch = PitchExtractor.extract_pitch(
                    audio.squeeze(0),  # Remove channel dim
                    sample_rate=self.config.sample_rate,
                    hop_length=self.config.hop_length
                )

                # Match length to mel frames
                if len(pitch) > num_mel_frames:
                    pitch = pitch[:num_mel_frames]
                elif len(pitch) < num_mel_frames:
                    padding = torch.zeros(num_mel_frames - len(pitch), device=pitch.device)
                    pitch = torch.cat([pitch, padding])

                # Extract energy from mel spectrogram
                energy = EnergyExtractor.extract_energy_from_mel(mel_spec)

                # Ensure energy matches mel frames
                if len(energy) > num_mel_frames:
                    energy = energy[:num_mel_frames]
                elif len(energy) < num_mel_frames:
                    padding = torch.zeros(num_mel_frames - len(energy), device=energy.device)
                    energy = torch.cat([energy, padding])

            except Exception as e:
                logger.warning(f"Failed to extract pitch/energy for {sample['audio_file']}: {e}")
                # Use zeros as fallback
                pitch = torch.zeros(num_mel_frames)
                energy = torch.zeros(num_mel_frames)
        else:
            # Variance prediction disabled, use zeros
            pitch = torch.zeros(num_mel_frames)
            energy = torch.zeros(num_mel_frames)

        return {
            'mel_spec': mel_spec,
            'phoneme_indices': phoneme_indices_tensor,
            'phoneme_durations': phoneme_durations,
            'stop_token_targets': stop_token_targets,
            'pitch': pitch,
            'energy': energy,
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

    # Pitch and energy
    pitches = [item['pitch'] for item in batch]
    energies = [item['energy'] for item in batch]

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
    pitches_padded = pad_sequence(pitches, batch_first=True, padding_value=0.0)
    energies_padded = pad_sequence(energies, batch_first=True, padding_value=0.0)

    return {
        'mel_specs': mel_specs_padded,
        'phoneme_indices': phoneme_indices_padded,
        'phoneme_durations': phoneme_durations_padded,
        'stop_token_targets': stop_token_targets_padded,
        'pitches': pitches_padded,
        'energies': energies_padded,
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


class DynamicFrameBatchSampler(Sampler):
    """
    Dynamic batch sampler that groups samples by total frame count.

    Instead of fixed batch size, batches are created to fit within a maximum
    frame budget, allowing longer samples to have smaller batch sizes and
    shorter samples to have larger batch sizes for optimal GPU utilization.

    Args:
        dataset: Dataset with samples that have 'mel_length' attribute
        max_frames: Maximum total mel frames per batch
        min_batch_size: Minimum number of samples per batch
        max_batch_size: Maximum number of samples per batch
        drop_last: Drop incomplete batches
        shuffle: Shuffle batches
    """

    def __init__(self,
                 dataset: Dataset,
                 max_frames: int = 20000,
                 min_batch_size: int = 4,
                 max_batch_size: int = 32,
                 drop_last: bool = False,
                 shuffle: bool = True):
        self.dataset = dataset
        self.max_frames = max_frames
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Create batches based on frame budget
        self.batches = self._create_batches()

        # Calculate statistics
        self._log_statistics()

    def _get_sample_frames(self, idx: int) -> int:
        """Get number of mel frames for a sample"""
        # Access the sample's audio_length (which is mel frames)
        sample = self.dataset.samples[idx]
        return sample['audio_length']

    def _create_batches(self) -> List[List[int]]:
        """Create batches that fit within frame budget"""
        batches = []
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            # Shuffle within length-similar windows to maintain some locality
            window_size = 1000
            num_windows = len(indices) // window_size
            shuffled_indices = []

            for i in range(num_windows):
                window = indices[i * window_size : (i + 1) * window_size]
                random.shuffle(window)
                shuffled_indices.extend(window)

            # Add remaining indices
            remaining = indices[num_windows * window_size:]
            random.shuffle(remaining)
            shuffled_indices.extend(remaining)
            indices = shuffled_indices

        # Group samples into batches based on frame budget
        current_batch = []
        current_frames = 0

        for idx in indices:
            sample_frames = self._get_sample_frames(idx)

            # Check if adding this sample would exceed limits
            would_exceed_frames = (current_frames + sample_frames) > self.max_frames
            would_exceed_max_batch = len(current_batch) >= self.max_batch_size

            # Start new batch if we exceed frame budget or max batch size
            if current_batch and (would_exceed_frames or would_exceed_max_batch):
                # Only add batch if it meets minimum size requirement
                if len(current_batch) >= self.min_batch_size:
                    batches.append(current_batch)
                elif not self.drop_last:
                    # Add small batch if not dropping
                    batches.append(current_batch)

                current_batch = []
                current_frames = 0

            # Add sample to current batch
            current_batch.append(idx)
            current_frames += sample_frames

        # Handle remaining samples
        if current_batch:
            if len(current_batch) >= self.min_batch_size or not self.drop_last:
                batches.append(current_batch)

        # Shuffle batch order
        if self.shuffle:
            random.shuffle(batches)

        return batches

    def _log_statistics(self):
        """Log batching statistics for monitoring"""
        if not self.batches:
            return

        batch_sizes = [len(batch) for batch in self.batches]
        batch_frames = []

        for batch in self.batches:
            total_frames = sum(self._get_sample_frames(idx) for idx in batch)
            batch_frames.append(total_frames)

        logger.info("Dynamic Batching Statistics:")
        logger.info(f"  Total batches: {len(self.batches)}")
        logger.info(f"  Batch sizes - Min: {min(batch_sizes)}, Max: {max(batch_sizes)}, "
                   f"Avg: {sum(batch_sizes)/len(batch_sizes):.1f}")
        logger.info(f"  Frames per batch - Min: {min(batch_frames)}, Max: {max(batch_frames)}, "
                   f"Avg: {sum(batch_frames)/len(batch_frames):.1f}")
        logger.info(f"  Frame budget: {self.max_frames}")
        logger.info(f"  Batch size range: [{self.min_batch_size}, {self.max_batch_size}]")

    def __iter__(self):
        """Iterate over batches"""
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        """Return number of batches"""
        return len(self.batches)
