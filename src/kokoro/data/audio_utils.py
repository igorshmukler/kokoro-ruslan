#!/usr/bin/env python3
"""
AudioUtils - Utilities for audio processing and saving
Handles multiple audio saving backends with fallbacks
"""

import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path
from typing import Optional

try:
    import soundfile as sf
    SOUND_FILE_AVAILABLE = True
except (ImportError, OSError):
    sf = None
    SOUND_FILE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioUtils:
    """Utility class for audio processing operations"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    @staticmethod
    def normalize_audio(audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to prevent clipping"""
        max_val = torch.max(torch.abs(audio))
        if max_val < 1e-8:
            return audio
        return audio / max_val

    @staticmethod
    def ensure_mono(audio: torch.Tensor) -> torch.Tensor:
        """Ensure audio is mono (1D tensor)"""
        if len(audio.shape) == 3:  # Remove batch dimension if present
            audio = audio.squeeze(0)
        if len(audio.shape) == 2:  # Remove channel dimension if present
            audio = audio.squeeze(0)
        return audio

    def save_audio(self, audio: torch.Tensor, output_path: str) -> bool:
        """Save audio with multiple fallback methods"""
        # Ensure audio is properly formatted
        audio = self.ensure_mono(audio)
        audio = self.normalize_audio(audio)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try multiple saving methods
        success = (
            self._save_with_torchaudio(audio, output_path) or
            self._save_with_soundfile(audio, output_path) or
            self._save_with_scipy(audio, output_path) or
            self._save_as_numpy(audio, output_path)
        )

        if not success:
            logger.error("All audio saving methods failed")
            raise RuntimeError("Could not save audio file with any method")

        return success

    def _save_with_torchaudio(self, audio: torch.Tensor, output_path: Path) -> bool:
        try:
            torchaudio.save(
                str(output_path),
                audio.unsqueeze(0),
                self.sample_rate,
                format="wav",
                encoding="PCM_S",
                bits_per_sample=16,
            )
            logger.info(f"Audio saved using torchaudio: {output_path}")
            return True
        except Exception as e:
            logger.warning(f"torchaudio.save failed: {e}")
            return False

    def _save_with_soundfile(self, audio: torch.Tensor, output_path: Path) -> bool:
        """Try saving with soundfile"""
        if not SOUND_FILE_AVAILABLE:
            logger.debug("soundfile backend unavailable, skipping")
            return False

        try:
            audio_np = audio.numpy()
            sf.write(str(output_path), audio_np, self.sample_rate)
            logger.info(f"Audio saved using soundfile: {output_path}")
            return True
        except Exception as e:
            logger.warning(f"soundfile failed: {e}")
            return False

    def _save_with_scipy(self, audio: torch.Tensor, output_path: Path) -> bool:
        """Try saving with scipy"""
        try:
            from scipy.io import wavfile
            # Convert to 16-bit integer
            audio_np = audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wavfile.write(str(output_path), self.sample_rate, audio_int16)
            logger.info(f"Audio saved using scipy: {output_path}")
            return True
        except Exception as e:
            logger.warning(f"scipy failed: {e}")
            return False

    def _save_as_numpy(self, audio: torch.Tensor, output_path: Path) -> bool:
        """Save as numpy array (debugging fallback)"""
        try:
            audio_np = audio.numpy()
            numpy_path = output_path.with_suffix('.npy')
            np.save(numpy_path, audio_np)
            logger.info(f"Audio saved as numpy array: {numpy_path}")
            logger.info("Note: You can convert the .npy file to WAV using external tools")
            return True
        except Exception as e:
            logger.warning(f"numpy save failed: {e}")
            return False

    @staticmethod
    def detect_device() -> str:
        """Auto-detect the best available device"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    @staticmethod
    def validate_device(device: Optional[str]) -> str:
        """Validate and return appropriate device"""
        if device is None:
            return AudioUtils.detect_device()

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"

        return device


class PhonemeProcessorUtils:
    """Helper class for phoneme processing operations"""

    @staticmethod
    def flatten_phoneme_output(raw_output) -> list:
        """Flatten phoneme processor output into a single list of phoneme strings"""
        phoneme_sequence = []

        if isinstance(raw_output, list):
            for item in raw_output:
                if isinstance(item, str):
                    # Single phoneme string
                    phoneme_sequence.append(item)
                elif isinstance(item, list):
                    # List of phonemes
                    for sub_item in item:
                        if isinstance(sub_item, str):
                            phoneme_sequence.append(sub_item)
                        elif isinstance(sub_item, list):
                            # Handle deeper nesting
                            for deepest_item in sub_item:
                                if isinstance(deepest_item, str):
                                    phoneme_sequence.append(deepest_item)
                                else:
                                    logger.warning(f"Unexpected item type in deepest level: {type(deepest_item)} - {deepest_item}")
                        else:
                            logger.warning(f"Unexpected item type in sub_item: {type(sub_item)} - {sub_item}")
                elif isinstance(item, tuple) and len(item) == 3:
                    # Tuple format: (word, word_phonemes, stress_info)
                    if isinstance(item[1], list):
                        for sub_phoneme in item[1]:
                            if isinstance(sub_phoneme, str):
                                phoneme_sequence.append(sub_phoneme)
                            else:
                                logger.warning(f"Unexpected phoneme type in tuple's phoneme list: {type(sub_phoneme)} - {sub_phoneme}")
                    else:
                        logger.warning(f"Unexpected type for word_phonemes in tuple: {type(item[1])} - {item[1]}")
                else:
                    logger.warning(f"Unexpected item type from phoneme processor: {type(item)} - {item}")
        else:
            logger.error(f"Phoneme processor returned unexpected top-level type: {type(raw_output)}. Expected a list.")
            raise TypeError("Phoneme processor output is not a list.")

        return phoneme_sequence

    @staticmethod
    def flatten_phoneme_output_with_sil(raw_output, phoneme_to_id: dict) -> list:
        """
        Flatten process_text() output, inserting <sil> between words.

        During training, MFA alignments include <sil> tokens at word boundaries.
        This method replicates that distribution at inference time so the model
        sees the same token patterns it was trained on.

        Falls back to plain flattening with a WARNING if <sil> is absent from
        phoneme_to_id (processor predates the <sil> vocab addition).

        Args:
            raw_output: Output of RussianPhonemeProcessor.process_text() —
                        list of (word, [phonemes], stress_info) tuples.
            phoneme_to_id: Vocabulary mapping from the loaded processor.
        """
        if '<sil>' not in phoneme_to_id:
            logger.warning(
                "flatten_phoneme_output_with_sil: '<sil>' not in phoneme_to_id. "
                "Falling back to plain flatten. Reload with a processor whose "
                "_build_vocab includes '<sil>' to enable silence injection."
            )
            return PhonemeProcessorUtils.flatten_phoneme_output(raw_output)

        result = []
        word_count = 0

        for item in raw_output:
            if isinstance(item, tuple) and len(item) == 3:
                word, word_phonemes, _ = item
                if not isinstance(word_phonemes, list):
                    logger.warning(
                        f"flatten_phoneme_output_with_sil: unexpected phoneme list "
                        f"type for word '{word}': {type(word_phonemes)}"
                    )
                    continue
                if word_count > 0:        # <sil> before every word except the first
                    result.append('<sil>')
                for ph in word_phonemes:
                    if isinstance(ph, str) and ph:
                        result.append(ph)
                word_count += 1
            else:
                # Non-tuple: fall back to plain logic for this element
                logger.warning(
                    f"flatten_phoneme_output_with_sil: unexpected item type "
                    f"{type(item)}, skipping sil injection for this item"
                )
                result.extend(PhonemeProcessorUtils.flatten_phoneme_output([item]))

        return result

    @staticmethod
    def phonemes_to_indices(phoneme_sequence: list, phoneme_to_id: dict) -> list:
        """Convert phoneme strings to indices, ensuring 1:1 length mapping"""
        phoneme_indices = []

        # Get ID for unknown or silence as fallback
        unk_id = phoneme_to_id.get('<unk>', phoneme_to_id.get('<sil>', 0))

        for p in phoneme_sequence:
            if p in phoneme_to_id:
                phoneme_indices.append(phoneme_to_id[p])
            else:
                logger.warning(f"Phoneme '{p}' not in vocab! Mapping to ID {unk_id}")
                phoneme_indices.append(unk_id)

        if not phoneme_indices:
            logger.error("No valid phoneme indices generated.")
            raise ValueError("No valid phoneme indices generated.")

        return phoneme_indices
