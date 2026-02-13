#!/usr/bin/env python3
"""
Montreal Forced Aligner (MFA) Integration for Russian TTS
Handles forced alignment to extract accurate phoneme durations
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)


@dataclass
class PhonemeAlignment:
    """Single phoneme alignment information"""
    phoneme: str
    start_time: float  # in seconds
    end_time: float    # in seconds
    duration: float    # in seconds

    @property
    def duration_frames(self) -> int:
        """Duration in mel spectrogram frames (assuming 256 hop length, 22050 sr)"""
        # hop_length=256, sr=22050 -> ~11.6ms per frame
        return int(self.duration * 22050 / 256)


@dataclass
class WordAlignment:
    """Word-level alignment with phonemes"""
    word: str
    start_time: float
    end_time: float
    phonemes: List[PhonemeAlignment]


class MFAIntegration:
    """
    Montreal Forced Aligner integration for extracting phoneme durations
    """

    def __init__(self,
                 corpus_dir: str,
                 output_dir: str,
                 acoustic_model: str = "russian_mfa",
                 dictionary: str = "russian_mfa",
                 hop_length: int = 256,
                 sample_rate: int = 22050):
        """
        Initialize MFA integration

        Args:
            corpus_dir: Directory containing audio and text files
            output_dir: Directory for alignment outputs
            acoustic_model: MFA acoustic model name (default: russian_mfa)
            dictionary: MFA dictionary name (default: russian_mfa)
            hop_length: Audio hop length for mel spectrogram
            sample_rate: Audio sample rate
        """
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Create output directories
        self.alignment_dir = self.output_dir / "alignments"
        self.cache_dir = self.output_dir / "alignment_cache"
        self.alignment_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check MFA installation
        self.mfa_command = None
        self.mfa_available = self._check_mfa_installation()

    def _find_mfa_in_conda(self) -> Optional[str]:
        """Try to find MFA in common conda installation locations"""
        import os

        # Common conda locations
        home = Path.home()
        conda_locations = [
            home / "homebrew" / "anaconda3" / "bin" / "mfa",
            home / "anaconda3" / "bin" / "mfa",
            home / "miniconda3" / "bin" / "mfa",
            home / ".conda" / "envs" / "base" / "bin" / "mfa",
            Path("/opt/homebrew/anaconda3/bin/mfa"),
            Path("/opt/anaconda3/bin/mfa"),
        ]

        # Also check CONDA_PREFIX environment variable
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_locations.insert(0, Path(conda_prefix) / "bin" / "mfa")

        for mfa_path in conda_locations:
            if mfa_path.exists() and mfa_path.is_file():
                logger.info(f"Found MFA at: {mfa_path}")
                return str(mfa_path)

        return None

    def _check_mfa_installation(self) -> bool:
        """Check if MFA is installed and available"""
        # First try 'mfa' in PATH
        try:
            result = subprocess.run(
                ["mfa", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"MFA found in PATH: {result.stdout.strip()}")
                self.mfa_command = "mfa"
                return True
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            logger.error("MFA command timed out")
            return False

        # Try to find MFA in conda locations
        mfa_path = self._find_mfa_in_conda()
        if mfa_path:
            try:
                result = subprocess.run(
                    [mfa_path, "version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"MFA version: {result.stdout.strip()}")
                    self.mfa_command = mfa_path
                    return True
                else:
                    logger.error(f"MFA found but failed to run: {result.stderr}")
                    logger.error(
                        "\nMFA installation may be incomplete. Try:\n"
                        "  conda install -c conda-forge montreal-forced-aligner --force-reinstall\n"
                        "Or create a fresh environment:\n"
                        "  conda create -n mfa -c conda-forge montreal-forced-aligner\n"
                        "  conda activate mfa"
                    )
                    return False
            except Exception as e:
                logger.error(f"Error running MFA from {mfa_path}: {e}")
                return False

        # MFA not found anywhere
        logger.error(
            "MFA not found in PATH or common conda locations.\n"
            "Install with:\n"
            "  conda install -c conda-forge montreal-forced-aligner\n"
            "Then run: mfa version\n"
            "to verify installation."
        )
        return False

    def download_models(self) -> bool:
        """Download required MFA models if not present"""
        if not self.mfa_available:
            logger.error("MFA is not installed. Cannot download models.")
            logger.info("Install MFA with: conda install -c conda-forge montreal-forced-aligner")
            return False

        logger.info("Downloading MFA Russian models...")

        # Download acoustic model
        try:
            subprocess.run(
                [self.mfa_command, "model", "download", "acoustic", self.acoustic_model],
                check=True,
                timeout=300
            )
            logger.info(f"Acoustic model '{self.acoustic_model}' ready")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download acoustic model: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Model download timed out")
            return False

        # Download dictionary
        try:
            subprocess.run(
                [self.mfa_command, "model", "download", "dictionary", self.dictionary],
                check=True,
                timeout=300
            )
            logger.info(f"Dictionary '{self.dictionary}' ready")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download dictionary: {e}")
            return False

        return True

    def prepare_corpus_for_mfa(self, metadata_file: str) -> Path:
        """
        Prepare corpus in MFA-compatible format

        MFA expects:
        - Audio files (.wav)
        - Corresponding text files (.txt) with same base name

        Args:
            metadata_file: Path to metadata CSV (filename|transcription)

        Returns:
            Path to prepared corpus directory
        """
        if not self.mfa_available:
            logger.error("MFA is not installed. Cannot prepare corpus.")
            raise RuntimeError("MFA not available")

        mfa_corpus_dir = self.output_dir / "mfa_corpus"
        mfa_corpus_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preparing MFA corpus at {mfa_corpus_dir}")

        # Read metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        prepared_count = 0
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue

            audio_file_stem = parts[0]
            text = parts[1]

            # Source audio file
            audio_src = self.corpus_dir / "wavs" / f"{audio_file_stem}.wav"
            if not audio_src.exists():
                logger.warning(f"Audio file not found: {audio_src}")
                continue

            # Copy or symlink audio file
            audio_dst = mfa_corpus_dir / f"{audio_file_stem}.wav"
            if not audio_dst.exists():
                try:
                    # Use symlink for efficiency
                    audio_dst.symlink_to(audio_src.absolute())
                except OSError:
                    # Fall back to copy if symlink fails
                    shutil.copy2(audio_src, audio_dst)

            # Create text file
            text_dst = mfa_corpus_dir / f"{audio_file_stem}.txt"
            with open(text_dst, 'w', encoding='utf-8') as f:
                f.write(text)

            prepared_count += 1

        logger.info(f"Prepared {prepared_count} files for MFA alignment")
        return mfa_corpus_dir

    def run_alignment(self, mfa_corpus_dir: Optional[Path] = None,
                     num_jobs: int = 4) -> bool:
        """
        Run MFA alignment on the corpus

        Args:
            mfa_corpus_dir: Path to MFA-formatted corpus (if None, will use default)
            num_jobs: Number of parallel jobs

        Returns:
            True if alignment succeeded
        """
        if not self.mfa_available:
            logger.error("MFA is not installed. Cannot run alignment.")
            return False

        if mfa_corpus_dir is None:
            mfa_corpus_dir = self.output_dir / "mfa_corpus"

        if not mfa_corpus_dir.exists():
            logger.error(f"MFA corpus directory not found: {mfa_corpus_dir}")
            return False

        logger.info(f"Running MFA alignment on {mfa_corpus_dir}")
        logger.info(f"Output will be saved to {self.alignment_dir}")

        # Build MFA align command
        cmd = [
            self.mfa_command, "align",
            str(mfa_corpus_dir),
            self.dictionary,
            self.acoustic_model,
            str(self.alignment_dir),
            "--clean",
            "--num_jobs", str(num_jobs),
            "--verbose"
        ]

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info("MFA alignment completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"MFA alignment failed with code {result.returncode}")
                logger.error(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            logger.error("MFA alignment timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Error running MFA alignment: {e}")
            return False

    def parse_textgrid(self, textgrid_path: Path) -> List[WordAlignment]:
        """
        Parse TextGrid file to extract alignment information

        Args:
            textgrid_path: Path to TextGrid file

        Returns:
            List of WordAlignment objects
        """
        try:
            import tgt  # TextGridTools library
        except ImportError:
            logger.error("tgt library not installed. Install with: pip install tgt")
            return []

        try:
            textgrid = tgt.io.read_textgrid(str(textgrid_path))

            # Find phoneme and word tiers
            phone_tier = None
            word_tier = None

            for tier in textgrid.tiers:
                if tier.name.lower() in ['phones', 'phone', 'phonemes']:
                    phone_tier = tier
                elif tier.name.lower() in ['words', 'word']:
                    word_tier = tier

            if phone_tier is None:
                logger.error(f"No phone tier found in {textgrid_path}")
                return []

            # Extract phoneme alignments
            phoneme_alignments = []
            for interval in phone_tier:
                if interval.text and interval.text.strip():  # Skip empty intervals
                    phoneme_alignments.append(PhonemeAlignment(
                        phoneme=interval.text.strip(),
                        start_time=interval.start_time,
                        end_time=interval.end_time,
                        duration=interval.end_time - interval.start_time
                    ))

            # If word tier exists, group phonemes by words
            if word_tier:
                word_alignments = []
                for word_interval in word_tier:
                    if not word_interval.text or not word_interval.text.strip():
                        continue

                    # Find phonemes within this word's time range
                    word_phonemes = [
                        p for p in phoneme_alignments
                        if p.start_time >= word_interval.start_time and
                           p.end_time <= word_interval.end_time
                    ]

                    word_alignments.append(WordAlignment(
                        word=word_interval.text.strip(),
                        start_time=word_interval.start_time,
                        end_time=word_interval.end_time,
                        phonemes=word_phonemes
                    ))

                return word_alignments
            else:
                # If no word tier, create a single word alignment with all phonemes
                return [WordAlignment(
                    word="<utterance>",
                    start_time=phoneme_alignments[0].start_time if phoneme_alignments else 0,
                    end_time=phoneme_alignments[-1].end_time if phoneme_alignments else 0,
                    phonemes=phoneme_alignments
                )]

        except Exception as e:
            logger.error(f"Error parsing TextGrid {textgrid_path}: {e}")
            return []

    def get_phoneme_durations(self, audio_file_stem: str) -> Optional[List[int]]:
        """
        Get phoneme durations in frames for a specific audio file

        Args:
            audio_file_stem: Base name of audio file (without extension)

        Returns:
            List of durations in mel spectrogram frames, or None if not found
        """
        # Check cache first
        cache_file = self.cache_dir / f"{audio_file_stem}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # Look for TextGrid file
        textgrid_path = self.alignment_dir / f"{audio_file_stem}.TextGrid"
        if not textgrid_path.exists():
            logger.warning(f"TextGrid not found: {textgrid_path}")
            return None

        # Parse TextGrid
        word_alignments = self.parse_textgrid(textgrid_path)
        if not word_alignments:
            return None

        # Extract all phoneme durations in frames
        durations = []
        for word_align in word_alignments:
            for phoneme_align in word_align.phonemes:
                durations.append(phoneme_align.duration_frames)

        # Cache the result
        with open(cache_file, 'wb') as f:
            pickle.dump(durations, f)

        return durations

    def validate_alignments(self, metadata_file: str) -> Dict[str, any]:
        """
        Validate alignment results

        Args:
            metadata_file: Path to metadata CSV

        Returns:
            Validation statistics
        """
        with open(metadata_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_files = 0
        aligned_files = 0
        failed_files = []
        duration_stats = []

        for line in lines:
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue

            audio_file_stem = parts[0]
            total_files += 1

            durations = self.get_phoneme_durations(audio_file_stem)
            if durations:
                aligned_files += 1
                duration_stats.extend(durations)
            else:
                failed_files.append(audio_file_stem)

        stats = {
            'total_files': total_files,
            'aligned_files': aligned_files,
            'failed_files': len(failed_files),
            'alignment_rate': aligned_files / total_files if total_files > 0 else 0,
            'failed_file_list': failed_files[:10],  # First 10 failed files
            'avg_duration_frames': sum(duration_stats) / len(duration_stats) if duration_stats else 0,
            'min_duration_frames': min(duration_stats) if duration_stats else 0,
            'max_duration_frames': max(duration_stats) if duration_stats else 0,
        }

        logger.info(f"Alignment validation: {aligned_files}/{total_files} files aligned "
                   f"({stats['alignment_rate']*100:.1f}%)")
        if failed_files:
            logger.warning(f"Failed alignments: {failed_files[:5]}...")

        return stats


def setup_mfa_for_corpus(corpus_dir: str,
                        output_dir: str,
                        metadata_file: str = "metadata_RUSLAN_22200.csv",
                        num_jobs: int = 4) -> MFAIntegration:
    """
    Complete MFA setup and alignment workflow

    Args:
        corpus_dir: Path to corpus directory
        output_dir: Path to output directory
        metadata_file: Name of metadata file
        num_jobs: Number of parallel jobs

    Returns:
        MFAIntegration instance with alignments ready
    """
    logger.info("="*60)
    logger.info("MFA Integration Setup")
    logger.info("="*60)

    # Initialize MFA integration
    mfa = MFAIntegration(corpus_dir, output_dir)

    # Check if MFA is available
    if not mfa.mfa_available:
        logger.error("""\nMFA is not installed!\n\nTo use MFA for accurate phoneme alignments, install it with:\n  conda install -c conda-forge montreal-forced-aligner\n\nAlternatively, the training pipeline can work without MFA using\nestimated durations from the forced aligner or preprocessing.\n""")
        return None

    # Download models
    logger.info("\n[1/4] Downloading MFA models...")
    if not mfa.download_models():
        logger.error("Failed to download models")
        return None

    # Prepare corpus
    logger.info("\n[2/4] Preparing corpus for MFA...")
    metadata_path = Path(corpus_dir) / metadata_file
    mfa_corpus_dir = mfa.prepare_corpus_for_mfa(str(metadata_path))

    # Run alignment
    logger.info("\n[3/4] Running forced alignment...")
    if not mfa.run_alignment(mfa_corpus_dir, num_jobs=num_jobs):
        logger.error("Alignment failed")
        return None

    # Validate results
    logger.info("\n[4/4] Validating alignments...")
    stats = mfa.validate_alignments(str(metadata_path))

    logger.info("\n" + "="*60)
    logger.info("MFA Integration Complete")
    logger.info("="*60)
    logger.info(f"Alignment rate: {stats['alignment_rate']*100:.1f}%")
    logger.info(f"Average phoneme duration: {stats['avg_duration_frames']:.1f} frames")
    logger.info(f"Alignments saved to: {mfa.alignment_dir}")
    logger.info("="*60)

    return mfa


if __name__ == "__main__":
    """Run MFA alignment as standalone script"""
    import argparse

    parser = argparse.ArgumentParser(description="Run MFA alignment on Russian corpus")
    parser.add_argument("--corpus", default="./ruslan_corpus",
                       help="Path to corpus directory")
    parser.add_argument("--output", default="./mfa_output",
                       help="Path to output directory")
    parser.add_argument("--metadata", default="metadata_RUSLAN_22200.csv",
                       help="Metadata file name")
    parser.add_argument("--jobs", type=int, default=4,
                       help="Number of parallel jobs")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run MFA setup
    mfa = setup_mfa_for_corpus(
        args.corpus,
        args.output,
        args.metadata,
        args.jobs
    )

    if mfa:
        logger.info("\nMFA alignment completed successfully!")
        logger.info(f"Use alignments from: {mfa.alignment_dir}")
    else:
        logger.error("\nMFA alignment failed!")
        exit(1)
