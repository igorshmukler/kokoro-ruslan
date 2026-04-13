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
import re
import unicodedata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MFA russian_mfa → text-processor phoneme normalisation
# ---------------------------------------------------------------------------
# The MFA `russian_mfa` model uses IPA symbols with combining diacritics and
# several phoneme symbols that differ from the text processor's inventory.
# This mapping converts MFA phones to their text-processor equivalents so that
# the positional duration list aligns 1-to-1 with the phoneme sequence
# produced by RussianPhonemeProcessor + flatten_phoneme_output_with_sil().
_MFA_PHONE_MAP: Dict[str, str] = {
    # Dental diacritics → plain
    's\u0320': 's',   # s̪  (s + combining minus sign below)
    't\u0320': 't',   # t̪
    'n\u0320': 'n',   # n̪
    'd\u0320': 'd',   # d̪
    'z\u0320': 'z',   # z̪
    # Dental affricate
    't\u0320s\u0320': 'ts',  # t̪s̪
    # Laterals
    '\u026b': 'l',    # ɫ (dark L) → plain l
    '\u028e': 'lʲ',   # ʎ (palatal lateral) → palatalized l
    # Nasals
    '\u0272': 'nʲ',   # ɲ (palatal nasal) → palatalized n
    # Fricatives
    '\u0282': 'ʃ',    # ʂ (retroflex) → postalveolar
    '\u0255\u02d0': 'ʃtʃ',  # ɕː (long alveolopalatal) → щ
    '\u00e7': 'xʲ',   # ç (voiceless palatal fricative) → palatalized x
    # Stops
    '\u0261': 'g',    # ɡ (IPA g U+0261) → ASCII g
    'c': 'kʲ',        # voiceless palatal stop → palatalized k
    '\u025f': 'gʲ',   # ɟ (voiced palatal stop) → palatalized g
    # Affricates
    't\u0255': 'tʃ',  # tɕ (voiceless alveolopalatal affricate) → "ч"
    't\u0282\u02d0': 'tʃ',  # tʂː (long retroflex affricate)
    'd\u0290\u02d0': 'ʐ',   # dʐː (long voiced retroflex)
    '\u0291\u02d0': 'zʲ',   # ʑː (long voiced alveolopalatal)
    # Vowels
    '\u025b': 'e',    # ɛ (open-mid front) → close-mid front
    '\u028a': 'u',    # ʊ (near-close back) → close back
    '\u00e6': 'a',    # æ (near-open front) → open central
    '\u0289': 'u',    # ʉ (close central) → close back
    '\u0275': 'o',    # ɵ (close-mid central) → close-mid back
}

# Vowels that can merge with a preceding 'j' to form an iotated vowel.
# Text processor emits ja/jo/ju/je/jɐ/jɪ/jə as single tokens for word-initial
# iotated vowels; MFA splits them into j + vowel.
_IOTATION_MERGE: Dict[str, str] = {
    'a': 'ja', 'o': 'jo', 'u': 'ju', 'e': 'je',
    'ɐ': 'jɐ', 'ɪ': 'jɪ', 'ə': 'jə',
}


def _normalize_mfa_phone(label: str) -> str:
    """Map an MFA phone label to its text-processor equivalent."""
    # Try the raw label first (covers multi-char phones like t̪s̪).
    mapped = _MFA_PHONE_MAP.get(label)
    if mapped is not None:
        return mapped

    # Normalise Unicode (NFC) and retry — some TextGrid editors save
    # combining diacritics in different orders.
    nfc = unicodedata.normalize('NFC', label)
    mapped = _MFA_PHONE_MAP.get(nfc)
    if mapped is not None:
        return mapped

    # Handle length mark (ː, U+02D0): normalise the base phone and
    # re-append ː so that e.g. tɕː → tʃː and nʲː → nʲː.
    if nfc.endswith('\u02d0'):
        base = nfc[:-1]
        base_norm = _normalize_mfa_phone(base)
        if base_norm != base:
            return base_norm + '\u02d0'

    # Strip combining characters and retry — catches dental diacritics that
    # were not in the map (e.g. rare combining forms).
    stripped = ''.join(
        ch for ch in nfc
        if unicodedata.category(ch) not in ('Mn', 'Mc', 'Me')  # combining marks
    )
    if stripped != nfc:
        mapped = _MFA_PHONE_MAP.get(stripped)
        if mapped is not None:
            return mapped
        # If the stripped form is a known text-proc phone, use it directly.
        # (e.g. "rʲ" with an extra combining mark → stripped "rʲ" already in vocab)
        return stripped

    return label  # return as-is; DP alignment will handle any residual mismatch


# ---------------------------------------------------------------------------
# Reverse iotation map: text-proc iotated vowel → expected MFA vowel
# ---------------------------------------------------------------------------
_IOTATION_COMPONENTS: Dict[str, str] = {v: k for k, v in _IOTATION_MERGE.items()}
# {'ja': 'a', 'jo': 'o', 'ju': 'u', 'je': 'e', 'jɐ': 'ɐ', 'jɪ': 'ɪ', 'jə': 'ə'}

_PROSODY_TOKENS = frozenset({'<period>', '<exclaim>', '<question>', '<comma>'})


def _phones_equivalent(mfa_phone: str, tp_phone: str) -> bool:
    """Return True if the normalised MFA phone and text-processor phone
    represent the same sound."""
    if mfa_phone == tp_phone:
        return True
    # Geminate base match: strip ː from MFA phone
    if mfa_phone.endswith('\u02d0') and mfa_phone[:-1] == tp_phone:
        return True
    return False


def align_durations(
    mfa_labeled: List[Tuple[str, int]],
    text_phones: List[str],
) -> Optional[List[int]]:
    """Align MFA phone-duration pairs to a text-processor phoneme sequence
    using Needleman-Wunsch dynamic programming with extensions for:

    - 2→1 iotation merge  (MFA ``j`` + vowel → text-proc iotated vowel)
    - 1→2 geminate split   (MFA ``Xː`` → text-proc ``X X``)
    - 1→N spn expansion    (MFA ``spn`` → K consecutive text-proc phones)
    - 0-cost insertion of prosody tokens (``<period>``, etc.)
    - Low-cost insertion of inter-word ``<sil>``

    Returns a duration list with exactly ``len(text_phones)`` entries,
    or *None* if the sequences are fundamentally incompatible.
    """
    n = len(mfa_labeled)
    m = len(text_phones)
    if m == 0:
        return []
    if n == 0:
        return [0] * m

    INF = float('inf')

    # ── cost constants ────────────────────────────────────────────────
    MATCH        = 0.0
    MISMATCH     = 3.0
    SKIP_MFA     = 1.5   # drop an MFA phone
    SKIP_SIL     = 0.05  # insert a text-proc <sil> with no MFA counterpart
    SKIP_PROSODY = 0.0   # insert a prosody token (always expected)
    SKIP_PHONE   = 2.0   # insert a real text-proc phone with no MFA match
    IOTATION     = 0.0   # merge MFA j+V → text-proc jV
    GEMINATE     = 0.0   # split MFA Xː  → text-proc X X
    SPN_PER_PHONE = 0.2  # cost per phone in spn 1:N expansion
    MAX_SPN_SPAN  = 40   # max phones a single spn can expand to

    # ── forward DP ────────────────────────────────────────────────────
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            c = dp[i][j]
            if c >= INF:
                continue

            # 1:1 match / substitution
            if i < n and j < m:
                eq = _phones_equivalent(mfa_labeled[i][0], text_phones[j])
                nc = c + (MATCH if eq else MISMATCH)
                if nc < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = nc
                    bp[i + 1][j + 1] = (i, j, 'M')

            # skip MFA phone (gap in text-proc)
            if i < n:
                nc = c + SKIP_MFA
                if nc < dp[i + 1][j]:
                    dp[i + 1][j] = nc
                    bp[i + 1][j] = (i, j, 'Dm')

            # skip text-proc phone (gap in MFA)
            if j < m:
                tp = text_phones[j]
                gc = SKIP_PROSODY if tp in _PROSODY_TOKENS else (
                     SKIP_SIL if tp == '<sil>' else SKIP_PHONE)
                nc = c + gc
                if nc < dp[i][j + 1]:
                    dp[i][j + 1] = nc
                    bp[i][j + 1] = (i, j, 'Dt')

            # 2:1 iotation merge — MFA j + vowel → text-proc jV
            if (i + 1 < n and j < m
                    and text_phones[j] in _IOTATION_COMPONENTS
                    and mfa_labeled[i][0] == 'j'
                    and mfa_labeled[i + 1][0] == _IOTATION_COMPONENTS[text_phones[j]]):
                nc = c + IOTATION
                if nc < dp[i + 2][j + 1]:
                    dp[i + 2][j + 1] = nc
                    bp[i + 2][j + 1] = (i, j, 'I')

            # 1:2 geminate split — MFA Xː → text-proc X X
            if (i < n and j + 1 < m
                    and '\u02d0' in mfa_labeled[i][0]):
                base = mfa_labeled[i][0].replace('\u02d0', '')
                if text_phones[j] == base and text_phones[j + 1] == base:
                    nc = c + GEMINATE
                    if nc < dp[i + 1][j + 2]:
                        dp[i + 1][j + 2] = nc
                        bp[i + 1][j + 2] = (i, j, 'G')

            # 1:N spn expansion — MFA spoken-noise → K text-proc phones
            if i < n and mfa_labeled[i][0] == 'spn':
                max_k = min(m - j, MAX_SPN_SPAN)
                for k in range(1, max_k + 1):
                    nc = c + SPN_PER_PHONE * k
                    if nc < dp[i + 1][j + k]:
                        dp[i + 1][j + k] = nc
                        bp[i + 1][j + k] = (i, j, f'S{k}')

    if dp[n][m] >= INF:
        return None

    # ── traceback ─────────────────────────────────────────────────────
    ops: List[Tuple[int, int, str]] = []
    ci, cj = n, m
    while ci > 0 or cj > 0:
        entry = bp[ci][cj]
        if entry is None:
            return None
        pi, pj, op = entry
        ops.append((pi, pj, op))
        ci, cj = pi, pj
    ops.reverse()

    # ── assign durations ──────────────────────────────────────────────
    durations = [0] * m
    pending = 0  # accumulated frames from skipped MFA phones

    for pi, pj, op in ops:
        if op == 'M':          # 1:1
            durations[pj] = mfa_labeled[pi][1] + pending
            pending = 0
        elif op == 'Dm':       # skip MFA phone
            pending += mfa_labeled[pi][1]
        elif op == 'Dt':       # skip text-proc phone (prosody / sil / unmatched)
            durations[pj] = 0
        elif op == 'I':        # iotation merge
            durations[pj] = mfa_labeled[pi][1] + mfa_labeled[pi + 1][1] + pending
            pending = 0
        elif op == 'G':        # geminate split
            total = mfa_labeled[pi][1] + pending
            half = total // 2
            durations[pj] = half
            durations[pj + 1] = total - half
            pending = 0
        elif op[0] == 'S':     # spn 1:N expansion
            k = int(op[1:])
            total = mfa_labeled[pi][1] + pending
            pending = 0
            per_phone = total // k
            remainder = total % k
            for offset in range(k):
                durations[pj + offset] = per_phone + (1 if offset < remainder else 0)

    # Flush any remaining pending frames into the last text-proc phone
    if pending > 0:
        durations[-1] += pending

    return durations


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
                clean_text = re.sub(r'[^\w\s]', '', text).lower()
                f.write(clean_text)

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

    def parse_textgrid(self, textgrid_path: Path) -> Tuple[List[WordAlignment], List[Tuple[float, float]]]:
        """Parse a TextGrid file and return phone alignments + word boundaries.

        Returns:
            (word_alignments, word_boundaries)
            word_boundaries is a list of (start_time, end_time) for each word
            in the word tier (excluding trailing empty intervals).
        """
        import tgt
        try:
            textgrid = tgt.io.read_textgrid(str(textgrid_path))
            phone_tier = textgrid.get_tier_by_name('phones')

            phoneme_alignments = []
            for interval in phone_tier:
                label = interval.text.strip()
                if not label or label.lower() in ['sil', 'sp', '']:
                    label = '<sil>'

                phoneme_alignments.append(PhonemeAlignment(
                    phoneme=label,
                    start_time=interval.start_time,
                    end_time=interval.end_time,
                    duration=interval.end_time - interval.start_time
                ))

            # Extract word boundaries from the word tier.
            word_boundaries: List[Tuple[float, float]] = []
            try:
                word_tier = textgrid.get_tier_by_name('words')
                for interval in word_tier:
                    text = interval.text.strip()
                    if text:  # skip trailing empty intervals
                        word_boundaries.append((interval.start_time, interval.end_time))
            except Exception:
                pass  # word tier missing — word_boundaries stays empty

            word_alignments = [WordAlignment(
                word="<utterance>",
                start_time=phoneme_alignments[0].start_time,
                end_time=phoneme_alignments[-1].end_time,
                phonemes=phoneme_alignments
            )]
            return word_alignments, word_boundaries
        except Exception as e:
            logger.error(f"Error parsing TextGrid {textgrid_path}: {e}")
            return [], []

    def get_phoneme_durations(
        self,
        audio_file_stem: str,
        actual_mel_len: Optional[int] = None,
        strip_outer_silences: bool = False,
    ) -> Optional[List[int]]:
        """Return per-phoneme frame counts from the TextGrid alignment.

        The returned duration list is normalised to match the phoneme sequence
        produced by ``flatten_phoneme_output_with_sil()``:

        1. MFA phones are mapped to the text-processor IPA inventory via
           ``_MFA_PHONE_MAP`` (dental diacritics, palatal symbols, etc.).
        2. Consecutive ``j`` + vowel pairs that the text processor represents
           as a single iotated-vowel token (``ja``, ``jo``, …) are merged and
           their frame counts summed.
        3. ``<sil>`` tokens are inserted at word boundaries so inter-word
           silence alignment matches the text-processor convention.
        4. If any ``spn`` (spoken-noise) interval is present, the alignment
           is considered unreliable and ``None`` is returned so the caller
           falls back to estimated durations.

        Args:
            audio_file_stem: Base name of the audio file (no extension).
            actual_mel_len: When provided, the last duration is adjusted so
                that ``sum(durations) == actual_mel_len``.
            strip_outer_silences: When True, leading and trailing ``<sil>``
                intervals are removed and their frame counts are absorbed into
                the first/last real phoneme respectively.
        """
        textgrid_path = self.alignment_dir / f"{audio_file_stem}.TextGrid"
        if not textgrid_path.exists():
            return None

        word_alignments, word_boundaries = self.parse_textgrid(textgrid_path)
        if not word_alignments:
            return None

        # Build flat (label, frames, start_time) list for the whole utterance.
        flat: List[Tuple[str, int, float]] = [
            (p.phoneme, p.duration_frames, p.start_time)
            for w in word_alignments
            for p in w.phonemes
        ]

        # ── Reject utterances with spn (spoken noise) ────────────────────
        # spn means MFA couldn't align one or more words; the alignment is
        # unreliable so we fall back to estimated durations instead of using
        # corrupted positional targets.
        if any(label == 'spn' for label, _, _ in flat):
            return None

        # ── Strip outer silences ──────────────────────────────────────────
        if strip_outer_silences and flat:
            while len(flat) > 1 and flat[0][0] == '<sil>':
                _, sil_dur, _ = flat.pop(0)
                lbl, dur, st = flat[0]
                flat[0] = (lbl, dur + sil_dur, st)

            while len(flat) > 1 and flat[-1][0] == '<sil>':
                _, sil_dur, _ = flat.pop()
                lbl, dur, st = flat[-1]
                flat[-1] = (lbl, dur + sil_dur, st)

        # ── Normalise MFA phone labels ────────────────────────────────────
        flat = [(_normalize_mfa_phone(lbl), dur, st) for lbl, dur, st in flat]

        # ── Merge iotated vowels (j + vowel → single token) ──────────────
        # The text processor emits ja/jo/ju/je/jɐ/jɪ/jə as one phoneme for
        # word-initial iotated Cyrillic vowels (я, ё, ю, е).  MFA always
        # splits them into two phones: j + vowel.  We merge when 'j' starts
        # at a word boundary (= word-initial iotation) to match the text
        # processor.  We do NOT merge mid-word 'j' + vowel because that can
        # come from 'й' before a vowel (e.g. "район" → r a j ə n) where the
        # text processor keeps them separate.
        word_start_times = {wb[0] for wb in word_boundaries} if word_boundaries else set()
        merged: List[Tuple[str, int]] = []
        i = 0
        while i < len(flat):
            lbl, dur, st = flat[i]
            if (lbl == 'j'
                    and i + 1 < len(flat)
                    and flat[i + 1][0] in _IOTATION_MERGE
                    and st in word_start_times):
                # Merge j + vowel → iotated vowel, sum durations
                vowel_lbl, vowel_dur, _ = flat[i + 1]
                merged.append((_IOTATION_MERGE[vowel_lbl], dur + vowel_dur))
                i += 2
            else:
                merged.append((lbl, dur))
                i += 1

        # ── Insert inter-word <sil> tokens ────────────────────────────────
        # The text processor inserts <sil> before every word except the first.
        # MFA doesn't emit explicit silence between contiguous words, but the
        # word-tier boundaries tell us where one word ends and the next begins.
        # We insert <sil> with 0-frame duration at each word boundary.
        if word_boundaries and len(word_boundaries) > 1:
            # Build a set of phone-index positions where a new word starts.
            # Walk the merged list and match start-of-word to word_boundaries.
            # First, we need the start times from the flat (pre-merge) list,
            # but after merging j+vowel the indexing shifts.  Instead we use
            # the word boundary end-times: when the cumulative phone index
            # crosses a word boundary, we insert <sil>.
            #
            # Simpler approach: build from word boundaries directly.
            # word_boundaries[k] = (start_k, end_k).  The first phone of
            # word k+1 is the first phone whose original start_time == start_{k+1}.
            # We need to find that position in the *merged* list.
            #
            # Re-derive start_times for the merged list:
            merged_start_times: List[float] = []
            j = 0
            mi = 0
            while mi < len(merged) and j < len(flat):
                merged_start_times.append(flat[j][2])
                # If this merged entry consumed 2 flat entries (iotation merge)
                lbl_m, _ = merged[mi]
                if lbl_m in _IOTATION_MERGE.values() and j + 1 < len(flat):
                    j += 2
                else:
                    j += 1
                mi += 1

            # Find insertion points for <sil> (before word k for k >= 1).
            sil_insert_positions: List[int] = []
            wb_starts = {wb[0] for wb in word_boundaries[1:]}  # skip first word
            for pos, st in enumerate(merged_start_times):
                if st in wb_starts:
                    sil_insert_positions.append(pos)

            # Insert <sil> in reverse order so indices stay valid.
            for pos in reversed(sil_insert_positions):
                merged.insert(pos, ('<sil>', 0))

        durations = [dur for _, dur in merged]

        if actual_mel_len is not None:
            current_sum = sum(durations)
            diff = actual_mel_len - current_sum
            if diff != 0 and len(durations) > 0:
                durations[-1] = max(1, durations[-1] + diff)

        return durations

    def get_aligned_durations(
        self,
        audio_file_stem: str,
        phoneme_sequence: List[str],
    ) -> Optional[List[int]]:
        """Return MFA durations aligned to the text-processor phoneme sequence.

        Uses :func:`align_durations` (Needleman-Wunsch DP) to produce a
        duration list whose length **exactly** matches ``phoneme_sequence``.
        Handles iotation merges, geminate splits, spn expansion, prosody-token
        insertions and inter-word ``<sil>`` gaps automatically.

        Returns ``None`` when the TextGrid is missing or produces no valid
        alignment — the caller should fall back to estimated durations.
        """
        textgrid_path = self.alignment_dir / f"{audio_file_stem}.TextGrid"
        if not textgrid_path.exists():
            return None

        word_alignments, _ = self.parse_textgrid(textgrid_path)
        if not word_alignments:
            return None

        # Build flat (label, frames) list for the whole utterance.
        flat: List[Tuple[str, int]] = [
            (p.phoneme, p.duration_frames)
            for w in word_alignments
            for p in w.phonemes
        ]

        # Strip leading/trailing <sil> (recording padding — not present in
        # the text-processor output).  Absorb their frames into the adjacent
        # real phone so total frame count is preserved.
        while len(flat) > 1 and flat[0][0] == '<sil>':
            _, sil_dur = flat.pop(0)
            lbl, dur = flat[0]
            flat[0] = (lbl, dur + sil_dur)
        while len(flat) > 1 and flat[-1][0] == '<sil>':
            _, sil_dur = flat.pop()
            lbl, dur = flat[-1]
            flat[-1] = (lbl, dur + sil_dur)

        # Normalise phone labels (dental diacritics, palatal symbols, etc.)
        flat = [(_normalize_mfa_phone(lbl), dur) for lbl, dur in flat]

        # DP alignment
        durations = align_durations(flat, phoneme_sequence)
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
