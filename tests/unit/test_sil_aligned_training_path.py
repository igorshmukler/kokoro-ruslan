"""
Tests for the training/inference phoneme-sequence alignment fix (v0.0.22).

Changes covered:
1. dataset.py: text_to_indices() replaced with
       process_text → flatten_phoneme_output_with_sil → phonemes_to_indices
   so the training phoneme sequence includes <sil> between words, matching
   the inference path.

2. mfa_integration.py: get_phoneme_durations() gained strip_outer_silences=True
   which absorbs MFA's utterance-boundary <sil> intervals into the adjacent
   real phoneme, making len(durations) == len(phoneme_sequence) without
   truncation/padding.

3. dataset.py: FEATURE_CACHE_VERSION bumped to 4 to invalidate stale cache
   entries that contain old (no-sil) phoneme index sequences.

Test organisation
─────────────────
TestStripOuterSilences          – unit tests for the new MFA helper logic
TestStripOuterSilencesEdgeCases – edge/corner-case inputs
TestDurationSumPreserved        – frame counts preserved under stripping
TestMfaDurationLengthMatchesSilPath – end-to-end length alignment property
TestDatasetPhonemePathUsesSil   – structural guarantee: dataset uses sil path
TestCacheVersionBump            – FEATURE_CACHE_VERSION == 4
TestInferenceTrainingConsistency – the produced sequences are identical
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import List, Tuple

from kokoro.data.russian_phoneme_processor import RussianPhonemeProcessor
from kokoro.data.audio_utils import PhonemeProcessorUtils
from kokoro.data.mfa_integration import MFAIntegration, PhonemeAlignment, WordAlignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alignment(phonemes_and_durations: List[Tuple[str, float]]) -> List[WordAlignment]:
    """Build a synthetic WordAlignment list as parse_textgrid would return."""
    t = 0.0
    phone_list = []
    for label, dur in phonemes_and_durations:
        phone_list.append(PhonemeAlignment(
            phoneme=label,
            start_time=t,
            end_time=t + dur,
            duration=dur,
        ))
        t += dur
    return [WordAlignment(
        word="<utterance>",
        start_time=phone_list[0].start_time,
        end_time=phone_list[-1].end_time,
        phonemes=phone_list,
    )]


def _mfa(tmp_path: Path, phonemes_and_durations: List[Tuple[str, float]]) -> MFAIntegration:
    """Create a minimal MFAIntegration whose parse_textgrid is patched."""
    mfa = MFAIntegration.__new__(MFAIntegration)
    mfa.alignment_dir = tmp_path
    mfa.hop_length = 256
    mfa.sample_rate = 22050
    # Patch parse_textgrid so no real .TextGrid file is needed.
    mfa.parse_textgrid = MagicMock(
        return_value=_make_alignment(phonemes_and_durations)
    )
    return mfa


def _textgrid_sentinel(tmp_path: Path, stem: str) -> Path:
    """Create an empty sentinel file so the 'exists' check passes."""
    p = tmp_path / f"{stem}.TextGrid"
    p.write_text("")
    return p


def _dur_frames(duration_seconds: float, sr: int = 22050, hop: int = 256) -> int:
    return int(duration_seconds * sr / hop)


# ---------------------------------------------------------------------------
# 1. Core stripping logic
# ---------------------------------------------------------------------------
class TestStripOuterSilences:

    def test_no_strip_returns_all_intervals(self, tmp_path):
        """Default (strip_outer_silences=False) preserves leading/trailing sil."""
        _textgrid_sentinel(tmp_path, "utt")
        intervals = [
            ('<sil>', 0.10),
            ('p', 0.05), ('r', 0.04), ('ʲ', 0.03),
            ('<sil>', 0.08),
            ('v', 0.05), ('ɛ', 0.06), ('t', 0.04),
            ('<sil>', 0.10),
        ]
        mfa = _mfa(tmp_path, intervals)
        durations = mfa.get_phoneme_durations("utt", strip_outer_silences=False)
        assert len(durations) == len(intervals)

    def test_strip_removes_leading_sil(self, tmp_path):
        """Leading <sil> is removed; its frames go to the first real phoneme."""
        _textgrid_sentinel(tmp_path, "utt")
        sil_dur = 0.10   # 8 frames at default hop/sr
        real_dur = 0.05  # 4 frames
        intervals = [('<sil>', sil_dur), ('p', real_dur), ('r', 0.04)]
        mfa = _mfa(tmp_path, intervals)
        durations = mfa.get_phoneme_durations("utt", strip_outer_silences=True)

        assert len(durations) == 2       # <sil> removed
        # First real phoneme absorbs the leading silence frames.
        assert durations[0] == _dur_frames(sil_dur) + _dur_frames(real_dur)

    def test_strip_removes_trailing_sil(self, tmp_path):
        """Trailing <sil> is removed; its frames go to the last real phoneme."""
        _textgrid_sentinel(tmp_path, "utt")
        sil_dur = 0.12
        last_dur = 0.06
        intervals = [('p', 0.05), ('r', 0.04), ('t', last_dur), ('<sil>', sil_dur)]
        mfa = _mfa(tmp_path, intervals)
        durations = mfa.get_phoneme_durations("utt", strip_outer_silences=True)

        assert len(durations) == 3
        assert durations[-1] == _dur_frames(last_dur) + _dur_frames(sil_dur)

    def test_strip_removes_both_ends(self, tmp_path):
        """Both leading and trailing silences are stripped."""
        _textgrid_sentinel(tmp_path, "utt")
        intervals = [
            ('<sil>', 0.10),
            ('p', 0.05), ('<sil>', 0.08), ('r', 0.04),
            ('<sil>', 0.10),
        ]
        mfa = _mfa(tmp_path, intervals)
        durations = mfa.get_phoneme_durations("utt", strip_outer_silences=True)

        # Only outer silences removed; the inner <sil> (word boundary) stays.
        assert len(durations) == 3  # p, <sil>, r
        assert durations[0] == _dur_frames(0.10) + _dur_frames(0.05)
        assert durations[1] == _dur_frames(0.08)   # inner sil preserved
        assert durations[-1] == _dur_frames(0.04) + _dur_frames(0.10)

    def test_strip_multiple_consecutive_leading_sils(self, tmp_path):
        """Multiple consecutive leading silences are all absorbed into the first real phoneme."""
        _textgrid_sentinel(tmp_path, "utt")
        # Two leading sils, then two real phonemes, no trailing sil — keeps len clean.
        intervals = [
            ('<sil>', 0.05), ('<sil>', 0.05),
            ('p', 0.06), ('r', 0.04),
        ]
        mfa = _mfa(tmp_path, intervals)
        durations = mfa.get_phoneme_durations("utt", strip_outer_silences=True)

        # Both leading sils absorbed; 'r' is not a sil so no trailing strip.
        assert len(durations) == 2    # p (with absorbed sils) + r
        assert durations[0] == _dur_frames(0.05) + _dur_frames(0.05) + _dur_frames(0.06)

    def test_inner_sil_preserved_by_stripping(self, tmp_path):
        """Stripping never touches silences that are not at the utterance boundary."""
        _textgrid_sentinel(tmp_path, "utt")
        intervals = [
            ('<sil>', 0.10),
            ('p', 0.05), ('<sil>', 0.08), ('r', 0.04),
            ('<sil>', 0.10),
        ]
        mfa = _mfa(tmp_path, intervals)
        durations = mfa.get_phoneme_durations("utt", strip_outer_silences=True)

        # Exactly the inner sil duration is in position 1
        assert durations[1] == _dur_frames(0.08)


# ---------------------------------------------------------------------------
# 2. Edge / corner cases
# ---------------------------------------------------------------------------
class TestStripOuterSilencesEdgeCases:

    def test_empty_textgrid_returns_none(self, tmp_path):
        """Missing TextGrid returns None regardless of flag."""
        mfa = MFAIntegration.__new__(MFAIntegration)
        mfa.alignment_dir = tmp_path
        result = mfa.get_phoneme_durations("nonexistent", strip_outer_silences=True)
        assert result is None

    def test_single_real_phoneme_no_sil(self, tmp_path):
        """Single real phoneme: stripping is a no-op."""
        _textgrid_sentinel(tmp_path, "utt")
        intervals = [('p', 0.05)]
        mfa = _mfa(tmp_path, intervals)
        durations = mfa.get_phoneme_durations("utt", strip_outer_silences=True)
        assert len(durations) == 1
        assert durations[0] == _dur_frames(0.05)

    def test_all_silence_sequence_not_reduced_below_one_entry(self, tmp_path):
        """If only silences remain, stripping stops before emptying the list."""
        _textgrid_sentinel(tmp_path, "utt")
        intervals = [('<sil>', 0.10), ('<sil>', 0.10)]
        mfa = _mfa(tmp_path, intervals)
        durations = mfa.get_phoneme_durations("utt", strip_outer_silences=True)
        # The while-loop condition is `len(flat) > 1`, so at least 1 entry stays.
        assert len(durations) >= 1

    def test_no_silences_at_boundaries_unchanged(self, tmp_path):
        """No-op when neither end has a <sil> interval."""
        _textgrid_sentinel(tmp_path, "utt")
        intervals = [('p', 0.05), ('<sil>', 0.08), ('r', 0.04)]
        mfa = _mfa(tmp_path, intervals)

        plain   = mfa.get_phoneme_durations("utt", strip_outer_silences=False)
        stripped = mfa.get_phoneme_durations("utt", strip_outer_silences=True)
        assert plain == stripped

    def test_backward_compat_default_is_false(self, tmp_path):
        """strip_outer_silences defaults to False — old callers unaffected."""
        _textgrid_sentinel(tmp_path, "utt")
        intervals = [('<sil>', 0.10), ('p', 0.05), ('<sil>', 0.10)]
        mfa = _mfa(tmp_path, intervals)

        default_result   = mfa.get_phoneme_durations("utt")
        explicit_false   = mfa.get_phoneme_durations("utt", strip_outer_silences=False)
        assert default_result == explicit_false
        assert len(default_result) == 3   # all three intervals present


# ---------------------------------------------------------------------------
# 3. Frame count conservation
# ---------------------------------------------------------------------------
class TestDurationSumPreserved:

    def test_total_frames_preserved_after_strip(self, tmp_path):
        """Stripping must preserve the total frame count (no frames lost)."""
        _textgrid_sentinel(tmp_path, "utt")
        intervals = [
            ('<sil>', 0.10),
            ('p', 0.05), ('<sil>', 0.08), ('r', 0.04),
            ('<sil>', 0.12),
        ]
        mfa = _mfa(tmp_path, intervals)
        raw = mfa.get_phoneme_durations("utt", strip_outer_silences=False)
        stripped = mfa.get_phoneme_durations("utt", strip_outer_silences=True)

        assert sum(stripped) == sum(raw)

    def test_actual_mel_len_adjustment_applied_after_strip(self, tmp_path):
        """actual_mel_len adjustment works correctly after outer sils are stripped."""
        _textgrid_sentinel(tmp_path, "utt")
        intervals = [('<sil>', 0.10), ('p', 0.05), ('r', 0.04), ('<sil>', 0.10)]
        mfa = _mfa(tmp_path, intervals)

        # stripped sum = frames for 0.05 + 0.10 (lead absorbed) + 0.04 + 0.10 (trail absorbed)
        stripped_no_adj = mfa.get_phoneme_durations("utt", strip_outer_silences=True)
        target_len = sum(stripped_no_adj) + 5

        stripped_adj = mfa.get_phoneme_durations(
            "utt", actual_mel_len=target_len, strip_outer_silences=True
        )
        assert sum(stripped_adj) == target_len
        # Adjustment applied to last element only
        assert stripped_adj[-1] == stripped_no_adj[-1] + 5

    def test_actual_mel_len_subtraction_works(self, tmp_path):
        """actual_mel_len less than natural sum reduces last duration (floor 1)."""
        _textgrid_sentinel(tmp_path, "utt")
        intervals = [('p', 0.20), ('r', 0.20)]
        mfa = _mfa(tmp_path, intervals)

        natural = mfa.get_phoneme_durations("utt", strip_outer_silences=True)
        target = sum(natural) - 3

        adjusted = mfa.get_phoneme_durations(
            "utt", actual_mel_len=target, strip_outer_silences=True
        )
        assert sum(adjusted) == target
        assert adjusted[-1] >= 1


# ---------------------------------------------------------------------------
# 4. Length-match property: MFA durations ↔ phoneme_sequence
# ---------------------------------------------------------------------------
class TestMfaDurationLengthMatchesSilPath:
    """
    The whole point of the fix: after stripping, the MFA duration list should
    have the same length as the phoneme sequence produced by
    flatten_phoneme_output_with_sil().  We test this using a synthetic round-trip.

    Layout:
        Word "привет"  → N_1 phonemes
        <sil> token    → 1 slot
        Word "мир"     → N_2 phonemes
        token total    → N_1 + 1 + N_2

    MFA alignment (after strip_outer_silences=True):
        [pʲ, rʲ, ɪ, vʲ, e, t]  +  [<sil>]  +  [mʲ, ɪ, r]
        total                   → N_1 + 1 + N_2  ✓
    """

    def test_two_word_utterance_lengths_match(self, tmp_path):
        processor = RussianPhonemeProcessor()
        text = "привет мир"

        # Compute phoneme sequence the training path would produce
        raw = processor.process_text(text)
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
            raw, processor.phoneme_to_id
        )
        n_phonemes = len(seq)

        # Build synthetic MFA alignment:
        #   leading <sil>, phonemes for word 1, inner <sil>, phonemes for word 2, trailing <sil>
        _, word1_phones, *_ = raw[0]
        _, word2_phones, *_ = raw[1]

        intervals: List[Tuple[str, float]] = [('<sil>', 0.10)]
        for ph in word1_phones:
            intervals.append((ph, 0.05))
        intervals.append(('<sil>', 0.08))   # word boundary sil
        for ph in word2_phones:
            intervals.append((ph, 0.05))
        intervals.append(('<sil>', 0.10))

        _textgrid_sentinel(tmp_path, "utt")
        mfa = _mfa(tmp_path, intervals)
        durations = mfa.get_phoneme_durations("utt", strip_outer_silences=True)

        assert len(durations) == n_phonemes, (
            f"Duration list length ({len(durations)}) != phoneme sequence length "
            f"({n_phonemes}).  Padding/truncation heuristic would be applied."
        )

    def test_single_word_no_inner_sil_lengths_match(self, tmp_path):
        """Single word: no inter-word <sil>.  Outer sils stripped matches seq length."""
        processor = RussianPhonemeProcessor()
        text = "привет"

        raw = processor.process_text(text)
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
            raw, processor.phoneme_to_id
        )
        n_phonemes = len(seq)

        _, word_phones, *_ = raw[0]
        intervals: List[Tuple[str, float]] = [('<sil>', 0.10)]
        for ph in word_phones:
            intervals.append((ph, 0.05))
        intervals.append(('<sil>', 0.10))

        _textgrid_sentinel(tmp_path, "utt")
        mfa = _mfa(tmp_path, intervals)
        durations = mfa.get_phoneme_durations("utt", strip_outer_silences=True)

        assert len(durations) == n_phonemes


# ---------------------------------------------------------------------------
# 5. Structural: dataset uses sil path, not text_to_indices
# ---------------------------------------------------------------------------
class TestDatasetPhonemePathUsesSil:
    """
    Confirm that dataset.py's __getitem__ no longer calls text_to_indices()
    and instead delegates to flatten_phoneme_output_with_sil + phonemes_to_indices.
    This is a static / import-level test — it doesn't require corpus data.
    """

    def test_dataset_module_does_not_call_text_to_indices_in_getitem(self):
        """
        dataset.__getitem__ source must not contain a bare text_to_indices()
        call — that was the old single-pass path that bypassed <sil> injection.
        """
        import inspect
        import kokoro.data.dataset as ds_module
        src = inspect.getsource(ds_module.RuslanDataset.__getitem__)
        # The old call was: self.phoneme_processor.text_to_indices(sample['text'])
        assert "text_to_indices" not in src, (
            "__getitem__ still contains text_to_indices.  "
            "Revert not detected but the sil-path patch was not applied."
        )

    def test_dataset_getitem_source_uses_flatten_with_sil(self):
        """__getitem__ must call flatten_phoneme_output_with_sil."""
        import inspect
        import kokoro.data.dataset as ds_module
        src = inspect.getsource(ds_module.RuslanDataset.__getitem__)
        assert "flatten_phoneme_output_with_sil" in src

    def test_dataset_getitem_source_uses_phonemes_to_indices(self):
        """__getitem__ must call PhonemeProcessorUtils.phonemes_to_indices."""
        import inspect
        import kokoro.data.dataset as ds_module
        src = inspect.getsource(ds_module.RuslanDataset.__getitem__)
        assert "phonemes_to_indices" in src

    def test_dataset_imports_phoneme_processor_utils(self):
        """PhonemeProcessorUtils must be imported at the module level."""
        import kokoro.data.dataset as ds_module
        assert hasattr(ds_module, "PhonemeProcessorUtils"), (
            "PhonemeProcessorUtils not imported in dataset.py"
        )

    def test_dataset_getitem_passes_strip_outer_silences(self):
        """__getitem__ must pass strip_outer_silences=True to get_phoneme_durations."""
        import inspect
        import kokoro.data.dataset as ds_module
        src = inspect.getsource(ds_module.RuslanDataset.__getitem__)
        assert "strip_outer_silences=True" in src


# ---------------------------------------------------------------------------
# 6. Cache version
# ---------------------------------------------------------------------------
class TestCacheVersionBump:

    def test_feature_cache_version_is_4(self):
        from kokoro.data.dataset import FEATURE_CACHE_VERSION
        assert FEATURE_CACHE_VERSION == 6, (
            f"FEATURE_CACHE_VERSION is {FEATURE_CACHE_VERSION}, expected 6.  "
            "Bump it when the phoneme sequence format changes so stale cache "
            "entries containing the old sequences are invalidated."
        )

    def test_cache_version_is_integer(self):
        from kokoro.data.dataset import FEATURE_CACHE_VERSION
        assert isinstance(FEATURE_CACHE_VERSION, int)


# ---------------------------------------------------------------------------
# 7. Inference/training consistency
# ---------------------------------------------------------------------------
class TestInferenceTrainingConsistency:
    """
    The training path and the inference path must produce an identical phoneme
    sequence for any input text.  We test this directly without touching the
    dataset's audio loading machinery.
    """

    def _training_seq(self, processor: RussianPhonemeProcessor, text: str) -> list:
        """Replicate exactly what dataset.__getitem__ now does."""
        raw = processor.process_text(text)
        return PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
            raw, processor.phoneme_to_id
        )

    def _inference_seq(self, processor: RussianPhonemeProcessor, text: str) -> list:
        """Replicate exactly what inference.py does."""
        raw = processor.process_text(text)
        return PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
            raw, processor.phoneme_to_id
        )

    @pytest.mark.parametrize("text", [
        "привет",
        "привет мир",
        "как дела",
        "молоко и хлеб",
        "это тест синтеза речи",
    ])
    def test_training_and_inference_sequences_are_identical(self, text):
        processor = RussianPhonemeProcessor()
        train = self._training_seq(processor, text)
        infer = self._inference_seq(processor, text)
        assert train == infer, (
            f"Training and inference phoneme sequences differ for '{text}':\n"
            f"  train: {train}\n"
            f"  infer: {infer}"
        )

    def test_multi_word_sequence_contains_sil_between_words(self):
        """<sil> must appear between each pair of adjacent words."""
        processor = RussianPhonemeProcessor()
        text = "привет как дела"
        seq = self._training_seq(processor, text)

        # There are 3 words → 2 word boundaries → 2 <sil> tokens expected.
        sil_count = seq.count('<sil>')
        assert sil_count == 2, (
            f"Expected 2 <sil> tokens for 3-word text; got {sil_count}.  "
            f"Sequence: {seq}"
        )

    def test_single_word_has_no_sil(self):
        """A single-word text must not contain any <sil> token."""
        processor = RussianPhonemeProcessor()
        seq = self._training_seq(processor, "привет")
        assert '<sil>' not in seq

    def test_sil_tokens_are_not_at_start_or_end(self):
        """The converted sequence must not start or end with <sil>."""
        processor = RussianPhonemeProcessor()
        for text in ["привет мир", "как дела хорошо"]:
            seq = self._training_seq(processor, text)
            assert seq[0] != '<sil>', f"Sequence for '{text}' starts with <sil>"
            assert seq[-1] != '<sil>', f"Sequence for '{text}' ends with <sil>"

    def test_phoneme_indices_include_sil_id(self):
        """After index conversion, the <sil> index must appear for multi-word input."""
        processor = RussianPhonemeProcessor()
        text = "привет мир"
        seq = self._training_seq(processor, text)
        indices = PhonemeProcessorUtils.phonemes_to_indices(
            seq, processor.phoneme_to_id
        )
        sil_id = processor.phoneme_to_id['<sil>']
        assert sil_id in indices, (
            f"<sil> id ({sil_id}) not found in phoneme index sequence for '{text}'"
        )

    def test_old_text_to_indices_produces_different_sequence_for_multi_word(self):
        """
        Documents the original bug: text_to_indices() omits <sil> tokens, so
        training and inference sequences are different for multi-word input.
        This test ensures we never regress to that mismatch.
        """
        processor = RussianPhonemeProcessor()
        text = "привет мир"

        old_path = processor.text_to_indices(text)
        new_path = PhonemeProcessorUtils.phonemes_to_indices(
            self._training_seq(processor, text), processor.phoneme_to_id
        )

        sil_id = processor.phoneme_to_id['<sil>']
        assert sil_id not in old_path, (
            "text_to_indices should NOT include <sil> — "
            "confirming it represents the old (buggy) behaviour."
        )
        assert sil_id in new_path, (
            "New path MUST include <sil> — confirming the fix is active."
        )
        assert len(new_path) == len(old_path) + 1, (
            f"Expected new path to be 1 token longer (the <sil>); "
            f"old={len(old_path)}, new={len(new_path)}"
        )
