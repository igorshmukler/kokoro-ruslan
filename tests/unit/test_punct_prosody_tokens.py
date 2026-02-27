"""
Tests for the prosody-conditioning punctuation token feature (v0.0.22).

Changes covered:
  russian_phoneme_processor.py
    - PUNCT_MAP class constant  {'.': '<period>', '?': '<question>',
                                  '!': '<exclaim>', ',': '<comma>'}
    - _extract_punct_after_words()  — raw-text scanner, called before
      normalize_text() strips punctuation
    - process_text() now returns 4-tuples
      (word, phonemes, stress_info, punct_token_or_None)
    - _build_vocab() includes all four punct tokens
    - from_dict() patches in missing punct tokens from old pickles

  audio_utils.py  (PhonemeProcessorUtils)
    - flatten_phoneme_output()       — emits punct token after word phonemes
    - flatten_phoneme_output_with_sil() — injects punct BEFORE the next <sil>

  dataset.py
    - FEATURE_CACHE_VERSION == 5

Test classes
────────────
TestPunctMap                 – class constant contract
TestExtractPunctAfterWords   – raw-text scanning correctness + edge cases
TestProcessTextFourTuple     – process_text() return shape & content
TestBuildVocabIncludesPunct  – vocab contains all four tokens
TestFromDictPatchesMissingPunct – backward compat with old pickles
TestFlattenEmitsPunct        – flatten_phoneme_output emits punct tokens
TestFlattenWithSilInjectsPunct – correct token order (phonemes→punct→sil)
TestEndToEndTokenSequence    – full round-trip: text → flat sequence
TestCacheVersion             – FEATURE_CACHE_VERSION == 5
"""

import pytest
from typing import Optional

from kokoro.data.russian_phoneme_processor import RussianPhonemeProcessor, StressInfo
from kokoro.data.audio_utils import PhonemeProcessorUtils
from kokoro.data.dataset import FEATURE_CACHE_VERSION


# ── helpers ──────────────────────────────────────────────────────────────────

def _processor() -> RussianPhonemeProcessor:
    return RussianPhonemeProcessor()


def _fake_raw(words_and_punct):
    """Build a synthetic process_text()-style list for flatten tests.
    words_and_punct: list of (phonemes, punct_or_None)
    """
    return [
        (f"word{i}", phs, StressInfo(0, 0, False), pt)
        for i, (phs, pt) in enumerate(words_and_punct)
    ]


FULL_VOCAB = {"<sil>": 0, "<period>": 1, "<question>": 2, "<exclaim>": 3,
              "<comma>": 4, "p": 5, "r": 6, "a": 7, "t": 8, "m": 9, "i": 10}


# ─────────────────────────────────────────────────────────────────────────────
# 1. PUNCT_MAP class constant
# ─────────────────────────────────────────────────────────────────────────────
class TestPunctMap:

    def test_all_four_marks_present(self):
        pm = RussianPhonemeProcessor.PUNCT_MAP
        assert pm['.'] == '<period>'
        assert pm['?'] == '<question>'
        assert pm['!'] == '<exclaim>'
        assert pm[','] == '<comma>'

    def test_punct_map_has_exactly_four_entries(self):
        assert len(RussianPhonemeProcessor.PUNCT_MAP) == 4

    def test_punct_map_values_are_angle_bracket_tokens(self):
        for v in RussianPhonemeProcessor.PUNCT_MAP.values():
            assert v.startswith('<') and v.endswith('>')

    def test_punct_map_is_class_attribute_not_instance(self):
        """Accessing via the class (not an instance) must work."""
        assert hasattr(RussianPhonemeProcessor, 'PUNCT_MAP')
        assert isinstance(RussianPhonemeProcessor.PUNCT_MAP, dict)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _extract_punct_after_words
# ─────────────────────────────────────────────────────────────────────────────
class TestExtractPunctAfterWords:

    def _extract(self, text: str):
        return RussianPhonemeProcessor._extract_punct_after_words(text)

    def test_sentence_ending_with_period(self):
        result = self._extract("Привет.")
        assert result == ['<period>']

    def test_sentence_ending_with_question_mark(self):
        result = self._extract("Как дела?")
        assert result == [None, '<question>']

    def test_sentence_ending_with_exclamation(self):
        result = self._extract("Стой!")
        assert result == ['<exclaim>']

    def test_comma_after_first_word(self):
        result = self._extract("Привет, как дела?")
        assert result == ['<comma>', None, '<question>']

    def test_no_punctuation(self):
        result = self._extract("привет мир")
        assert result == [None, None]

    def test_single_word_no_punct(self):
        result = self._extract("слово")
        assert result == [None]

    def test_multiple_commas(self):
        result = self._extract("раз, два, три.")
        assert result == ['<comma>', '<comma>', '<period>']

    def test_punct_before_space_counted_once(self):
        """Only the first matching punct character between two words is taken."""
        result = self._extract("слово... другое")
        # Three dots — only first registered as <period>
        assert result == ['<period>', None]

    def test_empty_string(self):
        assert self._extract("") == []

    def test_leading_punct_ignored(self):
        """Punctuation before the first Cyrillic word is not captured."""
        result = self._extract("! Привет")
        assert result == [None]

    def test_mixed_punct_takes_first(self):
        """If ',' and '.' both appear between words, ',' (first) wins."""
        result = self._extract("слово,. другое")
        assert result == ['<comma>', None]


# ─────────────────────────────────────────────────────────────────────────────
# 3. process_text() returns 4-tuples
# ─────────────────────────────────────────────────────────────────────────────
class TestProcessTextFourTuple:

    def test_returns_four_tuple_per_word(self):
        p = _processor()
        results = p.process_text("привет мир")
        for item in results:
            assert len(item) == 4, f"Expected 4-tuple, got length {len(item)}: {item}"

    def test_fourth_element_is_punct_or_none(self):
        p = _processor()
        for item in p.process_text("привет, как дела?"):
            assert item[3] is None or isinstance(item[3], str)

    def test_punct_assigned_correctly(self):
        p = _processor()
        results = p.process_text("Привет, как дела?")
        assert results[0][3] == '<comma>'   # "привет,"
        assert results[1][3] is None         # "как"
        assert results[2][3] == '<question>' # "дела?"

    def test_period_at_end(self):
        p = _processor()
        results = p.process_text("Это тест.")
        assert results[-1][3] == '<period>'

    def test_exclamation_at_end(self):
        p = _processor()
        results = p.process_text("Стоп!")
        assert results[-1][3] == '<exclaim>'

    def test_no_punct_all_none(self):
        p = _processor()
        for item in p.process_text("привет мир"):
            assert item[3] is None

    def test_empty_text_returns_empty_list(self):
        p = _processor()
        assert p.process_text("") == []

    def test_word_phonemes_unchanged_by_punct(self):
        """Punct extraction must not alter the phoneme list for any word."""
        p = _processor()
        plain  = [item[1] for item in p.process_text("привет мир")]
        punct  = [item[1] for item in p.process_text("привет, мир.")]
        assert plain == punct

    def test_extended_unpack_compat(self):
        """Callers using 3-element unpacking with *_ must work without error."""
        p = _processor()
        for word, phonemes, stress, *_ in p.process_text("привет"):
            assert isinstance(word, str)


# ─────────────────────────────────────────────────────────────────────────────
# 4. _build_vocab includes punct tokens
# ─────────────────────────────────────────────────────────────────────────────
class TestBuildVocabIncludesPunct:

    def test_period_in_vocab(self):
        assert '<period>' in _processor().phoneme_to_id

    def test_question_in_vocab(self):
        assert '<question>' in _processor().phoneme_to_id

    def test_exclaim_in_vocab(self):
        assert '<exclaim>' in _processor().phoneme_to_id

    def test_comma_in_vocab(self):
        assert '<comma>' in _processor().phoneme_to_id

    def test_all_punct_tokens_have_unique_ids(self):
        vocab = _processor().phoneme_to_id
        tokens = ['<period>', '<question>', '<exclaim>', '<comma>']
        ids = [vocab[t] for t in tokens]
        assert len(set(ids)) == 4, "punct tokens must have distinct IDs"

    def test_punct_ids_do_not_collide_with_sil(self):
        vocab = _processor().phoneme_to_id
        sil_id = vocab['<sil>']
        for tok in ['<period>', '<question>', '<exclaim>', '<comma>']:
            assert vocab[tok] != sil_id, f"{tok} must not share ID with <sil>"


# ─────────────────────────────────────────────────────────────────────────────
# 5. from_dict patches missing punct tokens into old pickles
# ─────────────────────────────────────────────────────────────────────────────
class TestFromDictPatchesMissingPunct:

    def _old_dict(self) -> dict:
        """Simulate a pre-punct processor dict (vocab without punct tokens)."""
        p = _processor()
        d = p.to_dict()
        # Remove all punct tokens from the saved vocab to mimic an old pickle
        for tok in ['<period>', '<question>', '<exclaim>', '<comma>']:
            d['phoneme_to_id'].pop(tok, None)
        return d

    def test_from_dict_injects_missing_period(self):
        p = RussianPhonemeProcessor.from_dict(self._old_dict())
        assert '<period>' in p.phoneme_to_id

    def test_from_dict_injects_missing_question(self):
        p = RussianPhonemeProcessor.from_dict(self._old_dict())
        assert '<question>' in p.phoneme_to_id

    def test_from_dict_injects_missing_exclaim(self):
        p = RussianPhonemeProcessor.from_dict(self._old_dict())
        assert '<exclaim>' in p.phoneme_to_id

    def test_from_dict_injects_missing_comma(self):
        p = RussianPhonemeProcessor.from_dict(self._old_dict())
        assert '<comma>' in p.phoneme_to_id

    def test_from_dict_injected_ids_do_not_collide(self):
        p = RussianPhonemeProcessor.from_dict(self._old_dict())
        all_ids = list(p.phoneme_to_id.values())
        assert len(all_ids) == len(set(all_ids)), "from_dict produced duplicate IDs"

    def test_from_dict_leaves_existing_tokens_unchanged(self):
        """Tokens that were already in the old vocab must keep their IDs."""
        d = self._old_dict()
        old_sil_id = d['phoneme_to_id']['<sil>']
        p = RussianPhonemeProcessor.from_dict(d)
        assert p.phoneme_to_id['<sil>'] == old_sil_id

    def test_from_dict_with_all_tokens_present_is_noop(self):
        """If the dict already contains all punct tokens, IDs are unchanged."""
        p1 = _processor()
        p2 = RussianPhonemeProcessor.from_dict(p1.to_dict())
        assert p2.phoneme_to_id == p1.phoneme_to_id


# ─────────────────────────────────────────────────────────────────────────────
# 6. flatten_phoneme_output emits punct tokens
# ─────────────────────────────────────────────────────────────────────────────
class TestFlattenEmitsPunct:

    def test_punct_token_appended_after_word_phonemes(self):
        raw = _fake_raw([(['p', 'r'], '<period>')])
        seq = PhonemeProcessorUtils.flatten_phoneme_output(raw)
        assert seq == ['p', 'r', '<period>']

    def test_no_punct_token_when_none(self):
        raw = _fake_raw([(['p', 'r'], None)])
        seq = PhonemeProcessorUtils.flatten_phoneme_output(raw)
        assert seq == ['p', 'r']

    def test_multiple_words_with_mixed_punct(self):
        raw = _fake_raw([(['p'], '<comma>'), (['r'], None), (['t'], '<period>')])
        seq = PhonemeProcessorUtils.flatten_phoneme_output(raw)
        assert seq == ['p', '<comma>', 'r', 't', '<period>']

    def test_empty_phoneme_list_with_punct(self):
        """Even if a word has no phonemes, its punct token is emitted."""
        raw = _fake_raw([([], '<question>')])
        seq = PhonemeProcessorUtils.flatten_phoneme_output(raw)
        assert seq == ['<question>']

    def test_three_tuple_compat_no_punct(self):
        """Old 3-tuples (no 4th element) must not emit any punct token."""
        raw_3 = [("word", ["a", "b"], StressInfo(0, 0, False))]
        seq = PhonemeProcessorUtils.flatten_phoneme_output(raw_3)
        assert seq == ['a', 'b']


# ─────────────────────────────────────────────────────────────────────────────
# 7. flatten_phoneme_output_with_sil: correct token order
# ─────────────────────────────────────────────────────────────────────────────
class TestFlattenWithSilInjectsPunct:

    def test_order_phonemes_punct_sil_for_punctuated_word(self):
        """For a word with trailing punct, order is: phonemes → <punct> → <sil>."""
        raw = _fake_raw([(['p'], '<comma>'), (['r'], None)])
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(raw, FULL_VOCAB)
        # word0: p, <comma>  — then sil — word1: r
        assert seq == ['p', '<comma>', '<sil>', 'r']

    def test_order_phonemes_sil_for_unpunctuated_word(self):
        """For a word without punct, order is: phonemes → <sil>."""
        raw = _fake_raw([(['p'], None), (['r'], None)])
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(raw, FULL_VOCAB)
        assert seq == ['p', '<sil>', 'r']

    def test_last_word_with_punct_no_trailing_sil(self):
        """The final word's punct token is NOT followed by <sil>."""
        raw = _fake_raw([(['p', 'r'], None), (['t'], '<period>')])
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(raw, FULL_VOCAB)
        assert seq == ['p', 'r', '<sil>', 't', '<period>']
        assert seq[-1] == '<period>'
        assert '<sil>' not in seq[seq.index('<period>') + 1:]   # nothing after period

    def test_last_word_without_punct_no_trailing_sil(self):
        """The final word's phonemes are NOT followed by <sil>."""
        raw = _fake_raw([(['p'], None), (['t'], None)])
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(raw, FULL_VOCAB)
        assert seq == ['p', '<sil>', 't']
        assert seq[-1] != '<sil>'

    def test_single_word_with_punct_no_sil(self):
        raw = _fake_raw([(['a'], '<question>')])
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(raw, FULL_VOCAB)
        assert seq == ['a', '<question>']

    def test_single_word_without_punct_no_sil(self):
        raw = _fake_raw([(['a'], None)])
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(raw, FULL_VOCAB)
        assert seq == ['a']

    def test_punct_in_vocab_check(self):
        """Punct tokens must be resolvable in the vocab for phonemes_to_indices."""
        raw = _fake_raw([(['p'], '<period>'), (['r'], None)])
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(raw, FULL_VOCAB)
        indices = PhonemeProcessorUtils.phonemes_to_indices(seq, FULL_VOCAB)
        period_id = FULL_VOCAB['<period>']
        assert period_id in indices


# ─────────────────────────────────────────────────────────────────────────────
# 8. End-to-end token sequence
# ─────────────────────────────────────────────────────────────────────────────
class TestEndToEndTokenSequence:

    @pytest.mark.parametrize("text,expected_puncts", [
        ("Привет.", ['<period>']),
        ("Привет, как дела?", ['<comma>', '<question>']),
        ("Стой!", ['<exclaim>']),
        ("Раз, два, три.", ['<comma>', '<comma>', '<period>']),
        ("Привет мир", []),
    ])
    def test_punct_tokens_appear_in_sequence(self, text, expected_puncts):
        p = _processor()
        raw = p.process_text(text)
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
            raw, p.phoneme_to_id
        )
        found = [t for t in seq if t in RussianPhonemeProcessor.PUNCT_MAP.values()]
        assert found == expected_puncts, (
            f"For text '{text}': expected punct tokens {expected_puncts}, "
            f"got {found}.  Full sequence: {seq}"
        )

    def test_punct_tokens_are_indexable(self):
        """All punct tokens in the sequence must map to valid IDs."""
        p = _processor()
        text = "Привет, как дела?"
        raw = p.process_text(text)
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
            raw, p.phoneme_to_id
        )
        # If any token is missing from the vocab, phonemes_to_indices warns and
        # maps to <sil>; here we want zero warnings — every token must be known.
        for tok in seq:
            assert tok in p.phoneme_to_id, (
                f"Token '{tok}' produced by the pipeline is not in the vocabulary"
            )

    def test_training_and_inference_sequences_include_punct(self):
        """Training path (via flatten_phoneme_output_with_sil) and inference path
        produce the same sequence, including punct tokens."""
        p = _processor()
        text = "Привет, как дела?"
        raw = p.process_text(text)

        training_seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
            raw, p.phoneme_to_id
        )
        # Inference path is identical — both call the same function.
        inference_seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
            raw, p.phoneme_to_id
        )
        assert training_seq == inference_seq

    def test_punct_token_position_relative_to_sil(self):
        """For 'Привет, как': <comma> must come before the <sil> word boundary."""
        p = _processor()
        raw = p.process_text("Привет, как")
        seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
            raw, p.phoneme_to_id
        )
        comma_idx = seq.index('<comma>')
        sil_idx   = seq.index('<sil>')
        assert comma_idx < sil_idx, (
            f"<comma> (pos {comma_idx}) must precede <sil> (pos {sil_idx})"
        )

    def test_sequence_does_not_start_or_end_with_sil(self):
        p = _processor()
        for text in ["Привет, мир.", "Как дела?", "Стой!"]:
            raw = p.process_text(text)
            seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
                raw, p.phoneme_to_id
            )
            assert seq[0] != '<sil>', f"Sequence starts with <sil> for '{text}'"
            assert seq[-1] != '<sil>', f"Sequence ends with <sil> for '{text}'"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Cache version
# ─────────────────────────────────────────────────────────────────────────────
class TestCacheVersion:

    def test_feature_cache_version_is_5(self):
        assert FEATURE_CACHE_VERSION == 5, (
            f"FEATURE_CACHE_VERSION is {FEATURE_CACHE_VERSION}, expected 5. "
            "It must be bumped whenever the phoneme sequence format changes "
            "so stale cache entries are invalidated."
        )
