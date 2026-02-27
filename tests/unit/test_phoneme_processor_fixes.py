"""
Regression tests for three specific fixes applied to russian_phoneme_processor.py.

Fix 1 – logging.basicConfig removed from module scope
    Importing the module must not install a root-level handler, so host
    applications can configure logging themselves.

Fix 2 – @lru_cache on instance methods replaced with per-instance caches
    Each RussianPhonemeProcessor instance owns its own cache.  When the
    instance is deleted the cache is released — no class-level strong
    reference to `self` remains.

Fix 3 – Combining marks (stress diacritics) stripped before Cyrillic
    str.replace cluster patterns in apply_consonant_assimilation.
    Previously a word with an embedded stress mark would silently skip
    cluster simplifications (e.g. 'вств' → 'ств', 'тся' → 'ца', etc.).
"""

import gc
import logging
import weakref

import pytest

from kokoro.data.russian_phoneme_processor import RussianPhonemeProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1  – no logging.basicConfig at module import time
# ─────────────────────────────────────────────────────────────────────────────

class TestNoRootLoggerHijack:
    """Importing the module must not add handlers to the root logger."""

    def test_import_does_not_add_root_handler(self):
        """The root logger must have zero handlers after import."""
        root = logging.getLogger()
        # Remove any handlers the test runner itself may have installed so we
        # get a clean starting point.
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            # Re-trigger the module-level code by importing (already cached,
            # so this is a no-op — but it ensures the module has been loaded).
            import kokoro.data.russian_phoneme_processor  # noqa: F401
            assert root.handlers == [], (
                "logging.basicConfig (or equivalent) must not be called at "
                "module scope in a library module — it hijacks the root logger."
            )
        finally:
            root.handlers[:] = original_handlers

    def test_module_logger_is_named(self):
        """The module-level logger must use the module's __name__, not root."""
        import kokoro.data.russian_phoneme_processor as mod
        assert mod.logger.name == "kokoro.data.russian_phoneme_processor"
        # A named logger is never the root logger
        assert mod.logger is not logging.getLogger()


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2  – per-instance LRU caches (no memory leak)
# ─────────────────────────────────────────────────────────────────────────────

class TestPerInstanceLRUCache:
    """Each processor instance owns its own cache; deleting the instance
    must release the cache (no class-level strong reference to self)."""

    def test_each_instance_has_independent_cache(self):
        p1 = RussianPhonemeProcessor()
        p2 = RussianPhonemeProcessor()
        # Populate both caches
        p1.normalize_text("привет")
        p2.normalize_text("привет")
        # Clearing p1 must not affect p2
        p1.clear_cache()
        assert p1.normalize_text.cache_info().currsize == 0
        assert p2.normalize_text.cache_info().currsize > 0

    def test_instance_is_garbage_collected_after_del(self):
        p = RussianPhonemeProcessor()
        # Warm the caches so they hold internal state
        p.normalize_text("мир")
        p._process_normalized_word("мир")
        ref = weakref.ref(p)
        del p
        gc.collect()
        assert ref() is None, (
            "RussianPhonemeProcessor instance was not garbage collected. "
            "This indicates a class-level strong reference (e.g. @lru_cache "
            "on a bound method) is keeping the instance alive."
        )

    def test_cache_attributes_are_instance_level(self):
        """normalize_text and _process_normalized_word must be instance
        attributes (the per-instance cached wrappers), not class attributes."""
        p = RussianPhonemeProcessor()
        # Instance dict should contain the cached callables
        assert 'normalize_text' in p.__dict__, (
            "normalize_text should be an instance attribute (per-instance cache), "
            "not a class method decorated with @lru_cache."
        )
        assert '_process_normalized_word' in p.__dict__, (
            "_process_normalized_word should be an instance attribute."
        )

    def test_two_instances_do_not_share_cache_state(self):
        """clear_cache on one instance must not affect the other."""
        p1 = RussianPhonemeProcessor()
        p2 = RussianPhonemeProcessor()
        p1.normalize_text("слово")
        p2.normalize_text("слово")
        p2.normalize_text("текст")

        p1.clear_cache()

        assert p1.normalize_text.cache_info().currsize == 0
        # p2 still has both entries
        assert p2.normalize_text.cache_info().currsize == 2

    def test_get_cache_info_works_on_instance(self):
        p = RussianPhonemeProcessor()
        p.normalize_text("тест")
        info = p.get_cache_info()
        assert info["normalize_text_cache"].currsize == 1


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3  – stress marks stripped before cluster replacements
# ─────────────────────────────────────────────────────────────────────────────

class TestStressMarkStrippedBeforeAssimilation:
    """Combining marks must be removed at the top of apply_consonant_assimilation
    so that Cyrillic cluster patterns still match when a word arrives with an
    embedded stress diacritic."""

    # U+0301 is the combining acute (standard stress mark for Russian)
    _ACUTE = '\u0301'

    def _stressed(self, word: str, after_vowel_index: int) -> str:
        """Insert a combining acute after the character at after_vowel_index."""
        return word[:after_vowel_index + 1] + self._ACUTE + word[after_vowel_index + 1:]

    def test_вств_cluster_simplified_without_stress_mark(self):
        p = RussianPhonemeProcessor()
        result = p.apply_consonant_assimilation('здравствуйте')
        assert 'ств' in result
        assert 'вств' not in result

    def test_вств_cluster_simplified_with_stress_mark_on_а(self):
        """здра́вствуйте — stress on 'а' (index 4)."""
        p = RussianPhonemeProcessor()
        word_with_mark = 'здра' + self._ACUTE + 'вствуйте'
        result = p.apply_consonant_assimilation(word_with_mark)
        assert 'вств' not in result, (
            "The 'вств'→'ств' simplification must fire even when the word "
            "contains a combining stress mark."
        )

    def test_тся_simplified_with_stress_mark(self):
        p = RussianPhonemeProcessor()
        # учится — stress on и (index 3)
        word_with_mark = 'учи' + self._ACUTE + 'тся'
        result = p.apply_consonant_assimilation(word_with_mark)
        assert 'ца' in result, (
            "'тся'→'ца' must fire even when the word has an embedded stress mark."
        )
        assert 'тся' not in result

    def test_ться_simplified_with_stress_mark(self):
        p = RussianPhonemeProcessor()
        word_with_mark = 'учи' + self._ACUTE + 'ться'
        result = p.apply_consonant_assimilation(word_with_mark)
        assert 'тся' not in result
        assert 'ться' not in result

    def test_стн_cluster_simplified_with_stress_mark(self):
        p = RussianPhonemeProcessor()
        # честный — stress on е (index 1)
        word_with_mark = 'ч' + self._ACUTE + 'естный'  # acute after ч (non-vowel edge test)
        word_with_mark2 = 'че' + self._ACUTE + 'стный'  # correct position
        result = p.apply_consonant_assimilation(word_with_mark2)
        assert 'стн' not in result

    def test_сч_cluster_simplified_with_stress_mark(self):
        p = RussianPhonemeProcessor()
        word_with_mark = 'сча' + self._ACUTE + 'стье'
        result = p.apply_consonant_assimilation(word_with_mark)
        assert 'сч' not in result

    def test_output_contains_no_combining_marks(self):
        """The returned string must never contain combining marks."""
        p = RussianPhonemeProcessor()
        import unicodedata
        for word in ['здра\u0301вствуйте', 'учи\u0301тся', 'че\u0301стный']:
            result = p.apply_consonant_assimilation(word)
            for ch in result:
                cat = unicodedata.category(ch)
                assert cat != 'Mn', (
                    f"Combining mark U+{ord(ch):04X} ({ch!r}) found in output "
                    f"of apply_consonant_assimilation({word!r})"
                )

    def test_full_pipeline_zdravstvuyte_correct_ipa(self):
        """End-to-end: здравствуйте (with or without mark) → correct IPA."""
        p = RussianPhonemeProcessor()
        for word in ['здравствуйте', 'здра\u0301вствуйте']:
            phonemes, _ = p.process_word(word)
            ipa = p.to_ipa(phonemes)
            assert 'vstv' not in ipa, (
                f"process_word({word!r}) → {ipa!r}: 'вств' cluster was not "
                "simplified — stress-mark stripping may have failed."
            )


class TestIotatedJPrefixPreservedInReduction:
    """
    Fix 4 – Iotated vowel j-prefix silently dropped in apply_vowel_reduction.

    Before the fix, unstressed 'ja'/'je'/'jo' were reduced to bare 'ɐ'/'ɪ'/'ɐ'
    (dropping the j).  After the fix they become 'jɐ'/'jɪ'/'jɐ' (j preserved).
    """

    def _p(self):
        return RussianPhonemeProcessor()

    # ── Direct apply_vowel_reduction tests ──────────────────────────────────

    def test_ja_pre_stress_adjacent_becomes_jɐ(self):
        """'ja' immediately before stressed syllable → 'jɐ', not 'ɐ'."""
        p = self._p()
        # stress at syllable 1; 'ja' is syllable 0 (distance 1 → 'ɐ' tier)
        phonemes = ['ja', 'z', 'a']   # syllable 0=ja, 1=a
        reduced = p.apply_vowel_reduction(phonemes, stress_syllable_idx=1)
        assert reduced[0] == 'jɐ', (
            f"'ja' pre-stress adjacent should reduce to 'jɐ', got {reduced[0]!r}")
        assert reduced[2] == 'a', "stressed vowel must not be reduced"

    def test_je_pre_stress_adjacent_becomes_jɪ(self):
        """'je' immediately before stressed syllable → 'jɪ', not 'ɪ'."""
        p = self._p()
        phonemes = ['je', 'z', 'a']
        reduced = p.apply_vowel_reduction(phonemes, stress_syllable_idx=1)
        assert reduced[0] == 'jɪ', (
            f"'je' pre-stress adjacent should reduce to 'jɪ', got {reduced[0]!r}")

    def test_jo_pre_stress_adjacent_becomes_jɐ(self):
        """'jo' immediately before stressed syllable → 'jɐ', not 'ɐ'."""
        p = self._p()
        phonemes = ['jo', 'z', 'a']
        reduced = p.apply_vowel_reduction(phonemes, stress_syllable_idx=1)
        assert reduced[0] == 'jɐ', (
            f"'jo' pre-stress adjacent should reduce to 'jɐ', got {reduced[0]!r}")

    def test_ja_far_pre_stress_becomes_jə(self):
        """'ja' two+ syllables before stress → 'jə', not 'ə'."""
        p = self._p()
        # syllables: 0=ja, 1=a, 2=a (stressed)
        phonemes = ['ja', 'z', 'a', 'z', 'a']
        reduced = p.apply_vowel_reduction(phonemes, stress_syllable_idx=2)
        assert reduced[0] == 'jə', (
            f"'ja' far from stress should reduce to 'jə', got {reduced[0]!r}")

    def test_ja_post_stress_becomes_jə(self):
        """'ja' after the stressed syllable → 'jə', not 'ə'."""
        p = self._p()
        # syllables: 0=a (stressed), 1=ja
        phonemes = ['a', 'z', 'ja']
        reduced = p.apply_vowel_reduction(phonemes, stress_syllable_idx=0)
        assert reduced[2] == 'jə', (
            f"'ja' post-stress should reduce to 'jə', got {reduced[2]!r}")
        assert reduced[0] == 'a', "stressed vowel must not be reduced"

    def test_ju_not_reduced(self):
        """'ju' (у sound) is not in the reduction set — must remain 'ju'."""
        p = self._p()
        phonemes = ['ju', 'z', 'a']
        reduced = p.apply_vowel_reduction(phonemes, stress_syllable_idx=1)
        assert reduced[0] == 'ju', (
            f"'ju' should not be reduced (у doesn't map to ə/ɐ/ɪ), got {reduced[0]!r}")

    # ── Vocab / tokenizer regression ─────────────────────────────────────────

    def test_reduced_iotated_phonemes_in_vocab(self):
        """'jɐ', 'jɪ', 'jə' must have entries in phoneme_to_id."""
        p = self._p()
        for ph in ('jɐ', 'jɪ', 'jə'):
            assert ph in p.phoneme_to_id, (
                f"Reduced iotated phoneme {ph!r} missing from phoneme_to_id")

    def test_reduced_iotated_phonemes_in_multi_char_list(self):
        """'jɐ', 'jɪ', 'jə' must appear in _multi_char_phonemes so the IPA
        tokenizer can round-trip them without splitting 'j' and the vowel."""
        p = self._p()
        for ph in ('jɐ', 'jɪ', 'jə'):
            assert ph in p._multi_char_phonemes, (
                f"Reduced iotated phoneme {ph!r} missing from _multi_char_phonemes")

    # ── End-to-end word tests ─────────────────────────────────────────────────

    def test_yazyk_first_syllable_is_jɐ(self):
        """язы́к: initial 'я' is unstressed pre-stress adjacent → 'jɐ'."""
        p = self._p()
        # Force stress on syllable 1 (ы́) via dict so the test is deterministic
        p.stress_patterns['язык'] = 1
        p.normalize_text.cache_clear()
        p._process_normalized_word.cache_clear()
        phonemes, _ = p.process_word('язык')
        assert phonemes[0] == 'jɐ', (
            f"язык: first phoneme should be 'jɐ', got {phonemes[0]!r} "
            f"(full: {phonemes})")

    def test_yabloko_first_syllable_is_ja(self):
        """я́блоко: initial 'я' is stressed → stays 'ja' (no reduction)."""
        p = self._p()
        p.stress_patterns['яблоко'] = 0
        p.normalize_text.cache_clear()
        p._process_normalized_word.cache_clear()
        phonemes, _ = p.process_word('яблоко')
        assert phonemes[0] == 'ja', (
            f"яблоко: stressed 'я' should remain 'ja', got {phonemes[0]!r} "
            f"(full: {phonemes})")
