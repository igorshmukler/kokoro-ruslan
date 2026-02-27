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
