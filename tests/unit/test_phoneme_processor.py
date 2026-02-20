import pytest

from kokoro.data.russian_phoneme_processor import RussianPhonemeProcessor


def test_tokenizer_multichar_and_single_chars():
    p = RussianPhonemeProcessor()
    ipa = 'bʲtʃstf'
    tokens = p._tokenize_ipa_string(ipa)
    # Expect multi-char tokens like 'bʲ', 'tʃ', 'stf' to appear
    assert 'bʲ' in tokens
    assert 'tʃ' in tokens
    # 'stf' may be tokenized as a single multi-char or as separate 's','t','f'
    joined = ''.join(tokens)
    assert 'stf' in joined
    # Combined tokens should recompose to original when joined (excluding removed marks)
    assert joined.startswith('bʲtʃ')


def test_apply_vowel_reduction_index_safety():
    p = RussianPhonemeProcessor()
    # Case 1: no vowels -> should return input unchanged
    phonemes = ['t', 's', 'k']
    out = p.apply_vowel_reduction(phonemes.copy(), stress_syllable_idx=0)
    assert out == phonemes

    # Case 2: mismatched internal bookkeeping shouldn't raise
    phonemes = ['b', 'ja', 't', 'o', 's']  # contains iotated and simple vowels
    # Call with a stress index larger than vowel count to exercise fallback
    out2 = p.apply_vowel_reduction(phonemes.copy(), stress_syllable_idx=10)
    assert isinstance(out2, list)
    assert len(out2) == len(phonemes)


def test_normalize_text_handles_yo_and_stress_marks():
    p = RussianPhonemeProcessor()
    # 'ё' should be converted to 'е' with stress mark
    normalized = p.normalize_text('ё')
    assert 'е' in normalized
    # Ensure a combining acute exists in the normalized output
    assert any(ch in normalized for ch in p.STRESS_MARKS)


@pytest.mark.parametrize("word,expected_ipa", [
    ("привет", "prʲɪvʲet"),
    ("как", "kak"),
    ("дела", "dʲɪla"),
    ("молоко", "məlɐko"),
    ("хорошо", "xərɐʃo"),
    ("сегодня", "sʲɪvodʲnʲə"),
    ("здравствуйте", "zdrəstvuitʲə"),
])
def test_process_word_expected_ipa(word, expected_ipa):
    p = RussianPhonemeProcessor()
    phonemes, stress_info = p.process_word(word)
    ipa = p.to_ipa(phonemes)
    # Compare normalized strings (no slashes)
    assert isinstance(ipa, str)
    assert ipa == expected_ipa


def test_process_text_end_to_end():
    p = RussianPhonemeProcessor()
    text = "Привет, как дела?"
    results = p.process_text(text)
    assert len(results) == 3
    for word, phonemes, stress in results:
        assert isinstance(word, str)
        assert isinstance(phonemes, list)
        assert isinstance(stress.position, int)
