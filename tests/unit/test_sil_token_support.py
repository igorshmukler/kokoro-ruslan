import pytest

from kokoro.data.audio_utils import PhonemeProcessorUtils


def test_flatten_with_sil_in_vocab_and_mapping():
    # Simulate processor output: two words, each with a phoneme list
    raw_output = [
        ("hello", ["h", "e"], None),
        ("world", ["w", "o"], None),
    ]

    phoneme_to_id = {"h": 1, "e": 2, "w": 3, "o": 4, "<sil>": 5}

    seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(raw_output, phoneme_to_id)
    # Expect sil injected between the two words
    assert seq == ["h", "e", "<sil>", "w", "o"]

    idx = PhonemeProcessorUtils.phonemes_to_indices(seq, phoneme_to_id)
    assert idx == [1, 2, 5, 3, 4]


def test_flatten_without_sil_in_vocab_falls_back():
    raw_output = [
        ("foo", ["f", "u"], None),
        ("bar", ["b", "a"], None),
    ]

    # No '<sil>' in vocab; include an <unk> mapping to check fallback mapping
    phoneme_to_id = {"f": 10, "u": 11, "b": 12, "a": 13, "<unk>": 0}

    seq = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(raw_output, phoneme_to_id)
    # Should not inject <sil> when it's not present in the vocab
    assert seq == ["f", "u", "b", "a"]

    idx = PhonemeProcessorUtils.phonemes_to_indices(seq, phoneme_to_id)
    assert idx == [10, 11, 12, 13]
