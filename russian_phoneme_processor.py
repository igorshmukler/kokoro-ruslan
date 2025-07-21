#!/usr/bin/env python3
"""
Russian Phoneme Processor with context-based G2P disambiguation.
Supports soft/hard consonant rules and stress mark handling.
"""

from typing import Dict, List
import unicodedata
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RussianPhonemeProcessor:
    """
    Rule-based Russian grapheme-to-phoneme converter.
    Handles palatalization (soft consonants) and explicit stress marks.
    """

    def __init__(self) -> None:
        # Basic grapheme-to-phoneme mappings
        self.vowels: Dict[str, str] = {
            'а': 'a', 'о': 'o', 'у': 'u', 'ы': 'i', 'э': 'e',
            'я': 'ja', 'ё': 'jo', 'ю': 'ju', 'и': 'i', 'е': 'je'
        }

        self.consonants: Dict[str, str] = {
            'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ж': 'zh',
            'з': 'z', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
            'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'ф': 'f',
            'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch'
        }

        self.special_chars: Dict[str, str] = {
            'ь': '', 'ъ': '', ' ': ' ', '.': '.', ',': ',',
            '!': '!', '?': '?', '-': '-'
        }

        # G2P enhancements
        self.soft_vowel_triggers = {'е', 'ё', 'и', 'ю', 'я'}
        self.softenable_consonants = {
            'б', 'в', 'г', 'д', 'ж', 'з',
            'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т', 'ф', 'х'
        }

        self.soft_marker = 'ʲ'  # palatalization (soft consonant)
        self.stress_marker = 'ˈ'  # primary stress

        self.palatalized_suffix = lambda p: f"{p}{self.soft_marker}"

        # Build full phoneme vocabulary
        palatalized = [
            self.palatalized_suffix(self.consonants[c])
            for c in self.softenable_consonants
        ]
        stressed_vowels = [f"{self.stress_marker}{v}" for v in self.vowels.values()]

        self.phonemes: List[str] = sorted(set(
            list(self.vowels.values()) +
            list(self.consonants.values()) +
            palatalized +
            stressed_vowels +
            [p for p in self.special_chars.values() if p]
        ))

        self.phoneme_to_idx: Dict[str, int] = {p: i for i, p in enumerate(self.phonemes)}
        self.idx_to_phoneme: Dict[int, str] = {i: p for p, i in self.phoneme_to_idx.items()}

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert Russian text to a list of phonemes.
        Handles:
        - Palatalization (soft consonants before certain vowels or 'ь')
        - Stress marks (acute accent \u0301)
        """
        text = unicodedata.normalize('NFD', text.lower().strip())
        phonemes = []
        i = 0

        while i < len(text):
            char = text[i]
            next_char = text[i + 1] if i + 1 < len(text) else ''

            is_stressed = next_char == '\u0301'
            skip_next = False

            if char in self.vowels:
                phoneme = self.vowels[char]
                if is_stressed:
                    phoneme = self.stress_marker + phoneme
                    skip_next = True
                phonemes.append(phoneme)

            elif char in self.consonants:
                base_phoneme = self.consonants[char]
                following_char = text[i + 1] if i + 1 < len(text) else ''
                if char in self.softenable_consonants and (following_char in self.soft_vowel_triggers or following_char == 'ь'):
                    phonemes.append(self.palatalized_suffix(base_phoneme))
                else:
                    phonemes.append(base_phoneme)

            elif char in self.special_chars:
                val = self.special_chars[char]
                if val:
                    phonemes.append(val)

            elif unicodedata.combining(char):
                pass  # skip isolated combining marks

            else:
                logger.debug(f"Unknown character skipped: '{char}'")
                phonemes.append(' ')

            i += 2 if skip_next else 1

        return phonemes

    def phonemes_to_indices(self, phonemes: List[str]) -> List[int]:
        """Convert list of phonemes to list of indices"""
        return [self.phoneme_to_idx.get(p, 0) for p in phonemes]

    def indices_to_phonemes(self, indices: List[int]) -> List[str]:
        """Convert list of indices back to phoneme list"""
        return [self.idx_to_phoneme.get(idx, ' ') for idx in indices]

    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to phoneme indices"""
        return self.phonemes_to_indices(self.text_to_phonemes(text))

    def get_vocab_size(self) -> int:
        """Return phoneme vocabulary size"""
        return len(self.phonemes)

    def to_dict(self) -> Dict:
        """Serialize processor state to dictionary"""
        return {
            'vowels': self.vowels,
            'consonants': self.consonants,
            'special_chars': self.special_chars,
            'phonemes': self.phonemes,
            'phoneme_to_idx': self.phoneme_to_idx,
            'idx_to_phoneme': self.idx_to_phoneme,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RussianPhonemeProcessor':
        """Reconstruct processor from saved dictionary"""
        processor = cls.__new__(cls)
        processor.vowels = data['vowels']
        processor.consonants = data['consonants']
        processor.special_chars = data['special_chars']
        processor.phonemes = data['phonemes']
        processor.phoneme_to_idx = data['phoneme_to_idx']
        processor.idx_to_phoneme = data['idx_to_phoneme']

        # Reinitialize derived properties
        processor.soft_vowel_triggers = {'е', 'ё', 'и', 'ю', 'я'}
        processor.softenable_consonants = {
            'б', 'в', 'г', 'д', 'ж', 'з',
            'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т', 'ф', 'х'
        }
        processor.soft_marker = 'ʲ'
        processor.stress_marker = 'ˈ'
        processor.palatalized_suffix = lambda p: f"{p}{processor.soft_marker}"

        return processor

    def __repr__(self) -> str:
        return f"<RussianPhonemeProcessor vocab_size={self.get_vocab_size()}>"

    def __str__(self) -> str:
        return f"Russian Phoneme Processor with {self.get_vocab_size()} phonemes"

def test_processor():
    """Test function for the phoneme processor"""
    processor = RussianPhonemeProcessor()

    test_texts = [
        "привет мир",
        "как дела?",
        "спасибо большое!",
        "до свидания"
    ]

    print(f"Processor: {processor}")
    print(f"Vocabulary size: {processor.get_vocab_size()}")
    print(f"Phonemes: {processor.phonemes}")
    print()

    for text in test_texts:
        phonemes = processor.text_to_phonemes(text)
        indices = processor.phonemes_to_indices(phonemes)
        reconstructed = processor.indices_to_phonemes(indices)

        print(f"Text: '{text}'")
        print(f"Phonemes: {phonemes}")
        print(f"Indices: {indices}")
        print(f"Reconstructed: {reconstructed}")
        print("-" * 50)


if __name__ == "__main__":
    test_processor()
