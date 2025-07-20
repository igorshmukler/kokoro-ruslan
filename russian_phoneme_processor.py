#!/usr/bin/env python3
"""
Russian Phoneme Processor for Kokoro Language Model
Standalone module for Russian grapheme-to-phoneme conversion
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RussianPhonemeProcessor:
    """
    Russian phoneme processor without espeak dependency
    Uses rule-based grapheme-to-phoneme conversion for Russian
    """

    def __init__(self):
        # Russian phoneme mapping (simplified)
        self.vowels = {
            'а': 'a', 'о': 'o', 'у': 'u', 'ы': 'i', 'э': 'e',
            'я': 'ja', 'ё': 'jo', 'ю': 'ju', 'и': 'i', 'е': 'je'
        }

        self.consonants = {
            'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ж': 'zh',
            'з': 'z', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
            'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'ф': 'f',
            'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch'
        }

        self.special_chars = {
            'ь': '', 'ъ': '', ' ': ' ', '.': '.', ',': ',',
            '!': '!', '?': '?', '-': '-'
        }

        # Combined phoneme vocabulary
        self.phonemes = list(self.vowels.values()) + list(self.consonants.values()) + \
                       list(self.special_chars.values())
        self.phonemes = list(set(self.phonemes))  # Remove duplicates

        # Create phoneme to index mapping
        self.phoneme_to_idx = {p: i for i, p in enumerate(self.phonemes)}
        self.idx_to_phoneme = {i: p for i, p in enumerate(self.phonemes)}

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert Russian text to phonemes"""
        text = text.lower().strip()
        phonemes = []

        for char in text:
            if char in self.vowels:
                phonemes.append(self.vowels[char])
            elif char in self.consonants:
                phonemes.append(self.consonants[char])
            elif char in self.special_chars:
                if self.special_chars[char]:  # Skip empty mappings
                    phonemes.append(self.special_chars[char])
            else:
                # Unknown character, skip or replace with space
                phonemes.append(' ')

        return phonemes

    def phonemes_to_indices(self, phonemes: List[str]) -> List[int]:
        """Convert phonemes to indices"""
        return [self.phoneme_to_idx.get(p, 0) for p in phonemes]

    def indices_to_phonemes(self, indices: List[int]) -> List[str]:
        """Convert indices back to phonemes"""
        return [self.idx_to_phoneme.get(idx, ' ') for idx in indices]

    def text_to_indices(self, text: str) -> List[int]:
        """Convert text directly to indices (convenience method)"""
        phonemes = self.text_to_phonemes(text)
        return self.phonemes_to_indices(phonemes)

    def get_vocab_size(self) -> int:
        """Get the size of the phoneme vocabulary"""
        return len(self.phonemes)

    def to_dict(self) -> Dict:
        """Convert processor to dictionary for safe serialization"""
        return {
            'vowels': self.vowels,
            'consonants': self.consonants,
            'special_chars': self.special_chars,
            'phonemes': self.phonemes,
            'phoneme_to_idx': self.phoneme_to_idx,
            'idx_to_phoneme': self.idx_to_phoneme
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RussianPhonemeProcessor':
        """Create processor from dictionary"""
        processor = cls.__new__(cls)  # Create instance without calling __init__
        processor.vowels = data['vowels']
        processor.consonants = data['consonants']
        processor.special_chars = data['special_chars']
        processor.phonemes = data['phonemes']
        processor.phoneme_to_idx = data['phoneme_to_idx']
        processor.idx_to_phoneme = data['idx_to_phoneme']
        return processor

    def __repr__(self) -> str:
        return f"RussianPhonemeProcessor(vocab_size={len(self.phonemes)})"

    def __str__(self) -> str:
        return f"Russian Phoneme Processor with {len(self.phonemes)} phonemes"


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
