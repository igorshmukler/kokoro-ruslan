import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StressInfo:
    """Information about stress in a word"""
    position: int  # Position of stressed syllable (0-based)
    vowel_index: int  # Index of stressed vowel in the word
    is_marked: bool  # Whether stress was explicitly marked

    def __post_init__(self):
        """Validate stress info after initialization"""
        if self.position < 0:
            raise ValueError("Stress position cannot be negative")
        if self.vowel_index < 0:
            raise ValueError("Vowel index cannot be negative")

class RussianPhonemeProcessor:
    """
    Enhanced Russian phoneme processor with comprehensive stress detection
    and pronunciation rules for TTS systems.
    """

    # Class constants to avoid repeated dict creation
    STRESS_MARKS = ['\u0301', '\u0300', '\u0341']  # Acute, grave, combining acute
    VOWEL_LETTERS = {'а', 'о', 'у', 'ы', 'э', 'я', 'ё', 'ю', 'и', 'е'}

    def __init__(self, stress_dict_path: Optional[str] = None):
        """
        Initialize the processor.

        Args:
            stress_dict_path: Optional path to external stress dictionary
        """
        # Vowel mappings with stress-dependent pronunciation
        self.vowels = {
            'а': 'a', 'о': 'o', 'у': 'u', 'ы': 'ɨ', 'э': 'e',
            'я': 'ja', 'ё': 'jo', 'ю': 'ju', 'и': 'i', 'е': 'je'
        }

        # Consonant mappings
        self.consonants = {
            'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ж': 'ʐ', 'з': 'z',
            'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'п': 'p', 'р': 'r',
            'с': 's', 'т': 't', 'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'tʃ',
            'ш': 'ʃ', 'щ': 'ʃtʃ'
        }

        # Palatalized consonants
        self.palatalized = {
            'б': 'bʲ', 'в': 'vʲ', 'г': 'gʲ', 'д': 'dʲ', 'з': 'zʲ',
            'к': 'kʲ', 'л': 'lʲ', 'м': 'mʲ', 'н': 'nʲ', 'п': 'pʲ',
            'р': 'rʲ', 'с': 'sʲ', 'т': 'tʲ', 'ф': 'fʲ', 'х': 'xʲ'
        }

        # Vowel reduction patterns (unstressed vowels)
        self.vowel_reduction = {
            # First pretonic and post-tonic positions
            'о': 'ə',  # o -> schwa
            'а': 'ə',  # a -> schwa (in some positions)
            'е': 'ɪ',  # e -> reduced i
            'я': 'ɪ',  # ya -> reduced i
            # Second pretonic and further positions
            'о_weak': 'ə',
            'а_weak': 'ə',
            'е_weak': 'ə',
            'я_weak': 'ə'
        }

        # Hard consonants (never palatalized)
        self.hard_consonants = {'ж', 'ш', 'ц'}

        # Soft consonants (always palatalized)
        self.soft_consonants = {'ч', 'щ', 'й'}

        # Voicing assimilation rules
        self.voiced_consonants = {'б', 'в', 'г', 'д', 'ж', 'з'}
        self.voiceless_consonants = {'п', 'ф', 'к', 'т', 'ш', 'с'}

        self.voicing_map = {
            'б': 'п', 'в': 'ф', 'г': 'к', 'д': 'т', 'ж': 'ш', 'з': 'с',
            'п': 'б', 'ф': 'в', 'к': 'г', 'т': 'd', 'ш': 'ж', 'с': 'з'
        }

        # Load stress patterns
        self.stress_patterns = self._load_stress_patterns(stress_dict_path)

        # Pronunciation exceptions
        self.exceptions = {
            'что': 'ʃto',
            'чтобы': 'ʃtobi',
            'конечно': 'kʌnʲeʃnə',
            'скучно': 'skutʃnə',
            'его': 'jɪvo',
            'сегодня': 'sʲɪvodʲnʲə'
        }

        # Build vocabulary after all mappings are set
        self.phoneme_to_id = self._build_vocab()

        # Cache for processed words to improve performance
        self._word_cache: Dict[str, Tuple[List[str], StressInfo]] = {}

    def _load_stress_patterns(self, dict_path: Optional[str] = None) -> Dict[str, int]:
        """
        Load stress patterns from file or use built-in patterns.

        Args:
            dict_path: Path to external stress dictionary file

        Returns:
            Dictionary mapping words to stress positions
        """
        patterns = {
            # Common monosyllabic words (always stressed)
            'дом': 0, 'кот': 0, 'мир': 0, 'лес': 0,
            # Common patterns for frequent words
            'говорить': 2, 'работать': 1, 'человек': 2,
            'хорошо': 2, 'плохо': 1, 'быстро': 1,
            'медленно': 1, 'красиво': 2, 'интересно': 2,
            # Verb endings patterns
            'делает': 1, 'говорит': 2, 'работает': 1,
            'понимает': 2, 'знает': 1, 'играет': 1
        }

        if dict_path:
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                word, stress_pos = parts[0], parts[1]
                                try:
                                    patterns[word] = int(stress_pos)
                                except ValueError:
                                    logger.warning(f"Invalid stress position for word {word}: {stress_pos}")
            except FileNotFoundError:
                logger.warning(f"Stress dictionary file not found: {dict_path}")
            except Exception as e:
                logger.error(f"Error loading stress dictionary: {e}")

        return patterns

    @lru_cache(maxsize=1000)
    def normalize_text(self, text: str) -> str:
        """
        Normalize Russian text for phoneme processing.
        Cached for performance on repeated texts.
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove combining marks and normalize Unicode
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')

        # Handle yo (ё) - always stressed
        text = text.replace('ё', 'о́')  # Convert to stressed o

        # Remove punctuation except stress marks
        text = re.sub(r'[^\w\s\u0301\u0300\u0341]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        return text

    def detect_stress(self, word: str) -> StressInfo:
        """
        Detect stress position in a Russian word.
        Priority: 1) Explicit marks, 2) Dictionary, 3) Heuristics
        """
        if not word:
            return StressInfo(0, 0, False)

        # Check for explicit stress marks
        for mark in self.STRESS_MARKS:
            if mark in word:
                clean_word = word.replace(mark, '')
                stress_pos = self._find_stress_position(word, mark)
                syllable_pos = self._syllable_of_vowel(clean_word, stress_pos)
                return StressInfo(
                    position=syllable_pos,
                    vowel_index=stress_pos,
                    is_marked=True
                )

        # Remove any remaining diacritics for dictionary lookup
        clean_word = re.sub(r'[\u0300-\u036f]', '', word)

        # Check dictionary
        if clean_word in self.stress_patterns:
            syllable_pos = self.stress_patterns[clean_word]
            vowel_index = self._vowel_index_from_syllable(clean_word, syllable_pos)
            return StressInfo(
                position=syllable_pos,
                vowel_index=vowel_index,
                is_marked=False
            )

        # Apply heuristic rules
        return self._apply_stress_heuristics(clean_word)

    def _find_stress_position(self, word: str, stress_mark: str) -> int:
        """Find the position of the vowel with stress mark"""
        for i, char in enumerate(word):
            if i + 1 < len(word) and word[i + 1] == stress_mark:
                return i
        return 0

    def _syllable_of_vowel(self, word: str, vowel_index: int) -> int:
        """Find which syllable a vowel belongs to"""
        if vowel_index >= len(word):
            return 0

        vowel_count = 0
        for i, char in enumerate(word[:vowel_index + 1]):
            if char.lower() in self.VOWEL_LETTERS:
                if i == vowel_index:
                    return vowel_count
                vowel_count += 1
        return 0

    def _vowel_index_from_syllable(self, word: str, syllable_pos: int) -> int:
        """Find the index of vowel in specified syllable"""
        vowel_count = 0
        for i, char in enumerate(word):
            if char.lower() in self.VOWEL_LETTERS:
                if vowel_count == syllable_pos:
                    return i
                vowel_count += 1
        return 0

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        return sum(1 for char in word if char.lower() in self.VOWEL_LETTERS)

    def _apply_stress_heuristics(self, word: str) -> StressInfo:
        """Apply heuristic rules for stress placement"""
        syllable_count = self._count_syllables(word)

        if syllable_count <= 1:
            vowel_index = self._vowel_index_from_syllable(word, 0)
            return StressInfo(position=0, vowel_index=vowel_index, is_marked=False)

        # Common heuristics for Russian stress
        if word.endswith(('ать', 'еть', 'ить', 'ыть', 'уть')):
            # Infinitive verbs - often stress the ending
            stress_pos = syllable_count - 1
        elif word.endswith(('ный', 'ная', 'ное', 'ные')):
            # Adjectives - often stress the root
            stress_pos = max(0, syllable_count - 2)
        elif word.endswith(('ость', 'есть')):
            # Abstract nouns - often stress the root
            stress_pos = max(0, syllable_count - 2)
        else:
            # Default: stress the penultimate syllable
            stress_pos = max(0, syllable_count - 2)

        vowel_index = self._vowel_index_from_syllable(word, stress_pos)
        return StressInfo(position=stress_pos, vowel_index=vowel_index, is_marked=False)

    def apply_vowel_reduction(self, word: str, stress_info: StressInfo) -> str:
        """Apply vowel reduction based on stress position"""
        if not word:
            return word

        result = list(word)
        vowel_positions = []

        # Find all vowel positions
        for i, char in enumerate(word):
            if char.lower() in self.vowels:
                vowel_positions.append(i)

        # Apply reduction to unstressed vowels
        for i, pos in enumerate(vowel_positions):
            if i != stress_info.position:  # Not the stressed vowel
                char = word[pos].lower()
                distance = abs(i - stress_info.position)

                # Apply different reduction based on distance from stress
                if distance == 1:  # First pretonic/post-tonic
                    if char in ['о', 'а']:
                        result[pos] = 'ə'
                    elif char in ['е', 'я']:
                        result[pos] = 'ɪ'
                else:  # Further positions - stronger reduction
                    if char in ['о', 'а', 'е', 'я']:
                        result[pos] = 'ə'

        return ''.join(result)

    def apply_consonant_assimilation(self, word: str) -> str:
        """Apply voicing assimilation and other consonant changes"""
        if len(word) < 2:
            return word

        result = list(word.lower())

        # Voicing assimilation
        for i in range(len(result) - 1):
            current = result[i]
            next_char = result[i + 1]

            if current in self.consonants and next_char in self.consonants:
                # Assimilate voicing
                if (current in self.voiced_consonants and
                    next_char in self.voiceless_consonants and
                    current in self.voicing_map):
                    result[i] = self.voicing_map[current]
                elif (current in self.voiceless_consonants and
                      next_char in self.voiced_consonants and
                      current in self.voicing_map):
                    result[i] = self.voicing_map[current]

        # Word-final devoicing
        if result and result[-1] in self.voiced_consonants:
            if result[-1] in self.voicing_map:
                result[-1] = self.voicing_map[result[-1]]

        return ''.join(result)

    def apply_palatalization(self, word: str) -> List[str]:
        """Apply palatalization rules and convert to phonemes"""
        if not word:
            return []

        processed_phonemes = []
        i = 0

        while i < len(word):
            char = word[i].lower()

            if char in self.consonants:
                phoneme = self._process_consonant(word, i)
                processed_phonemes.append(phoneme)
            elif char in self.vowels:
                phoneme = self._process_vowel(word, i)
                processed_phonemes.append(phoneme)
            elif char in ['ь', 'ъ']:
                # Soft/hard signs are handled during consonant processing
                pass

            i += 1

        return [p for p in processed_phonemes if p]

    def _process_consonant(self, word: str, pos: int) -> str:
        """Process a single consonant with palatalization rules"""
        char = word[pos].lower()

        if char not in self.consonants:
            return char

        # Check for palatalization triggers
        is_palatalized_by_vowel = (
            pos + 1 < len(word) and
            word[pos + 1].lower() in ['е', 'и', 'ё', 'ю', 'я']
        )

        is_palatalized_by_soft_sign = (
            pos + 1 < len(word) and
            word[pos + 1].lower() == 'ь'
        )

        # Apply palatalization rules
        if char in self.hard_consonants:
            return self.consonants[char]
        elif char in self.soft_consonants:
            return self.consonants[char]
        elif ((is_palatalized_by_vowel or is_palatalized_by_soft_sign) and
              char in self.palatalized):
            return self.palatalized[char]
        else:
            return self.consonants[char]

    def _process_vowel(self, word: str, pos: int) -> str:
        """Process a single vowel with context-dependent rules"""
        char = word[pos].lower()

        if char not in self.vowels and char != 'ə':
            return char

        if char == 'ə':
            return 'ə'

        phoneme = self.vowels[char]

        # Handle iotated vowels after consonants
        if char in ['я', 'ю', 'е', 'ё'] and pos > 0:
            prev_char = word[pos - 1].lower()
            if prev_char in self.consonants:
                # After soft consonants, iotated vowels lose their j-sound
                if prev_char not in self.hard_consonants:
                    vowel_map = {'я': 'a', 'ю': 'u', 'е': 'e', 'ё': 'o'}
                    phoneme = vowel_map.get(char, phoneme)
                # After hard consonants
                elif prev_char in self.hard_consonants:
                    vowel_map_after_hard = {'я': 'a', 'ю': 'u', 'е': 'e', 'ё': 'o', 'и': 'ɨ'}
                    phoneme = vowel_map_after_hard.get(char, phoneme)

        # Special case for 'и' after hard consonants (ж, ш, ц)
        if (char == 'и' and pos > 0 and
            word[pos - 1].lower() in self.hard_consonants):
            phoneme = 'ɨ'  # ы sound

        return phoneme


    # Alternative simpler process_word method for debugging:
    def process_word_debug(self, word: str) -> Tuple[List[str], StressInfo]:
        """Simplified word processing for debugging"""
        if not word:
            return [], StressInfo(0, 0, False)

        # Check for exceptions first
        clean_word = re.sub(r'[\u0300-\u036f]', '', word.lower())
        if clean_word in self.exceptions:
            ipa_string = self.exceptions[clean_word]
            tokenized_ipa = self._tokenize_ipa_string(ipa_string)
            return tokenized_ipa, StressInfo(0, 0, True)

        # Normalize the word
        normalized = self.normalize_text(word)
        if not normalized:
            return [], StressInfo(0, 0, False)

        # Detect stress
        stress_info = self.detect_stress(normalized)

        # SIMPLIFIED PROCESSING - skip reduction and assimilation for now
        # Apply only palatalization
        phonemes = []
        for i, char in enumerate(normalized.lower()):
            if char in self.consonants:
                phoneme = self._process_consonant(normalized, i)
                if phoneme:
                    phonemes.append(phoneme)
            elif char in self.vowels:
                phoneme = self._process_vowel(normalized, i)
                if phoneme:
                    phonemes.append(phoneme)
            elif char in self.STRESS_MARKS:
                continue  # Skip stress marks
            # Skip other characters like ь, ъ

        return phonemes, stress_info

    @lru_cache(maxsize=500)
    def process_word(self, word: str) -> Tuple[List[str], StressInfo]:
        """Process a single word and return phonemes with stress info (cached)"""
        if not word:
            return [], StressInfo(0, 0, False)

        # Check cache first (if not using lru_cache)
        # if word in self._word_cache:
        #     return self._word_cache[word]

        # Check for exceptions first
        clean_word = re.sub(r'[\u0300-\u036f]', '', word.lower())
        if clean_word in self.exceptions:
            ipa_string = self.exceptions[clean_word]
            tokenized_ipa = self._tokenize_ipa_string(ipa_string)
            result = (tokenized_ipa, StressInfo(0, 0, True))
            return result

        # Normalize the word
        normalized = self.normalize_text(word)
        if not normalized:
            return [], StressInfo(0, 0, False)

        try:
            # Detect stress
            stress_info = self.detect_stress(normalized)

            # Apply phonetic changes
            with_reduction = self.apply_vowel_reduction(normalized, stress_info)
            with_assimilation = self.apply_consonant_assimilation(with_reduction)

            # Convert to phonemes
            phonemes = self.apply_palatalization(with_assimilation)

            result = (phonemes, stress_info)
            # Cache the result (if not using lru_cache)
            # self._word_cache[word] = result
            return result

        except Exception as e:
            logger.error(f"Error processing word '{word}': {e}")
            return [], StressInfo(0, 0, False)

    def _tokenize_ipa_string(self, ipa_string: str) -> List[str]:
        """
        Tokenize an IPA string into individual phonemes.
        Improved version with better multi-character phoneme handling.
        """
        if not ipa_string:
            return []

        phonemes = []
        i = 0

        while i < len(ipa_string):
            # Try to match longest possible phoneme first
            matched = False

            # Check for three-character combinations
            if i + 2 < len(ipa_string):
                three_char = ipa_string[i:i+3]
                if three_char in ['ʃtʃ']:  # щ
                    phonemes.append(three_char)
                    i += 3
                    matched = True

            # Check for two-character combinations
            if not matched and i + 1 < len(ipa_string):
                two_char = ipa_string[i:i+2]
                # Palatalized consonants
                if two_char.endswith('ʲ') and two_char in self.palatalized.values():
                    phonemes.append(two_char)
                    i += 2
                    matched = True
                # Other two-character phonemes
                elif two_char in ['ts', 'tʃ']:
                    phonemes.append(two_char)
                    i += 2
                    matched = True

            # Single character
            if not matched:
                phonemes.append(ipa_string[i])
                i += 1

        return [p for p in phonemes if p and p != 'ʲ']  # Remove isolated palatalization marks

    def process_text(self, text: str) -> List[Tuple[str, List[str], StressInfo]]:
        """Process full text and return word-phoneme-stress tuples"""
        if not text:
            return []

        normalized = self.normalize_text(text)
        words = normalized.split()
        results = []

        for word in words:
            if word:
                try:
                    phonemes, stress_info = self.process_word(word)
                    results.append((word, phonemes, stress_info))
                except Exception as e:
                    logger.error(f"Error processing word '{word}': {e}")
                    # Add empty result to maintain word order
                    results.append((word, [], StressInfo(0, 0, False)))

        return results

    def to_ipa(self, phonemes: List[str]) -> str:
        """Convert internal phoneme representation to IPA"""
        return ''.join(phonemes) if phonemes else ""

    def get_stress_pattern(self, text: str) -> List[int]:
        """Get stress pattern for text (for TTS models)"""
        results = self.process_text(text)
        stress_pattern = []

        for word, phonemes, stress_info in results:
            word_pattern = [0] * len(phonemes)
            # Map syllable stress to phoneme stress (simplified)
            if 0 <= stress_info.position < len(phonemes):
                word_pattern[stress_info.position] = 1
            stress_pattern.extend(word_pattern)

        return stress_pattern

    def get_vocab_size(self) -> int:
        """Return the size of the phoneme vocabulary"""
        return len(self.phoneme_to_id)

    def get_phoneme_list(self) -> List[str]:
        """Return sorted list of all phonemes in vocabulary"""
        return sorted(self.phoneme_to_id.keys())

    def _build_vocab(self) -> Dict[str, int]:
        """Build complete phoneme vocabulary"""
        phoneme_set = set()

        # Add base phonemes
        phoneme_set.update(self.vowels.values())
        phoneme_set.update(self.vowel_reduction.values())
        phoneme_set.update(self.consonants.values())
        phoneme_set.update(self.palatalized.values())

        # Add phonemes from exceptions
        for ipa_string in self.exceptions.values():
            exception_phonemes = self._tokenize_ipa_string(ipa_string)
            phoneme_set.update(exception_phonemes)

        # Add commonly used phonemes that might be missing
        additional_phonemes = {'ʌ', 'j', 'ə', 'ɪ', 'ʐ'}
        phoneme_set.update(additional_phonemes)

        # Clean up the set
        phoneme_set.discard('')  # Remove empty strings
        phoneme_set.discard('ʲ')  # Remove isolated palatalization marks

        # Convert to sorted list and create mapping
        phoneme_list = sorted(phoneme_set)
        return {phoneme: idx for idx, phoneme in enumerate(phoneme_list)}

    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to phoneme indices"""
        results = self.process_text(text)
        indices = []

        for word, phonemes, _ in results:
            for phoneme in phonemes:
                idx = self.phoneme_to_id.get(phoneme)
                if idx is not None:
                    indices.append(idx)
                else:
                    logger.warning(f"Unknown phoneme '{phoneme}' in word '{word}'")
                    # Optionally add a default/unknown phoneme index
                    # indices.append(self.phoneme_to_id.get('<UNK>', 0))

        return indices

    def to_dict(self) -> Dict:
        """Serialize processor state to dictionary"""
        return {
            "vowels": self.vowels,
            "consonants": self.consonants,
            "palatalized": self.palatalized,
            "vowel_reduction": self.vowel_reduction,
            "hard_consonants": list(self.hard_consonants),
            "soft_consonants": list(self.soft_consonants),
            "voiced_consonants": list(self.voiced_consonants),
            "voiceless_consonants": list(self.voiceless_consonants),
            "voicing_map": self.voicing_map,
            "stress_patterns": self.stress_patterns,
            "exceptions": self.exceptions,
            "phoneme_to_id": self.phoneme_to_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RussianPhonemeProcessor":
        """Recreate processor from dictionary"""
        instance = cls()

        # Restore all attributes
        for key, value in data.items():
            if key in ["hard_consonants", "soft_consonants", "voiced_consonants", "voiceless_consonants"]:
                setattr(instance, key, set(value))
            else:
                setattr(instance, key, value)

        return instance

    def clear_cache(self):
        """Clear internal caches"""
        self.normalize_text.cache_clear()
        self.process_word.cache_clear()
        self._word_cache.clear()

    def get_cache_info(self) -> Dict:
        """Get cache statistics for debugging"""
        return {
            "normalize_text_cache": self.normalize_text.cache_info(),
            "process_word_cache": self.process_word.cache_info(),
            "word_cache_size": len(self._word_cache)
        }


# Example usage and testing
if __name__ == "__main__":
    processor = RussianPhonemeProcessor()

    # Test examples
    test_words = [
        "привет",      # hello
        "говорить",    # to speak
        "красиво",     # beautiful
        "что",         # what (exception)
        "москва",      # Moscow
        "работает",    # works
        "хорошо",      # good
        "понимать",    # to understand
        "будешь",      # The word that caused an error
        "конечно",     # The word that caused two errors - 'ʌ' error and now 'ʲ' error
        "и",
        "сегодня"      # Explicitly test problematic words
    ]

    print("Russian Phoneme Processing Examples:")
    print("=" * 50)

    for word in test_words:
        phonemes, stress_info = processor.process_word(word)
        ipa = processor.to_ipa(phonemes)

        print(f"Word: {word}")
        print(f"Phonemes: {phonemes}")
        print(f"IPA: /{ipa}/")
        print(f"Stress: syllable {stress_info.position}, vowel {stress_info.vowel_index}")
        print(f"Explicit stress: {stress_info.is_marked}")
        print("-" * 30)

    # Test full text processing
    test_text = "Привет, как дела? Говорить по-русски очень интересно! Будешь ли ты сегодня там? Конечно!"
    print(f"\nFull text: {test_text}")
    print("Processing results:")

    results = processor.process_text(test_text)
    for word, phonemes, stress_info in results:
        ipa = processor.to_ipa(phonemes)
        print(f"{word} -> /{ipa}/ (stress: {stress_info.position})")
