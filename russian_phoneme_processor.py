import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

@dataclass
class StressInfo:
    """Information about stress in a word"""
    position: int  # Position of stressed syllable (0-based)
    vowel_index: int  # Index of stressed vowel in the word
    is_marked: bool  # Whether stress was explicitly marked

class RussianPhonemeProcessor:
    """
    Enhanced Russian phoneme processor with comprehensive stress detection
    and pronunciation rules for TTS systems.
    """

    def __init__(self):
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

        # Common stress patterns for Russian words
        self.stress_patterns = self._load_stress_patterns()

        # Pronunciation exceptions
        self.exceptions = {
            'что': 'ʃto',
            'чтобы': 'ʃtobi',
            'конечно': 'kʌnʲeʃnə',
            'скучно': 'skutʃnə',
            'его': 'jɪvo',
            'сегодня': 'sʲɪvodʲnʲə'
        }

        self.phoneme_to_id = self._build_vocab()

    def _load_stress_patterns(self) -> Dict[str, int]:
        """
        Load common stress patterns. In a real implementation,
        this would load from a comprehensive stress dictionary.
        """
        return {
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

    def normalize_text(self, text: str) -> str:
        """Normalize Russian text for phoneme processing"""
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
        original_word = word

        # Check for explicit stress marks
        stress_marks = ['\u0301', '\u0300', '\u0341']  # Acute, grave, combining acute
        for mark in stress_marks:
            if mark in word:
                # Find position of stress mark
                clean_word = word.replace(mark, '')
                stress_pos = self._find_stress_position(word, mark)
                return StressInfo(
                    position=self._syllable_of_vowel(clean_word, stress_pos),
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
        vowel_count = 0
        for i, char in enumerate(word[:vowel_index + 1]):
            if char.lower() in self.vowels:
                if i == vowel_index:
                    return vowel_count
                vowel_count += 1
        return 0

    def _vowel_index_from_syllable(self, word: str, syllable_pos: int) -> int:
        """Find the index of vowel in specified syllable"""
        vowel_count = 0
        for i, char in enumerate(word):
            if char.lower() in self.vowels:
                if vowel_count == syllable_pos:
                    return i
                vowel_count += 1
        return 0

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        return sum(1 for char in word if char.lower() in self.vowels)

    def _apply_stress_heuristics(self, word: str) -> StressInfo:
        """Apply heuristic rules for stress placement"""
        syllable_count = self._count_syllables(word)

        if syllable_count <= 1:
            return StressInfo(position=0, vowel_index=0, is_marked=False)

        # Common heuristics for Russian stress
        # Rule 1: Many words stress the second-to-last syllable
        if syllable_count >= 2:
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
        else:
            stress_pos = 0

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
                if current in self.voiced_consonants and next_char in self.voiceless_consonants:
                    if current in self.voicing_map:
                        result[i] = self.voicing_map[current]
                elif current in self.voiceless_consonants and next_char in self.voiced_consonants:
                    if current in self.voicing_map:
                        result[i] = self.voicing_map[current]

        # Word-final devoicing
        if result and result[-1] in self.voiced_consonants:
            if result[-1] in self.voicing_map:
                result[-1] = self.voicing_map[result[-1]]

        return ''.join(result)

    def apply_palatalization(self, word: str) -> List[str]:
        """Apply palatalization rules and convert to phonemes"""
        processed_phonemes = []
        j = 0
        while j < len(word):
            char = word[j].lower()

            if char in self.consonants:
                is_palatalized_by_vowel = False
                is_palatalized_by_soft_sign = False

                if j + 1 < len(word):
                    next_char = word[j + 1].lower()
                    if next_char in ['е', 'и', 'ё', 'ю', 'я']:
                        is_palatalized_by_vowel = True
                    elif next_char == 'ь':
                        is_palatalized_by_soft_sign = True

                # Determine if consonant is hard and thus unpalatalizable
                if char in self.hard_consonants:
                    processed_phonemes.append(self.consonants[char])
                # Determine if consonant is soft and thus always palatalized (or already represented as soft)
                elif char in self.soft_consonants:
                    processed_phonemes.append(self.consonants[char])
                # For other consonants, apply palatalization if triggered
                elif (is_palatalized_by_vowel or is_palatalized_by_soft_sign) and char in self.palatalized:
                    processed_phonemes.append(self.palatalized[char])
                else:
                    processed_phonemes.append(self.consonants[char])

            elif char in self.vowels:
                # Handle vowel pronunciation based on context
                phoneme = self.vowels[char]
                if char in ['я', 'ю', 'е', 'ё'] and j > 0:
                    prev_char = word[j-1].lower()
                    if prev_char in self.consonants and prev_char not in self.hard_consonants:
                        vowel_map = {'я': 'a', 'ю': 'u', 'е': 'e', 'ё': 'o'}
                        phoneme = vowel_map.get(char, phoneme)
                    elif prev_char in self.hard_consonants:
                        vowel_map_after_hard_consonant = {'я': 'a', 'ю': 'u', 'е': 'e', 'ё': 'o', 'и': 'ɨ'}
                        phoneme = vowel_map_after_hard_consonant.get(char, phoneme)

                # Special case for 'и' after hard consonant
                if char == 'и' and j > 0 and word[j-1].lower() in self.hard_consonants:
                    phoneme = 'ɨ' # 'ы' sound

                processed_phonemes.append(phoneme)

            elif char == 'ь':
                # Soft sign itself is not a phoneme in IPA, it just modifies the preceding consonant.
                # Its effect is already handled when processing the preceding consonant by looking ahead.
                pass
            elif char == 'ъ':
                # Hard sign - also not a phoneme, marks a boundary or separation.
                pass

            j += 1

        return [p for p in processed_phonemes if p]

    def process_word(self, word: str) -> Tuple[List[str], StressInfo]:
        """Process a single word and return phonemes with stress info"""
        if not word:
            return [], StressInfo(0, 0, False)

        # Check for exceptions first
        clean_word = re.sub(r'[\u0300-\u036f]', '', word.lower())
        if clean_word in self.exceptions:
            # If an exception is found, its value is directly an IPA string.
            # We need to tokenize this IPA string into individual phonemes.
            ipa_string = self.exceptions[clean_word]
            tokenized_ipa = self._tokenize_ipa_string(ipa_string)
            return tokenized_ipa, StressInfo(0, 0, True)

        # Normalize the word
        normalized = self.normalize_text(word)

        # Detect stress
        stress_info = self.detect_stress(normalized)

        # Apply phonetic changes
        with_reduction = self.apply_vowel_reduction(normalized, stress_info)
        with_assimilation = self.apply_consonant_assimilation(with_reduction)

        # Convert to phonemes
        phonemes = self.apply_palatalization(with_assimilation)

        return phonemes, stress_info

    def _tokenize_ipa_string(self, ipa_string: str) -> List[str]:
        """
        Tokenizes an IPA string into a list of individual phonemes.
        This is a simplified tokenizer for the specific phonemes used in this processor.
        For a general IPA parser, a more comprehensive library is needed.
        """
        phonemes = []
        i = 0
        while i < len(ipa_string):
            char = ipa_string[i]

            # Prioritize matching multi-character phonemes first
            # Check for two-character phonemes ending in 'ʲ'
            if i + 1 < len(ipa_string) and ipa_string[i+1] == 'ʲ':
                combined_char = char + 'ʲ'
                # Check if this combined phoneme is a *value* in your palatalized map
                if combined_char in self.palatalized.values() or \
                   combined_char in [v for k,v in self.palatalized.items() if k==char]: # More robust check for palatalized
                    phonemes.append(combined_char)
                    i += 2
                    continue

            # Check for 'ts' (ц)
            if char == 't' and i + 1 < len(ipa_string) and ipa_string[i+1] == 's':
                phonemes.append('ts')
                i += 2
                continue

            # Check for 'tʃ' (ч)
            if char == 't' and i + 1 < len(ipa_string) and ipa_string[i+1] == 'ʃ':
                phonemes.append('tʃ')
                i += 2
                continue

            # Check for 'ʃtʃ' (щ)
            if char == 'ʃ' and i + 1 < len(ipa_string) and ipa_string[i+1] == 't' and \
               i + 2 < len(ipa_string) and ipa_string[i+2] == 'ʃ':
                phonemes.append('ʃtʃ')
                i += 3
                continue

            # Add other specific multi-character IPA combinations if they appear
            # For example, if you had 'dz' or similar in exceptions, add checks here.

            # Default to single character phoneme
            phonemes.append(char)
            i += 1
        return phonemes

    def process_text(self, text: str) -> List[Tuple[str, List[str], StressInfo]]:
        """Process full text and return word-phoneme-stress tuples"""
        normalized = self.normalize_text(text)
        words = normalized.split()
        results = []

        for word in words:
            if word:
                phonemes, stress_info = self.process_word(word)
                results.append((word, phonemes, stress_info))

        return results

    def to_ipa(self, phonemes: List[str]) -> str:
        """Convert internal phoneme representation to IPA"""
        return ''.join(phonemes)

    def get_stress_pattern(self, text: str) -> List[int]:
        """Get stress pattern for text (for TTS models)"""
        results = self.process_text(text)
        stress_pattern = []

        for word, phonemes, stress_info in results:
            word_pattern = [0] * len(phonemes)
            # This logic for mapping syllable stress to phoneme index remains a simplification.
            # A proper mapping requires syllable segmentation of the phoneme sequence.
            # For now, it will mark the phoneme at `stress_info.position` if it's within bounds.
            if stress_info.position < len(phonemes):
                word_pattern[stress_info.position] = 1
            stress_pattern.extend(word_pattern)

        return stress_pattern

    def get_vocab_size(self) -> int:
        """Return the size of the phoneme vocabulary"""
        phoneme_set = set()

        # Add static phonemes
        phoneme_set.update(self.vowels.values())
        phoneme_set.update(self.vowel_reduction.values())
        phoneme_set.update(self.consonants.values())
        phoneme_set.update(self.palatalized.values()) # Adds 'nʲ' etc. as whole units

        # Add phonemes found in exceptions by tokenizing their IPA strings
        for val in self.exceptions.values():
            tokenized_exception_phonemes = self._tokenize_ipa_string(val)
            phoneme_set.update(tokenized_exception_phonemes)

        # Explicitly add any other known single phonemes that might be missed but are used
        phoneme_set.add('ʌ') # From 'конечно'
        phoneme_set.add('j') # From 'й' or initial soft vowels
        phoneme_set.add('ə') # Schwa
        phoneme_set.add('ɪ') # Reduced i
        # Also ensure 'ʐ' (ж) from consonants is added

        # Remove empty strings if present (e.g., from hard sign processing if it results in '')
        phoneme_set.discard('')

        # Final check: remove the isolated 'ʲ' if it somehow got in (it shouldn't with _tokenize_ipa_string)
        phoneme_set.discard('ʲ')

        phoneme_list = sorted(list(phoneme_set)) # Convert to list then sort
        return len(phoneme_list) # Return the count

    def to_dict(self) -> dict:
        """Serialize the processor's state to a dictionary."""
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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RussianPhonemeProcessor":
        """Recreate the processor from a dictionary."""
        instance = cls()
        instance.vowels = data["vowels"]
        instance.consonants = data["consonants"]
        instance.palatalized = data["palatalized"]
        instance.vowel_reduction = data["vowel_reduction"]
        instance.hard_consonants = set(data["hard_consonants"])
        instance.soft_consonants = set(data["soft_consonants"])
        instance.voiced_consonants = set(data["voiced_consonants"])
        instance.voiceless_consonants = set(data["voiceless_consonants"])
        instance.voicing_map = data["voicing_map"]
        instance.stress_patterns = data["stress_patterns"]
        instance.exceptions = data["exceptions"]
        # Rebuild phoneme_to_id after loading all attributes
        instance.phoneme_to_id = instance._build_vocab()
        return instance

    def _build_vocab(self) -> Dict[str, int]:
        """Build full phoneme vocabulary, including dynamically generated palatalized forms."""
        phoneme_set = set()

        # Add static phonemes
        phoneme_set.update(self.vowels.values())
        phoneme_set.update(self.vowel_reduction.values())
        phoneme_set.update(self.consonants.values())
        phoneme_set.update(self.palatalized.values()) # Adds 'nʲ' etc. as whole units

        # Add phonemes found in exceptions by tokenizing their IPA strings
        for val in self.exceptions.values():
            tokenized_exception_phonemes = self._tokenize_ipa_string(val)
            phoneme_set.update(tokenized_exception_phonemes)

        # Explicitly add any other known single phonemes that might be missed but are used
        phoneme_set.add('ʌ') # From 'конечно'
        phoneme_set.add('j') # From 'й' or initial soft vowels
        phoneme_set.add('ə') # Schwa
        phoneme_set.add('ɪ') # Reduced i
        # Ensure 'ʐ' is explicitly considered if it's not guaranteed by self.consonants.values()
        # It should be covered by consonants.values(), but explicit adds can act as a safeguard.
        phoneme_set.add('ʐ')


        # Remove empty strings if present (e.g., from hard sign processing if it results in '')
        phoneme_set.discard('')

        # Final check: remove the isolated 'ʲ' if it somehow got in (it shouldn't with _tokenize_ipa_string)
        phoneme_set.discard('ʲ')

        phoneme_list = sorted(list(phoneme_set)) # Convert to list then sort
        return {phoneme: idx for idx, phoneme in enumerate(phoneme_list)}


    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to a list of phoneme indices"""
        results = self.process_text(text)
        indices = []

        for word, phonemes, _ in results:
            for p in phonemes:
                idx = self.phoneme_to_id.get(p)
                if idx is not None:
                    indices.append(idx)
                else:
                    # This is the line causing the error. Debugging here.
                    raise ValueError(f"Unknown phoneme: {p} in word: {word}")

        return indices


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
        "будешь",       # The word that caused the error
        "конечно",     # The word that caused the 'ʌ' error and now 'ʲ' error
        "и",
        "сегодня" # Explicitly test the problematic word
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
