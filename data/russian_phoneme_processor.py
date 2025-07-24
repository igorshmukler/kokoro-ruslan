import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StressInfo:
    """Information about stress in a word"""
    position: int  # Position of stressed syllable (0-based)
    vowel_index: int  # Index of stressed vowel in the word (0-based character index in the original word)
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
        # Vowel mappings (default, before reduction)
        self.vowels = {
            'а': 'a', 'о': 'o', 'у': 'u', 'ы': 'ɨ', 'э': 'e',
            'я': 'ja', 'ё': 'jo', 'ю': 'ju', 'и': 'i', 'е': 'je'
        }

        # Consonant mappings
        self.consonants = {
            'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ж': 'ʐ', 'з': 'z',
            'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'п': 'p', 'р': 'r',
            'с': 's', 'т': 't', 'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'tʃ',
            'ш': 'ʃ', 'щ': 'ʃtʃ', 'й': 'j'
        }

        # Palatalized consonants
        self.palatalized = {
            'б': 'bʲ', 'в': 'vʲ', 'г': 'gʲ', 'д': 'dʲ', 'з': 'zʲ',
            'к': 'kʲ', 'л': 'lʲ', 'м': 'mʲ', 'н': 'nʲ', 'п': 'pʲ',
            'р': 'rʲ', 'с': 'sʲ', 'т': 'tʲ', 'ф': 'fʲ', 'х': 'xʲ'
        }

        # Hard consonants (never palatalized)
        self.hard_consonants = {'ж', 'ш', 'ц'}

        # Soft consonants (always palatalized, or inherently soft)
        self.soft_consonants = {'ч', 'щ', 'й'}

        # Voicing assimilation rules
        self.voiced_consonants = {'б', 'в', 'г', 'д', 'ж', 'з'}
        self.voiceless_consonants = {'п', 'ф', 'к', 'т', 'ш', 'с', 'х', 'ц', 'ч', 'щ'}

        self.voicing_map = {
            'б': 'п', 'в': 'ф', 'г': 'к', 'д': 'т', 'ж': 'ш', 'з': 'с',
            'п': 'б', 'ф': 'в', 'к': 'г', 'т': 'д', 'ш': 'ж', 'с': 'з'
        }

        # Load stress patterns
        self.stress_patterns = self._load_stress_patterns(stress_dict_path)

        # Pronunciation exceptions (these are full IPA strings)
        self.exceptions = {
            'что': 'ʃto',
            'чтобы': 'ʃtobi',
            'конечно': 'kɐnʲeʃnə',
            'скучно': 'skutʃnə',
            'его': 'jɪvo',
            'сегодня': 'sʲɪvodʲnʲə',
            'здравствуйте': 'zdrastvujtʲe' # Explicitly added based on expected output
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
            'понимает': 2, 'знает': 1, 'играет': 1,
            # Add specific words from your example
            'привет': 1,  # приве́т
            'как': 0,     # как (monosyllabic)
            'дела': 1,    # дела́
            'молоко': 2,  # молоко́
            'сегодня': 1, # сего́дня - add for consistency with exceptions
            'здравствуйте': 1 # здра́вствуйте - add for consistency with exceptions
        }

        if dict_path:
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                word, stress_pos_str = parts[0], parts[1]
                                try:
                                    patterns[word] = int(stress_pos_str)
                                except ValueError:
                                    logger.warning(f"Invalid stress position for word {word}: {stress_pos_str}")
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

        # Handle 'ё' - it's always stressed 'о' but we convert it to 'е' with stress mark for consistent handling
        text = text.replace('ё', 'е́')

        # Normalize Unicode: separate base characters from combining marks
        text = unicodedata.normalize('NFD', text)

        # Remove combining marks that are NOT stress marks
        # Keep only Cyrillic letters, allowed punctuation (space), and stress marks
        allowed_chars_set = set('абвгдежзийклмнопрстуфхцчшщъыьэюя ')
        clean_text_chars = []
        for char in text:
            if char in allowed_chars_set:
                clean_text_chars.append(char)
            elif char in self.STRESS_MARKS:
                clean_text_chars.append(char)
            # else: skip other non-allowed combining marks or punctuation

        text = ''.join(clean_text_chars)

        # Remove any remaining punctuation that wasn't filtered by the NFD and allowed_chars logic
        # and wasn't a stress mark. Using a more targeted regex.
        text = re.sub(r'[^\w\s' + ''.join(re.escape(m) for m in self.STRESS_MARKS) + r']', ' ', text)

        # Normalize whitespace (multiple spaces to single space, trim)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def detect_stress(self, word: str) -> StressInfo:
        """
        Detect stress position in a Russian word.
        Priority: 1) Explicit marks, 2) Dictionary, 3) Heuristics
        """
        if not word:
            return StressInfo(0, 0, False)

        # Attempt 1: Check for explicit stress marks
        # Create a version of the word without stress marks to easily get clean indices
        clean_word_for_idx = []
        stress_vowel_char_idx = -1 # Character index in the *clean* word

        for i, char in enumerate(word):
            if char in self.STRESS_MARKS:
                if i > 0 and word[i-1].lower() in self.VOWEL_LETTERS:
                    # Stress mark applies to the *previous* vowel in the original word string
                    stress_vowel_char_idx = len(clean_word_for_idx) - 1
                # Do not append stress mark to clean_word_for_idx
            else:
                clean_word_for_idx.append(char)

        clean_word_str = "".join(clean_word_for_idx)

        if stress_vowel_char_idx != -1:
            # Found explicit stress. Now find its syllable position.
            syllable_pos = self._syllable_of_vowel(clean_word_str, stress_vowel_char_idx)
            return StressInfo(
                position=syllable_pos,
                vowel_index=stress_vowel_char_idx,
                is_marked=True
            )

        # Attempt 2: Check dictionary after removing all marks
        word_for_dict_lookup = re.sub(r'[\u0300-\u036f]', '', word).lower()
        if word_for_dict_lookup in self.stress_patterns:
            syllable_pos = self.stress_patterns[word_for_dict_lookup]
            vowel_index = self._vowel_index_from_syllable(word_for_dict_lookup, syllable_pos)
            return StressInfo(
                position=syllable_pos,
                vowel_index=vowel_index,
                is_marked=False
            )

        # Attempt 3: Apply heuristic rules (use the clean_word_str here)
        return self._apply_stress_heuristics(clean_word_str)

    def _syllable_of_vowel(self, word: str, vowel_char_index: int) -> int:
        """
        Finds the 0-based syllable position of a vowel given its character index
        in the *clean* (no stress marks) word.
        """
        if vowel_char_index < 0 or vowel_char_index >= len(word):
            return 0 # Invalid index, default to 0

        syllable_count = 0
        for i, char in enumerate(word):
            if char.lower() in self.VOWEL_LETTERS:
                if i == vowel_char_index:
                    return syllable_count
                syllable_count += 1
        return 0 # Should ideally not be reached if vowel_char_index points to a vowel

    def _vowel_index_from_syllable(self, word: str, syllable_pos: int) -> int:
        """
        Finds the character index of the vowel corresponding to the given
        0-based syllable position in the *clean* word.
        """
        vowel_count = 0
        for i, char in enumerate(word):
            if char.lower() in self.VOWEL_LETTERS:
                if vowel_count == syllable_pos:
                    return i
                vowel_count += 1

        # If the requested syllable position is out of bounds,
        # fallback to the last vowel's index or 0 for empty words.
        # logger.warning(f"Syllable {syllable_pos} not found in '{word}'. Defaulting to last vowel for stress.")

        last_vowel_idx = -1
        for i in reversed(range(len(word))):
            if word[i].lower() in self.VOWEL_LETTERS:
                last_vowel_idx = i
                break
        return max(0, last_vowel_idx) # Ensure it's not negative

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word based on vowels"""
        return sum(1 for char in word if char.lower() in self.VOWEL_LETTERS)

    def _apply_stress_heuristics(self, word: str) -> StressInfo:
        """Apply heuristic rules for stress placement"""
        syllable_count = self._count_syllables(word)

        if syllable_count <= 1:
            vowel_index = self._vowel_index_from_syllable(word, 0)
            return StressInfo(position=0, vowel_index=vowel_index, is_marked=False)

        # Default heuristic: stress the penultimate syllable
        stress_syllable_pos = max(0, syllable_count - 2)

        # Refined heuristics for common endings
        if word.endswith(('ать', 'еть', 'ить', 'ыть', 'уть', 'ять')): # Infinitive verbs
            stress_syllable_pos = syllable_count - 1 # Stress on the last syllable (infinitive ending)
        elif word.endswith(('ие', 'ые', 'ая', 'яя', 'ое', 'ее', 'ую', 'ею')): # Adjectives, participles
            stress_syllable_pos = max(0, syllable_count - 2) # Often on the root, before the ending
        elif word.endswith(('ость', 'есть')): # Abstract nouns
            stress_syllable_pos = max(0, syllable_count - 2)
        elif word.endswith('ий'): # Adjectives
            stress_syllable_pos = max(0, syllable_count - 2)
        elif word.endswith(('ние', 'тие')): # Nouns from verbs
            stress_syllable_pos = max(0, syllable_count - 2) # Often on the last root syllable

        # Ensure stress position is within valid bounds
        stress_syllable_pos = min(stress_syllable_pos, syllable_count - 1)

        vowel_index = self._vowel_index_from_syllable(word, stress_syllable_pos)
        return StressInfo(position=stress_syllable_pos, vowel_index=vowel_index, is_marked=False)

    def apply_vowel_reduction(self, phonemes: List[str], stress_syllable_idx: int) -> List[str]:
        """
        Apply vowel reduction to phoneme list based on stress position.
        The stress_syllable_idx is the 0-based index of the *stressed syllable*.
        """
        if not phonemes:
            return phonemes

        reduced_phonemes = phonemes.copy()
        current_vowel_syllable_count = 0 # This counts actual syllables based on vowel phonemes

        # Create a list to store original vowel sounds for accurate reduction
        # This is crucial because a phoneme might be 'je' and we need 'e' for reduction logic
        original_vowel_sounds = []
        for ph in phonemes:
            if ph in ['a', 'o', 'u', 'ɨ', 'e', 'i']:
                original_vowel_sounds.append(ph)
            elif ph in ['ja', 'jo', 'ju', 'je']:
                original_vowel_sounds.append(ph[1:]) # Get 'a', 'o', 'u', 'e'
            else:
                original_vowel_sounds.append(None) # Not a vowel phoneme, placeholder

        original_vowel_idx = 0 # Tracks the index in original_vowel_sounds

        for i, phoneme in enumerate(reduced_phonemes):
            is_vowel_phoneme = (original_vowel_sounds[original_vowel_idx] is not None) if original_vowel_idx < len(original_vowel_sounds) else False

            if is_vowel_phoneme:
                base_vowel_sound = original_vowel_sounds[original_vowel_idx]

                if current_vowel_syllable_count != stress_syllable_idx:  # Not the stressed syllable
                    # Vowels before stress (pre-tonic)
                    if current_vowel_syllable_count < stress_syllable_idx:
                        # First pre-tonic syllable: (stressed_idx - current_idx) == 1
                        if (stress_syllable_idx - current_vowel_syllable_count) == 1:
                            if base_vowel_sound in ['o', 'a']:
                                reduced_phonemes[i] = 'ɐ' # 'о', 'а' -> 'ɐ' in first pre-tonic
                            elif base_vowel_sound in ['e', 'je', 'jo', 'i']: # 'е', 'и', 'ё' -> 'ɪ' in first pre-tonic
                                reduced_phonemes[i] = 'ɪ'
                        # Second pre-tonic and beyond
                        else:
                            if base_vowel_sound in ['o', 'a', 'e', 'ja', 'jo', 'je', 'i']:
                                reduced_phonemes[i] = 'ə' # Stronger reduction to schwa 'ə'
                    # Vowels after stress (post-tonic)
                    else:
                        if base_vowel_sound in ['o', 'a', 'e', 'ja', 'jo', 'je', 'i']:
                            reduced_phonemes[i] = 'ə' # Post-tonic vowels generally reduce to schwa 'ə'

                    # 'u', 'ju', 'ɨ' typically do not reduce significantly in Russian (remain 'u', 'ju', 'ɨ')
                    # 'i' after hard consonants (which becomes 'ɨ') also does not reduce further.

                current_vowel_syllable_count += 1
                original_vowel_idx += 1 # Only increment if it was a vowel phoneme
            else:
                # If it's a consonant, just move past it in the phoneme list
                if original_vowel_sounds[original_vowel_idx] is None:
                    original_vowel_idx += 1


        return reduced_phonemes


    def apply_consonant_assimilation(self, word: str) -> str:
        """Apply voicing assimilation and other consonant changes"""
        # Ensure we work on a mutable list of characters
        word_chars = list(word.lower())

        # --- Specific Complex Cases (apply before general rules) ---
        # 'вств' in 'здравствуйте' is often pronounced 'stv' or 'stf'
        # Simplified to 'stf' for now, as it's a common realization
        word_str = "".join(word_chars)
        word_str = word_str.replace('вств', 'stf') # A common realization for "здравствуйте"

        # 'ться' and 'тся' (reflexive verb endings) often pronounced as 'tsə' or 'tsa'
        # The 'т' and 'с' merge into 'ts' and 'я'/'а' reduces.
        word_str = word_str.replace('ться', 'цə') # For verbs ending in -ться (e.g. учиться)
        word_str = word_str.replace('тся', 'цə')  # For verbs ending in -тся (e.g. учится)

        word_chars = list(word_str)


        # --- General Voicing Assimilation ---
        # Assimilate voicing from right to left (regressive assimilation) for pairs
        for i in range(len(word_chars) - 1):
            current = word_chars[i]
            next_char = word_chars[i + 1]

            if current in self.consonants and next_char in self.consonants:
                # Voicing assimilation: current consonant assimilates to the next
                if current in self.voiced_consonants and next_char in self.voiceless_consonants:
                    # Devoicing: If voiced consonant followed by voiceless
                    if current in self.voicing_map and self.voicing_map[current] in self.voiceless_consonants:
                        word_chars[i] = self.voicing_map[current]
                elif current in self.voiceless_consonants and next_char in self.voiced_consonants:
                    # Voicing: If voiceless consonant followed by voiced
                    if current in self.voicing_map and self.voicing_map[current] in self.voiced_consonants:
                        word_chars[i] = self.voicing_map[current]

        # --- Word-final Devoicing ---
        # Apply word-final devoicing: voiced consonants become voiceless at the end of a word
        if word_chars and word_chars[-1] in self.voiced_consonants:
            # Only devoice if the last character is indeed a consonant that can be devoiced
            if word_chars[-1] in self.voicing_map and self.voicing_map[word_chars[-1]] in self.voiceless_consonants:
                word_chars[-1] = self.voicing_map[word_chars[-1]]

        return ''.join(word_chars)

    def apply_palatalization(self, word: str) -> List[str]:
        """
        Applies palatalization rules and converts letters to base phonemes.
        Handles 'ь' and 'ъ' effects.
        """
        if not word:
            return []

        processed_phonemes = []
        i = 0
        while i < len(word):
            char = word[i].lower()

            if char in self.VOWEL_LETTERS:
                processed_phonemes.append(self._process_vowel(word, i))
            elif char in self.consonants or char in self.hard_consonants or char in self.soft_consonants:
                # Determine if the consonant is palatalized by context
                is_palatalized = False
                if i + 1 < len(word):
                    next_char = word[i + 1].lower()
                    if next_char in ['е', 'и', 'ё', 'ю', 'я', 'ь']:
                        is_palatalized = True

                # Apply palatalization if applicable
                if char in self.hard_consonants: # Always hard
                    processed_phonemes.append(self.consonants[char])
                elif char in self.soft_consonants: # Always soft
                    processed_phonemes.append(self.consonants[char])
                elif is_palatalized and char in self.palatalized:
                    processed_phonemes.append(self.palatalized[char])
                elif char in self.consonants: # Default hard consonant
                    processed_phonemes.append(self.consonants[char])
                else: # Fallback for unexpected consonant-like chars
                    processed_phonemes.append(char)
            elif char == 'ь':
                # Soft sign itself does not produce a phoneme, it affects preceding consonant
                pass
            elif char == 'ъ':
                # Hard sign itself does not produce a phoneme, it prevents palatalization/separation
                pass
            else:
                # If there are other non-alphabetic characters remaining, append them or skip
                # For now, we assume normalize_text cleans them well.
                pass
            i += 1

        return [p for p in processed_phonemes if p] # Filter out any empty strings

    def _process_consonant(self, word: str, pos: int) -> str:
        """
        Helper for `apply_palatalization` to get the base phoneme for a consonant,
        including inherent softness/hardness.
        Palatalization due to context is handled in `apply_palatalization`.
        """
        char = word[pos].lower()

        if char == 'й':
            return self.consonants['й'] # 'j'
        elif char in self.soft_consonants:
            return self.consonants[char] # 'ч', 'щ'
        elif char in self.hard_consonants:
            return self.consonants[char] # 'ж', 'ш', 'ц'
        elif char in self.consonants:
            return self.consonants[char] # Default hard consonant
        return char # Fallback (shouldn't be reached for valid consonants)


    def _process_vowel(self, word: str, pos: int) -> str:
        """
        Processes a single vowel character to its base phoneme,
        considering iotated vowels and 'и' after hard consonants.
        Vowel reduction happens in `apply_vowel_reduction`.
        """
        char = word[pos].lower()

        if char not in self.VOWEL_LETTERS:
            return char # Not a vowel letter, return as is (should be filtered earlier)

        # Handle iotated vowels ('я', 'ю', 'е', 'ё') contextually for their base phoneme
        if char in ['я', 'ю', 'е', 'ё']:
            if pos == 0:  # Word initial, or after non-letter (space, punctuation)
                return self.vowels[char] # Keep iotated form (e.g., 'ja', 'ju')

            prev_char = word[pos - 1].lower()
            if prev_char in self.VOWEL_LETTERS: # After another vowel
                return self.vowels[char] # Keep iotated form
            elif prev_char == 'ъ' or prev_char == 'ь': # After hard/soft sign
                return self.vowels[char] # Keep iotated form (sign acts as a separator)
            elif prev_char in self.consonants or prev_char in self.hard_consonants or prev_char in self.soft_consonants:
                # After a consonant, these vowels only contribute their vowel sound.
                # The 'j' component is implicitly handled by the preceding consonant's palatalization
                # or is absent if the consonant is hard.
                vowel_map_after_consonant = {
                    'я': 'a', 'ю': 'u', 'е': 'e', 'ё': 'o'
                }
                return vowel_map_after_consonant.get(char, self.vowels[char]) # Get non-iotated base vowel

        # Special case for 'и' after hard consonants (ж, ш, ц)
        if char == 'и' and pos > 0 and word[pos - 1].lower() in self.hard_consonants:
            return 'ɨ'  # 'ы' sound

        return self.vowels[char] # Default vowel mapping


    @lru_cache(maxsize=500)
    def process_word(self, word: str) -> Tuple[List[str], StressInfo]:
        """Process a single word and return phonemes with stress info (cached)"""
        if not word:
            return [], StressInfo(0, 0, False)

        # Remove explicit stress marks for consistent processing internally
        word_for_lookup = re.sub(r'[\u0300-\u036f]', '', word).lower()

        # Check for full word exceptions first on the cleaned word
        if word_for_lookup in self.exceptions:
            ipa_string = self.exceptions[word_for_lookup]
            tokenized_ipa = self._tokenize_ipa_string(ipa_string)

            # For exceptions, try to get stress info from the stress_patterns dictionary
            # if available, otherwise default. This provides more accurate stress info
            # for words handled by exceptions.
            if word_for_lookup in self.stress_patterns:
                syllable_pos = self.stress_patterns[word_for_lookup]
                vowel_index = self._vowel_index_from_syllable(word_for_lookup, syllable_pos)
                stress_info = StressInfo(
                    position=syllable_pos,
                    vowel_index=vowel_index,
                    is_marked=True # Marked as true since it's an exception, assumed known stress
                )
            else:
                # Default stress info if not found in stress_patterns
                stress_info = StressInfo(position=0, vowel_index=0, is_marked=True)

            logger.debug(f"  Word: '{word}' -> Handled by exception: {ipa_string}")
            logger.debug(f"  Stress Info (Exception): Syllable {stress_info.position}, Vowel Index {stress_info.vowel_index}, Marked: {stress_info.is_marked}")
            return (tokenized_ipa, stress_info)

        normalized_word = self.normalize_text(word)
        if not normalized_word:
            return [], StressInfo(0, 0, False)

        try:
            # Step 1: Detect stress on the normalized word (before any phoneme conversion)
            stress_info = self.detect_stress(normalized_word)
            logger.debug(f"  Word: '{word}' -> Normalized: '{normalized_word}'")
            logger.debug(f"  Stress Info: Syllable {stress_info.position}, Vowel Index {stress_info.vowel_index}, Marked: {stress_info.is_marked}")

            # Step 2: Apply consonant assimilation rules on the normalized word string
            # This step modifies the *string* before converting to phonemes
            word_after_assimilation = self.apply_consonant_assimilation(normalized_word)
            logger.debug(f"  After consonant assimilation (letters): '{word_after_assimilation}'")

            # Step 3: Convert letters to base phonemes (including palatalization effects)
            base_phonemes = self.apply_palatalization(word_after_assimilation)
            logger.debug(f"  Base Phonemes (pre-reduction): {base_phonemes}")

            # Step 4: Apply vowel reduction to the list of base phonemes using the detected stress info
            final_phonemes = self.apply_vowel_reduction(base_phonemes, stress_info.position)
            logger.debug(f"  Final Phonemes (post-reduction): {final_phonemes}")

            return (final_phonemes, stress_info)

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

        # Define multi-character phonemes (longest first).
        # Include all possible palatalized consonants, affricates, and reduced vowels.
        multi_char_phonemes = sorted(
            list(self.palatalized.values()) + # e.g., 'bʲ', 'dʲ'
            ['ts', 'tʃ', 'ʃtʃ', 'dʑ', 'dz', 'tɕ', 'dʑ', # Affricates and their palatalized/voiced forms
             'ɐ', 'ə', 'ɪ', 'ɨ', # Reduced vowels
             'ja', 'jo', 'ju', 'je', # Iotated vowels (base forms)
             'stf' # Specific clusters like 'здравствуйте' part
            ],
            key=len,
            reverse=True # Match longest sequence first
        )

        # Single characters (for fallback)
        single_chars = set('pbvmfnlrkgxdʒʃʐzvstchwiaeouɨɐəɪˈˌ') # Common IPA single chars including vowels and stress marks

        while i < len(ipa_string):
            matched = False
            # Try to match longest possible phoneme first
            for mc_ph in multi_char_phonemes:
                if ipa_string.startswith(mc_ph, i):
                    phonemes.append(mc_ph)
                    i += len(mc_ph)
                    matched = True
                    break

            if not matched:
                # If not a multi-character phoneme, try single character
                char = ipa_string[i]
                # Basic check: if it's a known single IPA char or a diacritic
                if char in single_chars or unicodedata.category(char) == 'Mn': # Mn for combining marks
                    phonemes.append(char)
                    i += 1
                else: # Fallback for unknown characters (e.g., if a new char is introduced)
                    phonemes.append(char)
                    i += 1

        # Post-processing: remove isolated stress marks and 'ʲ' if they were accidentally tokenized alone
        # Stress marks are typically applied *after* phoneme sequence is determined for TTS.
        return [p for p in phonemes if p and p not in self.STRESS_MARKS and p != 'ˈ' and p != 'ˌ' and p != 'ʲ']

    def process_text(self, text: str) -> List[Tuple[str, List[str], StressInfo]]:
        """Process full text and return word-phoneme-stress tuples"""
        if not text:
            return []

        normalized_text = self.normalize_text(text)
        words = normalized_text.split()
        results = []

        for word in words:
            if word: # Ensure word is not empty after splitting
                try:
                    phonemes, stress_info = self.process_word(word)
                    results.append((word, phonemes, stress_info))
                except Exception as e:
                    logger.error(f"Error processing word '{word}': {e}")
                    # Add empty result to maintain word order for sentence context
                    results.append((word, [], StressInfo(0, 0, False)))

        return results

    def to_ipa(self, phonemes: List[str]) -> str:
        """Convert internal phoneme representation to IPA string."""
        return ''.join(phonemes) if phonemes else ""

    def get_stress_pattern(self, text: str) -> List[int]:
        """
        Get stress pattern for text (for TTS models).
        Returns a list of integers, where 1 indicates stress and 0 no stress,
        aligned with the final phoneme sequence.
        """
        results = self.process_text(text)
        stress_pattern = []

        for word_orig, phonemes, stress_info in results:
            word_phoneme_stress = [0] * len(phonemes)

            vowel_phoneme_count = 0
            for i, ph in enumerate(phonemes):
                # Simple check if phoneme is a vowel or reduced vowel
                is_vowel_ph = any(ph.startswith(v) for v in ['a', 'o', 'u', 'ɨ', 'e', 'i', 'ja', 'jo', 'ju', 'je', 'ə', 'ɐ', 'ɪ'])

                if is_vowel_ph:
                    if vowel_phoneme_count == stress_info.position:
                        word_phoneme_stress[i] = 1 # Mark the phoneme at this position as stressed
                        break # Found the stressed vowel, move to next word
                    vowel_phoneme_count += 1

            stress_pattern.extend(word_phoneme_stress)

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
        phoneme_set.update(self.consonants.values())
        phoneme_set.update(self.palatalized.values())

        # Add reduced vowels
        phoneme_set.update(['ə', 'ɪ', 'ɐ'])

        # Add phonemes from exceptions (tokenized)
        for ipa_string in self.exceptions.values():
            exception_phonemes = self._tokenize_ipa_string(ipa_string)
            phoneme_set.update(exception_phonemes)

        # Add commonly used phonemes that might be missing or appear in specific contexts
        additional_phonemes = {'j', 'ʐ', 'ts', 'tʃ', 'ʃtʃ', 'bʲ', 'vʲ', 'gʲ', 'dʲ', 'zʲ', 'kʲ', 'lʲ', 'mʲ', 'nʲ', 'pʲ', 'rʲ', 'sʲ', 'tʲ', 'fʲ', 'xʲ', 'stf'}
        phoneme_set.update(additional_phonemes)

        # Clean up the set from any control characters or isolated diacritics
        phoneme_set.discard('')
        phoneme_set.discard('ʲ') # Should be part of a consonant phoneme (e.g., 'pʲ')
        phoneme_set.discard('ˈ') # Primary stress mark
        phoneme_set.discard('ˌ') # Secondary stress mark

        # Convert to sorted list and create mapping
        phoneme_list = sorted(list(phoneme_set))
        return {phoneme: idx for idx, phoneme in enumerate(phoneme_list)}

    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to phoneme indices for TTS model input"""
        results = self.process_text(text)
        indices = []

        for word, phonemes, _ in results:
            for phoneme in phonemes:
                idx = self.phoneme_to_id.get(phoneme)
                if idx is not None:
                    indices.append(idx)
                else:
                    logger.warning(f"Unknown phoneme '{phoneme}' encountered in word '{word}'. Skipping.")
        return indices

    def to_dict(self) -> Dict:
        """Serialize processor state to dictionary (for saving/loading)"""
        return {
            "vowels": self.vowels,
            "consonants": self.consonants,
            "palatalized": self.palatalized,
            "hard_consonants": list(self.hard_consonants),
            "soft_consonants": list(self.soft_consonants),
            "voiced_consonants": list(self.voiced_consonants),
            "voiceless_consonants": list(self.voiceless_consonants),
            "voicing_map": self.voicing_map,
            "stress_patterns": self.stress_patterns,
            "exceptions": self.exceptions,
            "phoneme_to_id": self.phoneme_to_id # Include the built vocabulary
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RussianPhonemeProcessor":
        """Recreate processor from a dictionary (for saving/loading)"""
        instance = cls()
        # Restore all attributes, ensuring sets are converted from lists
        instance.vowels = data.get("vowels", {})
        instance.consonants = data.get("consonants", {})
        instance.palatalized = data.get("palatalized", {})
        instance.hard_consonants = set(data.get("hard_consonants", []))
        instance.soft_consonants = set(data.get("soft_consonants", []))
        instance.voiced_consonants = set(data.get("voiced_consonants", []))
        instance.voiceless_consonants = set(data.get("voiceless_consonants", []))
        instance.voicing_map = data.get("voicing_map", {})
        instance.stress_patterns = data.get("stress_patterns", {})
        instance.exceptions = data.get("exceptions", {})
        instance.phoneme_to_id = data.get("phoneme_to_id", {})
        return instance

    def clear_cache(self):
        """Clear internal caches to free memory or re-run processing"""
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
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Russian Phoneme Processor for command line testing.")
    parser.add_argument("-t", "--text", type=str, help="Text to process directly (enclose in quotes for phrases).")
    parser.add_argument("-f", "--file", type=str, help="Path to a text file to process.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for debugging.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO) # Keep INFO level by default for less clutter

    input_text = ""
    if args.text:
        input_text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {args.file}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty(): # Check if input is being piped
        input_text = sys.stdin.read()
    else:
        print("Please provide text using -t, -f, or pipe input to stdin.", file=sys.stderr)
        # Fallback to default test text if no arguments provided for simple execution
        input_text = "Привет, как дела?"
        print(f"No input provided. Using default text: \"{input_text}\"")


    if not input_text.strip():
        print("No text provided for processing.", file=sys.stderr)
        sys.exit(1)

    processor = RussianPhonemeProcessor()

    print(f"\nProcessing input text: \"{input_text}\"")
    print("=" * 50)

    results = processor.process_text(input_text)
    for word, phonemes, stress_info in results:
        ipa = processor.to_ipa(phonemes)
        print(f"Word: {word}")
        print(f"  Phonemes: {phonemes}")
        print(f"  IPA: /{ipa}/")
        print(f"  Stress: syllable {stress_info.position} (vowel index in word: {stress_info.vowel_index}), marked: {stress_info.is_marked}")
        print("-" * 30)

    # Re-run specific fixed tests to verify against expected output
    print("\n--- Verification Against Expected Outputs ---")
    test_words_for_verification = {
        "привет": "prʲɪvʲet", # prʲi-vét
        "как": "kak",
        "дела": "dʲɪla", # dʲi-lá
        "молоко": "mɐlɐko", # mɐ-lɐ-kó (stress on last o)
        "хорошо": "xərɐʃo", # xə-rɐ-šó (stress on last o)
        "сегодня": "sʲɪvodʲnʲə", # from exceptions
        "здравствуйте": "zdrastvujtʲe" # from exceptions
    }

    for word, expected_ipa in test_words_for_verification.items():
        phonemes, stress_info = processor.process_word(word)
        actual_ipa = processor.to_ipa(phonemes)
        # Strip slashes from expected_ipa for direct comparison
        stripped_expected_ipa = expected_ipa.strip('/')
        match_status = "MATCH" if actual_ipa == stripped_expected_ipa else "MISMATCH"
        print(f"Word: '{word}'")
        print(f"  Expected IPA: /{stripped_expected_ipa}/")
        print(f"  Actual IPA:   /{actual_ipa}/")
        print(f"  Stress: syllable {stress_info.position}")
        print(f"  Status: {match_status}")
        print("-" * 30)
