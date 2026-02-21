import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
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

        self._multi_char_phonemes = sorted(
            list(self.palatalized.values()) +
            ['ts', 'tʃ', 'ʃtʃ', 'dʑ', 'dz', 'tɕ', 'ɐ', 'ə', 'ɪ', 'ɨ', 'ja', 'jo', 'ju', 'je'],
            key=len, reverse=True
        )

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
        }

        # Build vocabulary after all mappings are set
        self.phoneme_to_id = self._build_vocab()

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
        VOWEL_BASES = {'a', 'o', 'u', 'ɨ', 'e', 'i', 'ja', 'jo', 'ju', 'je'}
        reduced = phonemes.copy()
        syllable = 0
        for i, ph in enumerate(reduced):
            if ph in VOWEL_BASES:
                if syllable != stress_syllable_idx:
                    dist = stress_syllable_idx - syllable
                    base = ph[1:] if ph.startswith('j') else ph
                    if syllable < stress_syllable_idx:
                        if dist == 1:
                            reduced[i] = 'ɐ' if base in ('o', 'a') else 'ɪ' if base in ('e', 'i') else ph
                        else:
                            reduced[i] = 'ə' if base in ('o', 'a', 'e', 'i') else ph
                    else:
                        reduced[i] = 'ə' if base in ('o', 'a', 'e', 'i') else ph
                syllable += 1
        return reduced

    def apply_consonant_assimilation(self, word: str) -> str:
        """
        Apply voicing assimilation and consonant cluster simplifications.
        All substitutions remain in Cyrillic — IPA conversion happens downstream
        in apply_palatalization.
        """
        word = word.lower()

        # --- 1. The "Г" Exceptions ---

        # A. Genitive endings: -ого/-его -> -ово/-ево
        # (Applies to pronouns and adjectives: красного, его, синего)
        # We exclude common adverbs/nouns where 'г' is hard: много, строго, дорого
        hard_g_exceptions = {
            'много', 'немного', 'строго', 'дорого', 'лого', 'иго', 'благо', 'танго',
            'манго', 'лего', 'карго', 'арго', 'индиго', 'фламинго', 'маренго',
            'конго', 'альтер-эго', 'убого', 'полого', 'разноголосо', 'гюго', 'чикаго',
            'живаго', 'сан-диего', 'ого'
        }
        if word.endswith(('ого', 'его')) and word not in hard_g_exceptions:
            # Only replace the 'г' in the last 3 characters
            word = word[:-3] + word[-3:].replace('г', 'в')

        # B. The Г -> Х shift (Specific clusters)
        # e.g., легко -> лехко, мягко -> мяхко
        word = word.replace('легк', 'лехк')
        word = word.replace('мягк', 'мяхк')
        # Also handles comparative: легче -> лехче
        word = word.replace('легч', 'лехч')
        word = word.replace('мягч', 'мяхч')

        # 2. Affricate Merging (Merging two letters into one sound)
        # These are high-impact for naturalness
        word = word.replace('сч', 'щ')   # счастье -> щастье
        word = word.replace('зч', 'щ')   # извозчик -> извощик
        word = word.replace('тч', 'ч')   # отчим -> очим
        word = word.replace('дч', 'ч')   # докладчик -> доклачик
        word = word.replace('дс', 'ц')   # детский -> децкий

        # 3. Additional Silent Consonants
        word = word.replace('рдц', 'рц') # сердце -> серце
        word = word.replace('стл', 'сл') # счастливый -> счасливый
        word = word.replace('нтск', 'нск') # гигантский -> гиганский
        word = word.replace('ндск', 'нск') # голландский -> голланский

        # --- Cluster simplifications (Cyrillic only) ---

        # 'вств' cluster: first 'в' is typically silent in spoken Russian
        # e.g. здравствуйте → здраствуйте
        word = word.replace('вств', 'ств')

        # Reflexive endings: 'тся'/'ться' → 'ца'/'ця'
        # The т+с merge into ц; ь palatalizes it
        word = word.replace('ться', 'ця')
        word = word.replace('тся', 'ца')

        # 'стн', 'здн' — silent consonant clusters (e.g. честный, поздно)
        word = word.replace('стн', 'сн')
        word = word.replace('здн', 'зн')

        word = word.replace('тск', 'цк')   # советский → совецкий, детский → децкий
        word = word.replace('дск', 'цк')   # городской → горо(д→ц)кой

        # 'лнц' — silent л (e.g. солнце)
        word = word.replace('лнц', 'нц')

        # remove combining marks for assimilation logic, but keep the base characters
        word = re.sub(r'[\u0300-\u036f]', '', word.lower())

        # --- Voicing assimilation (regressive, right-to-left) ---
        chars = list(word)
        for i in range(len(chars) - 1):
            cur = chars[i]
            nxt = chars[i + 1]

            # Only assimilate between two Cyrillic consonants
            if cur not in self.consonants or nxt not in self.consonants:
                continue

            if cur in self.voiced_consonants and nxt in self.voiceless_consonants:
                # Devoice: voiced before voiceless
                devoiced = self.voicing_map.get(cur)
                if devoiced and devoiced in self.voiceless_consonants:
                    chars[i] = devoiced

            elif cur in self.voiceless_consonants and nxt in self.voiced_consonants:
                # Voice: voiceless before voiced (except в which doesn't trigger voicing)
                if nxt != 'в':
                    voiced = self.voicing_map.get(cur)
                    if voiced and voiced in self.voiced_consonants:
                        chars[i] = voiced

        # --- Word-final devoicing ---
        if chars and chars[-1] in self.voiced_consonants:
            devoiced = self.voicing_map.get(chars[-1])
            if devoiced and devoiced in self.voiceless_consonants:
                chars[-1] = devoiced

        return ''.join(chars)

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
    def _process_normalized_word(self, word: str) -> Tuple[Tuple[str, ...], StressInfo]:
        """
        Process a single already-normalized word. Cached on the normalized form.
        Returns a tuple of phonemes (not a list) because lru_cache requires hashable
        return values to avoid cache mutation bugs. Callers convert to list at the boundary.
        """
        word_clean = re.sub(r'[\u0300-\u036f]', '', word).lower()
        if word_clean in self.exceptions:
            ipa_string = self.exceptions[word_clean]
            tokenized = tuple(self._tokenize_ipa_string(ipa_string))
            if word_clean in self.stress_patterns:
                syllable_pos = self.stress_patterns[word_clean]
                vowel_index = self._vowel_index_from_syllable(word_clean, syllable_pos)
                stress_info = StressInfo(position=syllable_pos, vowel_index=vowel_index, is_marked=True)
            else:
                stress_info = StressInfo(position=0, vowel_index=0, is_marked=True)

            return tokenized, stress_info

        try:
            stress_info = self.detect_stress(word)
            word_after_assimilation = self.apply_consonant_assimilation(word)
            base_phonemes = self.apply_palatalization(word_after_assimilation)
            final_phonemes = self.apply_vowel_reduction(base_phonemes, stress_info.position)

            return tuple(final_phonemes), stress_info
        except Exception as e:
            logger.error(f"Error processing word '{word}': {e}")

            return tuple(), StressInfo(0, 0, False)

    def process_word(self, word: str) -> Tuple[List[str], StressInfo]:
        """Public entry point — normalizes the word then processes it."""
        if not word:
            return [], StressInfo(0, 0, False)
        normalized = self.normalize_text(word)
        if not normalized:
            return [], StressInfo(0, 0, False)

        phonemes, stress_info = self._process_normalized_word(normalized)
        return list(phonemes), stress_info

    def process_text(self, text: str) -> List[Tuple[str, List[str], StressInfo]]:
        """Process full text and return word-phoneme-stress tuples."""
        if not text:
            return []

        normalized_text = self.normalize_text(text)
        results = []

        for word in normalized_text.split():
            try:
                phonemes, stress_info = self._process_normalized_word(word)
                results.append((word, list(phonemes), stress_info))
            except Exception as e:
                logger.error(f"Error processing word '{word}': {e}")
                results.append((word, [], StressInfo(0, 0, False)))

        return results

    def _tokenize_ipa_string(self, ipa_string: str) -> List[str]:
        """
        Tokenize an IPA string into individual phonemes.
        Improved version with better multi-character phoneme handling.
        """
        if not ipa_string:
            return []

        phonemes = []
        i = 0

        # Single characters (for fallback)
        single_chars = set('pbvmfnlrkgxdʒʃʐzvstchwiaeouɨɐəɪˈˌ') # Common IPA single chars including vowels and stress marks

        while i < len(ipa_string):
            matched = False
            # Try to match longest possible phoneme first
            for mc_ph in self._multi_char_phonemes:
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
        additional_phonemes = {'j', 'ʐ', 'ts', 'tʃ', 'ʃtʃ', 'bʲ', 'vʲ', 'gʲ', 'dʲ', 'zʲ', 'kʲ', 'lʲ', 'mʲ', 'nʲ', 'pʲ', 'rʲ', 'sʲ', 'tʲ', 'fʲ', 'xʲ'}
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
        instance.palatalized = data.get("palatalized", {})

        # Must rebuild after restoring palatalized:
        instance._multi_char_phonemes = sorted(
            list(instance.palatalized.values()) +
            ['ts', 'tʃ', 'ʃtʃ', 'dʑ', 'dz', 'tɕ', 'ɐ', 'ə', 'ɪ', 'ɨ', 'ja', 'jo', 'ju', 'je'],
            key=len, reverse=True
        )
        instance.multi_char_phonemes = instance._multi_char_phonemes

        # Restore all attributes, ensuring sets are converted from lists
        instance.vowels = data.get("vowels", {})
        instance.consonants = data.get("consonants", {})
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
        self.normalize_text.cache_clear()
        self._process_normalized_word.cache_clear()  # this is where real caching happens

    def get_cache_info(self) -> Dict:
        """Get cache statistics for debugging"""
        return {
            "normalize_text_cache": self.normalize_text.cache_info(),
            "_process_normalized_word_cache": self._process_normalized_word.cache_info(),
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
