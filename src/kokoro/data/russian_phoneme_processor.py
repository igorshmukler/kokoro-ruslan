import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import logging

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

    # Terminal / clause punctuation mapped to prosody-conditioning tokens.
    # These tokens are injected into the phoneme sequence at word boundaries
    # so the model can condition duration, pitch and energy on sentence type.
    PUNCT_MAP = {'.': '<period>', '?': '<question>', '!': '<exclaim>', ',': '<comma>'}

    # ── Number-to-words tables (Russian nominative case) ─────────────────────
    _UNITS_M = ('ноль', 'один', 'два', 'три', 'четыре', 'пять', 'шесть',
                'семь', 'восемь', 'девять')
    _UNITS_F = ('ноль', 'одна', 'две', 'три', 'четыре', 'пять', 'шесть',
                'семь', 'восемь', 'девять')
    _TEENS   = ('десять', 'одиннадцать', 'двенадцать', 'тринадцать',
                'четырнадцать', 'пятнадцать', 'шестнадцать', 'семнадцать',
                'восемнадцать', 'девятнадцать')
    _TENS    = ('', '', 'двадцать', 'тридцать', 'сорок', 'пятьдесят',
                'шестьдесят', 'семьдесят', 'восемьдесят', 'девяносто')
    _HUNDREDS = ('', 'сто', 'двести', 'триста', 'четыреста', 'пятьсот',
                 'шестьсот', 'семьсот', 'восемьсот', 'девятьсот')

    # ── Abbreviation expansion table (longest / most specific first) ─────────
    # Each entry is (compiled_regex, replacement_string).
    # Patterns use word boundaries and are case-insensitive Cyrillic.
    _ABBREV_TABLE = [
        (re.compile(r'\bт\.\s*е\.', re.IGNORECASE),   'то есть'),
        (re.compile(r'\bт\.\s*д\.', re.IGNORECASE),   'так далее'),
        (re.compile(r'\bт\.\s*п\.', re.IGNORECASE),   'тому подобное'),
        (re.compile(r'\bмлрд\b',     re.IGNORECASE),   'миллиардов'),
        (re.compile(r'\bмлн\b',      re.IGNORECASE),   'миллионов'),
        (re.compile(r'\bтыс\b',      re.IGNORECASE),   'тысяч'),
        (re.compile(r'\bкм\b',       re.IGNORECASE),   'километров'),
        (re.compile(r'\bкг\b',       re.IGNORECASE),   'килограммов'),
        (re.compile(r'\bмм\b',       re.IGNORECASE),   'миллиметров'),
        (re.compile(r'\bсм\b',       re.IGNORECASE),   'сантиметров'),
        (re.compile(r'\bкв\b',       re.IGNORECASE),   'квадратных'),
        (re.compile(r'\bруб\b',      re.IGNORECASE),   'рублей'),
        (re.compile(r'\bкоп\b',      re.IGNORECASE),   'копеек'),
        (re.compile(r'\bмин\b',      re.IGNORECASE),   'минут'),
        (re.compile(r'\bсек\b',      re.IGNORECASE),   'секунд'),
        (re.compile(r'\bчел\b',      re.IGNORECASE),   'человек'),
        (re.compile(r'\bул\b',       re.IGNORECASE),   'улица'),
        (re.compile(r'\bпр\b',       re.IGNORECASE),   'проспект'),
    ]

    # ── Numeric unit forms: (is_feminine, nominative_sg, genitive_sg, genitive_pl)
    # Used by expand_digits_and_abbrevs to pick the correct case based on number.
    # Multiplier abbreviations (тыс/млн/млрд) are included here so that
    # "N млн" is handled by the same _select_case path as physical units,
    # avoiding the need to compute N×10^6 (which can overflow _int_to_words).
    _UNIT_FORMS: Dict[str, tuple] = {
        # Multipliers
        'млрд': (False, 'миллиард',   'миллиарда',   'миллиардов'),
        'млн':  (False, 'миллион',    'миллиона',    'миллионов'),
        'тыс':  (True,  'тысяча',     'тысячи',      'тысяч'),
        # Physical / monetary units
        'км':   (False, 'километр',   'километра',   'километров'),
        'кг':   (False, 'килограмм',  'килограмма',  'килограммов'),
        'мм':   (False, 'миллиметр',  'миллиметра',  'миллиметров'),
        'см':   (False, 'сантиметр',  'сантиметра',  'сантиметров'),
        'руб':  (False, 'рубль',      'рубля',       'рублей'),
        'коп':  (True,  'копейка',    'копейки',     'копеек'),
        'мин':  (True,  'минута',     'минуты',      'минут'),
        'сек':  (True,  'секунда',    'секунды',     'секунд'),
        'чел':  (False, 'человек',    'человека',    'человек'),
        'г':    (False, 'грамм',      'грамма',      'граммов'),
        'м':    (False, 'метр',       'метра',       'метров'),
        'л':    (False, 'литр',       'литра',       'литров'),
    }

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
            ['ts', 'tʃ', 'ʃtʃ', 'dʑ', 'dz', 'tɕ', 'ɐ', 'ə', 'ɪ', 'ɨ',
             'ja', 'jo', 'ju', 'je', 'jɐ', 'jɪ', 'jə'],
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

        # Per-instance LRU caches — stored on the instance rather than the class so
        # that `self` is not permanently retained in a class-level cache key, which
        # would prevent garbage collection when multiple processor instances are used.
        self.normalize_text = lru_cache(maxsize=1000)(self._normalize_text_impl)
        self._process_normalized_word = lru_cache(maxsize=500)(self._process_normalized_word_impl)

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

    # ── Text pre-processing helpers ───────────────────────────────────────────

    @staticmethod
    def _int_to_words(n: int, feminine: bool = False) -> str:
        """Expand a non-negative integer to Russian words (nominative case).

        Handles 0 – 999 999 999 999 correctly.  Numbers ≥ 10^12 fall back to
        digit-by-digit spelling as a safe upper bound.  The ``feminine`` flag
        switches unit words 1/2 to their feminine forms (одна/две), needed
        for thousands and feminine units (тысяча, минута, etc.).
        """
        if n < 0:
            return 'минус ' + RussianPhonemeProcessor._int_to_words(-n, feminine)
        if n == 0:
            return 'ноль'
        # Extremely large numbers (>999 billion): spell each digit individually
        # as a safe fallback.  All practical numbers (up to hundreds of billions)
        # are covered by the millions/thousands recursion above.
        if n >= 1_000_000_000_000:
            return ' '.join(
                RussianPhonemeProcessor._UNITS_M[int(d)]
                for d in str(n) if d.isdigit()
            )

        units = RussianPhonemeProcessor._UNITS_F if feminine else RussianPhonemeProcessor._UNITS_M
        parts: List[str] = []

        # Billions
        if n >= 1_000_000_000:
            b = n // 1_000_000_000
            n %= 1_000_000_000
            b_word = RussianPhonemeProcessor._int_to_words(b, feminine=False)
            last2, last1 = b % 100, b % 10
            if 11 <= last2 <= 19:     suffix = 'миллиардов'
            elif last1 == 1:          suffix = 'миллиард'
            elif 2 <= last1 <= 4:     suffix = 'миллиарда'
            else:                     suffix = 'миллиардов'
            parts.append(f'{b_word} {suffix}')

        # Millions
        if n >= 1_000_000:
            m = n // 1_000_000
            n %= 1_000_000
            m_word = RussianPhonemeProcessor._int_to_words(m, feminine=False)
            last2, last1 = m % 100, m % 10
            if 11 <= last2 <= 19:     suffix = 'миллионов'
            elif last1 == 1:          suffix = 'миллион'
            elif 2 <= last1 <= 4:     suffix = 'миллиона'
            else:                     suffix = 'миллионов'
            parts.append(f'{m_word} {suffix}')

        # Thousands
        if n >= 1_000:
            k = n // 1_000
            n %= 1_000
            k_word = RussianPhonemeProcessor._int_to_words(k, feminine=True)
            last2, last1 = k % 100, k % 10
            if 11 <= last2 <= 19:     suffix = 'тысяч'
            elif last1 == 1:          suffix = 'тысяча'
            elif 2 <= last1 <= 4:     suffix = 'тысячи'
            else:                     suffix = 'тысяч'
            parts.append(f'{k_word} {suffix}')

        # Hundreds
        if n >= 100:
            parts.append(RussianPhonemeProcessor._HUNDREDS[n // 100])
            n %= 100

        # Tens and units
        if n >= 20:
            parts.append(RussianPhonemeProcessor._TENS[n // 10])
            n %= 10
        if n >= 10:
            parts.append(RussianPhonemeProcessor._TEENS[n - 10])
            n = 0
        if n > 0:
            parts.append(units[n])

        return ' '.join(p for p in parts if p)

    @staticmethod
    def _select_case(n: int, singular: str, paucal: str, plural: str) -> str:
        """Pick the correct Russian noun form based on the governing number.

        Returns *singular* for 1, *paucal* (genitive singular) for 2-4,
        and *plural* (genitive plural) for 0 and 5+.  Teen numbers (11-19)
        always take the plural form.
        """
        last2 = abs(n) % 100
        if 11 <= last2 <= 19:
            return plural
        last1 = abs(n) % 10
        if last1 == 1:
            return singular
        if 2 <= last1 <= 4:
            return paucal
        return plural

    def expand_digits_and_abbrevs(self, text: str) -> str:
        """Pre-pass applied *before* ``normalize_text``.

        Expands:

        * **Cardinal digits** – ``"12 км"`` → ``"двенадцать километров"``
        * **Common Cyrillic abbreviations** – ``"руб"`` → ``"рублей"``

        Punctuation is intentionally left intact so that
        ``_extract_punct_after_words()`` can still capture it afterwards.
        Numbers ≥ 10^12 fall back to digit-by-digit spelling.
        """
        if not text:
            return text

        # 1. Expand numeric unit/multiplier compounds: "N км" → "пять километров",
        #    "2 тыс" → "две тысячи", "100 млн" → "сто миллионов".
        #    Unit keys are sorted by length (longest first) to prevent
        #    single-char keys like "м" from shadowing "мм", "мин", "млн", etc.
        def _expand_unit(m: re.Match) -> str:
            n = int(m.group(1))
            fem, sg, pauc, pl = self._UNIT_FORMS[m.group(2).lower()]
            num_word  = self._int_to_words(n, feminine=fem)
            unit_word = self._select_case(n, sg, pauc, pl)
            return f'{num_word} {unit_word}'

        _unit_keys = '|'.join(sorted(self._UNIT_FORMS, key=len, reverse=True))
        text = re.sub(
            rf'(\d+)\s*({_unit_keys})\b',
            _expand_unit,
            text,
            flags=re.IGNORECASE | re.UNICODE,
        )

        # 3. Expand remaining non-numeric abbreviations (т.е., ул., etc.) and
        #    standalone unit/multiplier abbreviations without a preceding digit.
        for pattern, replacement in self._ABBREV_TABLE:
            text = pattern.sub(replacement, text)

        # 4. Expand any remaining bare digit runs.
        text = re.sub(r'\d+', lambda m: self._int_to_words(int(m.group())), text)

        return text

    def _normalize_text_impl(self, text: str) -> str:
        """
        Normalize Russian text for phoneme processing.
        Called through the per-instance LRU cache ``self.normalize_text``.
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
            elif char == '\u0306':
                # Combining breve — the NFD decomposition of й is и + U+0306.
                # Preserve it so the NFC recomposition below can restore й.
                clean_text_chars.append(char)
            # else: skip other non-allowed combining marks or punctuation

        text = unicodedata.normalize('NFC', ''.join(clean_text_chars))

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
                    is_iotated = ph.startswith('j')
                    base = ph[1:] if is_iotated else ph
                    if syllable < stress_syllable_idx:
                        if dist == 1:
                            reduced_base = 'ɐ' if base in ('o', 'a') else 'ɪ' if base in ('e', 'i') else None
                        else:
                            reduced_base = 'ə' if base in ('o', 'a', 'e', 'i') else None
                    else:
                        reduced_base = 'ə' if base in ('o', 'a', 'e', 'i') else None
                    if reduced_base is not None:
                        reduced[i] = ('j' + reduced_base) if is_iotated else reduced_base
                syllable += 1
        return reduced

    def apply_consonant_assimilation(self, word: str) -> str:
        """
        Apply voicing assimilation and consonant cluster simplifications.
        All substitutions remain in Cyrillic — IPA conversion happens downstream
        in apply_palatalization.
        """
        word = word.lower()
        # Strip combining marks (stress marks, etc.) before any Cyrillic substitutions
        # so that cluster patterns like 'вств' match even when a vowel in the word
        # carries an explicit stress mark that would otherwise split the sequence.
        word = re.sub(r'[\u0300-\u036f]', '', word)

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
        word = word.replace('отч', 'оч')   # отчим, отчаянный — prefix от+ч
        word = word.replace('дчик', 'чик')   # докладчик, переводчик


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



    def _process_normalized_word_impl(self, word: str) -> Tuple[Tuple[str, ...], StressInfo]:
        """
        Process a single already-normalized word. Called through the per-instance LRU
        cache ``self._process_normalized_word``.
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

    @staticmethod
    def _extract_punct_after_words(text: str) -> List[Optional[str]]:
        """
        Scan *raw* text character-by-character and return, in order, the first
        PUNCT_MAP token that appears after each Cyrillic word before the next
        Cyrillic word (or end of string).  Returns None for words with no
        following punctuation.

        Example:  "Привет, как дела?"  →  ['<comma>', None, '<question>']
        """
        punct_map = RussianPhonemeProcessor.PUNCT_MAP
        result: List[Optional[str]] = []
        i = 0
        n = len(text)
        while i < n:
            # Advance past non-Cyrillic characters
            if not ('\u0400' <= text[i] <= '\u04FF'):
                i += 1
                continue
            # Consume one Cyrillic word (base letters + combining stress marks)
            while i < n and ('\u0400' <= text[i] <= '\u04FF' or text[i] in '\u0301\u0300\u0341'):
                i += 1
            # Collect the first matching punctuation before the next Cyrillic word
            punct: Optional[str] = None
            while i < n and not ('\u0400' <= text[i] <= '\u04FF'):
                if punct is None and text[i] in punct_map:
                    punct = punct_map[text[i]]
                i += 1
            result.append(punct)
        return result

    def process_text(self, text: str) -> List[Tuple]:
        """
        Process full text and return tuples of
        ``(word, phonemes, stress_info, punct_token_or_None)``.

        The optional 4th element is a prosody token from PUNCT_MAP ('<period>',
        '<question>', '<exclaim>', '<comma>') when a punctuation mark follows
        the word in the raw input, otherwise ``None``.  Callers that only need
        the first three fields can use extended unpacking::

            word, phonemes, stress, *_ = item
        """
        if not text:
            return []

        # Pre-pass: expand digits and abbreviations while punctuation is still
        # present, so that _extract_punct_after_words sees the correct token
        # boundaries and normalize_text receives clean Cyrillic words.
        text = self.expand_digits_and_abbrevs(text)

        # Extract punctuation associations from the raw text BEFORE normalization
        # strips all non-Cyrillic characters.
        punct_list = self._extract_punct_after_words(text)

        normalized_text = self.normalize_text(text)
        results = []

        for idx, word in enumerate(normalized_text.split()):
            try:
                phonemes, stress_info = self._process_normalized_word(word)
            except Exception as e:
                logger.error(f"Error processing word '{word}': {e}")
                phonemes, stress_info = (), StressInfo(0, 0, False)
            punct_token: Optional[str] = punct_list[idx] if idx < len(punct_list) else None
            results.append((word, list(phonemes), stress_info, punct_token))

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

        for word_orig, phonemes, stress_info, *_ in results:
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

        # ADD SPECIAL TOKENS FIRST
        phoneme_set.update(['<pad>', '<sil>', '<sp>'])

        # Prosody-conditioning punctuation tokens
        phoneme_set.update(['<period>', '<question>', '<exclaim>', '<comma>'])

        # Add base phonemes
        phoneme_set.update(self.vowels.values())
        phoneme_set.update(self.consonants.values())
        phoneme_set.update(self.palatalized.values())

        # Add reduced vowels (plain and iotated)
        phoneme_set.update(['ə', 'ɪ', 'ɐ', 'jɐ', 'jɪ', 'jə'])

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

        for word, phonemes, *_ in results:
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
        # Rebuild to include iotated reduced vowels added after the original
        # _multi_char_phonemes was serialised.
        instance._multi_char_phonemes = sorted(
            list(instance.palatalized.values()) +
            ['ts', 'tʃ', 'ʃtʃ', 'dʑ', 'dz', 'tɕ', 'ɐ', 'ə', 'ɪ', 'ɨ',
             'ja', 'jo', 'ju', 'je', 'jɐ', 'jɪ', 'jə'],
            key=len, reverse=True
        )

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

        # Forward-compatibility patch: if the pickled vocab pre-dates the
        # prosody-token addition or the iotated-reduced-vowel addition,
        # inject the missing tokens with fresh IDs so they are always usable
        # at inference time without retraining.
        punct_tokens = list(RussianPhonemeProcessor.PUNCT_MAP.values())
        # Also ensure the base special tokens and reduced iotated vowels are present.
        required_tokens = ['<pad>', '<sil>', '<sp>'] + punct_tokens + ['jɐ', 'jɪ', 'jə']
        next_id = max(instance.phoneme_to_id.values(), default=-1) + 1
        for tok in required_tokens:
            if tok not in instance.phoneme_to_id:
                instance.phoneme_to_id[tok] = next_id
                next_id += 1

        # Flush any cached results from __init__'s default state; the restored
        # exceptions/stress_patterns may differ from the constructor defaults.
        instance.clear_cache()
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
    for word, phonemes, stress_info, *_ in results:
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
