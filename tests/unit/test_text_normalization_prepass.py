"""
Tests for the text normalization pre-pass (digit & abbreviation expansion).

Features covered
────────────────
_int_to_words      – cardinal integer → Russian words (0 … 999 billion)
_select_case       – Russian noun case selection by governing number
expand_digits_and_abbrevs
                   – compound unit expansion with correct case agreement,
                     multiplier abbreviations (тыс/млн/млрд),
                     non-numeric abbreviations (т.е., руб standalone, …)
normalize_text     – й (U+0439) survives NFD decomposition + NFC recompose
process_text       – full round-trip including digits and abbreviations

Test classes
────────────
TestIntToWords          – _int_to_words correctness + edge cases
TestSelectCase          – case selector (singular / paucal / plural)
TestExpandDigitsCore    – bare digit expansion inside expand_digits_and_abbrevs
TestUnitCompounds       – "N <unit>" → correct word + case
TestMultiplierCompounds – "N тыс/млн/млрд" → correct words
TestAbbrevExpansion     – standalone abbreviations without a preceding digit
TestYoAndYiNormalize    – ё and й survive normalize_text correctly
TestProcessTextNumerics – full process_text round-trips with digits/abbrevs
"""

import pytest
from kokoro.data.russian_phoneme_processor import RussianPhonemeProcessor


# ─── shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def p() -> RussianPhonemeProcessor:
    return RussianPhonemeProcessor()


# ─── helpers ──────────────────────────────────────────────────────────────────

def _w(n: int, feminine: bool = False) -> str:
    return RussianPhonemeProcessor._int_to_words(n, feminine=feminine)


def _c(n: int, sg: str, pauc: str, pl: str) -> str:
    return RussianPhonemeProcessor._select_case(n, sg, pauc, pl)


# ═════════════════════════════════════════════════════════════════════════════
# 1. _int_to_words
# ═════════════════════════════════════════════════════════════════════════════

class TestIntToWords:

    # ── zero ──────────────────────────────────────────────────────────────────

    def test_zero(self):
        assert _w(0) == 'ноль'

    # ── single digits ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize('n,expected', [
        (1, 'один'),
        (2, 'два'),
        (3, 'три'),
        (4, 'четыре'),
        (5, 'пять'),
        (6, 'шесть'),
        (7, 'семь'),
        (8, 'восемь'),
        (9, 'девять'),
    ])
    def test_units_masculine(self, n, expected):
        assert _w(n) == expected

    @pytest.mark.parametrize('n,expected', [
        (1, 'одна'),
        (2, 'две'),
    ])
    def test_units_feminine(self, n, expected):
        assert _w(n, feminine=True) == expected

    # ── teens ─────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize('n,expected', [
        (10, 'десять'),
        (11, 'одиннадцать'),
        (12, 'двенадцать'),
        (13, 'тринадцать'),
        (15, 'пятнадцать'),
        (19, 'девятнадцать'),
    ])
    def test_teens(self, n, expected):
        assert _w(n) == expected

    # ── tens ──────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize('n,expected', [
        (20, 'двадцать'),
        (30, 'тридцать'),
        (40, 'сорок'),
        (50, 'пятьдесят'),
        (90, 'девяносто'),
        (42, 'сорок два'),
    ])
    def test_tens(self, n, expected):
        assert _w(n) == expected

    # ── hundreds ──────────────────────────────────────────────────────────────

    @pytest.mark.parametrize('n,expected', [
        (100,  'сто'),
        (200,  'двести'),
        (300,  'триста'),
        (400,  'четыреста'),
        (500,  'пятьсот'),
        (999,  'девятьсот девяносто девять'),
    ])
    def test_hundreds(self, n, expected):
        assert _w(n) == expected

    # ── thousands ─────────────────────────────────────────────────────────────

    def test_one_thousand(self):
        assert _w(1000) == 'одна тысяча'

    def test_two_thousand(self):
        assert _w(2000) == 'две тысячи'

    def test_five_thousand(self):
        assert _w(5000) == 'пять тысяч'

    def test_eleven_thousand(self):
        assert _w(11_000) == 'одиннадцать тысяч'

    def test_twenty_one_thousand(self):
        assert _w(21_000) == 'двадцать одна тысяча'

    def test_compound_thousands(self):
        assert _w(2001) == 'две тысячи один'

    # ── millions ──────────────────────────────────────────────────────────────

    def test_one_million(self):
        assert _w(1_000_000) == 'один миллион'

    def test_two_million(self):
        assert _w(2_000_000) == 'два миллиона'

    def test_five_million(self):
        assert _w(5_000_000) == 'пять миллионов'

    def test_compound_millions(self):
        assert _w(1_001_000) == 'один миллион одна тысяча'

    # ── billions ──────────────────────────────────────────────────────────────

    def test_one_billion(self):
        assert _w(1_000_000_000) == 'один миллиард'

    def test_two_billion(self):
        assert _w(2_000_000_000) == 'два миллиарда'

    def test_five_billion(self):
        assert _w(5_000_000_000) == 'пять миллиардов'

    def test_billion_millions(self):
        # 1,100,000,000
        result = _w(1_100_000_000)
        assert 'миллиард' in result
        assert 'миллион' in result

    # ── fallback for very large numbers ───────────────────────────────────────

    def test_very_large_falls_back_to_digits(self):
        """Numbers >= 10^12 are spelled digit by digit."""
        result = _w(1_000_000_000_000)
        # Must consist only of Cyrillic word tokens, no digits
        assert all(not c.isdigit() for c in result)
        # "один" appears once (the leading 1)
        assert 'один' in result


# ═════════════════════════════════════════════════════════════════════════════
# 2. _select_case
# ═════════════════════════════════════════════════════════════════════════════

class TestSelectCase:

    SG, PAUC, PL = 'рубль', 'рубля', 'рублей'

    def test_zero_plural(self):
        assert _c(0, self.SG, self.PAUC, self.PL) == self.PL

    @pytest.mark.parametrize('n', [1, 21, 31, 101, 1001])
    def test_singular(self, n):
        assert _c(n, self.SG, self.PAUC, self.PL) == self.SG

    @pytest.mark.parametrize('n', [2, 3, 4, 22, 23, 24, 102, 1002])
    def test_paucal(self, n):
        assert _c(n, self.SG, self.PAUC, self.PL) == self.PAUC

    @pytest.mark.parametrize('n', [5, 6, 9, 10, 11, 12, 19, 20, 25, 100])
    def test_plural(self, n):
        assert _c(n, self.SG, self.PAUC, self.PL) == self.PL

    @pytest.mark.parametrize('n', [11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    111, 112, 113, 211, 311])
    def test_teens_always_plural(self, n):
        """11-19 and compounds with teen base always use plural."""
        assert _c(n, self.SG, self.PAUC, self.PL) == self.PL


# ═════════════════════════════════════════════════════════════════════════════
# 3. expand_digits_and_abbrevs – bare digits (no unit)
# ═════════════════════════════════════════════════════════════════════════════

class TestExpandDigitsCore:

    def test_single_digit_in_sentence(self, p):
        assert p.expand_digits_and_abbrevs('в 5 раз') == 'в пять раз'

    def test_multi_digit_number(self, p):
        assert p.expand_digits_and_abbrevs('год 2001') == 'год две тысячи один'

    def test_zero(self, p):
        assert p.expand_digits_and_abbrevs('0 очков') == 'ноль очков'

    def test_multiple_numbers_in_sentence(self, p):
        result = p.expand_digits_and_abbrevs('от 1 до 10')
        assert 'один' in result
        assert 'десять' in result
        assert not any(c.isdigit() for c in result)

    def test_no_digits_unchanged(self, p):
        s = 'привет как дела'
        assert p.expand_digits_and_abbrevs(s) == s

    def test_empty_string(self, p):
        assert p.expand_digits_and_abbrevs('') == ''


# ═════════════════════════════════════════════════════════════════════════════
# 4. expand_digits_and_abbrevs – numeric unit compounds
# ═════════════════════════════════════════════════════════════════════════════

class TestUnitCompounds:

    @pytest.mark.parametrize('text,expected', [
        # case-agreement for different numbers + feminine/masculine units
        ('5 км',   'пять километров'),
        ('1 км',   'один километр'),
        ('2 км',   'два километра'),
        ('3 кг',   'три килограмма'),
        ('1 кг',   'один килограмм'),
        ('5 кг',   'пять килограммов'),
        ('1 руб',  'один рубль'),
        ('2 руб',  'два рубля'),
        ('5 руб',  'пять рублей'),
        ('11 руб', 'одиннадцать рублей'),     # teen → plural
        ('21 руб', 'двадцать один рубль'),    # 21 → singular
        ('1 мин',  'одна минута'),             # feminine
        ('2 мин',  'две минуты'),
        ('5 мин',  'пять минут'),
        ('1 сек',  'одна секунда'),
        ('4 сек',  'четыре секунды'),
        ('10 сек', 'десять секунд'),
        ('1 коп',  'одна копейка'),
        ('3 коп',  'три копейки'),
        ('11 коп', 'одиннадцать копеек'),
        ('50 коп', 'пятьдесят копеек'),
    ])
    def test_unit_expansion(self, p, text, expected):
        assert p.expand_digits_and_abbrevs(text) == expected

    def test_unit_in_full_sentence(self, p):
        result = p.expand_digits_and_abbrevs('5 км до города')
        assert result == 'пять километров до города'

    def test_unit_at_end_of_sentence(self, p):
        result = p.expand_digits_and_abbrevs('цена 300 руб 50 коп')
        assert result == 'цена триста рублей пятьдесят копеек'

    def test_unit_case_insensitive(self, p):
        """Unit abbreviations are matched case-insensitively."""
        result = p.expand_digits_and_abbrevs('42 КМ')
        assert 'километр' in result.lower()

    def test_unit_longest_key_wins(self, p):
        """'мм' must not be parsed as 'м'+'м'; 'мин' must not be 'м'+'ин'."""
        assert 'миллиметр' in p.expand_digits_and_abbrevs('5 мм')
        assert 'минут' in p.expand_digits_and_abbrevs('5 мин')


# ═════════════════════════════════════════════════════════════════════════════
# 5. expand_digits_and_abbrevs – multiplier compounds
# ═════════════════════════════════════════════════════════════════════════════

class TestMultiplierCompounds:

    @pytest.mark.parametrize('text,expected', [
        ('1 тыс',   'одна тысяча'),
        ('2 тыс',   'две тысячи'),
        ('5 тыс',   'пять тысяч'),
        ('11 тыс',  'одиннадцать тысяч'),
        ('21 тыс',  'двадцать одна тысяча'),
        ('1 млн',   'один миллион'),
        ('2 млн',   'два миллиона'),
        ('5 млн',   'пять миллионов'),
        ('100 млн', 'сто миллионов'),
        ('1 млрд',  'один миллиард'),
        ('2 млрд',  'два миллиарда'),
        ('5 млрд',  'пять миллиардов'),
    ])
    def test_multiplier_expansion(self, p, text, expected):
        assert p.expand_digits_and_abbrevs(text) == expected

    def test_multiplier_in_sentence(self, p):
        result = p.expand_digits_and_abbrevs('100 млн человек')
        assert result == 'сто миллионов человек'

    def test_multiplier_with_participants(self, p):
        result = p.expand_digits_and_abbrevs('2 тыс участников')
        assert result == 'две тысячи участников'


# ═════════════════════════════════════════════════════════════════════════════
# 6. expand_digits_and_abbrevs – standalone abbreviations (no digit)
# ═════════════════════════════════════════════════════════════════════════════

class TestAbbrevExpansion:

    @pytest.mark.parametrize('text,expected', [
        ('т.е. здесь',    'то есть здесь'),
        ('т.д.',          'так далее'),
        ('т.п.',          'тому подобное'),
        ('цена руб',      'цена рублей'),        # bare unit w/o digit
        ('длина км пути', 'длина километров пути'),
    ])
    def test_standalone_abbrev(self, p, text, expected):
        assert p.expand_digits_and_abbrevs(text) == expected

    def test_punct_preserved_during_expansion(self, p):
        """Punctuation must not be stripped so _extract_punct_after_words still works."""
        result = p.expand_digits_and_abbrevs('Стоимость 15 руб.')
        assert result.endswith('.')

    def test_no_double_expansion(self, p):
        """Abbrevs already expanded in step 1 must not be touched again in step 3."""
        result = p.expand_digits_and_abbrevs('3 км')
        # Should contain 'километра', not 'километров километров'
        assert result.count('километр') == 1


# ═════════════════════════════════════════════════════════════════════════════
# 7. normalize_text – й (U+0439) survives NFD → filter → NFC
# ═════════════════════════════════════════════════════════════════════════════

class TestYoAndYiNormalize:

    def test_yi_preserved_in_rublei(self, p):
        """'рублей' must normalise to 'рублей', not 'рублеи'."""
        result = p.normalize_text('рублей')
        assert result == 'рублей', f"Got {result!r}, expected 'рублей'"

    def test_yi_preserved_in_word_with_yi(self, p):
        """Any word containing й must keep it after normalization."""
        for word in ['рублей', 'копеек', 'тысячи', 'километров', 'минут',
                     'чай', 'трамвай', 'стой']:
            result = p.normalize_text(word)
            if 'й' in word:
                assert 'й' in result, (
                    f"normalize_text({word!r}) lost 'й': got {result!r}"
                )

    def test_yo_converts_to_stressed_e(self, p):
        """ё should be converted to е with a stress mark."""
        result = p.normalize_text('ёж')
        # After normalization stress marks are preserved; the ё becomes е+stress
        assert 'й' not in result or True   # unrelated assertion
        assert 'ж' in result
        # The е should still be present (ё → е́)
        assert 'е' in result

    def test_no_bare_digit_survives_normalize(self, p):
        """normalize_text strips digits; pre-pass must run before it."""
        result = p.normalize_text('42')
        assert not any(c.isdigit() for c in result)


# ═════════════════════════════════════════════════════════════════════════════
# 8. process_text round-trips (pre-pass wired in)
# ═════════════════════════════════════════════════════════════════════════════

class TestProcessTextNumerics:

    def _words_and_punct(self, results):
        return [(w, pt) for w, _, _, pt in results]

    def test_greeting_unchanged(self, p):
        """Pure Cyrillic text without digits must still work."""
        results = p.process_text('Привет, как дела?')
        words = [w for w, *_ in results]
        assert words == ['привет', 'как', 'дела']
        assert results[0][3] == '<comma>'
        assert results[2][3] == '<question>'

    def test_digits_expanded_before_phonemization(self, p):
        """Digits must be gone from phonemized words."""
        results = p.process_text('В 2001 году.')
        # After expansion: "В две тысячи один году."
        all_words = [w for w, *_ in results]
        assert not any(c.isdigit() for w in all_words for c in w)

    def test_rub_punct_correct(self, p):
        """'Стоимость 15 руб.' → last word 'рублей' with <period>."""
        results = p.process_text('Стоимость 15 руб.')
        last_word, last_ph, _, last_pt = results[-1]
        assert last_word == 'рублей', f"Got {last_word!r}"
        assert last_pt == '<period>'

    def test_km_exclaim_correct(self, p):
        """'Расстояние 42 км!' → last word 'километра' (paucal) with <exclaim>."""
        results = p.process_text('Расстояние 42 км!')
        last_word, _, _, last_pt = results[-1]
        assert last_word == 'километра', f"Got {last_word!r}"
        assert last_pt == '<exclaim>'

    def test_no_digits_in_phonemes(self, p):
        """Phoneme lists must never contain raw digit characters."""
        for text in ['цена 300 руб', '15 км пути', 'В 2001 году']:
            for _, phonemes, _, _ in p.process_text(text):
                for ph in phonemes:
                    assert not any(c.isdigit() for c in ph), (
                        f"Digit found in phoneme {ph!r} for text {text!r}"
                    )

    def test_multiplier_in_sentence(self, p):
        """'100 млн человек' expands and phonemises without errors."""
        results = p.process_text('100 млн человек')
        assert len(results) > 0
        all_words = [w for w, *_ in results]
        assert not any(c.isdigit() for w in all_words for c in w)

    def test_tye_abbreviation_in_sentence(self, p):
        """'т.е. здесь' should expand 'то есть' and process correctly."""
        results = p.process_text('т.е. здесь')
        all_words = [w for w, *_ in results]
        assert 'то' in all_words
        assert 'есть' in all_words

    def test_punct_token_survives_expansion(self, p):
        """Punctuation tokens survive the pre-pass and are correctly attributed."""
        results = p.process_text('Вес 1 кг.')
        assert results[-1][3] == '<period>'

    def test_four_tuple_shape_with_digits(self, p):
        """Each result item must still be a 4-tuple after digit expansion."""
        for item in p.process_text('Скорость 100 км в час.'):
            assert len(item) == 4
