"""
Tests for the stress parallel embedding feature.

Changes covered:
  audio_utils.py  (PhonemeProcessorUtils)
    - get_stress_indices_with_sil()  — parallel-to-phoneme stress IDs
      0=unstressed/special, 1=primary_stress, 2=secondary_stress (reserved)

  model/model.py  (KokoroModel)
    - stress_embedding  nn.Embedding(3, hidden_dim, padding_idx=0)
    - encode_text()     adds stress embedding to text_emb when provided
    - forward_training/forward_inference/forward  accept stress_indices=None

  dataset.py
    - FEATURE_CACHE_VERSION == 6
    - __getitem__ returns 'stress_indices' parallel to 'phoneme_indices'
    - collate_fn batches 'stress_indices' → (B, P) LongTensor

Test classes
────────────
TestGetStressIndicesBasic        – consonants→0, stressed vowel→1
TestGetStressIndicesSilInjection – <sil> boundary tokens are 0
TestGetStressIndicesPunct        – punct tokens are 0
TestGetStressIndicesLength       – output length matches flatten_phoneme_output_with_sil
TestStressEmbeddingExists        – model has stress_embedding when flag=True
TestStressEmbeddingDisabled      – model has no stress_embedding when flag=False
TestEncodeTextStressAdded        – encode_text changes output when stress_indices given
TestEncodeTextStressNone         – stress_indices=None is a no-op (backward compat)
TestCollateStressIndices         – collate_fn produces (B, P) LongTensor
TestCacheVersion                 – FEATURE_CACHE_VERSION == 6
"""

import pytest
import torch
import torch.nn as nn
from typing import List

from kokoro.data.audio_utils import PhonemeProcessorUtils
from kokoro.data.russian_phoneme_processor import StressInfo
from kokoro.data.dataset import FEATURE_CACHE_VERSION, collate_fn
from kokoro.model.model import KokoroModel


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_raw(words_phonemes_stress, punct_tokens=None):
    """Build a process_text()-compatible list.

    Args:
        words_phonemes_stress: list of (phoneme_list, StressInfo)
        punct_tokens: optional list of punct-or-None, one per word
    """
    if punct_tokens is None:
        punct_tokens = [None] * len(words_phonemes_stress)
    return [
        (f"word{i}", phs, si, pt)
        for i, ((phs, si), pt) in enumerate(zip(words_phonemes_stress, punct_tokens))
    ]


# VOCAB that includes <sil> so sil-injection logic is exercised
VOCAB_WITH_SIL = {
    "<sil>": 0,
    "<period>": 1,
    "p": 2, "r": 3, "a": 4, "t": 5, "m": 6, "u": 7, "i": 8,
    "b": 9, "o": 10, "n": 11,
}

# VOCAB without <sil> (fallback path)
VOCAB_NO_SIL = {k: v for k, v in VOCAB_WITH_SIL.items() if k != "<sil>"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. TestGetStressIndicesBasic
# ─────────────────────────────────────────────────────────────────────────────
class TestGetStressIndicesBasic:
    """Consonants → 0, stressed vowel → 1, other vowels → 0."""

    def test_single_word_cvc_stress_on_only_vowel(self):
        # 'p' 'a' 't'  — stress position 0 = the 0th vowel → 'a' gets 1
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([(["p", "a", "t"], si)])
        result = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB_WITH_SIL)
        assert result == [0, 1, 0]

    def test_single_word_two_vowels_stress_first(self):
        # 'm' 'a' 't' 'u'  — stress position 0 → 'a' stressed
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([(["m", "a", "t", "u"], si)])
        result = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB_WITH_SIL)
        assert result == [0, 1, 0, 0]

    def test_single_word_two_vowels_stress_second(self):
        # 'm' 'a' 't' 'u'  — stress position 1 → 'u' stressed
        si = StressInfo(position=1, vowel_index=0, is_marked=False)
        raw = _make_raw([(["m", "a", "t", "u"], si)])
        result = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB_WITH_SIL)
        assert result == [0, 0, 0, 1]

    def test_all_consonants_all_zeros(self):
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([(["p", "r", "t"], si)])
        result = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB_WITH_SIL)
        assert result == [0, 0, 0]

    def test_only_vowel_is_stressed(self):
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([(["a"], si)])
        result = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB_WITH_SIL)
        assert result == [1]


# ─────────────────────────────────────────────────────────────────────────────
# 2. TestGetStressIndicesSilInjection
# ─────────────────────────────────────────────────────────────────────────────
class TestGetStressIndicesSilInjection:
    """<sil> boundary tokens receive stress-ID 0."""

    def test_two_words_sil_between_them(self):
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([
            (["p", "a"], si),   # word 0 → tokens: p a
            (["t", "u"], si),   # word 1 → tokens: <sil> t u
        ])
        result = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB_WITH_SIL)
        # expected: [0, 1, 0, 1, 0]  →  p a <sil> t u
        assert result[2] == 0, "sil token should have stress-ID 0"
        assert len(result) == 5

    def test_sil_not_injected_when_vocab_has_no_sil(self):
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([
            (["p", "a"], si),
            (["t", "u"], si),
        ])
        result_no_sil = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB_NO_SIL)
        result_with_sil = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB_WITH_SIL)
        # Without sil vocab → 4 tokens; with sil vocab → 5 tokens
        assert len(result_no_sil) == 4
        assert len(result_with_sil) == 5

    def test_first_word_has_no_preceding_sil(self):
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([(["a"], si)])
        result = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB_WITH_SIL)
        # single word → no leading sil
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. TestGetStressIndicesPunct
# ─────────────────────────────────────────────────────────────────────────────
class TestGetStressIndicesPunct:
    """Punctuation tokens receive stress-ID 0."""

    def test_punct_after_word_is_zero(self):
        VOCAB = {**VOCAB_WITH_SIL, "<period>": 12}
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([(["p", "a"], si)], punct_tokens=["<period>"])
        result = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB)
        assert result[-1] == 0, "punct token should have stress-ID 0"
        assert len(result) == 3  # p a <period>

    def test_punct_does_not_shift_sil(self):
        VOCAB = {**VOCAB_WITH_SIL, "<comma>": 13}
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([
            (["p", "a"], si),
            (["t", "u"], si),
        ], punct_tokens=["<comma>", None])
        result = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, VOCAB)
        # word0: p a <comma>  →  3 tokens
        # word1: <sil> t u    →  3 tokens
        assert len(result) == 6
        assert result[2] == 0  # <comma>
        assert result[3] == 0  # <sil>


# ─────────────────────────────────────────────────────────────────────────────
# 4. TestGetStressIndicesLength
# ─────────────────────────────────────────────────────────────────────────────
class TestGetStressIndicesLength:
    """Output length must exactly match flatten_phoneme_output_with_sil."""

    def _check_lengths_match(self, raw, vocab):
        flat = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(raw, vocab)
        stress = PhonemeProcessorUtils.get_stress_indices_with_sil(raw, vocab)
        assert len(flat) == len(stress), (
            f"length mismatch: flatten={len(flat)}, stress={len(stress)}\n"
            f"flat={flat}\nstress={stress}"
        )

    def test_single_word_no_punct(self):
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([(["m", "a", "p"], si)])
        self._check_lengths_match(raw, VOCAB_WITH_SIL)

    def test_two_words_with_sil(self):
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([
            (["p", "a"], si),
            (["b", "u"], si),
        ])
        self._check_lengths_match(raw, VOCAB_WITH_SIL)

    def test_two_words_punct_first(self):
        VOCAB = {**VOCAB_WITH_SIL, "<comma>": 13}
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([
            (["p", "a"], si),
            (["b", "u"], si),
        ], punct_tokens=["<comma>", None])
        self._check_lengths_match(raw, VOCAB)

    def test_three_words_punct_last(self):
        VOCAB = {**VOCAB_WITH_SIL, "<period>": 12}
        si = StressInfo(position=0, vowel_index=0, is_marked=False)
        raw = _make_raw([
            (["a"], si),
            (["b", "o"], si),
            (["r", "i"], si),
        ], punct_tokens=[None, None, "<period>"])
        self._check_lengths_match(raw, VOCAB)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TestStressEmbeddingExists
# ─────────────────────────────────────────────────────────────────────────────
class TestStressEmbeddingExists:
    """When use_stress_embedding=True, model has a stress_embedding layer."""

    def test_attribute_present(self):
        model = KokoroModel(vocab_size=20, hidden_dim=64, n_encoder_layers=1,
                            n_decoder_layers=1, use_stress_embedding=True)
        assert hasattr(model, 'stress_embedding')

    def test_is_nn_embedding(self):
        model = KokoroModel(vocab_size=20, hidden_dim=64, n_encoder_layers=1,
                            n_decoder_layers=1, use_stress_embedding=True)
        assert isinstance(model.stress_embedding, nn.Embedding)

    def test_embedding_shape(self):
        hidden = 64
        model = KokoroModel(vocab_size=20, hidden_dim=hidden, n_encoder_layers=1,
                            n_decoder_layers=1, use_stress_embedding=True)
        assert model.stress_embedding.num_embeddings == 3
        assert model.stress_embedding.embedding_dim == hidden

    def test_padding_idx_is_zero(self):
        model = KokoroModel(vocab_size=20, hidden_dim=64, n_encoder_layers=1,
                            n_decoder_layers=1, use_stress_embedding=True)
        assert model.stress_embedding.padding_idx == 0

    def test_flag_stored(self):
        model = KokoroModel(vocab_size=20, hidden_dim=64, n_encoder_layers=1,
                            n_decoder_layers=1, use_stress_embedding=True)
        assert model.use_stress_embedding is True


# ─────────────────────────────────────────────────────────────────────────────
# 6. TestStressEmbeddingDisabled
# ─────────────────────────────────────────────────────────────────────────────
class TestStressEmbeddingDisabled:
    """When use_stress_embedding=False, the layer must not exist."""

    def test_attribute_absent(self):
        model = KokoroModel(vocab_size=20, hidden_dim=64, n_encoder_layers=1,
                            n_decoder_layers=1, use_stress_embedding=False)
        assert not hasattr(model, 'stress_embedding')

    def test_flag_stored(self):
        model = KokoroModel(vocab_size=20, hidden_dim=64, n_encoder_layers=1,
                            n_decoder_layers=1, use_stress_embedding=False)
        assert model.use_stress_embedding is False


# ─────────────────────────────────────────────────────────────────────────────
# 7. TestEncodeTextStressAdded
# ─────────────────────────────────────────────────────────────────────────────
class TestEncodeTextStressAdded:
    """encode_text output differs when stress_indices contains 1s."""

    def _make_model(self):
        return KokoroModel(vocab_size=20, hidden_dim=32, n_encoder_layers=1,
                           n_decoder_layers=1, use_stress_embedding=True)

    def test_output_differs_with_stress(self):
        model = self._make_model()
        model.eval()
        ph = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)  # (1, 4)
        stress_zero  = torch.zeros(1, 4, dtype=torch.long)
        stress_one   = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)

        with torch.no_grad():
            out_zero = model.encode_text(ph, stress_indices=stress_zero)
            out_one  = model.encode_text(ph, stress_indices=stress_one)

        # The stressed position should differ; the whole tensor need not be equal
        assert not torch.equal(out_zero, out_one)

    def test_output_same_when_stress_all_zero(self):
        """If all stress IDs are 0, padding_idx means embedding values are 0 — same as None branch."""
        model = self._make_model()
        model.eval()
        ph = torch.tensor([[2, 3]], dtype=torch.long)
        stress_zero = torch.zeros(1, 2, dtype=torch.long)

        with torch.no_grad():
            out_with_zero_stress = model.encode_text(ph, stress_indices=stress_zero)
            out_no_stress = model.encode_text(ph, stress_indices=None)

        # padding_idx=0 → embedding(0) is zeroed, so adding it is a no-op
        assert torch.allclose(out_with_zero_stress, out_no_stress)


# ─────────────────────────────────────────────────────────────────────────────
# 8. TestEncodeTextStressNone
# ─────────────────────────────────────────────────────────────────────────────
class TestEncodeTextStressNone:
    """stress_indices=None skips stress embedding entirely (backward compat)."""

    def test_none_does_not_raise(self):
        model = KokoroModel(vocab_size=20, hidden_dim=32, n_encoder_layers=1,
                            n_decoder_layers=1, use_stress_embedding=True)
        model.eval()
        ph = torch.tensor([[2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            out = model.encode_text(ph, stress_indices=None)
        assert out.shape == (1, 3, 32)

    def test_disabled_model_none_does_not_raise(self):
        model = KokoroModel(vocab_size=20, hidden_dim=32, n_encoder_layers=1,
                            n_decoder_layers=1, use_stress_embedding=False)
        model.eval()
        ph = torch.tensor([[2, 3]], dtype=torch.long)
        with torch.no_grad():
            out = model.encode_text(ph, stress_indices=None)
        assert out.shape == (1, 2, 32)

    def test_disabled_model_with_stress_indices_ignores_them(self):
        """When use_stress_embedding=False, passing stress_indices is silently ignored."""
        model = KokoroModel(vocab_size=20, hidden_dim=32, n_encoder_layers=1,
                            n_decoder_layers=1, use_stress_embedding=False)
        model.eval()
        ph = torch.tensor([[2, 3]], dtype=torch.long)
        stress = torch.tensor([[0, 1]], dtype=torch.long)
        with torch.no_grad():
            out_none   = model.encode_text(ph, stress_indices=None)
            out_stress = model.encode_text(ph, stress_indices=stress)
        assert torch.equal(out_none, out_stress)


# ─────────────────────────────────────────────────────────────────────────────
# 9. TestCollateStressIndices
# ─────────────────────────────────────────────────────────────────────────────
class TestCollateStressIndices:
    """collate_fn must return 'stress_indices' as a (B, max_P) LongTensor."""

    def _make_batch_items(self, phoneme_lengths: List[int]) -> List[dict]:
        """Create minimal dataset item dicts with stress_indices."""
        items = []
        for length in phoneme_lengths:
            ph = torch.arange(length, dtype=torch.long)
            si = torch.zeros(length, dtype=torch.long)
            si[min(1, length - 1)] = 1  # put a stress marker in position 1 if possible
            mel_len = length * 5
            items.append({
                'phoneme_indices': ph,
                'stress_indices': si,
                'mel_spec': torch.zeros(80, mel_len),
                'phoneme_durations': torch.ones(length, dtype=torch.long) * 5,
                'stop_token_targets': torch.zeros(mel_len),
                'pitch': torch.zeros(mel_len),
                'energy': torch.zeros(mel_len),
                'mel_length': mel_len,
                'phoneme_length': length,
                'text': 'test',
                'audio_file': 'test.wav',
            })
        return items

    def test_stress_indices_key_present(self):
        items = self._make_batch_items([4, 6])
        batch = collate_fn(items)
        assert 'stress_indices' in batch

    def test_stress_indices_shape(self):
        items = self._make_batch_items([4, 6, 5])
        batch = collate_fn(items)
        B = len(items)
        max_P = max(4, 6, 5)
        assert batch['stress_indices'].shape == (B, max_P)

    def test_stress_indices_dtype_long(self):
        items = self._make_batch_items([3, 3])
        batch = collate_fn(items)
        assert batch['stress_indices'].dtype == torch.long

    def test_stress_indices_padded_with_zeros(self):
        items = self._make_batch_items([2, 5])
        batch = collate_fn(items)
        # item 0 has length 2 — positions 2..4 in row 0 should be 0
        assert batch['stress_indices'][0, 2:].sum() == 0

    def test_stress_values_transferred_correctly(self):
        items = self._make_batch_items([4])
        # item stress: [0, 1, 0, 0]
        batch = collate_fn(items)
        assert batch['stress_indices'][0, 1].item() == 1


# ─────────────────────────────────────────────────────────────────────────────
# 10. TestCacheVersion
# ─────────────────────────────────────────────────────────────────────────────
class TestCacheVersion:
    """FEATURE_CACHE_VERSION must be 6 now that stress_indices are in the cache."""

    def test_version_equals_6(self):
        assert FEATURE_CACHE_VERSION == 6
