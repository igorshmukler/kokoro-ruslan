# Compare phoneme coverage between train and validation splits.
# Reports unigram, bigram, and trigram coverage plus any phonemes
# present in one split but missing from the other.
import sys
import random
from collections import Counter
from itertools import islice

import numpy as np

sys.path.insert(0, 'src')

from kokoro.training.config import TrainingConfig
from kokoro.data.dataset import RuslanDataset
from kokoro.data.russian_phoneme_processor import RussianPhonemeProcessor
from kokoro.data.audio_utils import PhonemeProcessorUtils


def text_to_phonemes(processor, text):
    """Convert Russian text to flat phoneme list (with <sil> between words)."""
    raw = processor.process_text(text)
    return PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
        raw, processor.phoneme_to_id
    )


def ngrams(seq, n):
    """Yield n-grams as tuples from a sequence."""
    it = iter(seq)
    window = tuple(islice(it, n))
    if len(window) == n:
        yield window
    for item in it:
        window = window[1:] + (item,)
        yield window


def coverage_report(train_counter, val_counter, label):
    """Print coverage comparison for a given n-gram level."""
    train_types = set(train_counter.keys())
    val_types = set(val_counter.keys())
    all_types = train_types | val_types

    train_only = train_types - val_types
    val_only = val_types - train_types
    shared = train_types & val_types
    coverage = len(shared) / len(train_types) * 100 if train_types else 0

    print(f'\n{"=" * 60}')
    print(f'{label} coverage')
    print(f'{"=" * 60}')
    print(f'  Train unique: {len(train_types)}')
    print(f'  Val unique:   {len(val_types)}')
    print(f'  Shared:       {len(shared)}')
    print(f'  Val covers {coverage:.1f}% of train {label.lower()}s')

    if train_only:
        # Sort by frequency (most common missing ones first)
        missing = sorted(train_only, key=lambda x: train_counter[x], reverse=True)
        top = missing[:20]
        print(f'\n  In train but NOT in val ({len(train_only)} total):')
        for item in top:
            display = ' '.join(item) if isinstance(item, tuple) else item
            print(f'    {display:30s}  (train count: {train_counter[item]})')
        if len(missing) > 20:
            print(f'    ... and {len(missing) - 20} more')

    if val_only:
        missing = sorted(val_only, key=lambda x: val_counter[x], reverse=True)
        top = missing[:10]
        print(f'\n  In val but NOT in train ({len(val_only)} total):')
        for item in top:
            display = ' '.join(item) if isinstance(item, tuple) else item
            print(f'    {display:30s}  (val count: {val_counter[item]})')

    return coverage


def frequency_correlation(train_counter, val_counter, label):
    """Report rank correlation between train and val frequency distributions."""
    shared = set(train_counter.keys()) & set(val_counter.keys())
    if len(shared) < 3:
        return

    items = sorted(shared)
    train_total = sum(train_counter.values())
    val_total = sum(val_counter.values())
    train_freqs = np.array([train_counter[k] / train_total for k in items])
    val_freqs = np.array([val_counter[k] / val_total for k in items])

    # Pearson correlation on relative frequencies
    corr = np.corrcoef(train_freqs, val_freqs)[0, 1]
    print(f'  Frequency correlation (Pearson r): {corr:.4f}')

    # KL divergence (val || train), smoothed to avoid zeros
    eps = 1e-10
    p = train_freqs + eps
    q = val_freqs + eps
    p /= p.sum()
    q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    print(f'  KL divergence (train || val):      {kl:.6f}')


def main():
    data_dir = 'ruslan_corpus'
    cfg = TrainingConfig(data_dir=data_dir)

    print('Loading dataset...')
    full = RuslanDataset(cfg.data_dir, cfg)
    total = len(full.samples)
    print(f'Total samples: {total}')

    # Reproduce trainer split (same logic as trainer.py)
    val_split = getattr(cfg, 'validation_split', 0.1)
    indices = list(range(total))
    random.seed(42)
    random.shuffle(indices)
    split_idx = int(total * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_samples = [full.samples[i] for i in train_indices]
    val_samples = [full.samples[i] for i in val_indices]
    print(f'Train: {len(train_samples)}  Val: {len(val_samples)}')

    processor = RussianPhonemeProcessor()

    # Collect phoneme sequences for each split
    print('\nConverting texts to phonemes (train)...')
    train_unigrams = Counter()
    train_bigrams = Counter()
    train_trigrams = Counter()
    for i, sample in enumerate(train_samples):
        ph = text_to_phonemes(processor, sample['text'])
        train_unigrams.update(ph)
        train_bigrams.update(ngrams(ph, 2))
        train_trigrams.update(ngrams(ph, 3))
        if (i + 1) % 5000 == 0:
            print(f'  processed {i + 1}/{len(train_samples)}')

    print('Converting texts to phonemes (val)...')
    val_unigrams = Counter()
    val_bigrams = Counter()
    val_trigrams = Counter()
    for i, sample in enumerate(val_samples):
        ph = text_to_phonemes(processor, sample['text'])
        val_unigrams.update(ph)
        val_bigrams.update(ngrams(ph, 2))
        val_trigrams.update(ngrams(ph, 3))

    # Reports
    uni_cov = coverage_report(train_unigrams, val_unigrams, 'Unigram')
    frequency_correlation(train_unigrams, val_unigrams, 'Unigram')

    bi_cov = coverage_report(train_bigrams, val_bigrams, 'Bigram')
    frequency_correlation(train_bigrams, val_bigrams, 'Bigram')

    tri_cov = coverage_report(train_trigrams, val_trigrams, 'Trigram')
    frequency_correlation(train_trigrams, val_trigrams, 'Trigram')

    # Top phonemes comparison
    print(f'\n{"=" * 60}')
    print('Top 15 phonemes by frequency')
    print(f'{"=" * 60}')
    train_total = sum(train_unigrams.values())
    val_total = sum(val_unigrams.values())
    print(f'  {"Phoneme":10s} {"Train %":>10s} {"Val %":>10s} {"Diff":>10s}')
    print(f'  {"-" * 40}')
    for ph, cnt in train_unigrams.most_common(15):
        t_pct = cnt / train_total * 100
        v_pct = val_unigrams.get(ph, 0) / val_total * 100
        diff = v_pct - t_pct
        print(f'  {ph:10s} {t_pct:9.2f}% {v_pct:9.2f}% {diff:+9.2f}%')

    # Summary verdict
    print(f'\n{"=" * 60}')
    print('Summary')
    print(f'{"=" * 60}')
    print(f'  Unigram coverage: {uni_cov:.1f}%')
    print(f'  Bigram coverage:  {bi_cov:.1f}%')
    print(f'  Trigram coverage: {tri_cov:.1f}%')

    if uni_cov >= 99 and bi_cov >= 80 and tri_cov >= 60:
        print('\n  Verdict: Validation set has GOOD phoneme coverage.')
        if tri_cov < 75:
            print('  Note: trigram gaps are expected with a 10% split and affect only rare combinations.')
    elif uni_cov >= 95 and bi_cov >= 70:
        print('\n  Verdict: Validation set has ADEQUATE phoneme coverage.')
        print('  Consider increasing validation_split for better trigram representation.')
    else:
        print('\n  Verdict: Validation set has INSUFFICIENT phoneme coverage.')
        print('  Consider stratified splitting or increasing validation_split.')


if __name__ == '__main__':
    main()
