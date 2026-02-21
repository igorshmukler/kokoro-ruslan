# Quick script to compare train/val utterance length distributions
import sys
from statistics import mean, median, pstdev
import numpy as np

# Ensure local src is importable
sys.path.insert(0, 'src')

from kokoro.training.config import TrainingConfig
from kokoro.data.dataset import RuslanDataset


def percentiles(arr, qs=[10,25,50,75,90,95]):
    return {q: float(np.percentile(arr, q)) for q in qs}


def summarize(name, arr):
    arr = np.array(arr)
    s = {
        'count': int(arr.size),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std(ddof=0)),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }
    s.update(percentiles(arr, [10,25,50,75,90,95]))
    print(f"{name}: count={s['count']} mean={s['mean']:.2f} median={s['median']:.1f} std={s['std']:.2f} min={s['min']:.1f} max={s['max']:.1f}")
    print(f"  percentiles: 10={s[10]:.1f} 25={s[25]:.1f} 50={s[50]:.1f} 75={s[75]:.1f} 90={s[90]:.1f} 95={s[95]:.1f}")
    return s


def main():
    # Point to corpus in workspace
    data_dir = 'ruslan_corpus'
    cfg = TrainingConfig(data_dir=data_dir)

    print('Loading full dataset (this may take a minute the first time)')
    full = RuslanDataset(cfg.data_dir, cfg)
    total = len(full.samples)
    print(f'Total samples loaded: {total}')

    # Reproduce trainer split
    val_split = getattr(cfg, 'validation_split', 0.1)
    indices = list(range(total))
    import random
    random.seed(42)
    random.shuffle(indices)
    split_idx = int(total * (1 - val_split))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_samples = [full.samples[i] for i in train_idx if i < total]
    val_samples = [full.samples[i] for i in val_idx if i < total]

    print(f'Train samples: {len(train_samples)}  Val samples: {len(val_samples)}')

    # Extract audio_length and phoneme_length
    train_audio_lengths = [s['audio_length'] for s in train_samples]
    val_audio_lengths = [s['audio_length'] for s in val_samples]
    train_phoneme_lengths = [s['phoneme_length'] for s in train_samples]
    val_phoneme_lengths = [s['phoneme_length'] for s in val_samples]

    print('\nAudio length (mel frames) stats:')
    t_audio = summarize('Train audio', train_audio_lengths)
    v_audio = summarize('Val audio', val_audio_lengths)

    print('\nPhoneme length (tokens) stats:')
    t_ph = summarize('Train phoneme', train_phoneme_lengths)
    v_ph = summarize('Val phoneme', val_phoneme_lengths)

    # Simple comparison: relative median difference
    def rel_diff(a,b):
        return abs(a - b) / max(1e-9, (a+b)/2.0)

    audio_med_diff = rel_diff(t_audio['median'], v_audio['median'])
    ph_med_diff = rel_diff(t_ph['median'], v_ph['median'])

    print('\nRelative median differences:')
    print(f'  audio median rel_diff = {audio_med_diff*100:.2f}%')
    print(f'  phoneme median rel_diff = {ph_med_diff*100:.2f}%')

    if audio_med_diff < 0.05 and ph_med_diff < 0.05:
        print('\nConclusion: train and val splits are similarly distributed by length (medians within 5%).')
    else:
        print('\nConclusion: train and val splits show measurable differences by length (medians differ >5%).')


if __name__ == '__main__':
    main()
