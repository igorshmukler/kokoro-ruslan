# Feature Caching - Pre-computed Mel Spectrograms

## Overview

The feature caching system pre-computes and caches mel spectrograms, pitch, and energy features to disk, providing **5-10x faster training** by eliminating redundant audio processing on every epoch.

## How It Works

1. **First Access**: When a sample is accessed for the first time, the system:
   - Loads the audio file
   - Computes mel spectrogram, pitch, and energy
   - Saves these features to disk (`.feature_cache/`)
   - Returns the computed features

2. **Subsequent Access**: On future accesses:
   - Loads pre-computed features directly from cache
   - Skips all audio loading and processing
   - ~10x faster than computing from scratch

## Quick Start

### Option 1: Pre-compute Before Training (Recommended)

```shell
# Pre-compute all features before training starts
kokoro-precompute --corpus ./ruslan_corpus

# Then train normally - will automatically use cached features
kokoro-train --corpus ./ruslan_corpus
```

### Option 2: Compute During First Epoch

```shell
# Features will be cached automatically during first epoch
kokoro-train --corpus ./ruslan_corpus

# First epoch: Normal speed (computing + caching)
# Epochs 2+: 5-10x faster (loading from cache)
```

### Option 3: Disable Caching

```shell
# Disable feature caching in TrainingConfig (no dedicated train CLI flag)
python -c "from kokoro.training.config import TrainingConfig; print(TrainingConfig(use_feature_cache=False))"
```

## Commands

### Pre-compute Features

```shell
# Basic usage
kokoro-precompute --corpus ./ruslan_corpus

# Custom cache directory
kokoro-precompute --corpus ./ruslan_corpus --cache-dir /path/to/cache

# Force recomputation (overwrite existing cache)
kokoro-precompute --corpus ./ruslan_corpus --force

# Without variance prediction (pitch/energy)
kokoro-precompute --corpus ./ruslan_corpus --no-variance

# With MFA alignments
kokoro-precompute --corpus ./ruslan_corpus --use-mfa --mfa-alignment-dir ./mfa_output/alignments
```

Module form:

```shell
python -m kokoro.cli.precompute_features --corpus ./ruslan_corpus
```

### Check Cache Status

```shell
# View cache statistics
python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --status

# Output:
# ==========================================
# FEATURE CACHE STATUS
# ==========================================
# Cache directory: /path/to/ruslan_corpus/.feature_cache
# Cached files: 22200
# Total size: 1234.5 MB
# Average size per file: 0.06 MB
# ==========================================
```

### Clear Cache

```shell
# Clear all cached features
python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --clear

# Clear without confirmation
python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --clear --force
```

## Configuration

Add to your training config or use command-line arguments:

```python
from kokoro.training.config import TrainingConfig

config = TrainingConfig(
    data_dir="./ruslan_corpus",
    use_feature_cache=True,  # Enable caching (default: True)
    feature_cache_dir="./custom_cache",  # Custom location (optional)
    precompute_features=True,  # Pre-compute before training (default: False)
)
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `use_feature_cache` | `True` | Enable/disable feature caching |
| `feature_cache_dir` | `{corpus}/.feature_cache` | Cache directory path |
| `precompute_features` | `False` | Pre-compute all features before training starts |

## Performance Comparison

### Without Caching (Baseline)
```
Epoch 1: 45 minutes
Epoch 2: 45 minutes
Epoch 3: 45 minutes
Total for 100 epochs: ~75 hours
```

### With Caching (After First Epoch)
```
Epoch 1: 50 minutes (computing + caching)
Epoch 2: 5 minutes (loading from cache)
Epoch 3: 5 minutes (loading from cache)
Total for 100 epochs: ~9 hours
```

**Speedup: 8-9x faster overall training time!**

## Storage Requirements

For a typical corpus with 22,000 samples:
- **Cache size**: ~1-2 GB
- **Per sample**: ~50-100 KB
- **Variance features**: +20% size (pitch/energy)

Example for Ruslan corpus (22,200 samples):
```
Base features (mel spectrograms): ~900 MB
Variance features (pitch + energy): ~200 MB
Total cache size: ~1.1 GB
```

## Cache Structure

```
ruslan_corpus/
├── .feature_cache/           # Feature cache directory
│   ├── 005559_RUSLAN.pt     # Cached features for sample
│   ├── 005560_RUSLAN.pt
│   └── ...
├── wavs/
├── metadata_RUSLAN_22200.csv
└── ...
```

Each `.pt` file contains:
- `mel_spec`: Log mel spectrogram (80 x time_frames)
- `pitch`: Frame-level F0 values (time_frames)
- `energy`: Frame-level energy (time_frames)
- `phoneme_indices`: Text converted to phoneme indices
- `phoneme_durations`: MFA or estimated durations
- `stop_token_targets`: End-of-sequence targets
- `mel_length`: Actual mel frames
- `phoneme_length`: Actual phonemes

## Advanced Usage

### Parallel Pre-computation

For very large datasets, you can split pre-computation across multiple processes:

```shell
# Split dataset into chunks and process in parallel
# (Implementation coming soon)
```

### Selective Caching

Cache only specific features:

```python
# Custom dataset modification to cache only mel spectrograms
# (not pitch/energy for faster caching)
```

### Cache Warmup

Pre-load cache into RAM for even faster access:

```python
# Load entire cache into memory at startup
dataset.preload_cache_to_memory()
```

## Troubleshooting

### Cache Misses

If you're not seeing speedups:

```shell
# Check cache status
python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --status

# Verify features are being cached
ls -lh ruslan_corpus/.feature_cache/ | head

# Force recomputation if cache is corrupted
kokoro-precompute --corpus ./ruslan_corpus --force
```

### Disk Space Issues

If running out of disk space:

```shell
# Clear cache
python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --clear

# Disable caching through config (example)
python -c "from kokoro.training.config import TrainingConfig; cfg=TrainingConfig(use_feature_cache=False); print('use_feature_cache=', cfg.use_feature_cache)"

# Or use external drive for cache
kokoro-precompute --corpus ./ruslan_corpus --cache-dir /external/cache
```

### Stale Cache

If you modify audio files or configuration:

```shell
# Clear old cache
python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --clear

# Recompute with new settings
kokoro-precompute --corpus ./ruslan_corpus
```

## Best Practices

1. **Pre-compute before long training runs**
   ```shell
   kokoro-precompute --corpus ./ruslan_corpus
   kokoro-train --corpus ./ruslan_corpus --epochs 100
   ```

2. **Use dedicated cache directory for large datasets**
   ```shell
   kokoro-precompute --corpus ./ruslan_corpus --cache-dir /fast-ssd/cache
   ```

3. **Verify cache before training**
   ```shell
   python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --status
   ```

4. **Clear cache after major config changes**
   ```shell
   # If you change sample_rate, n_mels, hop_length, etc.
   python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --clear
   kokoro-precompute --corpus ./ruslan_corpus
   ```

## Implementation Details

The caching system uses PyTorch's `torch.save()` and `torch.load()` for efficient serialization. Features are stored as CPU tensors to avoid device-specific issues.

Cache loading is optimized with:
- In-memory LRU cache for frequently accessed samples
- Lazy loading (features loaded on-demand)
- Multi-process safe (each worker has its own cache instance)

## Future Enhancements

- [ ] Parallel pre-computation across multiple GPUs
- [ ] Compressed cache format for smaller disk usage
- [ ] Remote cache support (S3, cloud storage)
- [ ] Incremental cache updates (only recompute changed files)
- [ ] Cache versioning for config compatibility checks
