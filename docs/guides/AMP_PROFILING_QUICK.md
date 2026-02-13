# AMP Profiling Quick Reference

## One-Line Usage

```bash
# Profile and train
python3 training.py --corpus ./ruslan_corpus --profile-amp
```

## What It Does

Compares training speed **with** vs **without** mixed precision to show actual speedup on your hardware.

## Example Output

```
Without AMP: 45.32s (4.532s per batch)
With AMP:    25.18s (2.518s per batch)
Speedup:     1.80x ← YOUR ACTUAL SPEEDUP

✓ AMP provides significant speedup (1.80x faster)
  Recommendation: Keep AMP enabled
```

## Quick Decision Guide

| Speedup | What to Do |
|---------|------------|
| **> 1.5x** | ✓ Keep AMP (huge benefit!) |
| **1.2-1.5x** | ✓ Keep AMP (significant) |
| **1.0-1.2x** | ✓ Keep AMP (modest benefit) |
| **< 1.0x** | ✗ Disable AMP (slower) |

## Common Commands

```bash
# Basic profiling (10 batches, ~1 minute)
python3 training.py --corpus ./ruslan_corpus --profile-amp

# More accurate (50 batches, ~5 minutes)
python3 training.py --corpus ./ruslan_corpus --profile-amp --profile-amp-batches 50

# Quick test (5 batches, ~30 seconds)
python3 training.py --corpus ./ruslan_corpus --profile-amp --profile-amp-batches 5

# Test with different batch size
python3 training.py --corpus ./ruslan_corpus --profile-amp --batch-size 16

# Standalone test (no training)
python3 test_amp_profiling.py
```

## Expected Speedups by Hardware

| Hardware | Expected Speedup |
|----------|-----------------|
| RTX 3000/4000 (NVIDIA) | **1.5-2.5x** |
| RTX 2000 (NVIDIA) | **1.3-1.8x** |
| M1/M2/M3 Pro/Max (Apple) | **1.2-1.6x** |
| M1 base (Apple) | **1.1-1.3x** |
| GTX 1000 (NVIDIA) | **1.0-1.2x** |
| CPU | **Not supported** |

## How to Disable AMP

If profiling shows AMP is slower:

### Option 1: In config.py
```python
use_mixed_precision = False
```

### Option 2: Programmatically
```python
config = TrainingConfig(
    use_mixed_precision=False,
    # ... other settings
)
```

## Troubleshooting

**AMP shows slowdown:**
- ✓ Update GPU drivers
- ✓ Check if GPU is actually being used
- ✓ Try larger batch size
- ✓ Close background applications

**Profiling takes too long:**
- Use `--profile-amp-batches 5` for quick test

**Inconsistent results:**
- Use `--profile-amp-batches 20` for more accuracy

## When to Profile

✓ Before long training runs
✓ After hardware/driver updates
✓ When debugging performance
✗ During active training
✗ On CPU (not supported)

## Full Documentation

See [AMP_PROFILING.md](AMP_PROFILING.md) for complete details.
