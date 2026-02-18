# MPS Out of Memory Solutions

## Quick Fix

If you're getting OOM errors on Apple Silicon (MPS), run this **before training**:

```bash
# Set environment variable to allow more memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7

# OR run the optimizer helper
python3 -m kokoro.utils.mps_optimizer

# Then train with reduced settings (automatically applied)
kokoro-train --corpus ./ruslan_corpus
```

## Understanding the Error

```
ERROR: MPS backend out of memory (MPS allocated: 76.86 GiB, other allocations: 6.01 GiB, max allowed: 88.13 GiB)
```

This means:
- **MPS allocated**: 76.86 GB used by PyTorch
- **Other allocations**: 6.01 GB used by macOS/other apps
- **Max allowed**: 88.13 GB (your unified memory limit)
- **Tried to allocate**: 5.96 GB (not enough space!)

## Automatic Optimizations (Already Applied)

The code now **automatically** applies these settings when MPS is detected:

1. ✅ `max_frames_per_batch`: capped to **28000**
2. ✅ `max_seq_length`: capped to **1800**
3. ✅ `batch_size`: capped to **10**
4. ✅ `max_batch_size`: 32 → **16**
5. ✅ Aggressive cache clearing after every batch
6. ✅ Memory cleanup after backward pass

## Manual Settings (If Still Getting OOM)

### Option 1: Environment Variable

```bash
# Lower watermark ratio to allow more memory
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6

# Or disable the limit entirely (risky - may crash system)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

Add to `~/.zshrc` to make permanent:
```bash
echo 'export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7' >> ~/.zshrc
source ~/.zshrc
```

### Option 2: Further Reduce Batch Size

```bash
kokoro-train \
  --corpus ./ruslan_corpus \
    --batch-size 4
```

If you need higher effective batch size, set `gradient_accumulation_steps` in `TrainingConfig`.

### Option 3: Reduce Sequence Length

Edit `src/kokoro/training/config.py` or pass via command line:

```python
config = TrainingConfig(
    max_seq_length=800,  # Reduce from 1200
    max_frames_per_batch=16000,  # Reduce from 28000 default cap
)
```

### Option 4: Disable Variance Prediction

No dedicated train CLI flag exists; set it in config:

```python
config = TrainingConfig(
    use_variance_predictor=False  # Disable pitch/energy
)
```

## Memory by System Configuration

### 16 GB Unified Memory
```python
TrainingConfig(
    max_frames_per_batch=22000,
    max_seq_length=1600,
    batch_size=8,
    max_batch_size=12,
    gradient_accumulation_steps=6,  # Effective: 48
)
```

### 32 GB Unified Memory
```python
TrainingConfig(
    max_frames_per_batch=28000,
    max_seq_length=1800,
    batch_size=10,
    max_batch_size=16,
    gradient_accumulation_steps=4,  # Effective: 40
)
```

### 64+ GB Unified Memory
```python
TrainingConfig(
    max_frames_per_batch=28000,
    max_seq_length=1800,
    batch_size=10,
    max_batch_size=16,
    gradient_accumulation_steps=4,  # Effective: 40
)
```

## Monitoring Memory Usage

The progress bar shows memory pressure:

```
mem=low   ✅ Safe - plenty of memory
mem=mod   ⚠️  Moderate - watch for issues
mem=high  ⚠️  High - reduce batch size if OOM occurs
mem=cri   🚨 Critical - likely to OOM soon
mem=cri*  🚨 Critical + cleanup occurred
```

## Pre-compute Features (Highly Recommended)

Pre-computing features reduces memory usage during training:

```bash
# Pre-compute once
kokoro-precompute --corpus ./ruslan_corpus

# Then train with cached features (faster + less memory)
kokoro-train --corpus ./ruslan_corpus
```

Benefits:
- 5-10x faster data loading
- Lower memory spikes during data loading
- More consistent memory usage

## Troubleshooting Steps

### Step 1: Check System Memory
```bash
python3 -m kokoro.utils.mps_optimizer
```

### Step 2: Close Other Applications
- Quit Chrome, Slack, etc.
- Free up system memory for training

### Step 3: Pre-compute Features
```bash
kokoro-precompute --corpus ./ruslan_corpus
```

### Step 4: Set Environment Variable
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
```

### Step 5: Reduce Batch Size
```bash
kokoro-train --corpus ./ruslan_corpus --batch-size 4
```

### Step 6: Disable Variance Prediction
Set `use_variance_predictor=False` in `TrainingConfig` if needed.

### Step 7: Last Resort - Reduce Model Size
```python
TrainingConfig(
    hidden_dim=384,  # Reduce from 512
    n_encoder_layers=4,  # Reduce from 6
    n_decoder_layers=4,  # Reduce from 6
)
```

## Understanding Memory Cleanup

The code now clears MPS cache:

1. **After every backward pass** - Frees gradients
2. **After optimizer step** - Frees optimizer state
3. **Every N batches** - Deep cleanup

You'll see in logs:
```
INFO: Emergency cleanup triggered on mps
INFO: Emergency cleanup completed: freed 0.0MB in 1544.0ms
```

If cleanup freed **0.0MB**, it means memory is actually allocated and in use. This is expected - the cleanup can only free unused cached memory.

## Performance Impact

These optimizations will:
- ✅ **Prevent OOM** - Training will complete
- ⚠️  **Slightly slower** - Smaller batches = more steps
- ✅ **Same final quality** - Gradient accumulation compensates

Example timing:
- Before: OOM at batch 33 ❌
- After: 625 batches complete ✅ (slower but works)

## When to Use CUDA Instead

If you have access to an NVIDIA GPU:
- CUDA has better memory management
- Can use larger batch sizes
- Generally 2-3x faster for training

MPS is excellent for inference and smaller training runs, but large TTS models may benefit from CUDA for production training.

## Summary

**Quick commands to prevent OOM:**

```bash
# 1. Set environment variable
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7

# 2. Pre-compute features
kokoro-precompute --corpus ./ruslan_corpus

# 3. Train with defaults (auto-optimized for MPS)
kokoro-train --corpus ./ruslan_corpus

# 4. If still OOM, reduce batch size
kokoro-train --corpus ./ruslan_corpus --batch-size 4
```

**The automatic optimizations in the code should prevent OOM for most users with 16GB+ unified memory.**
