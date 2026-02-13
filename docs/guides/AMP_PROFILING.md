# Automatic Mixed Precision (AMP) Profiling

## Overview

AMP profiling allows you to measure the actual performance benefit of mixed precision training on your hardware before committing to a long training run. This helps you determine whether AMP provides a speedup or overhead for your specific setup.

## What is Mixed Precision Training?

Mixed precision training uses lower precision (float16) for some operations while maintaining full precision (float32) for others. This can:
- **Speed up training** by 1.5-3x on modern GPUs
- **Reduce memory usage** by ~40%
- **Maintain model quality** through careful gradient scaling

However, benefits vary by hardware:
- ✅ **NVIDIA GPUs** (especially with Tensor Cores): Significant speedup (1.5-2.5x)
- ✅ **Apple Silicon (M1/M2/M3/M4)**: Excellent speedup on MPS (2.5-4.0x with PyTorch 2.1+)
- ⚠️ **Older GPUs**: May see little or no benefit
- ❌ **CPU**: Not supported

## Why Profile AMP?

AMP performance varies based on:
1. **Hardware**: Tensor Cores, compute capability
2. **Model size**: Small models may not benefit as much
3. **Batch size**: Larger batches benefit more
4. **Sequence length**: Longer sequences benefit more

**Without profiling**, you might:
- Assume AMP is faster when it's actually slower on your hardware
- Waste time debugging when AMP causes issues
- Miss out on 2x+ speedup that AMP could provide

**With profiling**, you get:
- Concrete measurements of speedup (e.g., "1.8x faster")
- Confidence in your configuration
- Data-driven decisions about AMP settings

## Usage

### Quick Test (Command Line)

Profile AMP benefits before training:

```bash
# Basic profiling (10 batches)
python3 training.py --corpus ./ruslan_corpus --profile-amp

# More thorough profiling (50 batches)
python3 training.py --corpus ./ruslan_corpus --profile-amp --profile-amp-batches 50

# Profile and continue to training
python3 training.py --corpus ./ruslan_corpus --profile-amp --epochs 100
```

### Standalone Test Script

Test AMP profiling without full training setup:

```bash
python3 test_amp_profiling.py
```

This will:
1. Initialize a minimal trainer
2. Run 10 batches without AMP (baseline)
3. Run 10 batches with AMP
4. Compare performance and show speedup
5. Provide recommendations

### Programmatic Usage

In your own training scripts:

```python
from trainer import KokoroTrainer
from config import TrainingConfig

# Create config
config = TrainingConfig(
    data_dir='./ruslan_corpus',
    use_mixed_precision=True,
    # ... other settings
)

# Initialize trainer
trainer = KokoroTrainer(config)

# Profile AMP benefits
results = trainer.profile_amp_benefits(num_batches=10)

# Check results
if results['speedup'] > 1.2:
    print(f"✓ AMP provides {results['speedup']:.2f}x speedup")
    # Continue with training
    trainer.train()
else:
    print(f"⚠ AMP only provides {results['speedup']:.2f}x speedup")
    # Maybe disable AMP
    config.use_mixed_precision = False
    trainer = KokoroTrainer(config)
    trainer.train()
```

## Understanding Results

### Example Output

```
AUTOMATIC MIXED PRECISION (AMP) PROFILING
============================================================
Device: MPS
Testing with 10 batches
Mixed precision dtype: torch.float16

[1/2] Running WITHOUT AMP (baseline)...
Timing training loop WITHOUT AMP for 10 batches...
  Processed 10 batches in 45.32s (4.532s per batch)

[2/2] Running WITH AMP...
Timing training loop WITH AMP for 10 batches...
  Processed 10 batches in 25.18s (2.518s per batch)

============================================================
AMP PROFILING RESULTS
============================================================
Without AMP: 45.32s (4.532s per batch)
With AMP:    25.18s (2.518s per batch)
Speedup:     1.80x

✓ AMP provides significant speedup (1.80x faster)
  Recommendation: Keep AMP enabled (use_mixed_precision=True)
============================================================
```

### Interpreting Speedup

| Speedup | Interpretation | Recommendation |
|---------|----------------|----------------|
| **> 1.5x** | Excellent! | Definitely use AMP |
| **1.2-1.5x** | Significant benefit | Use AMP (recommended) |
| **1.05-1.2x** | Modest benefit | Use AMP (still worth it) |
| **0.95-1.05x** | Negligible impact | Either way is fine |
| **< 0.95x** | AMP is slower | Disable AMP |

### What Gets Measured

The profiler times:
1. **Data loading**: Transfer batches to GPU/MPS
2. **Forward pass**: Model computation
3. **Loss calculation**: All loss functions
4. **Backward pass**: Gradient computation
5. **Optimizer step**: Weight updates

Both runs use identical:
- Batch sizes
- Model architecture
- Optimizer settings
- Data samples

The only difference is AMP enabled vs disabled.

## Hardware-Specific Results

### NVIDIA GPUs

**Ampere/Ada (RTX 3000/4000, A100):**
- Expected speedup: **1.5-2.5x**
- Memory reduction: **~40%**
- Recommendation: **Always enable AMP**

**Turing (RTX 2000):**
- Expected speedup: **1.3-1.8x**
- Memory reduction: **~40%**
- Recommendation: **Enable AMP**

**Pascal/Older (GTX 1000):**
- Expected speedup: **1.0-1.2x** (minimal)
- Memory reduction: **~40%**
- Recommendation: Enable for memory savings

### Apple Silicon (M1/M2/M3)

**M1 Pro/Max/Ultra, M2 Pro/Max/Ultra, M3/M4:**
- Expected speedup: **2.5-4.0x** (PyTorch 2.1+)
- Memory: Unified memory (no separate GPU)
- Recommendation: **Definitely enable AMP**
- Note: Custom MPSGradScaler is used
- Real-world result: **3.89x** on M-series with PyTorch 2.1+

**M1 (base):**
- Expected speedup: **1.8-2.5x**
- Recommendation: **Enable AMP**

### CPU

- AMP: **Not supported**
- Profiler will skip testing
- Recommendation: `use_mixed_precision=False`

## Configuration Options

### In TrainingConfig

```python
config = TrainingConfig(
    # Enable/disable AMP
    use_mixed_precision=True,

    # Precision type (float16 or bfloat16)
    mixed_precision_dtype=torch.float16,

    # GradScaler settings (advanced)
    amp_init_scale=65536.0,      # Initial loss scale
    amp_growth_factor=2.0,        # Scale growth rate
    amp_backoff_factor=0.5,       # Scale reduction on overflow
    amp_growth_interval=2000,     # Steps between scale increases
)
```

### CLI Arguments

```bash
--profile-amp               # Enable AMP profiling before training
--profile-amp-batches N     # Number of batches to profile (default: 10)
```

## Advanced Usage

### Custom Batch Count

More batches = more accurate results but longer profiling time:

```bash
# Quick test (less accurate, ~30 seconds)
python3 training.py --corpus ./ruslan_corpus --profile-amp --profile-amp-batches 5

# Standard (good balance, ~1-2 minutes)
python3 training.py --corpus ./ruslan_corpus --profile-amp --profile-amp-batches 10

# Thorough (most accurate, ~5 minutes)
python3 training.py --corpus ./ruslan_corpus --profile-amp --profile-amp-batches 50
```

### Testing Different Batch Sizes

AMP benefits increase with larger batches:

```bash
# Small batches (may show less speedup)
python3 training.py --corpus ./ruslan_corpus --profile-amp --batch-size 4

# Medium batches (typical)
python3 training.py --corpus ./ruslan_corpus --profile-amp --batch-size 16

# Large batches (maximum speedup)
python3 training.py --corpus ./ruslan_corpus --profile-amp --batch-size 32
```

### Comparing Float16 vs BFloat16

In your config:

```python
# Test with float16 (default)
config.mixed_precision_dtype = torch.float16
trainer = KokoroTrainer(config)
results_fp16 = trainer.profile_amp_benefits()

# Test with bfloat16 (if supported)
config.mixed_precision_dtype = torch.bfloat16
trainer = KokoroTrainer(config)
results_bf16 = trainer.profile_amp_benefits()

# Compare
print(f"FP16 speedup:  {results_fp16['speedup']:.2f}x")
print(f"BF16 speedup:  {results_bf16['speedup']:.2f}x")
```

## Troubleshooting

### Issue: "AMP is not supported on this device (CPU)"

**Cause:** Running on CPU where AMP isn't available

**Solution:**
```python
config.use_mixed_precision = False
```

### Issue: Very slow profiling

**Cause:** Large model or long sequences

**Solutions:**
- Reduce `--profile-amp-batches` to 5
- Use smaller `--batch-size` temporarily
- Ensure GPU/MPS is actually being used

### Issue: Inconsistent results between runs

**Cause:** Warmup effects, background processes

**Solutions:**
- Increase `--profile-amp-batches` to 20+
- Close other applications
- Run profiling multiple times and average

### Issue: AMP shows slowdown unexpectedly

**Possible causes:**
1. **Model too small**: AMP overhead dominates
2. **Old hardware**: No Tensor Cores or equivalent
3. **CPU fallback**: GPU not being used
4. **Old PyTorch version**: MPS autocast requires PyTorch 2.1+
5. **Driver issues**: Update GPU drivers

**Check:**
```bash
# Verify device and PyTorch version
python3 -c "import torch; print(torch.cuda.is_available())"  # CUDA
python3 -c "import torch; print(torch.backends.mps.is_available())"  # MPS
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"  # Version

# For MPS, verify autocast support:
python3 -c "import torch; x=torch.randn(2,2,device='mps'); \
with torch.autocast('mps', dtype=torch.float16): y=x*2; print('MPS autocast: OK')"
```

## Implementation Details

### What Happens During Profiling

1. **Save current state**: Store AMP settings
2. **Warmup (2 batches)**: Avoid cold start effects
3. **Baseline run**: Train N batches without AMP
   - Time all operations
   - Synchronize GPU to ensure accurate timing
4. **Clear cache**: Reset memory state
5. **AMP run**: Train N batches with AMP
   - Time all operations with autocast enabled
   - Include GradScaler overhead
6. **Calculate speedup**: `baseline_time / amp_time`
7. **Restore state**: Return to original settings

### Timing Methodology

```python
# Without AMP
start = time.time()
for batch in batches:
    forward_pass()
    backward_pass()
    optimizer_step()
device.synchronize()  # Wait for GPU completion
baseline_time = time.time() - start

# With AMP
start = time.time()
for batch in batches:
    with autocast():
        forward_pass()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
device.synchronize()
amp_time = time.time() - start

speedup = baseline_time / amp_time
```

### Accuracy Considerations

The profiling is accurate because:
- ✓ Same batches used for both runs
- ✓ Warmup batches to stabilize GPU clocks
- ✓ Device synchronization for precise timing
- ✓ Multiple batches to average out variance
- ✓ Includes all overhead (data loading, scaler, etc.)

## Best Practices

### When to Profile

✓ **Before long training runs** (100+ epochs)
✓ **After hardware changes** (new GPU, driver update)
✓ **When switching model architectures**
✓ **When debugging performance issues**
✓ **After updating PyTorch version**

✗ **During training** (adds overhead)
✗ **On CPU** (not supported)
✗ **With tiny datasets** (not representative)

### Recommended Workflow

1. **Profile first**:
   ```bash
   python3 training.py --corpus ./ruslan_corpus --profile-amp --profile-amp-batches 20
   ```

2. **Review results**: Check speedup value

3. **Adjust config** if needed:
   ```python
   # If speedup < 1.0, disable AMP
   config.use_mixed_precision = False
   ```

4. **Train**:
   ```bash
   python3 training.py --corpus ./ruslan_corpus --epochs 100
   ```

### Tips for Best Results

1. **Use realistic settings**: Profile with your actual batch size
2. **Run multiple times**: Average results for consistency
3. **Close other apps**: Reduce GPU contention
4. **Use enough batches**: 10-20 is a good balance
5. **Check memory**: Ensure AMP fits in memory
6. **Test both dtypes**: Try float16 and bfloat16

## FAQ

**Q: How long does profiling take?**

A: With default settings (10 batches), about 1-2 minutes. Scale linearly with batch count.

**Q: Will profiling affect my training data?**

A: No, it only times operations. No checkpoints or model changes are saved.

**Q: Should I profile every time I train?**

A: No, only when:
- First setting up training
- Hardware changes
- Model architecture changes

**Q: Does profiling consume GPU memory?**

A: Yes, the same as normal training. If profiling fails with OOM, reduce batch size temporarily.

**Q: Can I profile with validation enabled?**

A: Yes, but it only profiles training batches, not validation.

**Q: What if speedup is exactly 1.0x?**

A: Run with more batches (`--profile-amp-batches 50`) for more accurate measurement.

**Q: Does profiling work with gradient accumulation?**

A: Yes, but profile measures per-step time, not effective batch time.

## See Also

- [README.md](README.md) - Main documentation
- [DYNAMIC_BATCHING.md](DYNAMIC_BATCHING.md) - Dynamic batching optimization
- [VALIDATION.md](VALIDATION.md) - Validation and early stopping
- [config.py](config.py) - Configuration options
