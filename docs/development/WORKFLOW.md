# Complete Workflow Guide

This guide walks you through the complete process from corpus to trained TTS model.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Install MFA (choose one method)
conda install -c conda-forge montreal-forced-aligner  # Recommended
# OR
pip install montreal-forced-aligner
```

## Step-by-Step Workflow

### Step 1: Prepare Your Corpus

Ensure your corpus follows this structure:

```
ruslan_corpus/
├── metadata_RUSLAN_22200.csv    # Format: filename|transcription
└── wavs/
    ├── 005559_RUSLAN.wav
    ├── 005560_RUSLAN.wav
    └── ...
```

**Metadata format:**
```
005559_RUSLAN|Привет, как дела?
005560_RUSLAN|Сегодня хорошая погода.
```

### Step 2: Run Preprocessing (Extract Phoneme Alignments)

```bash
# Option A: Use the preprocessing script (recommended)
kokoro-preprocess --corpus ./ruslan_corpus --output ./mfa_output --jobs 8

# Option B: Module form
python -m kokoro.cli.preprocess --corpus ./ruslan_corpus --output ./mfa_output --jobs 8

# Option C: Skip MFA and use estimated durations (faster but lower quality)
# No preprocessing needed, just go to Step 3 with --no-mfa flag
```

**Expected time:** 1-3 hours for 22k files (depends on CPU)

**What this does:**
- Downloads Russian acoustic model and pronunciation dictionary
- Runs forced alignment to extract exact phoneme timings
- Creates alignment cache for fast loading during training
- Validates results

### Step 3: Start Training

```bash
# Basic training with MFA alignments
kokoro-train --corpus ./ruslan_corpus --output ./my_model

# Full custom configuration
kokoro-train \
    --corpus ./ruslan_corpus \
    --output ./my_model \
    --batch-size 10 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --save-every 2

# Without MFA (estimated durations - lower quality)
kokoro-train --corpus ./ruslan_corpus --output ./my_model --no-mfa

# Resume from checkpoint
kokoro-train --corpus ./ruslan_corpus --output ./my_model --resume auto
```

### Step 4: Monitor Training

Training logs show:
```
Epoch 1/100: 100%|████████| 2775/2775 [12:34<00:00, 3.68it/s]
Total Loss: 2.345, Mel Loss: 1.234, Duration Loss: 0.567, Stop Loss: 0.544
Learning Rate: 0.0001000
✓ Checkpoint saved: ./kokoro_russian_model/checkpoint_epoch_1.pth
```

**Important metrics to watch:**
- **Total Loss**: Should decrease steadily
- **Mel Loss**: Most important - target < 1.0 for good quality
- **Duration Loss**: Should converge quickly with MFA
- **Stop Loss**: Should decrease to ~0.1

### Step 5: Validate Quality (Optional but Recommended)

During training, periodically run inference to check quality:

```bash
# Generate test samples
python -m kokoro.inference.inference \
    --model ./my_model \
    --text "Привет, как дела?" \
    --output test.wav
```

Listen to the output and compare with training data.

### Step 6: Final Model

After training completes:

```
my_model/
├── kokoro_russian_final.pth           # Final model
├── checkpoint_epoch_100.pth           # Latest checkpoint
├── phoneme_processor.pkl              # Phoneme processor state
└── checkpoint_epoch_*.pth             # All checkpoints
```

## Quick Start (Skip MFA)

If you want to start quickly without waiting for alignment:

```bash
# Install dependencies
pip install -r requirements.txt

# Start training immediately with estimated durations
kokoro-train --corpus ./ruslan_corpus --no-mfa

# Quality will be lower, but you can test the pipeline
```

Later, you can run MFA and restart training with better data:

```bash
# Run MFA alignment
kokoro-preprocess --corpus ./ruslan_corpus --output ./mfa_output

# Restart training with MFA alignments
kokoro-train --corpus ./ruslan_corpus --resume auto
```

## Common Workflows

### Fast Experimentation

```bash
# Small subset for testing (create subset first)
kokoro-train \
    --corpus ./ruslan_corpus_subset \
    --epochs 10 \
    --batch-size 4 \
    --no-mfa
```

### Production Training

```bash
# Step 1: Full preprocessing
kokoro-preprocess --corpus ./ruslan_corpus --output ./mfa_output --jobs 16

# Step 2: Train with optimal settings
kokoro-train \
    --corpus ./ruslan_corpus \
    --output ./production_model \
    --batch-size 10 \
    --epochs 200 \
    --learning-rate 1e-4 \
    --save-every 5
```

### Resume After Interruption

```bash
# Auto-resume from latest checkpoint
kokoro-train --corpus ./ruslan_corpus --resume auto

# Resume from specific checkpoint
kokoro-train --corpus ./ruslan_corpus --resume ./my_model/checkpoint_epoch_50.pth
```

## Troubleshooting

### MFA Alignment Fails

```bash
# Validate corpus first
kokoro-preprocess --corpus ./ruslan_corpus --validate-only

# Check MFA installation
mfa version

# Try with fewer jobs if memory issues
kokoro-preprocess --corpus ./ruslan_corpus --jobs 2
```

### Training Out of Memory

```bash
# Reduce batch size
kokoro-train --corpus ./ruslan_corpus --batch-size 4

# Or reduce sequence length in config.py
# max_seq_length: int = 1500  # Reduce from 2500
```

### Low Quality Output

**Checklist:**
1. ✓ Used MFA alignments? (--mfa-alignments should be set)
2. ✓ Trained for enough epochs? (At least 50-100)
3. ✓ Mel loss below 1.0?
4. ✓ Audio quality in training data good?
5. ✓ Transcriptions accurate?

### MPS Device Issues (Mac)

```bash
# Enable MPS fallback
PYTORCH_ENABLE_MPS_FALLBACK=1 kokoro-train --corpus ./ruslan_corpus
```

## Expected Timeline

For 22k samples on modern hardware:

| Task | Duration | Can Run Overnight? |
|------|----------|-------------------|
| MFA Alignment | 1-3 hours | ✓ |
| Training (100 epochs) | 12-24 hours | ✓ |
| Inference (per sample) | 1-2 seconds | N/A |

**Recommendation:** Run MFA overnight, then start training the next day.

## Next Steps

After successful training:

1. **Generate samples** to validate quality
2. **Fine-tune** on specific voices or styles
3. **Optimize** for faster inference
4. **Deploy** with your vocoder of choice

See [inference.md](../setup/inference.md) for deployment options.
