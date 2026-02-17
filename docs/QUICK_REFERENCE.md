# Quick Reference: MFA Integration

> **Note:** After package restructuring, use `kokoro-train`, `kokoro-preprocess`, etc. commands (after `pip install -e .`), or Python modules: `python3 -m kokoro.cli.training`

## Installation Check
```bash
python verify_setup.py
```

## One-Command Setup
```bash
# Install MFA
conda install -c conda-forge montreal-forced-aligner

# Run preprocessing
kokoro-preprocess --corpus ./ruslan_corpus
# Or: python3 -m kokoro.cli.preprocess --corpus ./ruslan_corpus

# Start training
kokoro-train
# Or: python3 -m kokoro.cli.training
```

## Command Reference

### Preprocessing
```bash
# Basic (using installed command)
kokoro-preprocess --corpus ./ruslan_corpus

# Or using Python module
python3 -m kokoro.cli.preprocess --corpus ./ruslan_corpus

# With more CPU cores
kokoro-preprocess --corpus ./ruslan_corpus --jobs 8

# Custom output
kokoro-preprocess --corpus ./ruslan_corpus --output ./my_mfa

# Validate only (check existing alignments)
kokoro-preprocess --corpus ./ruslan_corpus --validate-only

# Skip MFA
kokoro-preprocess --corpus ./ruslan_corpus --skip-mfa
```

### Training
```bash
# With MFA (default) - using installed command
kokoro-train

# Or using Python module
python3 -m kokoro.cli.training

# Custom alignment path
kokoro-train --mfa-alignments ./my_mfa/alignments

# Without MFA
kokoro-train --no-mfa

# Full custom
kokoro-train \
    --corpus ./ruslan_corpus \
    --output ./my_model \
    --batch-size 16 \
    --epochs 100 \
    --mfa-alignments ./mfa_output/alignments
```

### Programmatic MFA utilities

```bash
# Python module utilities
python3 -m kokoro.data.mfa_integration

# Direct MFA integration script
kokoro-preprocess --corpus ./ruslan_corpus --output ./mfa_output --jobs 8
```

## File Locations

### Input
```
ruslan_corpus/
├── metadata_RUSLAN_22200.csv
└── wavs/
    └── *.wav
```

### Output
```
mfa_output/
├── alignments/           # TextGrid files (use for training)
├── alignment_cache/      # Cached durations (auto-generated)
└── mfa_corpus/          # Temporary (can delete after alignment)
```

### Training Config
```python
# In config.py or CLI
use_mfa=True
mfa_alignment_dir="./mfa_output/alignments"
```

## Status Checks

### Check MFA Installation
```bash
mfa version
```

### Validate Alignments
```bash
kokoro-preprocess --corpus ./ruslan_corpus --validate-only
```

### Check Training Mode
```bash
# Look for this in training logs:
# "Using MFA forced alignment for phoneme durations (high quality)"
# OR
# "MFA alignment directory not found: ... Falling back to estimated durations"
```

## Typical Timeline

| Task | Time | Notes |
|------|------|-------|
| MFA installation | 5 min | One-time |
| Model download | 2 min | One-time |
| Alignment (22k files) | 1-3 hours | Depends on CPU |
| Training (100 epochs) | 12-24 hours | Depends on GPU |

## Troubleshooting Quick Fixes

### "MFA not found"
```bash
conda install -c conda-forge montreal-forced-aligner
```

### "tgt could not be resolved"
```bash
pip install tgt
```

### "Alignment directory not found"
```bash
# Run preprocessing first
kokoro-preprocess --corpus ./ruslan_corpus
```

### "Low alignment rate (<90%)"
```bash
# Check corpus structure
ls -l ruslan_corpus/wavs/ | head
cat ruslan_corpus/metadata_RUSLAN_22200.csv | head

# Validate
kokoro-preprocess --corpus ./ruslan_corpus --validate-only
```

### Out of Memory During Alignment
```bash
# Reduce parallel jobs
kokoro-preprocess --corpus ./ruslan_corpus --jobs 2
```

## Quality Indicators

### Good MFA Setup
- ✓ Alignment rate: >95%
- ✓ Average duration: 5-15 frames
- ✓ Training logs show "Using MFA"
- ✓ Durations vary (not uniform)

### Issues to Fix
- ✗ Alignment rate: <90%
- ✗ Many "TextGrid not found" warnings
- ✗ Training logs show "estimated durations"
- ✗ All durations nearly identical

## Advanced Options

### Custom MFA Models
```python
from kokoro.data.mfa_integration import MFAIntegration

mfa = MFAIntegration(
    corpus_dir="./corpus",
    output_dir="./output",
    acoustic_model="my_custom_model",
    dictionary="my_custom_dict"
)
```

### Programmatic Access
```python
from kokoro.data.mfa_integration import MFAIntegration

mfa = MFAIntegration("./corpus", "./output")

# Get durations for specific file
durations = mfa.get_phoneme_durations("audio_file_stem")
print(durations)  # [5, 8, 12, 6, ...]

# Validate all
stats = mfa.validate_alignments("./corpus/metadata.csv")
print(f"Aligned: {stats['alignment_rate']*100:.1f}%")
```

## Configuration Priority

```
CLI args > Config file > Defaults

# Priority order:
1. --mfa-alignments ./custom/path
2. config.mfa_alignment_dir
3. "./mfa_output/alignments" (default)
```

## Common Workflows

### First Time Setup
```bash
# Install package
pip install -e .

# Check dependencies
python3 verify_setup.py  # Or move to scripts/

# Run MFA preprocessing
kokoro-preprocess --corpus .

# Train
kokoro-train
```

### Quick Test (No MFA)
```bash
kokoro-train --no-mfa --epochs 5
```

### Production Training
```bash
kokoro-preprocess --corpus . --jobs 16
kokoro-train --batch-size 16 --epochs 200
```

### After Code Update
```bash
# Alignments are cached, just restart training
kokoro-train --resume auto
```

## Documentation Links

- **Full workflow**: [WORKFLOW.md](WORKFLOW.md)
- **MFA details**: [MFA_SETUP.md](MFA_SETUP.md)
- **Implementation**: [MFA_IMPLEMENTATION.md](MFA_IMPLEMENTATION.md)
- **Inference**: [inference.md](inference.md)
