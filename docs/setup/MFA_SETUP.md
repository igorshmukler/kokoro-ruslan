# Montreal Forced Aligner (MFA) Setup Guide

This guide explains how to set up and use Montreal Forced Aligner (MFA) to extract accurate phoneme durations for high-quality TTS training.

## Why MFA?

Montreal Forced Aligner provides **accurate phoneme-level alignments** by analyzing the audio and transcription together. This is crucial for TTS quality because:

- **Natural timing**: Phonemes have realistic durations instead of estimates
- **Better prosody**: The model learns actual speech patterns
- **Improved quality**: Results in more natural-sounding synthetic speech

## Installation

### Option 1: Conda (Recommended)

Conda provides the most reliable and reproducible installation for MFA and Kaldi dependencies (recommended for Linux and macOS):

```shell
# Create a dedicated environment (recommended Python 3.11)
conda create -n kokoro python=3.11 -y
conda activate kokoro

# Install MFA + Kaldi and helper packages from conda-forge
conda install -c conda-forge montreal-forced-aligner kalpy kaldi -y

# Optional: install TextGrid parsing helper
pip install tgt
```

Notes:
- `kaldi` is large and easiest to install via `conda-forge`.
- Use the conda environment when running `kokoro-preprocess`/`kokoro-train` so the MFA binaries are on `PATH`.

### Option 2: Pip (works but less predictable)

Installing via `pip` can work on many systems but often requires a system-level Kaldi installation or the use of prebuilt wheels. Use this if you cannot use conda:

```shell
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install montreal-forced-aligner tgt
```

If `pip` install fails due to missing Kaldi or system libraries, prefer the `conda` approach or use the Docker image below.

### Option 3: Docker (isolated, reproducible)

If you cannot modify the host environment, use the official MFA Docker image or a community image with Kaldi preinstalled. This is robust for CI or heterogeneous compute.

```shell
# Example (pseudo-commands; adapt to MFA docker image you choose):
docker run --rm -v $(pwd):/data montreal-forced-aligner/montreal-forced-aligner:latest \
    align /data/ruslan_corpus /data/mfa_output --config /data/mfa_config.json
```

## Quick Start

### 1. Run MFA Alignment

Use the standalone script to align your corpus:

```shell
# Basic usage
kokoro-preprocess --corpus ./ruslan_corpus --output ./mfa_output

# With custom settings
kokoro-preprocess --corpus ./ruslan_corpus --output ./mfa_output --metadata metadata_RUSLAN_22200.csv --jobs 8

# Module form
python -m kokoro.cli.preprocess --corpus ./ruslan_corpus --output ./mfa_output --jobs 8
```

**What this does:**
1. Downloads Russian MFA models (acoustic model + dictionary)
2. Prepares your corpus in MFA format
3. Runs forced alignment
4. Validates results and creates alignment cache

### 2. Train with MFA Alignments

Once alignment is complete, training automatically uses the alignments:

```shell
# MFA is enabled by default
kokoro-train --corpus ./ruslan_corpus --output ./my_model

# Specify custom alignment directory
kokoro-train \
    --corpus ./ruslan_corpus \
    --output ./my_model \
    --mfa-alignments ./mfa_output/alignments

# Disable MFA (use estimated durations)
kokoro-train --corpus ./ruslan_corpus --output ./my_model --no-mfa
```

## Expected Output

After running MFA alignment, you'll have:

```
mfa_output/
├── alignments/              # TextGrid files with phoneme timings
│   ├── 005559_RUSLAN.TextGrid
│   ├── 005560_RUSLAN.TextGrid
│   └── ...
├── alignment_cache/         # Cached duration tensors (faster loading)
│   ├── 005559_RUSLAN.pkl
│   ├── 005560_RUSLAN.pkl
│   └── ...
└── mfa_corpus/             # Temporary MFA-formatted corpus
        ├── 005559_RUSLAN.wav
        ├── 005559_RUSLAN.txt
        └── ...
```

Important filesystem conventions and what `kokoro` expects

- Default alignment directory used by `kokoro-preprocess` / `kokoro-train` is `./mfa_output/alignments` (TextGrid files).
- `kokoro` also creates/reads `./mfa_output/alignment_cache` containing pickled duration tensors for fast loading during training.
- File name matching: `kokoro` expects TextGrid stems to match the `audio_filename` values listed in your `metadata_*.csv` file (audio stem, without extension). Example:
    - metadata entry: `005559_RUSLAN.wav|Привет`
    - expected TextGrid: `mfa_output/alignments/005559_RUSLAN.TextGrid`
    - expected cache: `mfa_output/alignment_cache/005559_RUSLAN.pkl`
- TextGrid parsing is case-sensitive on some filesystems; ensure consistent stems and extensions.

How to point `kokoro-train` at a custom alignment directory:

```shell
kokoro-train --corpus ./ruslan_corpus --output ./my_model --mfa-alignments ./custom_mfa/alignments
```

Quick verification commands after alignment:

```shell
# Count TextGrid files
ls -1 mfa_output/alignments | wc -l

# Check that cached .pkl files exist and match count
ls -1 mfa_output/alignment_cache | wc -l

# Print a small excerpt of a TextGrid to confirm formatting
python - <<PY
from pathlib import Path
from tgt import read_textgrid
tg = read_textgrid('mfa_output/alignments/005559_RUSLAN.TextGrid')
print(tg.get_tier_by_name('phones').get_intervals()[:5])
PY
```

## Validation

The script automatically validates alignments and reports:

```
Alignment validation: 22150/22200 files aligned (99.8%)
Average phoneme duration: 8.3 frames
```

**Good alignment rates:** 95%+ is excellent, 90%+ is good

**If alignment rate is low:**
- Check audio quality (noisy files may fail)
- Verify transcriptions match audio
- Ensure metadata format is correct

## Troubleshooting

### MFA Command Not Found

```shell
# Check if MFA is installed
mfa version

# If not found, reinstall
conda install -c conda-forge montreal-forced-aligner
```

### Model Download Fails

```shell
# Manually download models
mfa model download acoustic russian_mfa
mfa model download dictionary russian_mfa

# List available models
mfa model list acoustic
mfa model list dictionary
```

### Low Alignment Success Rate

1. **Check audio quality**: Remove very noisy files
2. **Verify transcriptions**: Ensure text matches audio
3. **Check metadata format**: Should be `filename|transcription`

### Duration Mismatch Warnings

Occasional warnings like "MFA duration length != phoneme length" are normal due to:
- Different phoneme representations
- MFA's internal tokenization

The system automatically handles these mismatches.

## Advanced Usage

### Using Custom MFA Models

```python
from kokoro.data.mfa_integration import MFAIntegration

mfa = MFAIntegration(
    corpus_dir="./ruslan_corpus",
    output_dir="./mfa_output",
    acoustic_model="custom_russian_model",  # Your custom model
    dictionary="custom_dictionary"          # Your custom dictionary
)
```

### Extracting Phoneme Alignments Programmatically

```python
from kokoro.data.mfa_integration import MFAIntegration

mfa = MFAIntegration("./ruslan_corpus", "./mfa_output")

# Get durations for specific file
durations = mfa.get_phoneme_durations("005559_RUSLAN")
print(f"Phoneme durations (in frames): {durations}")

# Parse TextGrid directly
alignments = mfa.parse_textgrid("./mfa_output/alignments/005559_RUSLAN.TextGrid")
for word in alignments:
    print(f"Word: {word.word}, Time: {word.start_time:.2f}-{word.end_time:.2f}s")
    for phoneme in word.phonemes:
        print(f"  {phoneme.phoneme}: {phoneme.duration:.3f}s ({phoneme.duration_frames} frames)")
```

### Batch Processing Large Corpora

For very large datasets:

```shell
# Use more parallel jobs
kokoro-preprocess --corpus ./large_corpus --output ./mfa_output --jobs 16

# Or split into batches and process separately
kokoro-preprocess --corpus ./batch1 --output ./mfa_output_1 --jobs 8
kokoro-preprocess --corpus ./batch2 --output ./mfa_output_2 --jobs 8
```

## Performance Notes

- **First run**: Downloads models (~300MB), takes longer
- **Alignment time**: ~1-2 seconds per audio file (varies with CPU)
- **Storage**: TextGrid files are small (~1-5KB each)
- **Memory**: Minimal, can process large corpora

## Integration with Training

The training pipeline automatically:
1. Checks for MFA alignments in configured directory
2. Loads alignments from cache for fast iteration
3. Falls back to estimated durations if MFA unavailable
4. Logs which duration source is being used

**Training logs will show:**
```
Using MFA forced alignment for phoneme durations (high quality)
```

Or if MFA is not available:
```
MFA alignment directory not found: ./mfa_output/alignments
Falling back to estimated durations
```

## Quality Comparison

**Estimated Durations (without MFA):**
- All phonemes get roughly equal duration
- No variation in speech rate
- Less natural prosody

**MFA Durations:**
- Realistic phoneme-specific durations
- Natural speech rate variations
- Vowel lengthening in stressed syllables
- Better overall prosody

**Expected quality improvement**: 20-40% better naturalness scores (MOS)

## Resources

- [MFA Documentation](https://montreal-forced-aligner.readthedocs.io/)
- [MFA GitHub](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
- [Russian MFA Models](https://mfa-models.readthedocs.io/en/latest/acoustic/Russian/index.html)

## Support

If you encounter issues:
1. Check MFA logs in `./mfa_output/`
2. Validate your corpus format matches requirements
3. Try alignment on a small subset first
4. Check MFA GitHub issues for known problems
