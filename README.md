# Kokoro-Ruslan

Russian TTS training pipeline based on the Kokoro architecture, with strong Apple Silicon (MPS) support, MFA duration alignment, feature caching, and stability-focused training defaults.

## What’s New

- Stabilized training defaults for fewer practical gradient spikes (longer warmup + lower OneCycle peak LR pressure).
- Built-in spike safeguards for projection/attention layers and warmup-aware gradient explosion detection.
- Dynamic frame-based batching enabled by default with MPS-aware auto-caps.
- Cleaner diagnostics: detailed stabilization logs are now behind `--verbose`.
- Better CI portability in tests (device-safe behavior when MPS is unavailable).

## Key Features

- MPS-first training flow (with CUDA/CPU fallback).
- MFA integration for phoneme-duration supervision.
- Precomputed feature caching for faster epochs.
- Variance prediction (pitch + energy) for improved prosody.
- Validation + early stopping + checkpoint resume.
- Gradient checkpointing and adaptive memory management.

## Documentation

- [Workflow](docs/development/WORKFLOW.md)
- [Feature Caching](docs/FEATURE_CACHING.md)
- [MFA Setup](docs/setup/MFA_SETUP.md)
- [Variance Predictor](docs/architecture/VARIANCE_PREDICTOR.md)
- [Validation](docs/development/VALIDATION.md)
- [MPS OOM Solutions](docs/MPS_OOM_SOLUTIONS.md)
- [Inference](docs/setup/inference.md)

## Installation

```shell
pip install -r requirements.txt
pip install -e .
```

Recommended explicit environment setup (macOS / Linux)

```shell
# Use a supported Python version (recommended: 3.11)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install package requirements and the project in editable mode
pip install -r requirements.txt
pip install -e .

# Quick verification of core dependencies
python - <<PY
import sys, torch
print('Python', sys.version.split()[0])
print('PyTorch', torch.__version__)
print('MPS available:', getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available())
PY
```

Optional dev extras:

```shell
pip install -e .[dev]
```

## Dataset Layout

Expected corpus structure:

```text
ruslan_corpus/
├── metadata_RUSLAN_22200.csv
└── wavs/
    ├── 000001_RUSLAN.wav
    └── ...
```

Metadata format:

```text
audio_filename|transcription
```

## Quick Start

### 1) (Recommended) Run MFA alignment

```shell
kokoro-preprocess --corpus ./ruslan_corpus --output ./mfa_output
```

Quick verification and smoke tests (minimal reproducible checks)

Minimal smoke-run (small, non-production)

```shell
# If you want to run a very short training run as a sanity check, use --no-mfa and a tiny corpus.
# Replace ./my_small_corpus with a directory that contains a single wav and a metadata CSV in the expected format.
kokoro-train --corpus ./my_small_corpus --output ./tmp_kokoro_test --no-mfa --epochs 1 --batch-size 1 --verbose

# Expected: the run should start, print a few log lines including the model init and dataloader length,
# and either finish one epoch or fail gracefully with a clear error (dataset/corpus misconfiguration).
```

### 2) (Recommended) Precompute features

```shell
kokoro-precompute --corpus ./ruslan_corpus
```

### 3) Train

```shell
kokoro-train --corpus ./ruslan_corpus --output ./models/kokoro_russian
```

If you hit occasional MPS backend fallback issues:

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 kokoro-train --corpus ./ruslan_corpus
```

Module equivalents:

```shell
python -m kokoro.cli.precompute_features --corpus ./ruslan_corpus
python -m kokoro.cli.preprocess --corpus ./ruslan_corpus --output ./mfa_output
python -m kokoro.cli.training --corpus ./ruslan_corpus
```

## `kokoro-train` CLI

Complete CLI flags and interactions (concise):

- `--corpus, -c` (default: `./ruslan_corpus`): Corpus directory containing `metadata_*.csv` and `wavs/`.
- `--output, -o` (default: `./kokoro_russian_model`): Output model directory.
- `--resume, -r` (default: `None`): `auto` or explicit checkpoint path to resume training.
- `--batch-size, -b` (default: `8`): Per-device batch size (subject to dynamic batching overrides).
- `--epochs, -e` (default: `50`): Number of epochs.
- `--learning-rate, -lr` (default: `1e-4`): Base learning rate.
- `--save-every` (default: `2`): Save checkpoint every N epochs.
- `--mfa-alignments` (default: `auto`): Path to MFA `alignments/` directory. Use `--no-mfa` to disable MFA usage.
- `--no-mfa` (flag): Disable MFA and use estimated durations.
- `--val-split` (default: `0.1`): Validation split fraction.
- `--no-validation` (flag): Disable validation entirely.
- `--validation-interval` (default: `1`): Validate every N epochs.
- `--early-stopping-patience` (default: `10`): Early stopping patience.

- Dynamic batching and frame caps:
    - `--dynamic-batching` (enabled by default): Use frame-based dynamic batching.
    - `--no-dynamic-batching` (flag): Use fixed-size batching.
    - `--max-frames` (default: config-driven): Maximum mel frames allowed in a dynamic batch.
    - `--min-batch-size` (default: `4`): Minimum batch size under dynamic batching.
    - `--max-batch-size` (default: `32`): Maximum batch size under dynamic batching (may be auto-capped on MPS).

- Profiling / AMP:
    - `--profile-amp` (flag): Run AMP profiling to select stable AMP usage before training.
    - `--profile-amp-batches` (default: `10`): Number of batches used for AMP profiling.

- Optimizer / fused AdamW flags & interactions:
    - `--fused-adamw` (flag): Force-enable fused AdamW (may only be supported on some backends).
    - `--no-fused-adamw` (flag): Force-disable fused AdamW.
    - `--try-fused-adamw-mps` (default: `True`): Attempt to use fused AdamW on MPS.

Optimizer selection behavior summary:
    - If neither `--fused-adamw` nor `--no-fused-adamw` is set, selection is automatic: fused AdamW is used when the device and PyTorch version support it.
    - `--fused-adamw` forces attempted use; if unavailable it may raise when forced.
    - `--no-fused-adamw` forces the standard `torch.optim.AdamW` implementation.
    - On MPS, `--try-fused-adamw-mps` enables an experimental code path that attempts a fused variant; it will auto-fallback if unsupported.

- Diagnostics and memory:
    - `--verbose, -v` (flag): Enable verbose stabilization diagnostics (duration pred vs target stats, mask counts).
    - `--no-memory-cache` (flag): Disable in-memory feature caching (use on-disk cache only).

Examples:

```shell
# 1) Basic training with MFA (default) and dynamic batching
kokoro-train --corpus ./ruslan_corpus --output ./models/kokoro_russian --batch-size 8 --epochs 50

# 2) Force fused AdamW (may fail if unsupported) or force-disable it
kokoro-train --corpus ./ruslan_corpus --output ./models/kokoro_russian --fused-adamw
kokoro-train --corpus ./ruslan_corpus --output ./models/kokoro_russian --no-fused-adamw

# 3) Try fused AdamW on MPS (experimental) — auto-fallback if not supported
kokoro-train --corpus ./ruslan_corpus --output ./models/kokoro_russian --try-fused-adamw-mps

# 4) Minimal debugging run: single epoch, no MFA, verbose logs for duration diagnostics
kokoro-train --corpus ./my_small_corpus --output ./tmp_kokoro_test --no-mfa --epochs 1 --batch-size 1 --verbose

# 5) Explicit alignment directory
kokoro-train --corpus ./ruslan_corpus --output ./my_model --mfa-alignments ./mfa_output/alignments
```

Notes:
- If you see fused-optimizer errors on startup, pass `--no-fused-adamw` to force the fallback optimizer and avoid runtime crashes.
- The `--try-fused-adamw-mps` flag is safe: it will attempt the fused code path on Apple Silicon and fall back when necessary, but behavior can vary by PyTorch version.
- `--verbose` prints helpful diagnostics (duration pred vs target mean/std/min/max and phoneme mask counts) useful when diagnosing duration-loss convergence.

## Training Defaults (Current)

From `TrainingConfig` in `src/kokoro/training/config.py`:

- OneCycle LR enabled, `max_lr_multiplier=3.0`.
- Linear warmup enabled, `warmup_steps=1200`.
- Gradient accumulation default: `2`.
- Dynamic batching default: on.
- In-memory feature cache: enabled by default. Use `--no-memory-cache` to disable keeping precomputed features in RAM (reduces host memory usage at cost of slightly higher I/O and cache latency).
- Stability safeguards: projection/attention pre-clipping + warmup-aware explosion thresholds.
- MPS-aware auto-limits can reduce oversized values (e.g., frame caps/seq length/batch sizes).

## Useful Commands

```shell
# Verify feature cache health before training
python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --status

# Resume automatically from latest checkpoint
kokoro-train --corpus ./ruslan_corpus --output ./models/kokoro_russian --resume auto

# Train without MFA durations
kokoro-train --corpus ./ruslan_corpus --no-mfa

# Train with explicit dynamic batching bounds
kokoro-train --corpus ./ruslan_corpus --max-frames 18000 --min-batch-size 4 --max-batch-size 12

# Force fused AdamW (or force-disable it)
kokoro-train --corpus ./ruslan_corpus --fused-adamw
kokoro-train --corpus ./ruslan_corpus --no-fused-adamw

# Fused AdamW on MPS is enabled by default (experimental)
kokoro-train --corpus ./ruslan_corpus --try-fused-adamw-mps

# Inference from final model or latest checkpoint in a model directory
python -m kokoro.inference.inference --model ./my_model --text "Привет, это тест." --output output.wav --device mps

# Inference tuning (helps early checkpoints avoid very short outputs)
# Note: an explicit `--stop-threshold` passed on the CLI overrides any
# checkpoint-tuned or internal model default and will be honored during
# generation.
python -m kokoro.inference.inference --model ./my_model --text "Привет, это тест." --output output.wav --device mps --stop-threshold 0.6 --min-len-ratio 0.9 --max-len 1600

# Run focused unit tests
python -m pytest tests/unit/test_attention_operations.py tests/unit/test_multi_layer_attention.py tests/unit/test_trainer_adaptive_stabilization.py -q
```

# TensorBoard / Logs

```shell
# The trainer writes logs to `<output_dir>/logs` (SummaryWriter) and profiler traces to
# `<output_dir>/profiler_logs/<timestamp>`. Example:
tensorboard --logdir my_model/logs --bind_all

# To view profiler traces (TensorBoard Profiler) you can point TensorBoard at the profiler
# directory or the parent `profiler_logs` directory:
tensorboard --logdir my_model/profiler_logs --bind_all
```

What to look for in TensorBoard:
- Scalars: `total_loss`, `mel_loss`, `duration_loss`, `stop_token_loss`, `pitch_loss`, `energy_loss`.
- Histograms: model parameter distributions and gradients (if enabled).
- Graph: model graph (if exported) and profiler timelines (CPU/GPU/MPS activity).

Tips:
- If running on a remote machine, forward the tensorboard port (default 6006) to your local machine.
    Example: `ssh -L 6006:localhost:6006 user@remote` then run tensorboard on remote and open http://localhost:6006 locally.
- Use `--verbose` during a short validation run to get additional diagnostics printed to the training logs that complement TensorBoard (duration pred vs target stats, mask counts).


## Troubleshooting

- MPS out-of-memory: lower `--max-frames` and/or `--batch-size`; see [MPS OOM Solutions](docs/MPS_OOM_SOLUTIONS.md).
- Missing metadata/audio: verify corpus layout and `metadata_RUSLAN_22200.csv`.
- Slower-than-expected startup: first epoch may build caches; precompute features to speed up.
- Gradient spike warnings: use defaults first, then reduce `--learning-rate` or `--max-frames` if needed.

## Output Artifacts

Typical output directory:

```text
models/kokoro_russian/
├── checkpoint_epoch_2.pth
├── checkpoint_epoch_4.pth
├── ...
└── kokoro_russian_final.pth
```

## Contributing

PRs are welcome. For larger changes, open an issue first so implementation direction is aligned.

## License

This project is intended for educational and research use with the Ruslan corpus and Kokoro-style TTS training workflows. Contact the owner with questions and/or for commercial usage.
