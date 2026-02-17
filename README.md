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

```bash
pip install -r requirements.txt
pip install -e .
```

Optional dev extras:

```bash
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

### 1) (Recommended) Precompute features

```bash
kokoro-precompute --corpus ./ruslan_corpus
```

### 2) (Recommended) Run MFA alignment

```bash
kokoro-preprocess --corpus ./ruslan_corpus --output ./mfa_output
```

### 3) Train

```bash
kokoro-train --corpus ./ruslan_corpus --output ./models/kokoro_russian
```

If you hit occasional MPS backend fallback issues:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 kokoro-train --corpus ./ruslan_corpus
```

Module equivalents:

```bash
python -m kokoro.cli.precompute_features --corpus ./ruslan_corpus
python -m kokoro.cli.preprocess --corpus ./ruslan_corpus --output ./mfa_output
python -m kokoro.cli.training --corpus ./ruslan_corpus
```

## `kokoro-train` CLI

| Argument | Short | Default | Description |
|---|---|---|---|
| `--corpus` | `-c` | `./ruslan_corpus` | Corpus directory |
| `--output` | `-o` | `./kokoro_russian_model` | Output model directory |
| `--resume` | `-r` | `None` | Resume from `auto` or checkpoint path |
| `--batch-size` | `-b` | `8` | Batch size |
| `--epochs` | `-e` | `100` | Number of epochs |
| `--learning-rate` | `-lr` | `1e-4` | Base learning rate |
| `--save-every` |  | `2` | Save checkpoint every N epochs |
| `--mfa-alignments` |  | `auto` | MFA alignments directory |
| `--no-mfa` |  | `False` | Disable MFA durations |
| `--val-split` |  | `0.1` | Validation split |
| `--no-validation` |  | `False` | Disable validation |
| `--validation-interval` |  | `1` | Validate every N epochs |
| `--early-stopping-patience` |  | `10` | Early stopping patience |
| `--dynamic-batching` |  | `True` | Enable frame-based dynamic batching |
| `--no-dynamic-batching` |  | `False` | Use fixed-size batching |
| `--max-frames` |  | `config-driven` | Max mel frames per batch |
| `--min-batch-size` |  | `4` | Min dynamic batch size |
| `--max-batch-size` |  | `32` | Max dynamic batch size (may be auto-capped on MPS) |
| `--profile-amp` |  | `False` | Run AMP speed profiling before training |
| `--profile-amp-batches` |  | `10` | Number of AMP profiling batches |
| `--fused-adamw` |  | `False` | Force-enable fused AdamW optimizer (experimental on non-CUDA backends) |
| `--no-fused-adamw` |  | `False` | Force-disable fused AdamW optimizer |
| `--try-fused-adamw-mps` |  | `True` | Try fused AdamW on MPS (experimental, enabled by default, auto-fallback if unsupported) |
| `--verbose` | `-v` | `False` | Enable verbose stabilization diagnostics |

Optimizer flag behavior:
- If neither `--fused-adamw` nor `--no-fused-adamw` is passed, optimizer selection is automatic.
- `--fused-adamw` takes priority unless `--no-fused-adamw` is also passed (which forces disable).
- `--try-fused-adamw-mps` only affects MPS and is enabled by default.

## Training Defaults (Current)

From `TrainingConfig` in `src/kokoro/training/config.py`:

- OneCycle LR enabled, `max_lr_multiplier=3.0`.
- Linear warmup enabled, `warmup_steps=1200`.
- Gradient accumulation default: `2`.
- Dynamic batching default: on.
- Stability safeguards: projection/attention pre-clipping + warmup-aware explosion thresholds.
- MPS-aware auto-limits can reduce oversized values (e.g., frame caps/seq length/batch sizes).

## Useful Commands

```bash
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
python -m kokoro.inference.inference --model ./my_model --text "Привет, это тест." --output output.wav --device mps --stop-threshold 0.6 --min-len-ratio 0.9 --max-len 1600

# Run focused unit tests
python -m pytest tests/unit/test_attention_operations.py tests/unit/test_multi_layer_attention.py tests/unit/test_trainer_adaptive_stabilization.py -q
```

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

This project is intended for educational and research use with the Ruslan corpus and Kokoro-style TTS training workflows.
