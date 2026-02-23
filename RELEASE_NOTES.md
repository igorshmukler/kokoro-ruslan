# Release Notes

This file tracks releases based on `version=` changes in `setup.py`.

## 0.0.17 (2026-02-23)
- Pre-allocate chunk slices to reduce memory pressure

## 0.0.17 (2026-02-22)
- Inference improvements

## 0.0.16 (2026-02-22)
- Minor GPU memory optimizations

## 0.0.15 (2026-02-22)
- Implemented adaptive bucketed batching

## 0.0.14 (2026-02-22)
- Vectorized expansion in length regulator and variance adaptor

## 0.0.13 (2026-02-21)
- Improved pitch extractor
- Implemented length regulator
- Improved checkpointing
- Improved phoneme processor
- Vectorized average pitch by duration

## 0.0.12 (2026-02-18)
- Fixed pitch and energy normalization bugs

## 0.0.11 (2026-02-18)
- Improved feature cache (more work needed)
- Added auto EMA decay calculation

## 0.0.10 (2026-02-17)
- Make DataLoader workers configurable via `TrainingConfig.num_workers` and wire `prefetch_factor`/`persistent_workers` appropriately.
- Auto-tune inference controls per-checkpoint (`stop_threshold`, `min_len_ratio`, `max_len`, `min_len_floor`) from `model_metadata.inference_controls` with safe bounds and explicit-override behavior.
- Add epoch-level feature-cache hit/miss delta summaries and a final cumulative "FEATURE CACHE SUMMARY" at training completion for improved observability.
- Add/adjust unit tests covering metadata strictness, inference auto-tuning, and cache telemetry.

## 0.0.9 (2026-02-17)
- Save and restore model metadata with checkpoints. **BREAKING CHANGE**

## 0.0.8 (2026-02-17)
- Data loader optimizations.

## 0.0.7 (2026-02-17)
- Variance predictor rework.

## 0.0.6 (2026-02-17)
- Improved checkpointing, inference, and userland tooling.

## 0.0.5 (2026-02-16)
- AdamW optimizer enabled with MPS.

## 0.0.4 (2026-02-16)
- Documentation and unit-test cleanup.

## 0.0.3 (2026-02-14)
- Increased frame budget.
- Minor transformer improvements.

## 0.0.2 (2026-02-13)
- MPS memory cleanup.

## 0.0.1
- Moved to 0.0.x versioning.
- Refactored code.

## 0.1.0 (before 2026-02-12)
- Historical transition version recorded in `setup.py` history before moving to the 0.0.x versioning.
