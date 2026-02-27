# Release Notes

This file tracks releases based on `version=` changes in `setup.py`.

## 0.0.22 (2026-02-27)

### New feature — stress parallel embedding

`KokoroModel` now accepts an optional `stress_indices` tensor that is embedded via a dedicated `nn.Embedding` table and added to the encoder input in parallel with the phoneme embedding. This gives the model an explicit signal for which syllable carries lexical stress in each word.

- **`model.py`**: Added `stress_embedding: nn.Embedding(vocab_size, d_model)` to `KokoroModel.__init__`. `encode_text` sums `phoneme_embed + stress_embed` before the positional encoding layer. `stress_indices` defaults to `None` (zero-vector contribution) so the change is fully backward compatible with checkpoints that pre-date it.
- **`dataset.py`**: `RuslanDataset.__getitem__` now calls `audio_utils.get_stress_indices_with_sil` to produce a per-phoneme stress index tensor and stores it in the feature cache under key `stress_indices`. `collate_fn` pads and stacks the new field.
- **`trainer.py`**: All six `model(...)` call sites in `train_epoch`, `_run_single_batch`, and `validate_epoch` forward `stress_indices=stress_indices` from the batch.
- **`inference.py`**: `text_to_speech` constructs `stress_indices` from `RussianPhonemeProcessor.process_text` and passes it to the model.
- **`model_loader.py`**: Checkpoint metadata is extended to record stress embedding presence; missing keys are patched in at load time for smooth migration from old checkpoints.
- **`audio_utils.py`**: New `get_stress_indices_with_sil` helper builds a per-phoneme integer tensor from `StressInfo`, inserting `0` for silence tokens.
- **`FEATURE_CACHE_VERSION` bumped to 6** to invalidate cache entries that do not contain `stress_indices`.

### Bug fixes (phoneme processor)

- **Iotated vowel `j`-prefix dropped in `apply_vowel_reduction`** (`russian_phoneme_processor.py`): Unstressed iotated vowels (`ja`, `je`, `jo`) were reduced to bare `ɐ`/`ɪ`/`ə`, silently discarding the `j`. For example, the initial `я` in unstressed `язы́к` produced `ɐ` instead of `jɐ`. Fixed by tracking `is_iotated` before stripping the base, then prepending `j` to the reduced form when a reduction actually occurred. Non-reducible iotated vowels (e.g. `ju`) are left untouched. New reduced-iotated phonemes `jɐ`, `jɪ`, `jə` added to `_multi_char_phonemes`, `_build_vocab`, and the `from_dict` forward-compatibility patch.
- **`logging.basicConfig` removed from module scope** (`russian_phoneme_processor.py`): The module was unconditionally installing a `StreamHandler` on the root logger at import time, hijacking log configuration in any host application. Removed; module-level `logger = logging.getLogger(__name__)` retained.
- **`@lru_cache` memory leak on instance methods** (`russian_phoneme_processor.py`): Python's `functools.lru_cache` applied as a decorator to instance methods keeps a strong reference to `self` in every cache key, preventing garbage collection for the lifetime of the process. Replaced with per-instance caches created in `__init__` (`self.normalize_text = lru_cache(1000)(self._normalize_text_impl)`), so the cache is released when the instance is collected.
- **Combining marks stripped too late in `apply_consonant_assimilation`** (`russian_phoneme_processor.py`): NFD stress diacritics embedded in a word (e.g. `здра́вствуйте`) were stripped only after all Cyrillic `str.replace` cluster patterns, causing every cluster simplification (`вств→ств`, `тся→ца`, `стн→сн`, `сч→щ`, etc.) to silently fail on marked input. The `re.sub(r'[\u0300-\u036f]', '', word)` call is now the first operation after `word.lower()`.
- **`_int_to_words` missing billions tier** (`russian_phoneme_processor.py`): Numbers ≥ 1 000 000 000 fell into the thousands branch, producing nonsensical output (e.g. `1 000 000 000` → `"одна тысяча миллионов"`). Added a dedicated billions block with correct Russian склонение (`миллиард` / `миллиарда` / `миллиардов`).
- **`get_stress_indices_with_sil` crash on `stress_info=None`** (`audio_utils.py`): `DummyProcessor` returns 3-tuples with `None` as the stress field during testing. The vowel-count comparison `vowel_count == stress_info.position` raised `AttributeError`. Fixed by defaulting `stress_position = stress_info.position if stress_info is not None else -1`.

### Unit tests added

- `tests/unit/test_phoneme_processor_fixes.py` — 23 tests across four classes:
  - `TestNoRootLoggerHijack` — confirms root logger has zero handlers after module import.
  - `TestPerInstanceLRUCache` — `weakref` GC check, two-instance cache isolation, `clear_cache` scoping.
  - `TestStressMarkStrippedBeforeAssimilation` — cluster simplifications fire correctly on words with embedded combining marks; end-to-end IPA for `здравствуйте`.
  - `TestIotatedJPrefixPreservedInReduction` — each iotated vowel in each reduction tier, `ju` non-reduction, vocab/tokenizer presence, and end-to-end word tests (`язык`, `яблоко`).

## 0.0.21 (2026-02-26)

### Critical bug fixes (pipeline correctness)

- **Energy axis layout fix** (`dataset.py`): `torchaudio.MelSpectrogram` returns `(n_mels, T)` after squeeze; `EnergyExtractor.extract_energy_from_mel` expects `(..., n_mels)` on the last axis. Without the transpose, `mean(dim=-1)` averaged over the time axis and produced 80 per-band scalars instead of `T` per-frame energy values. Fixed by passing `mel_spec_linear[:, :num_mel_frames].T`.
- **`mel_spec_linear` clip sync fix** (`dataset.py`): `mel_spec` was clipped to `max_seq_length` frames but `mel_spec_linear` was not, leaving the two tensors out of sync. Both are now clipped together at the source.
- **`FEATURE_CACHE_VERSION` bumped to 3** (`dataset.py`): All cache entries written before the axis layout and clip fixes contain corrupted energy values. Bumping the version forces automatic re-computation of any stale cache entry rather than silently serving wrong data.
- **`duration_target` expansion bug** (`model.py`): `VarianceAdaptor.LengthRegulator` casts durations to `.long()` immediately. The old code passed `torch.log1p(phoneme_durations)` (e.g. `1.79` for a 5-frame phoneme), which truncated to `1` frame per phoneme. Fixed by passing raw integer frame counts `phoneme_durations.float()`; the log-domain target for the duration MSE loss is computed separately in the trainer.
- **Auto-recovery attribute path fix** (`trainer.py`): `self.model.pitch_predictor._init_weights()` was a silent no-op because `KokoroModel` has no top-level `pitch_predictor`. Fixed to `self.model.variance_adaptor.pitch_predictor._init_weights()`.
- **`forward_inference` discarded pitch/energy embeddings** (`model.py`): The inference path called `variance_adaptor` only to obtain predicted durations, then discarded `adapted_output` and re-ran `_length_regulate` on the bare encoder output — dropping all pitch and energy embeddings. Fixed by using `adapted_encoder_output` directly, matching training behavior.

### Unit tests added
- `tests/unit/test_dataset_energy_axis_layout.py` — 21 tests covering the `(n_mels, T)→(T, n_mels)` transpose contract, the pre-clip slice pattern `[:, :num_mel_frames].T`, per-frame energy content correctness, batch layout, and documentation of the pre-fix wrong-shape behaviour as a regression canary.
- `tests/unit/test_model_inference_adaptor_output.py` — fixed `test_variance_adaptor_called_exactly_once_during_inference`: replaced `patch.object(model, 'variance_adaptor', ...)` (rejected by `nn.Module.__setattr__`) with `patch.object(model.variance_adaptor, 'forward', ...)`.

## 0.0.20 (2026-02-25)
- Added unit tests for `<sil>` token handling and fallback behavior
- Added diagnostic logging for duration predictions vs. targets in the trainer (`_calculate_losses`), gated by `config.verbose` for local debugging of duration-loss convergence
- Fixed unit test expectations to match the trainer's current log-duration target computation (uses +1.0 in the log target), preventing false failures in CI/local runs

## 0.0.19 (2026-02-24)
- Improved preprocessing with better silence support
- Lowered default MPS_MAX_FRAMES_PER_BATCH
- Cleanup

## 0.0.18 (2026-02-23)
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
