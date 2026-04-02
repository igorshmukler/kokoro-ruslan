# Release Notes

This file tracks releases based on `version=` changes in `setup.py`.

## 0.0.34 (2026-04-02)

### Model size reduction (`config.py`)

- **`hidden_dim` 768 → 512**: cuts total parameters from ~35M to ~16M. With 22K training utterances (~1,600 params/sample at 35M), the model was deep in the overfitting regime. At 16M (~730 params/sample) the capacity/data ratio is far more tractable.
- **`encoder_ff_dim` / `decoder_ff_dim` 2048 → 1536**: GLU FFN expansion rebalanced from 5.3× to 4×, reducing FFN dominance in total parameter count.

### Decoder dropout separation (`config.py`, `model.py`, `trainer.py`)

- **`decoder_dropout: float = 0.25`** (new config field): dedicated dropout rate for decoder attention/FFN residual connections, separate from `encoder_dropout` (0.15). The decoder is more prone to overfitting due to teacher forcing and benefits from stronger regularization.
- **`decoder_input_dropout: float = 0.15`** (new config field): dropout on projected mel input before it enters the decoder (was hardcoded 0.1).
- **Model wiring** (`model.py`): `KokoroModel.__init__` accepts `decoder_dropout` (default `None` → falls back to `encoder_dropout` for backward compatibility). Stored as `self._decoder_dropout` and passed to `TransformerDecoder`.
- **Trainer wiring** (`trainer.py`): passes both `decoder_dropout` and `decoder_input_dropout` from config to `KokoroModel`.

### SpecAugment improvements (`config.py`, `trainer.py`)

- **Reduced masking intensity**: `spec_augment_time_mask_max` 30 → 10, `spec_augment_num_time_masks` 2 → 1, `spec_augment_freq_mask_max` 10 → 5. Worst-case temporal masking drops from ~43% to ~7%, preventing the catastrophic autoregressive pathway disruption observed in the previous run (Ep13 shock of +0.053 that never recovered).
- **Per-sample masking** (`trainer.py`): `_apply_spec_augment` now generates independent masks for each sample in the batch instead of applying one mask batch-wide. Improves gradient diversity across the batch.

### Speed perturbation augmentation (`config.py`, `dataset.py`, `trainer.py`)

- **Audio-level speed perturbation** (`dataset.py`): randomly resamples training audio by a factor in [0.9, 1.1] before feature extraction, effectively multiplying dataset diversity by ~1.5× without additional data. Applied per-sample in `__getitem__` after audio normalization but before mel computation via `torchaudio.functional.resample`.
- **MFA duration rescaling**: phoneme durations are scaled by `1/factor` (speed up → shorter durations) with `clamp(min=1)`. Existing frame-sum correction handles any rounding residual.
- **Cache bypass**: augmented samples skip both cache load and save (stochastic perturbation is incompatible with deterministic caching); unperturbed samples use the cache normally.
- **Training-only**: `is_training` flag added to `RuslanDataset.__init__`; trainer passes `is_training=True` for train, `is_training=False` for validation. Speed perturbation is never applied to validation data.
- New config fields: `use_speed_perturbation: bool = True`, `speed_perturb_range: float = 0.1`, `speed_perturb_prob: float = 0.5`.

### Training schedule (`config.py`)

- **`num_epochs` 60 → 100**: extends OneCycleLR to ~32,700 steps, adding ~67% more cosine-decay refinement time. Combined with the smaller model, estimated val_mel at Ep100: ~0.70 (central), range 0.65–0.74.
- **`early_stopping_patience` 10 → 15**: prevents premature stopping during the SpecAugment adaptation window.

## 0.0.33 (2026-03-28)

### QK-normalization (`transformers.py`, `model.py`, `config.py`, `trainer.py`)

- **Per-head RMSNorm on Q and K projections** (`transformers.py`): `MultiHeadAttentionImproved` now accepts a `qk_norm: bool = False` parameter. When enabled, `nn.RMSNorm(d_k)` is applied independently to Q and K after the linear projection but **before** RoPE. This decouples attention logit magnitudes from the projection weight norms, breaking the self-reinforcing growth loop (larger `w_o` → larger outputs → larger gradients → larger `w_o`) that caused unbounded decoder attention weight growth across training runs.
- **Threaded through all constructor chains**: `qk_norm` is propagated through `ImprovedTransformerEncoderBlock`, `ImprovedTransformerDecoderBlock` (both self-attn and cross-attn), `ImprovedTransformerDecoder`, and the compatibility wrappers `TransformerEncoderBlock` and `TransformerDecoder`.
- **Model constructor** (`model.py`): `KokoroModel.__init__` accepts `qk_norm: bool = False` and passes it to all encoder blocks and the decoder.
- **Config field** (`config.py`): `TrainingConfig.qk_norm: bool = False` added. Set to `True` to enable QK-norm for new training runs. Incompatible with prior checkpoints (new `q_norm`/`k_norm` parameters in state dict).
- **Trainer wiring** (`trainer.py`): `_setup_model()` passes `qk_norm` from config to `KokoroModel`.

## 0.0.32 (2026-03-13)

- Perform post-step norm clamping on all decoder layers
- Further tuning
- Separate LR for stop

## 0.0.31 (2026-03-11)

- Added training analysis script
- Spread heavy batches more evenly across the epoch to prevent clustering
- Added post-step max weight-norm clamp for decoder.layers.0.ff.linear1.weight
- Misc configuration changes to stabilize convergence

## 0.0.30 (2026-03-09)

- Improved tensorboard logs purge upon resuming from a checkpoint
- Misc configuration changes to stabilize convergence

## 0.0.29 (2026-03-08)

- Switched to Xavier init instead of Kaiming
- Implemented RoPE - relative displacement between positions, for MPS
- Misc bug-fixes and configuration adjustments

## 0.0.28 (2026-03-07)

- **Encoder/decoder parameter group split** (`trainer.py`): optimizer now uses two separate AdamW parameter groups. Encoder parameters (`text_embedding`, `stress_embedding`, positional encodings, `transformer_encoder_layers`) are trained at `encoder_lr_multiplier × base_lr` (default 3×); decoder and variance adaptor parameters use `base_lr`. Per-group `max_lr` values are set accordingly in the OneCycleLR schedule and manual warmup ramp. Previously both groups shared the same LR, starving the encoder of gradient signal.
- **Adaptive clip base and explosion floor raised** (`trainer.py`): base gradient clip norm increased from `0.5` to `1.0`; the hard floor applied after an explosion event raised from `0.05` to `0.3`. The `soft_mel_length` threshold was also raised from `900` to `1400` frames to stop penalising normally-sized Russian sequences.
- **`encoder_ffn_spike_clip_norm` loosened** (`config.py`): default raised from `10.0` to `100.0`. The old value was zeroing microscopically small but valid encoder FFN gradients at every step.
- **`duration_loss_weight` raised** (`config.py`): default raised from `0.1` to `0.35`, giving the encoder stronger alignment signal through the duration predictor path.
- **`decoder_input_dropout` reduced** (`model.py`): default in `KokoroModel.__init__` changed from `0.3` to `0.1`. The high value was over-regularising teacher-forced decoding during early training.
- **`max_lr_multiplier` raised** (`config.py`): default raised from `2.0` to `5.0`, widening the OneCycleLR peak to give the model more room to descend in the first epochs.
- **Checkpoint resume multi-group support** (`checkpoint_manager.py`): `resume_from_checkpoint` now reconstructs the OneCycleLR with per-group `max_lr` values (`[max_lr * encoder_lr_mult, max_lr]`) so LR schedules are correctly restored after resume.

### Spec augment gate delayed

- **`spec_augment_start_epoch` raised from 5 to 18** (`config.py`, `trainer.py`): empirical observation showed val_loss regressing from 1.87 → 2.04 across epochs 5–6 when spec augment activated at epoch 5 while the OneCycleLR was still ramping (peak at ~epoch 15 with `pct_start=0.3`, 50 epochs). Starting augmentation 3 epochs after the LR peak eliminates ramp-phase double-destabilisation. The fallback default in the trainer is updated to match.

### Transformer module refactor (`transformers.py`)

- **`GLUFeedForward` extracted as shared module**: the GLU feed-forward block (linear1 → split gate/linear → activation(gate) × linear → linear2) is now a standalone `nn.Module` used by both encoder and decoder layers. Previously each block inlined duplicate `linear1`/`linear2`/`dropout_ff`/`_init_weights` logic.
- **`_build_activation` factory**: activation selection for FFN blocks moved to a single `_build_activation(name)` factory function instead of being duplicated across encoder and decoder block constructors. Raises `ValueError` for unrecognised names.
- **`use_prenorm` removed**: `ImprovedTransformerEncoderBlock` and decoder equivalents no longer accept a `use_prenorm` parameter — pre-norm is now always applied (matching the 0.0.27 change that made pre-norm GELU the fixed architecture). The parameter was dead code.
- **`dropout_ff` deduplicated**: internal feed-forward dropout is now owned by `GLUFeedForward`; the redundant `self.dropout_ff` attribute on the block classes is removed.

### New config fields

| Field | Default | Description |
|-------|---------|-------------|
| `encoder_lr_multiplier` | `3.0` | LR multiplier for the encoder parameter group relative to `learning_rate` |
| `spec_augment_start_epoch` | `18` | Epoch at which SpecAugment is first applied |
| `max_lr_multiplier` | `5.0` | OneCycleLR peak multiplier (was `2.0`) |

### Unit tests added / updated

- `tests/unit/test_optimizer_param_groups.py` — **new**, 19 tests: verifies two-group structure, encoder LR multiplier, decoder base LR, fallback to single group when no encoder params are found, per-group `max_lr` in scheduler.
- `tests/unit/test_spec_augment.py` — added `TestSpecAugmentEpochGate` class (15 tests): gate suppresses augment before `spec_augment_start_epoch`, gate opens exactly at the threshold epoch, `use_spec_augment=False` always suppresses regardless of epoch.
- `tests/unit/test_config_pitch_extraction_defaults.py` — updated `test_training_config_convergence_fix_defaults` to assert new defaults (`spec_augment_start_epoch=18`, `max_lr_multiplier=5.0`, `duration_loss_weight=0.35`, `encoder_lr_multiplier=3.0`, `encoder_ffn_spike_clip_norm=100.0`).
- `tests/unit/test_trainer_adaptive_stabilization.py` — updated two clip-norm threshold assertions to match new values (`0.5→1.0`, `0.05→0.3`).
- `tests/unit/test_transformers.py` — expanded coverage for `GLUFeedForward` and `_build_activation`, removed tests for the deleted `use_prenorm` parameter.

## 0.0.27 (2026-03-06)

- Moved encoder/decoder to pre-norm GELU
- Fixed bugs in forward training, duration predictor
- Only advance scheduler/EMA on successful optimizer steps
- Adopt oversized sequences rather than skip

## 0.0.26 (2026-03-06)

- Refactored trainer

### Bug fixes (decoder regularization)

- **Decoder input dropout now honors constructor config** (`model.py`): `_prepare_training_decoder_inputs` previously applied `torch.nn.functional.dropout(..., p=0.3, ...)` with a hardcoded probability, ignoring `KokoroModel(decoder_input_dropout=...)`. This made architectural tuning ineffective and could over-regularize teacher-forced decoding. Fixed by using `self.decoder_input_dropout` at the dropout call site.

### Unit tests added

- `tests/unit/test_decoder_helpers.py` — added `test_prepare_training_decoder_inputs_uses_configured_decoder_input_dropout`, which monkeypatches `torch.nn.functional.dropout` and asserts that `_prepare_training_decoder_inputs` passes the configured `decoder_input_dropout` value (and `training=True`).

## 0.0.25 (2026-03-05)

### Model architecture update

- Increased default model capacity for new training runs:
  - `hidden_dim`: `512 -> 768`
  - `encoder_ff_dim`: `2048 -> 3072`
  - `decoder_ff_dim`: `2048 -> 3072`
- Updated both training defaults (`training/config.py`) and model construction defaults (`model_loader.py`) so train/infer paths stay aligned.
- Trainer model instantiation now forwards architecture fields (`n_encoder_layers`, `n_heads`, `encoder_ff_dim`, `encoder_dropout`, `n_decoder_layers`, `decoder_ff_dim`, `max_decoder_seq_len`) from `TrainingConfig` instead of relying on constructor defaults.

### Bug fixes

- **OneCycle warmup handoff continuity** (`trainer.py`): when manual warmup is enabled, OneCycleLR now uses `div_factor=max_lr_multiplier` so LR starts exactly at `learning_rate` after warmup, avoiding a warmup→scheduler LR jump.
- **Resume-time scheduler consistency** (`trainer.py`): OneCycle reconstruction now uses the same effective `div_factor` that was used at creation.
- **Stale metadata FF-dim mismatch guard** (`inference.py`): if checkpoint metadata `encoder_ff_dim` / `decoder_ff_dim` disagree with actual `linear1.weight` shapes, inference auto-corrects to weight-derived values and logs a warning, preventing shape-load failures from stale config metadata.

### Unit tests added

- `tests/unit/test_onecycle_warmup_continuity.py` — validates smooth LR continuity across linear warmup into OneCycle phase.

## 0.0.24 (2026-03-05)

### Refactor (model/inference structure)

- **Autoregressive inference extracted** (`model/generator.py`): introduced `KokoroGenerator` to encapsulate generation loop, stop criteria, cache precompute/updates, and inference-time housekeeping.
- **Encode/expand path unified** (`model.py`): added `_encode_and_expand` helper used by both training and inference to reduce divergence between code paths.
- **Training decoder input prep unified** (`model.py`): added `_prepare_training_decoder_inputs` helper for shift/pad, projection, dropout, positional encoding, and mask handling.
- **Duration adaptor interface unified** (`model.py`): both variance-enabled and fallback duration paths now flow through a common adaptor interface (`VarianceAdaptorWrapper` / `SimpleDurationAdaptor`) to keep behavior consistent.

### Bug fixes and compatibility

- **Legacy variance-adaptor checkpoint compatibility** (`trainer.py`): EMA checkpoint loading now handles older key layouts by performing a partial non-strict load and mapping compatible parameters into the current structure.
- **Stop-token loss rebalanced** (`training/config.py`): default `stop_token_loss_weight` reduced from `1.0` to `0.5`.

### Unit tests added

- `tests/unit/test_encode_and_expand.py`
- `tests/unit/test_generator.py`
- `tests/unit/test_model_refactors.py`
- `tests/unit/test_model_log_memory_refactor.py`
- `tests/unit/test_spec_augment.py`
- `tests/unit/test_decoder_helpers.py` (expanded for decoder helper coverage)

## 0.0.23 (2026-03-03)

### Training stability and observability

- **OneCycle tuning** (`training/config.py`): reduced aggressiveness (`max_lr_multiplier: 3.0 -> 2.0`, `pct_start: 0.4 -> 0.3`) for more stable early training.
- **Localized gradient clipping expanded** (`trainer.py`): added dedicated FFN clipping controls (`ffn_spike_clip_norm`, `encoder_ffn_spike_clip_norm`) and extended pre-clip logic to target known spike-prone `linear1/linear2` weights.
- **TensorBoard resume cleanup** (`trainer.py`): writer is reopened with `purge_step` on resume to hide stale events beyond resume step.
- **Scheduler resume reconstruction** (`trainer.py`): OneCycleLR is rebuilt from current config after checkpoint resume to avoid stale schedule boundaries from old optimizer metadata.
- **Histogram logging** (`trainer.py`): added epoch-level parameter histogram logging for better training diagnostics.
- **Checkpoint metrics** (`trainer.py`): `val_loss` is now included in checkpoint payload.

### Model behavior updates

- **Decoder pre-net dropout introduced** (`model.py`): training path applies dropout to projected decoder inputs (`p=0.3`) to reduce teacher-forcing over-reliance.
- **Inference collapse guard** (`model.py`): added energy-based early-stop fallback when recent generated frames indicate prolonged near-silence collapse.

### Unit tests added

- `tests/unit/test_trainer_adaptive_stabilization.py`
- `tests/unit/test_trainer_checkpoint_step_counters.py`
- `tests/unit/test_trainer_loss_stability.py`

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
