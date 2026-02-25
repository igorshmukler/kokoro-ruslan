# Pitch/Energy Tensor Shape Mismatch Fix

## Problem

Training crashed with the following error:
```
RuntimeError: The size of tensor a (157) must match the size of tensor b (1234) at non-singleton dimension 1
```

This occurred in the pitch loss calculation when comparing `predicted_pitch` and `pitch_targets`.

## Root Cause

The variance predictor (FastSpeech 2 style) operates at different granularities:

1. **Dataset** extracts pitch/energy at **mel-frame level** (one value per mel frame)
   - Shape: `[batch_size, mel_frames]` (e.g., `[32, 1234]`)
   - Represents continuous acoustic features

2. **Model's variance predictor** outputs predictions at **phoneme level** (one value per phoneme)
   - Shape: `[batch_size, phonemes]` (e.g., `[32, 157]`)
   - Represents linguistic features

3. **Trainer** was comparing mismatched levels:
   - `predicted_pitch`: phoneme-level (157 elements)
   - `pitch_targets`: mel-frame level (1234 elements)
   - **Result**: Shape mismatch error

## Why This Design?

The model predicts pitch/energy at phoneme level because:
- Phonemes are the linguistic units with semantic meaning
- Each phoneme can span multiple mel frames (via durations)
- This matches FastSpeech 2 architecture where variance is predicted per phoneme

The dataset provides mel-frame level targets because:
- Acoustic features (pitch, energy) are extracted from the audio waveform
- Extraction is frame-synchronous with mel spectrogram generation
- More granular data can be averaged to phoneme level

## Solution

Convert mel-frame level targets to phoneme-level **before** loss calculation by averaging frame-level values according to phoneme durations.

### Implementation

Added `_average_pitch_energy_by_duration()` method to Trainer class:

```python
def _average_pitch_energy_by_duration(self,
                                      values: torch.Tensor,
                                      durations: torch.Tensor,
                                      phoneme_lengths: torch.Tensor) -> torch.Tensor:
    """
    Average frame-level values (pitch/energy) to phoneme-level using durations

    For each phoneme:
    1. Get its duration (number of mel frames)
    2. Average the corresponding mel-frame values
    3. Return phoneme-level averaged value
    """
```

### Example

Given:
- Phoneme sequence: `['p', 'a', 't']` (3 phonemes)
- Durations: `[10, 15, 8]` (frames per phoneme)
- Pitch (mel-frame): `[100.5, 101.2, ..., 98.3]` (33 values)

Conversion:
- Phoneme 'p' (frames 0-9): pitch = avg(100.5, 101.2, ...) = 100.8
- Phoneme 'a' (frames 10-24): pitch = avg(...) = 102.5
- Phoneme 't' (frames 25-32): pitch = avg(...) = 98.1

Result:
- Pitch (phoneme-level): `[100.8, 102.5, 98.1]` (3 values)

## Changes Made

### File: trainer.py

1. **Added conversion method** (before `_calculate_losses`)
   - `_average_pitch_energy_by_duration()`: Converts mel-frame to phoneme-level

2. **Updated `train_epoch()`** (before loss calculation)
   - Convert `pitches` and `energies` from mel-frame to phoneme-level
   - Pass converted values to `_calculate_losses()`

3. **Updated `validate_epoch()`** (before loss calculation)
   - Same conversion for validation loop
   - Ensures consistent loss calculation

### Before (Broken)
```python
# Loss calculation with mel-frame level targets
total_loss, ... = self._calculate_losses(
    ...,
    predicted_pitch, predicted_energy,
    pitches, energies  # ❌ Shape: [batch, mel_frames]
)
```

### After (Fixed)
```python
# Convert targets to phoneme level
phoneme_pitches = self._average_pitch_energy_by_duration(
    pitches, phoneme_durations, phoneme_lengths
)
phoneme_energies = self._average_pitch_energy_by_duration(
    energies, phoneme_durations, phoneme_lengths
)

# Loss calculation with phoneme-level targets
total_loss, ... = self._calculate_losses(
    ...,
    predicted_pitch, predicted_energy,
    phoneme_pitches, phoneme_energies  # ✅ Shape: [batch, phonemes]
)
```

## Validation

Created `test_pitch_energy_conversion.py` to verify:
- ✅ Correct shape conversion (mel-frames → phonemes)
- ✅ Accurate averaging over duration windows
- ✅ Proper handling of batched data
- ✅ Padding handled correctly

## Impact

- **Training**: Now runs without shape mismatch errors
- **Validation**: Loss calculation matches training methodology
- **Performance**: Minimal overhead (simple averaging operation)
- **Accuracy**: Proper comparison of predicted vs target pitch/energy

## Testing

Run training to verify the fix:
```shell
kokoro-train --corpus ./ruslan_corpus --val-split 0.1
```

Expected behavior:
- No tensor shape mismatch errors
- Pitch/energy losses computed correctly
- Training proceeds normally

## Technical Details

### Duration Alignment

The conversion ensures phoneme-level targets align with predictions:

```
Mel frames:    [---10 frames---][------15 frames------][--8 frames--]
Pitch values:  [100.5 ... 101.2][102.3 ... 103.1]     [97.5 ... 98.3]
                      ↓                  ↓                    ↓
Phonemes:           'p'                'a'                  't'
Phoneme pitch:     100.8              102.5                98.1
                    ↕                  ↕                    ↕
Predicted:         101.2              102.3                97.8
```

Loss is computed per phoneme:
- `loss_pitch = MSE(predicted, target)` at phoneme level
- Masked to exclude padding phonemes
- Averaged across valid phonemes

### Gradient Flow

The conversion is done on targets (no gradients), so it doesn't affect backpropagation:
- Predictions: From model (requires_grad=True)
- Targets: Converted from dataset (requires_grad=False)
- Loss: MSE between predictions and converted targets
- Gradients: Flow back through predictions to model parameters

## Related Files

- `model.py`: Variance predictor outputs phoneme-level predictions
- `dataset.py`: Extracts mel-frame level pitch/energy
- `trainer.py`: Converts targets and calculates losses (FIXED)
- `test_pitch_energy_conversion.py`: Validation test

## Future Enhancements

Possible improvements:
1. Cache phoneme-level targets in dataset (avoid repeated conversion)
2. Weighted averaging (consider frame importance)
3. Alternative conversion methods (median, max, etc.)
4. GPU-optimized vectorized implementation
