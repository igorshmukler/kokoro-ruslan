# Model Training Guide

## Overview
This guide summarizes expected loss behavior during training, how to detect over-smoothing, epoch expectations, and quick commands to test a trained model.

### The Alignment Penalty (The "Blurry Boundary" Problem)
Energy changes drastically at phoneme boundaries — think of the jump from a silent "p" to a loud "a." When dur_loss is still relatively high, the model is likely predicting the timing of these transitions slightly off.

If the ground truth says the energy should drop at frame 40, but the duration predictor thinks it should drop at frame 42, the MSE loss sees a "huge" error for those 2 frames. Pitch is often more continuous across transitions, so it doesn't get "punished" as severely for minor timing offsets as energy does.

### Information Redundancy
Energy is "double-booked." It carries two types of information at once:

Phonetic Info: Consonants are quiet; vowels are loud.

Prosodic Info: We speak louder to emphasize specific words.

The model has to learn to separate the "volume" that comes from the letter being spoken from the "volume" that comes from the emotion or stress in the sentence. Pitch is mostly just prosodic, making it a "cleaner" signal for the model to isolate.

Once dur_loss drops below 0.3, the energy_loss should finally cave in and follow.

## Loss Phases (Mel Loss)
- 0.8 – 1.0 — "Alien": The model places sounds in roughly the right locations but the timbre and fine details are not learned yet.
- 0.4 – 0.6 — "Static": Words become intelligible, but the voice sounds like it's behind a wall of noise (a rain-like texture).
- Below 0.3 — Professional quality: Background noise and digital artifacts largely disappear.

Note: Mel Loss is a helpful coarse metric, but always interpret it together with Spectral Convergence (SpectralConv).

## Spectral Convergence & Over-smoothing
- If Mel Loss continues to drop while Spectral Convergence plateaus or rises, the model may be optimizing global spectral averages at the expense of fine details (a form of over-smoothing).
- When both Mel Loss and Spectral Convergence drop together, it generally indicates healthy learning.

Most definitive check for over-smoothing: inspect the Mel-spectrogram (visual check).

### How to read the spectrogram
- Healthy spectrogram: sharp vertical lines (glottal pulses) and distinct horizontal bands (formants).
- Over-smoothed spectrogram: blurry or "smudged" appearance where horizontal bands bleed together and vertical detail is lost.

## Epoch Expectations
- Around Epoch 20: the "digital crunch" artifacts should subside and you can judge whether the voice sounds "crisp" or "muffled".
- For a Russian corpus of ~2200 samples: expect roughly 30–50 epochs before the voice sounds convincingly human.
- Training in fp32 on MPS: convergence tends to be more stable but may appear slower compared to other setups.

## Quick Monitoring Tips
- Track both Mel Loss and Spectral Convergence concurrently — diverging trends are a red flag.
- If you see over-smoothing (spectrogram blur) while Mel Loss keeps improving, consider debugging by:
	- visualizing spectrograms from intermediate checkpoints
	- temporarily lowering the learning rate or adjusting loss weighting

## Example commands to generate test samples
Long test (Russian):

```bash
python -m kokoro.inference.inference --model ./my_model --text "В дебрях заброшенного леса слышны странные звуки, не так ли?" --output output.wav --device mps
```

Short test (Russian):

```bash
python -m kokoro.inference.inference --model ./my_model --text "Привет, это тест." --output output.wav --device mps
```

## Summary
Use Mel Loss as a coarse progress indicator and Spectral Convergence plus visual spectrogram inspection to validate fine spectral detail. Expect 30–50 epochs for a ~2200-sample Russian corpus; check samples around epoch 20 to evaluate artifact reduction.
