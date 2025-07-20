# Kokoro Russian TTS Inference Guide

This guide explains how to use your trained Kokoro Russian TTS model to convert text to speech.

## Overview

The inference script (`inference.py`) loads your trained Kokoro model and converts Russian text into speech using the same phoneme processing and model architecture used during training.

## Requirements

- Python 3.7+
- PyTorch with MPS/CUDA support (optional)
- torchaudio
- numpy
- Trained Kokoro model files

## File Structure

Your model directory should contain:
```
kokoro_russian_model/
├── phoneme_processor.pkl          # Phoneme processor from training
├── kokoro_russian_final.pth       # Final trained model (preferred)
└── checkpoint_epoch_*.pth         # Or checkpoint files
```

## Usage

### 1. Basic Text Conversion

Convert a single Russian text to speech:

```shell
python inference.py --model ./kokoro_russian_model --text "Привет, как дела?"
```

This will generate `output.wav` in the current directory.

### 2. Custom Output File

Specify a custom output file:

```shell
python inference.py --model ./kokoro_russian_model --text "Привет мир" --output hello.wav
```

### 3. Convert from Text File

Read text from a file and convert to speech:

```shell
python inference.py --model ./kokoro_russian_model --text-file input.txt --output output.wav
```

Create `input.txt` with your Russian text:
```
Добро пожаловать в мир синтеза речи на русском языке.
```

### 4. Interactive Mode

Enter text interactively for quick testing:

```shell
python inference.py --model ./kokoro_russian_model --interactive
```

Type Russian text and press Enter. Type `quit` to exit.

### 5. Specify Device

Force specific device usage:

```shell
# Use MPS (Mac Metal)
python inference.py --model ./kokoro_russian_model --text "Тест" --device mps

# Use CUDA (NVIDIA GPU)
python inference.py --model ./kokoro_russian_model --text "Тест" --device cuda

# Use CPU
python inference.py --model ./kokoro_russian_model --text "Тест" --device cpu
```

## Command Line Arguments

| Argument | Short | Description | Required |
|----------|-------|-------------|----------|
| `--model` | `-m` | Path to trained model directory | Yes |
| `--text` | `-t` | Text to convert to speech | No* |
| `--text-file` | `-f` | File containing text to convert | No* |
| `--output` | `-o` | Output audio file path (default: output.wav) | No |
| `--interactive` | `-i` | Interactive mode | No* |
| `--device` | | Device: cpu/cuda/mps (auto-detected) | No |

*At least one of `--text`, `--text-file`, or `--interactive` is required.

## Technical Details

### Audio Configuration

The inference uses the same audio settings as training:
- **Sample Rate**: 22,050 Hz
- **Hop Length**: 256 samples
- **Window Length**: 1,024 samples
- **FFT Size**: 1,024
- **Mel Bands**: 80
- **Frequency Range**: 0-8,000 Hz

### Phoneme Processing

The script uses the same Russian phoneme mapping as training:

**Vowels**: а→a, о→o, у→u, ы→i, э→e, я→ja, ё→jo, ю→ju, и→i, е→je

**Consonants**: б→b, в→v, г→g, д→d, ж→zh, з→z, к→k, л→l, м→m, н→n, п→p, р→r, с→s, т→t, ф→f, х→kh, ц→ts, ч→ch, ш→sh, щ→shch

**Special Characters**: Soft/hard signs are removed, punctuation is preserved

### Audio Generation Process

1. **Text → Phonemes**: Russian text converted to phoneme sequence
2. **Phonemes → Indices**: Phonemes mapped to numerical indices
3. **Model Inference**: Neural network generates mel spectrogram
4. **Mel → Audio**: Griffin-Lim algorithm converts spectrogram to waveform
5. **Output**: Normalized WAV file at 22,050 Hz

## Improving Audio Quality

The current implementation uses Griffin-Lim for mel-to-audio conversion. For higher quality audio, consider:

1. **Neural Vocoders**: Replace Griffin-Lim with HiFi-GAN, WaveGlow, or WaveNet
2. **Post-processing**: Apply noise reduction or audio enhancement
3. **Fine-tuning**: Additional training epochs may improve model quality

## Troubleshooting

### Common Issues

**Model not found**:
- Ensure model directory exists and contains `.pth` files
- Check that `phoneme_processor.pkl` exists

**Poor audio quality**:
- Model may need more training epochs
- Consider using a neural vocoder instead of Griffin-Lim
- Verify training data quality

**Memory errors**:
- Use CPU device: `--device cpu`
- Reduce maximum sequence length in the model code

**Unknown characters**:
- Script handles most Russian text automatically
- Unknown characters are converted to spaces

### Supported Text

The model works best with:
- Standard Russian Cyrillic text
- Common punctuation (. , ! ? -)
- Mixed case text (automatically lowercased)

## Example Scripts

### Batch Processing

Create multiple audio files:

```python
from inference import KokoroTTS

tts = KokoroTTS("./kokoro_russian_model")

texts = [
    "Первый пример текста.",
    "Второй пример текста.",
    "Третий пример текста."
]

tts.batch_text_to_speech(texts, "./output_dir")
```

### Custom Integration

Use in your own Python code:

```python
from inference import KokoroTTS
import torchaudio

# Initialize TTS
tts = KokoroTTS("./kokoro_russian_model")

# Generate speech
audio_tensor = tts.text_to_speech("Привет мир!")

# Save or process audio
torchaudio.save("output.wav", audio_tensor.unsqueeze(0), 22050)
```

## Performance

- **CPU**: ~2-5x real-time (depends on text length)
- **MPS (Mac M1/M2)**: ~5-10x real-time
- **CUDA (GPU)**: ~10-20x real-time

Longer texts take proportionally longer to generate.

## Next Steps

1. **Test with various Russian texts** to evaluate quality
2. **Experiment with different devices** for optimal performance
3. **Consider neural vocoder integration** for production use
4. **Fine-tune model** if audio quality needs improvement

## Support

For issues related to:
- **Model training**: Check the training script and logs
- **Audio quality**: Consider more training or neural vocoders  
- **Performance**: Try different devices or optimize model architecture
- **Text processing**: Verify phoneme mapping for specific characters
