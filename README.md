# Kokoro Language Model Training Script for Russian (Ruslan Corpus)

A Text-to-Speech (TTS) training script for Russian language using the Kokoro architecture, optimized for Mac MPS acceleration and compatible with the Ruslan corpus dataset.

## Features

- **Mac MPS Optimized**: Full support for Apple Silicon GPU acceleration
- **No espeak Dependency**: Uses rule-based Russian phoneme processing
- **Checkpoint Support**: Resume training from any saved checkpoint
- **Memory Efficient**: Optimized for limited VRAM with gradient clipping and memory management
- **Flexible Configuration**: Command-line arguments for easy customization

## Installation

```shell
pip install -r requirements.txt
```

## Dataset Structure

The script expects the Ruslan corpus to be organized as follows:

```
ruslan_corpus/
├── metadata_RUSLAN_22200.csv    # Main metadata file
├── wavs/                        # Audio files directory
│   ├── 005559_RUSLAN.wav
│   ├── 005560_RUSLAN.wav
│   └── ...
└── texts/                       # Optional: individual text files
    ├── 005559_RUSLAN.txt
    ├── 005560_RUSLAN.txt
    └── ...
```

The metadata CSV should be formatted as: `audio_filename|transcription`

## Usage

### Basic Training

```shell
# Using default settings
python ruslan-training-no-espeak.py

# Specify custom corpus and output directories
python ruslan-training-no-espeak.py --corpus /path/to/ruslan_corpus --output ./my_russian_model
```

### On Mac (with MPS acceleration)

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python ruslan-training-no-espeak.py
```

### Training with Custom Parameters

```shell
# Custom batch size and epochs
python ruslan-training-no-espeak.py --batch-size 16 --epochs 50

# Custom learning rate and save frequency
python ruslan-training-no-espeak.py --learning-rate 0.0001 --save-every 5

# Full custom configuration
python ruslan-training-no-espeak.py \
    --corpus ./data/ruslan_corpus \
    --output ./models/kokoro_russian \
    --batch-size 12 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --save-every 2
```

### Resume Training

```shell
# Auto-resume from latest checkpoint
python ruslan-training-no-espeak.py --resume auto

# Resume from specific checkpoint
python ruslan-training-no-espeak.py --resume ./models/checkpoint_epoch_10.pth

# Resume with different parameters
python ruslan-training-no-espeak.py --resume auto --batch-size 16 --epochs 200
```

## Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--corpus` | `-c` | `./ruslan_corpus` | Path to the corpus directory |
| `--output` | `-o` | `./kokoro_russian_model` | Path to the output model directory |
| `--resume` | `-r` | `None` | Resume from checkpoint (`auto` or path to `.pth` file) |
| `--batch-size` | `-b` | `8` | Batch size for training |
| `--epochs` | `-e` | `100` | Number of training epochs |
| `--learning-rate` | `-lr` | `1e-4` | Learning rate |
| `--save-every` | | `2` | Save checkpoint every N epochs |

## Model Architecture

The Kokoro model implements a sequence-to-sequence architecture with:

- **Text Encoder**: Bidirectional LSTM for phoneme sequence encoding
- **Attention Mechanism**: Multi-head attention for alignment between text and speech
- **Decoder**: LSTM-based mel-spectrogram generation
- **Mel Processing**: 80-dimensional mel-spectrograms at 22.05kHz

### Model Configuration

- **Sample Rate**: 22,050 Hz
- **Mel Channels**: 80
- **FFT Size**: 1024
- **Hop Length**: 256 samples (~11.6ms)
- **Hidden Dimensions**: 512

## Output Files

The training script generates several files in the output directory:

```
kokoro_russian_model/
├── checkpoint_epoch_2.pth       # Regular checkpoints
├── checkpoint_epoch_4.pth
├── ...
├── phoneme_processor.pkl        # Russian phoneme processor
└── kokoro_russian_final.pth     # Final trained model
```

### Checkpoint Contents

Each checkpoint contains:
- Model state dictionary
- Optimizer state
- Learning rate scheduler state
- Training configuration
- Current epoch and loss

## Memory Management

The script includes several optimizations for limited memory:

- **Gradient Clipping**: Prevents exploding gradients (max norm: 1.0)
- **Sequence Limiting**: Maximum 800 mel frames per sample
- **Cache Clearing**: Automatic MPS cache clearing every 50 batches
- **Mixed Precision**: Enabled for MPS acceleration

## Phoneme Processing

The script uses a rule-based Russian grapheme-to-phoneme converter:

### Supported Characters

- **Vowels**: а, о, у, ы, э, я, ё, ю, и, е
- **Consonants**: б, в, г, д, ж, з, к, л, м, н, п, р, с, т, ф, х, ц, ч, ш, щ
- **Special**: ь, ъ (soft/hard signs), punctuation, spaces

### Example Phoneme Conversion

```
Russian: "Привет мир"
Phonemes: ['p', 'r', 'i', 'v', 'je', 't', ' ', 'm', 'i', 'r']
```

## Training Progress

The script provides detailed logging:

```
INFO:__main__:Loaded 1234 samples from corpus at ./ruslan_corpus
INFO:__main__:Starting training on device: mps
INFO:__main__:Training from epoch 1 to 100
Epoch 1/100: 100%|██████████| 154/154 [05:23<00:00, 0.48it/s, loss=2.345]
INFO:__main__:Epoch 1 completed. Average loss: 2.345
INFO:__main__:Checkpoint saved: ./kokoro_russian_model/checkpoint_epoch_2.pth
```

## Troubleshooting

### Common Issues

1. **MPS Backend Error**: Use `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable
2. **Out of Memory**: Reduce `--batch-size` (try 4 or 6)
3. **Audio Loading Error**: Ensure `soundfile` and `librosa` are installed
4. **Missing Metadata**: Check that `metadata_RUSLAN_22200.csv` exists in corpus directory

### Performance Tips

- **Mac M1/M2**: Use batch size 8-12 for optimal performance
- **Limited RAM**: Reduce batch size and enable `num_workers=0`
- **Fast Storage**: Keep corpus on SSD for faster data loading
- **Long Training**: Use `--save-every 1` for frequent checkpoints

## Examples

### Quick Start
```shell
# Download and prepare Ruslan corpus, then:
python ruslan-training-no-espeak.py --corpus ./ruslan_corpus --epochs 10
```

### Production Training
```shell
# Full training run with checkpointing
PYTORCH_ENABLE_MPS_FALLBACK=1 python ruslan-training-no-espeak.py \
    --corpus ./data/ruslan_corpus \
    --output ./models/kokoro_russian_v1 \
    --batch-size 10 \
    --epochs 200 \
    --learning-rate 5e-5 \
    --save-every 5
```

### Resume Interrupted Training
```shell
# Auto-resume from latest checkpoint
PYTORCH_ENABLE_MPS_FALLBACK=1 python ruslan-training-no-espeak.py \
    --corpus ./data/ruslan_corpus \
    --output ./models/kokoro_russian_v1 \
    --resume auto
```

## Requirements

- Python 3.9+
- PyTorch 2.0+ with MPS support
- torchaudio 2.0+
- NumPy 1.21+
- soundfile 0.12+
- librosa 0.9+
- tqdm 4.64+

## License

This training script is designed for use with the Ruslan Russian speech corpus and implements the Kokoro TTS architecture for educational and research purposes.
For commercial license contact the author
