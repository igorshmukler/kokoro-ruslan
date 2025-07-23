# Kokoro-Ruslan: Russian TTS Training Script

A Text-to-Speech (TTS) training script for Russian language using the Kokoro architecture, optimized for Mac MPS acceleration and compatible with the Ruslan corpus dataset.

## Features

- **Mac MPS optimized**: Full support for Apple Silicon GPU acceleration
- **No espeak dependency**: Uses rule-based Russian phoneme processing
- **Checkpoint support**: Resume training from any saved checkpoint
- **Memory eficient**: Optimized for limited VRAM with gradient clipping and memory management
- **Flexible configuration**: Command-line arguments for easy customization

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

The script expects the Ruslan corpus to be organized as follows:

```
ruslan_corpus/
├── metadata_RUSLAN_22200.csv  # Main metadata file
├── wavs/                      # Audio files directory
│   ├── 005559_RUSLAN.wav
│   ├── 005560_RUSLAN.wav
│   └── ...
└── texts/                     # Optional: individual text files
    ├── 005559_RUSLAN.txt
    ├── 005560_RUSLAN.txt
    └── ...
```

The metadata CSV should be formatted as: `audio_filename|transcription`

## Quick Start

```bash
# Using default settings
python training.py

# Specify custom corpus and output directories
python training.py --corpus /path/to/ruslan_corpus --output ./my_russian_model

# For Mac users with MPS issues
PYTORCH_ENABLE_MPS_FALLBACK=1 python training.py
```

## Training Examples

```bash
# Custom batch size and epochs
python training.py --batch-size 16 --epochs 50

# Custom learning rate and save frequency
python training.py --learning-rate 0.0001 --save-every 5

# Full custom configuration
python training.py \
    --corpus ./data/ruslan_corpus \
    --output ./models/kokoro_russian \
    --batch-size 12 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --save-every 2
```

## Resume Training

```bash
# Auto-resume from latest checkpoint
python training.py --resume auto

# Resume from specific checkpoint
python training.py --resume ./models/checkpoint_epoch_10.pth

# Resume with different parameters
python training.py --resume auto --batch-size 16 --epochs 200
```

## Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--corpus` | `-c` | `./ruslan_corpus` | Path to the corpus directory |
| `--output` | `-o` | `./kokoro_russian_model` | Path to the output model directory |
| `--resume` | `-r` | `None` | Resume from checkpoint (auto or path to .pth file) |
| `--batch-size` | `-b` | `8` | Batch size for training |
| `--epochs` | `-e` | `100` | Number of training epochs |
| `--learning-rate` | `-lr` | `1e-4` | Learning rate |
| `--save-every` |  | `2` | Save checkpoint every N epochs |

## Model Architecture

The Kokoro model implements a modern Transformer-based sequence-to-sequence architecture with:

### Core Components

- **Text Encoder**: Stack of Transformer encoder blocks with multi-head self-attention
- **Duration Predictor**: Multi-layer perceptron for phoneme duration prediction
- **Length Regulator**: Expands encoder outputs based on predicted/ground-truth durations
- **Decoder**: Stack of Transformer decoder blocks with masked self-attention and cross-attention
- **Stop Token Predictor**: Linear layer for end-of-sequence prediction

### Architecture Details

- **Encoder Layers**: 6 Transformer encoder blocks (configurable)
- **Decoder Layers**: 6 Transformer decoder blocks (configurable)
- **Attention Heads**: 8 multi-head attention heads (configurable)
- **Hidden Dimensions**: 512 (d_model for Transformer layers)
- **Feed-Forward Dimensions**: 2048 for encoder, 2048 for decoder
- **Positional Encoding**: Sinusoidal encoding with dropout
- **Gradient Checkpointing**: Enabled for memory efficiency during training

### Audio Configuration

- **Sample Rate**: 22,050 Hz
- **Mel Channels**: 80
- **FFT Size**: 1024
- **Hop Length**: 256 samples (~11.6ms)
- **Maximum Sequence Length**: 1420 mel frames (configurable)

### Training Modes

The model supports two distinct forward pass modes:

#### Training Mode (Teacher Forcing)
- Uses ground-truth mel spectrograms as decoder input
- Applies causal masking for autoregressive training
- Predicts mel frames, durations, and stop tokens simultaneously
- Supports gradient checkpointing for memory efficiency

#### Inference Mode (Autoregressive)
- Generates mel spectrograms step-by-step
- Uses predicted durations for length regulation
- Implements stop token prediction for sequence termination
- Configurable generation length and stop threshold

### Memory Optimizations

The model includes several optimizations for efficient training:

- **Gradient Checkpointing**: Applied to all Transformer layers to reduce memory usage
- **Padding Masks**: Support for variable-length sequences with proper masking
- **Length Regulation**: Efficient expansion of encoder outputs based on durations
- **Causal Masking**: Proper masking for autoregressive decoder training
- **MPS Acceleration**: Optimized for Apple Silicon GPU training

### Key Features

- **Flexible Architecture**: Configurable number of layers, heads, and dimensions
- **Modern Design**: Full Transformer architecture replacing legacy LSTM components
- **Production Ready**: Supports both training and inference with proper masking
- **Memory Efficient**: Gradient checkpointing and optimized attention mechanisms
- **Robust Training**: Teacher forcing with ground-truth alignment during training

The training script generates several files in the output directory:

```
kokoro_russian_model/
├── checkpoint_epoch_2.pth      # Regular checkpoints
├── checkpoint_epoch_4.pth
├── ...
├── phoneme_processor.pkl       # Russian phoneme processor
└── kokoro_russian_final.pth    # Final trained model
```

Each checkpoint contains:
- Model state dictionary
- Optimizer state
- Learning rate scheduler state
- Training configuration
- Current epoch and loss

## Memory Optimizations

The script includes several optimizations for limited memory:

- **Gradient Clipping**: Prevents exploding gradients (max norm: 1.0)
- **Sequence Limiting**: Maximum 1420 mel frames per sample
- **Cache Clearing**: Automatic MPS cache clearing every 50 batches
- **Mixed Precision**: Enabled for MPS acceleration

## Russian Phoneme Processing

The script uses a rule-based Russian grapheme-to-phoneme converter:

- **Vowels**: а, о, у, ы, э, я, ё, ю, и, е
- **Consonants**: б, в, г, д, ж, з, к, л, м, н, п, р, с, т, ф, х, ц, ч, ш, щ
- **Special**: ь, ъ (soft/hard signs), punctuation, spaces

### Example

```
Russian: "Привет мир"
Phonemes: ['p', 'r', 'i', 'v', 'je', 't', ' ', 'm', 'i', 'r']
```

## Training Logs

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

- **MPS Backend Error**: Use `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable
- **Out of Memory**: Reduce `--batch-size` (try 4 or 6)
- **Audio Loading Error**: Ensure `soundfile` and `librosa` are installed
- **Missing Metadata**: Check that `metadata_RUSLAN_22200.csv` exists in corpus directory

### Performance Tips

- **Mac M1/M2**: Use batch size 8-12 for optimal performance
- **Limited RAM**: Reduce batch size and enable `num_workers=0`
- **Fast Storage**: Keep corpus on SSD for faster data loading
- **Long Training**: Use `--save-every 1` for frequent checkpoints

## Usage Examples

### Basic Training

```bash
# Download and prepare Ruslan corpus, then:
python training.py --corpus ./ruslan_corpus --epochs 10
```

### Production Training

```bash
# Full training run with checkpointing
PYTORCH_ENABLE_MPS_FALLBACK=1 python training.py \
    --corpus ./data/ruslan_corpus \
    --output ./models/kokoro_russian_v1 \
    --batch-size 10 \
    --epochs 200 \
    --learning-rate 5e-5 \
    --save-every 5
```

### Resume Training

```bash
# Auto-resume from latest checkpoint
PYTORCH_ENABLE_MPS_FALLBACK=1 python training.py \
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

For inference and deployment of trained models, check out:

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - The main Kokoro model on Hugging Face
- [Kokoros (Rust)](https://github.com/lucasjinreal/Kokoros) - Fast inference engine in Rust
- [Kokoro FastAPI](https://github.com/remsky/Kokoro-FastAPI) - Dockerized API wrapper
- [StreamingKokoroJS](https://github.com/rhulha/StreamingKokoroJS) - Browser-based inference

## License

This training script is designed for use with the Ruslan Russian speech corpus and implements the Text-To-Speech model architecture for educational and research purposes. For commercial license contact the author.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
