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
python training.py

# Specify custom corpus and output directories
python training.py --corpus /path/to/ruslan_corpus --output ./my_russian_model
```

### On Mac (with MPS acceleration)

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python training.py
```

### Training with Custom Parameters

```shell
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

### Resume Training

```shell
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

The script uses an advanced rule-based Russian grapheme-to-phoneme converter with comprehensive linguistic features:

### RussianPhonemeProcessor Features

- **Comprehensive Stress Detection**: Multi-level approach with explicit marks, dictionary lookup, and heuristic rules
- **Vowel Reduction**: Implements Russian vowel reduction patterns based on stress position
- **Consonant Assimilation**: Handles voicing assimilation and word-final devoicing
- **Palatalization Rules**: Complete palatalization system with soft/hard consonant distinctions
- **Performance Optimized**: LRU caching for frequently processed words and text normalization
- **Exception Handling**: Built-in dictionary for irregular pronunciations
- **IPA Output**: Full International Phonetic Alphabet representation

### Stress Detection System

The processor uses a three-tier stress detection approach:

1. **Explicit Stress Marks**: Recognizes combining acute (◌́), grave (◌̀), and acute accent marks
2. **Dictionary Lookup**: Built-in stress patterns for common words with optional external dictionary support
3. **Heuristic Rules**: Intelligent fallback based on morphological patterns:
   - Infinitive verbs (-ать, -еть, -ить): stress on ending
   - Adjectives (-ный, -ная): stress on root
   - Abstract nouns (-ость, -есть): stress on root
   - Default: penultimate syllable stress

### Phonetic Processes

#### Vowel System
- **Base Vowels**: а/a, о/o, у/u, ы/ɨ, э/e, я/ja, ё/jo, ю/ju, и/i, е/je
- **Vowel Reduction**:
  - First pretonic/post-tonic: о,а → ə; е,я → ɪ
  - Further positions: о,а,е,я → ə (stronger reduction)
- **Contextual Variants**: Iotated vowels after consonants (я→a, ю→u, etc.)

#### Consonant System
- **Base Consonants**: 20 primary consonants with IPA mapping
- **Palatalization**: Full palatalized variants (Cʲ) triggered by:
  - Following soft vowels (е, и, ё, ю, я)
  - Soft sign (ь)
- **Hard Consonants**: ж, ш, ц (never palatalized)
- **Soft Consonants**: ч, щ, й (always palatalized)

#### Phonetic Rules
- **Voicing Assimilation**: Consonant clusters assimilate voicing
- **Word-Final Devoicing**: Voiced consonants become voiceless at word end
- **Consonant Simplification**: Handles consonant cluster simplifications

### Supported Character Set

- **Vowels**: а, о, у, ы, э, я, ё, ю, и, е (10 letters)
- **Consonants**: б, в, г, д, ж, з, к, л, м, н, п, р, с, т, ф, х, ц, ч, ш, щ (20 letters)
- **Special Signs**: ь (soft sign), ъ (hard sign)
- **Stress Marks**: Combining diacritics (U+0301, U+0300, U+0341)

### Usage Examples

#### Basic Word Processing
```python
processor = RussianPhonemeProcessor()
phonemes, stress_info = processor.process_word("говорить")
# Returns: (['g', 'ə', 'v', 'ə', 'r', 'i', 'tʲ'], StressInfo(position=2, vowel_index=5, is_marked=False))

ipa = processor.to_ipa(phonemes)
# Returns: "gəvərʲitʲ"
```

#### Text Processing with Stress
```python
results = processor.process_text("Привет, как дела?")
for word, phonemes, stress_info in results:
    print(f"{word} -> /{processor.to_ipa(phonemes)}/ (stress: syllable {stress_info.position})")

# Output:
# привет -> /prʲivʲet/ (stress: syllable 1)
# как -> /kak/ (stress: syllable 0)
# дела -> /dʲɪla/ (stress: syllable 1)
```

#### Phoneme-to-Index Conversion
```python
# For neural network training
indices = processor.text_to_indices("говорить")
vocab_size = processor.get_vocab_size()  # Returns total phoneme count
phoneme_list = processor.get_phoneme_list()  # Returns all phonemes
```

### Exception Dictionary

Built-in irregular pronunciations for common words:
- что → /ʃto/ (not standard /tʃto/)
- конечно → /kənʲeʃnə/ (не-consonant simplification)
- его → /jɪvo/ (genitive case vowel change)
- сегодня → /sʲɪvodʲnʲə/ (historical sound changes)

### Performance Features

- **LRU Caching**: `@lru_cache` decorators on frequently called methods
- **Word Caching**: Internal cache for processed words
- **Batch Processing**: Optimized for processing multiple words/sentences
- **Memory Management**: Cache clearing and statistics monitoring

### Configuration Options

```python
# Initialize with external stress dictionary
processor = RussianPhonemeProcessor(stress_dict_path="./stress_dict.txt")

# Stress dictionary format (TSV):
# слово	2	# word with stress on syllable 2
# говорить	2
# красивый	1
```

### Advanced Features

#### Serialization Support
```python
# Save processor state
state_dict = processor.to_dict()

# Restore from state
processor = RussianPhonemeProcessor.from_dict(state_dict)
```

#### Cache Management
```python
# Clear all caches
processor.clear_cache()

# Get cache statistics
stats = processor.get_cache_info()
print(f"Cache hits: {stats['normalize_text_cache'].hits}")
```

#### Stress Pattern Generation
```python
# Get binary stress pattern for TTS models
stress_pattern = processor.get_stress_pattern("говорить по-русски")
# Returns: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]  # 1 = stressed phoneme
```

### Phoneme Vocabulary

The processor generates a complete phoneme vocabulary including:
- Base vowel and consonant phonemes
- Palatalized consonants (Cʲ)
- Reduced vowels (ə, ɪ)
- Multi-character phonemes (ts, tʃ, ʃtʃ)
- Exception phonemes (ʌ, ʐ, etc.)

Total vocabulary size: ~35-40 unique phonemes depending on exceptions and reductions used.

### Integration with Training

The phoneme processor integrates seamlessly with the TTS training pipeline:

```python
# In training script
processor = RussianPhonemeProcessor()

# Process text to phoneme indices
def text_to_phonemes(text):
    results = processor.process_text(text)
    phoneme_ids = []
    stress_pattern = []

    for word, phonemes, stress_info in results:
        word_ids = [processor.phoneme_to_id[p] for p in phonemes if p in processor.phoneme_to_id]
        word_stress = [1 if i == stress_info.vowel_index else 0 for i in range(len(phonemes))]

        phoneme_ids.extend(word_ids)
        stress_pattern.extend(word_stress)

    return phoneme_ids, stress_pattern
```

This comprehensive phoneme processing system ensures accurate Russian pronunciation modeling for high-quality TTS synthesis.

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
python training.py --corpus ./ruslan_corpus --epochs 10
```

### Production Training
```shell
# Full training run with checkpointing
PYTORCH_ENABLE_MPS_FALLBACK=1 python training.py \
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

## License

This training script is designed for use with the Ruslan Russian speech corpus and implements the Kokoro TTS architecture for educational and research purposes.
For commercial license contact the author
