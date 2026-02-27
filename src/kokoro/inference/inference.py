#!/usr/bin/env python3
"""
Kokoro Russian TTS Inference Script with HiFi-GAN Vocoder
Convert Russian text to speech using trained Kokoro model with neural vocoder
"""

import torch
import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Any, Dict
import logging

# Import our training configuration, model and phoneme processor
from kokoro.model.model import KokoroModel
from kokoro.data.russian_phoneme_processor import RussianPhonemeProcessor
from kokoro.inference.vocoder_manager import VocoderManager
from kokoro.data.audio_utils import AudioUtils, PhonemeProcessorUtils

import re

from kokoro.training.trainer import TrainingConfig

# This tells PyTorch it's safe to load our custom config class
torch.serialization.add_safe_globals([TrainingConfig])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KokoroTTS:
    """Main TTS inference class with neural vocoder support."""

    def __init__(
        self,
        model_dir: str,
        device: str = None,
        vocoder_type: str = "hifigan",
        vocoder_path: str = None,
        inference_max_len: Optional[int] = None,
        inference_stop_threshold: Optional[float] = None,
        inference_min_len_ratio: Optional[float] = None,
        inference_min_len_floor: Optional[int] = None,
        weights: str = 'auto',
    ):
        self.model_dir = Path(model_dir)

        # Determine device
        self.device = AudioUtils.validate_device(device)
        logger.info(f"Using device: {self.device}")

        # Audio configuration (should match training config)
        self.sample_rate = 22050
        self.hop_length = 256
        self.win_length = 1024
        self.n_fft = 1024
        self.n_mels = 80
        self.f_min = 0.0
        self.f_max = 8000.0

        # Inference generation controls (auto-tuned from checkpoint unless explicitly provided)
        self.inference_max_len = inference_max_len
        self.inference_stop_threshold = inference_stop_threshold
        self.inference_min_len_ratio = inference_min_len_ratio
        self.inference_min_len_floor = inference_min_len_floor

        # Keep explicit override flags so checkpoint-tuning does not override user intent
        self._explicit_inference_max_len = inference_max_len is not None
        self._explicit_inference_stop_threshold = inference_stop_threshold is not None
        self._explicit_inference_min_len_ratio = inference_min_len_ratio is not None
        self._explicit_inference_min_len_floor = inference_min_len_floor is not None

        # Initialize utility classes
        self.audio_utils = AudioUtils(self.sample_rate)

        # Weights preference for inference: 'auto'|'ema'|'model'
        if weights not in ('auto', 'ema', 'model'):
            raise ValueError("weights must be one of: 'auto', 'ema', 'model'")
        self.weights_preference = weights

        # Load phoneme processor
        self.phoneme_processor = self._load_phoneme_processor()

        # Load model
        self.model = self._load_model()

        # Initialize vocoder
        self.vocoder_manager = VocoderManager(vocoder_type, vocoder_path, self.device)

    def _load_phoneme_processor(self) -> RussianPhonemeProcessor:
        """Loads the phoneme processor from the model directory."""
        processor_path = self.model_dir / "phoneme_processor.pkl"

        if processor_path.exists():
            try:
                with open(processor_path, 'rb') as f:
                    processor_data = pickle.load(f)
                processor = RussianPhonemeProcessor.from_dict(processor_data)
                logger.info(f"Loaded phoneme processor from: {processor_path}")
            except Exception as e:
                logger.error(f"Error loading phoneme processor from {processor_path}: {e}")
                raise
        else:
            logger.warning("Phoneme processor not found at expected path. Creating a new one. This might lead to issues if the model was trained with a different vocabulary.")
            processor = RussianPhonemeProcessor()

        return processor

    def _load_model(self) -> KokoroModel:
        """Loads the trained Kokoro model with robust error handling."""
        final_model_path = self.model_dir / "kokoro_russian_final.pth"
        checkpoint_files = sorted(list(self.model_dir.glob("checkpoint_epoch_*.pth")),
                                  key=lambda x: int(x.stem.split('_')[-1]))

        model_path = None
        if final_model_path.exists():
            model_path = final_model_path
            logger.info(f"Attempting to load final model: {model_path}")
        elif checkpoint_files:
            model_path = checkpoint_files[-1] # Use the latest checkpoint
            logger.info(f"Final model not found, loading latest checkpoint: {model_path}")
        else:
            raise FileNotFoundError(f"No model files found in {self.model_dir}. Ensure 'kokoro_russian_final.pth' or 'checkpoint_epoch_*.pth' exists.")

        checkpoint = None
        try_methods = [
            lambda: torch.load(model_path, map_location='cpu', weights_only=True),
            lambda: torch.load(model_path, map_location='cpu', weights_only=False),
        ]

        # Try loading with various methods
        for i, load_func in enumerate(try_methods):
            try:
                checkpoint = load_func()
                logger.info(f"Successfully loaded checkpoint using method {i+1}.")

                break
            except Exception as e:
                logger.warning(f"Checkpoint load attempt {i+1} failed: {e}")

        if checkpoint is None:
            raise RuntimeError(f"Failed to load checkpoint from {model_path} with any attempted method. It might be corrupted or incompatible.")

        # Extract model state dictionary
        state_dict_to_load = None
        if 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict_to_load = checkpoint['model']
        elif isinstance(checkpoint, dict): # If the checkpoint itself is the state_dict
            state_dict_to_load = checkpoint

        if state_dict_to_load is None:
            raise RuntimeError("Checkpoint does not contain a recognized model state dictionary (expected 'model_state_dict' or 'model' key, or raw state dict).")

        # Decide which weights to load based on preference
        if isinstance(checkpoint, dict):
            pref = getattr(self, 'weights_preference', 'auto')
            if pref == 'ema':
                if 'ema_model_state_dict' in checkpoint:
                    state_dict_to_load = checkpoint['ema_model_state_dict']
                    logger.info("Using EMA weights from checkpoint for inference (requested 'ema').")
                else:
                    raise RuntimeError("EMA weights requested but 'ema_model_state_dict' not found in checkpoint.")
            elif pref == 'model':
                # Prefer explicit model key if present
                if 'model_state_dict' in checkpoint:
                    state_dict_to_load = checkpoint['model_state_dict']
                    logger.info("Using model_state_dict from checkpoint for inference (requested 'model').")
                elif 'model' in checkpoint:
                    state_dict_to_load = checkpoint['model']
                    logger.info("Using 'model' entry from checkpoint for inference (requested 'model').")
                else:
                    # Leave state_dict_to_load as discovered earlier (maybe raw state_dict)
                    logger.info("Requested 'model' weights but explicit keys not found; using available state dict.")
            else:  # auto
                if 'ema_model_state_dict' in checkpoint:
                    state_dict_to_load = checkpoint['ema_model_state_dict']
                    logger.info("EMA weights found in checkpoint — using EMA weights for inference (auto).")
                else:
                    # keep state_dict_to_load as discovered earlier
                    logger.info("No EMA weights in checkpoint — using standard model weights (auto).")

        # Prefer explicit architecture metadata saved with checkpoints
        metadata = checkpoint.get('model_metadata') if isinstance(checkpoint, dict) else None
        architecture = metadata.get('architecture', {}) if isinstance(metadata, dict) else {}

        self._apply_checkpoint_inference_controls(checkpoint)

        vocab_size_from_processor = len(self.phoneme_processor.phoneme_to_id)
        if architecture:
            required_fields = [
                'mel_dim', 'hidden_dim', 'n_encoder_layers', 'n_decoder_layers',
                'n_heads', 'encoder_ff_dim', 'decoder_ff_dim', 'encoder_dropout',
                'max_decoder_seq_len'
            ]
            missing_fields = [field for field in required_fields if field not in architecture]
            if missing_fields:
                raise RuntimeError(
                    f"Checkpoint metadata is incomplete. Missing fields: {missing_fields}. "
                    "Please retrain or resave checkpoint with full model metadata."
                )

            metadata_vocab_size = architecture.get('vocab_size')
            if metadata_vocab_size is not None and int(metadata_vocab_size) != vocab_size_from_processor:
                raise RuntimeError(
                    f"Vocabulary size mismatch: checkpoint metadata expects {int(metadata_vocab_size)}, "
                    f"phoneme processor has {vocab_size_from_processor}. "
                    "Use the matching phoneme_processor.pkl from the same training run."
                )

            model_kwargs = {
                'vocab_size': vocab_size_from_processor,
                'mel_dim': int(architecture['mel_dim']),
                'hidden_dim': int(architecture['hidden_dim']),
                'n_encoder_layers': int(architecture['n_encoder_layers']),
                'n_heads': int(architecture['n_heads']),
                'encoder_ff_dim': int(architecture['encoder_ff_dim']),
                'encoder_dropout': float(architecture['encoder_dropout']),
                'n_decoder_layers': int(architecture['n_decoder_layers']),
                'decoder_ff_dim': int(architecture['decoder_ff_dim']),
                'max_decoder_seq_len': int(architecture['max_decoder_seq_len']),
                'use_variance_predictor': bool(architecture.get('use_variance_predictor', True)),
                'variance_filter_size': int(architecture.get('variance_filter_size', 256)),
                'variance_kernel_size': int(architecture.get('variance_kernel_size', 3)),
                'variance_dropout': float(architecture.get('variance_dropout', 0.1)),
                'n_variance_bins': int(architecture.get('n_variance_bins', 256)),
                'pitch_min': float(architecture.get('pitch_min', 0.0)),
                'pitch_max': float(architecture.get('pitch_max', 1.0)),
                'energy_min': float(architecture.get('energy_min', 0.0)),
                'energy_max': float(architecture.get('energy_max', 1.0)),
                'use_stochastic_depth': bool(architecture.get('use_stochastic_depth', True)),
                'stochastic_depth_rate': float(architecture.get('stochastic_depth_rate', 0.1)),
                'enable_profiling': getattr(self, 'enable_profiling', False),
            }
        else:
            raise RuntimeError(
                "Checkpoint is missing required 'model_metadata.architecture'. "
                "Legacy checkpoints without metadata are not supported for strict inference load. "
                "Use a checkpoint and phoneme processor saved from the same metadata-enabled training run."
            )

        model = KokoroModel(
            **model_kwargs
        )

        # Strict load only: fail fast on architecture/state mismatch
        try:
            model.load_state_dict(state_dict_to_load, strict=True)
            logger.info("Model state dictionary loaded successfully (strict=True).")
        except RuntimeError as e:
            raise RuntimeError(
                "Strict model load failed due to architecture/state mismatch. "
                "Use a checkpoint and phoneme processor from the same run, and ensure metadata matches. "
                f"Original error: {e}"
            ) from e

        model.to(self.device)
        model.eval() # Set model to evaluation mode

        logger.info(f"Model '{model_path.name}' loaded successfully with vocab_size={vocab_size_from_processor}.")
        return model

    def _safe_float(self, value: Any, default: float, min_value: float, max_value: float) -> float:
        """Coerce a float control into a safe bounded range."""
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, min(max_value, value_f))

    def _safe_int(self, value: Any, default: int, min_value: int) -> int:
        """Coerce an int control into a safe bounded range."""
        try:
            value_i = int(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, value_i)

    def _apply_checkpoint_inference_controls(self, checkpoint: Dict[str, Any]):
        """Auto-tune inference controls from checkpoint metadata/config unless user-overridden."""
        default_controls = {
            'max_len': 1200,
            'stop_threshold': 0.45,
            'min_len_ratio': 0.7,
            'min_len_floor': 12,
        }

        explicit_max_len = bool(getattr(self, '_explicit_inference_max_len', False))
        explicit_stop_threshold = bool(getattr(self, '_explicit_inference_stop_threshold', False))
        explicit_min_len_ratio = bool(getattr(self, '_explicit_inference_min_len_ratio', False))
        explicit_min_len_floor = bool(getattr(self, '_explicit_inference_min_len_floor', False))

        metadata = checkpoint.get('model_metadata', {}) if isinstance(checkpoint, dict) else {}
        metadata_controls = metadata.get('inference_controls', {}) if isinstance(metadata, dict) else {}

        config_controls = {}
        config_obj = checkpoint.get('config') if isinstance(checkpoint, dict) else None
        if config_obj is not None:
            config_controls = {
                'max_len': getattr(config_obj, 'inference_max_len', None),
                'stop_threshold': getattr(config_obj, 'inference_stop_threshold', None),
                'min_len_ratio': getattr(config_obj, 'inference_min_len_ratio', None),
                'min_len_floor': getattr(config_obj, 'inference_min_len_floor', None),
            }

        # Priority for auto-tuned values: metadata inference_controls -> config fields -> defaults
        chosen_max_len = metadata_controls.get('max_len', config_controls.get('max_len', default_controls['max_len']))
        chosen_stop_threshold = metadata_controls.get(
            'stop_threshold', config_controls.get('stop_threshold', default_controls['stop_threshold'])
        )
        chosen_min_len_ratio = metadata_controls.get(
            'min_len_ratio', config_controls.get('min_len_ratio', default_controls['min_len_ratio'])
        )
        chosen_min_len_floor = metadata_controls.get(
            'min_len_floor', config_controls.get('min_len_floor', default_controls['min_len_floor'])
        )

        if not explicit_max_len:
            self.inference_max_len = self._safe_int(chosen_max_len, default_controls['max_len'], min_value=64)
        if not explicit_stop_threshold:
            self.inference_stop_threshold = self._safe_float(
                chosen_stop_threshold, default_controls['stop_threshold'], min_value=0.05, max_value=0.99
            )
        if not explicit_min_len_ratio:
            self.inference_min_len_ratio = self._safe_float(
                chosen_min_len_ratio, default_controls['min_len_ratio'], min_value=0.1, max_value=1.5
            )
        if not explicit_min_len_floor:
            self.inference_min_len_floor = self._safe_int(chosen_min_len_floor, default_controls['min_len_floor'], min_value=1)

        logger.info(
            "Inference controls: max_len=%d stop_threshold=%.3f min_len_ratio=%.3f min_len_floor=%d",
            int(self.inference_max_len),
            float(self.inference_stop_threshold),
            float(self.inference_min_len_ratio),
            int(self.inference_min_len_floor),
        )

    def split_text(self, text: str, max_chars: int = 150) -> List[str]:
        """
        Splits long Russian text into smaller chunks based on punctuation.
        Useful for preventing inference degradation on long paragraphs.
        """
        # Split by common sentence endings (. ! ? ;), keeping the delimiter
        sentences = re.split(r'([.!?;\n])', text)

        chunks = []
        current_chunk = ""

        # Reconstruct sentences (re.split with groups keeps the delimiters in the list)
        full_sentences = []
        for i in range(0, len(sentences)-1, 2):
            full_sentences.append(sentences[i] + sentences[i+1])
        if len(sentences) % 2 != 0:
            full_sentences.append(sentences[-1])

        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence exceeds max_chars, save current and start new
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


    def text_to_speech(
        self,
        text: str,
        output_path: Optional[str] = None,
        stop_threshold: Optional[float] = None,
        max_len: Optional[int] = None,
        min_len_ratio: Optional[float] = None,
        min_len_floor: Optional[int] = None,
    ) -> torch.Tensor:
        if not text:
            return torch.empty(0)

        chunks = self.split_text(text)
        all_audio_segments = []

        eff_stop_threshold = self.inference_stop_threshold if stop_threshold is None else stop_threshold
        eff_max_len = self.inference_max_len if max_len is None else max_len
        eff_min_len_ratio = self.inference_min_len_ratio if min_len_ratio is None else min_len_ratio
        eff_min_len_floor = self.inference_min_len_floor if min_len_floor is None else min_len_floor

        try:
            for i, chunk in enumerate(chunks):
                # Step 1 & 2: Phoneme Processing
                raw_processor_output = self.phoneme_processor.process_text(chunk)

                # Inject <sil> between words to match MFA training distribution.
                # Falls back to plain flatten automatically if vocab predates <sil>.
                phoneme_sequence = PhonemeProcessorUtils.flatten_phoneme_output_with_sil(
                    raw_processor_output, self.phoneme_processor.phoneme_to_id
                )

                phoneme_indices = PhonemeProcessorUtils.phonemes_to_indices(
                    phoneme_sequence, self.phoneme_processor.phoneme_to_id
                )
                phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long).unsqueeze(0).to(self.device)

                logger.info(f"--- Processing Chunk {i} ---")
                logger.info(f"Phoneme sequence length: {len(phoneme_indices)}")

                # Step 3: Generate Mel
                with torch.no_grad():
                    # If the stop threshold was explicitly provided by the user (either at
                    # initialization via CLI or as an argument to text_to_speech), ensure
                    # the post-expected-length threshold used by the model is not a lower
                    # default that would silently clamp the user's intent. Pass the
                    # same value for `post_expected_stop_threshold` when explicit.
                    explicit_stop = (stop_threshold is not None) or getattr(self, '_explicit_inference_stop_threshold', False)
                    kwargs = {
                        'phoneme_indices': phoneme_tensor,
                        'max_len': eff_max_len,
                        'stop_threshold': eff_stop_threshold,
                        'min_len_ratio': eff_min_len_ratio,
                        'min_len_floor': eff_min_len_floor,
                    }
                    if explicit_stop:
                        kwargs['post_expected_stop_threshold'] = eff_stop_threshold

                    mel_spec = self.model.forward_inference(**kwargs)

                # ========================================================
                # ENHANCED LOGGING & HEALTH CHECK
                # ========================================================
                if torch.isnan(mel_spec).any():
                    logger.error("CRITICAL: Mel-spectrogram contains NaNs!")

                m_min, m_max = mel_spec.min().item(), mel_spec.max().item()
                m_mean, m_std = mel_spec.mean().item(), mel_spec.std().item()

                logger.info(f"Raw Mel Shape: {mel_spec.shape}")
                logger.info(f"Value Distribution -> Min: {m_min:.4f}, Max: {m_max:.4f}, Mean: {m_mean:.4f}, Std: {m_std:.4f}")

                # Check for "Dead Model" Syndrome (all values identical or extremely close)
                if m_std < 1e-5:
                    logger.warning("WARNING: Mel-spectrogram has near-zero variance. Model output is flat.")

                # 1. Denormalization (SKIPPED based on your Dataset code)
                # If your dataset uses mel = torch.log(linear + 1e-9), DO NOT denormalize here.
                # mel_spec = self.denormalize_mel(mel_spec)

                # 2. Safety Clip
                # Based on torch.log(1e-9), the floor is -20.7, but typical speech is -11.5 to 0.
                mel_spec = torch.clamp(mel_spec, min=-11.5, max=2.0)

                # 3. Transposition Fix
                if mel_spec.shape[-1] == self.n_mels:
                    logger.info("Transposing dimensions: [Batch, Time, 80] -> [Batch, 80, Time]")
                    mel_spec = mel_spec.transpose(1, 2)

                # 4. Squeeze for Vocoder
                mel_spec_final = mel_spec.squeeze(0).cpu()
                logger.info(f"Vocoder Input Shape: {mel_spec_final.shape}")
                # ========================================================

                # Step 4: Vocoding
                chunk_audio = self.vocoder_manager.mel_to_audio(mel_spec_final)

                # Check audio stats
                a_max = chunk_audio.abs().max().item()
                logger.info(f"Generated Audio Peak Amplitude: {a_max:.4f}")
                if a_max < 1e-4:
                    logger.warning("WARNING: Generated audio is nearly silent.")

                # Add small silence gap (0.15s)
                all_audio_segments.append(chunk_audio)
                if i < len(chunks) - 1:   # only between chunks, not after the last one
                    silence = torch.zeros(int(self.sample_rate * 0.15))
                    all_audio_segments.append(silence)

            final_audio = torch.cat(all_audio_segments, dim=0)
            if output_path:
                self.audio_utils.save_audio(final_audio, output_path)
                logger.info(f"Successfully saved to {output_path}")
            return final_audio

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def batch_text_to_speech(self, texts: List[str], output_dir: str):
        """Converts multiple texts to speech, saving each to the specified output directory."""
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(texts):
            output_path = output_dir_path / f"output_{i:03d}.wav"
            try:
                self.text_to_speech(text, str(output_path))
                logger.info(f"Successfully converted text {i+1} to {output_path}")
            except Exception as e:
                logger.error(f"Failed to convert text '{text}' (item {i+1}): {e}")

def parse_arguments():
    """Parses command line arguments for the TTS inference script."""
    parser = argparse.ArgumentParser(
        description="Kokoro Russian TTS Inference Script with Neural Vocoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single text with the default HiFi-GAN vocoder (recommended for quality)
  python inference.py --model ./kokoro_russian_model --text "Привет, как дела?" --output my_speech.wav

  # Use a custom HiFi-GAN model path
  python inference.py --model ./kokoro_russian_model --text "Привет мир" --vocoder-path ./my_hifigan_model.pth

  # Fallback to Griffin-Lim vocoder (lower quality, but doesn't require vocoder model)
  python inference.py --model ./kokoro_russian_model --text "Привет мир" --vocoder griffin_lim

  # Convert text from a file
  python inference.py --model ./kokoro_russian_model --text-file input.txt --output file_output.wav

  # Run in interactive mode
  python inference.py --model ./kokoro_russian_model --interactive
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to the trained Kokoro model directory (containing .pth and phoneme_processor.pkl).'
    )

    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Single text string to convert to speech.'
    )

    parser.add_argument(
        '--text-file', '-f',
        type=str,
        help='Path to a file containing text(s) to convert. Each line will be processed if batch mode is used.'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output.wav',
        help='Output audio file path for single text conversion (default: output.wav). For --text-file, this will be overridden by batch naming.'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Enable interactive mode, allowing manual text input for continuous conversion.'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        help='Explicit device to use for inference (e.g., "cuda", "cpu", "mps"). Auto-detected if not specified.'
    )

    parser.add_argument(
        '--vocoder',
        type=str,
        choices=['hifigan', 'griffin_lim'],
        default='hifigan',
        help='Type of vocoder to use: "hifigan" (neural) or "griffin_lim" (algorithmic). Default is hifigan.'
    )

    parser.add_argument(
        '--vocoder-path',
        type=str,
        help='Path to a custom HiFi-GAN vocoder model checkpoint (.pt or .pth) if not using the default or if a specific one is required.'
    )

    parser.add_argument(
        '--stop-threshold',
        dest='stop_threshold',
        type=float,
        default=None,
        help='Stop-token threshold for autoregressive decoding (default: auto from checkpoint).'
    )

    parser.add_argument(
        '--max-len',
        type=int,
        default=None,
        help='Maximum number of mel frames to generate (default: auto from checkpoint).'
    )

    parser.add_argument(
        '--min-len-ratio',
        type=float,
        default=None,
        help='Minimum generated length ratio vs expected duration-based length (default: auto from checkpoint).'
    )

    parser.add_argument(
        '--min-len-floor',
        type=int,
        default=None,
        help='Minimum generated frames floor before allowing stop-token termination (default: auto from checkpoint).'
    )

    parser.add_argument(
        '--weights',
        choices=['auto', 'ema', 'model'],
        default='auto',
        help="Which weights to use for inference: 'auto' prefers EMA if present, 'ema' requires EMA, 'model' uses the trained model weights."
    )

    return parser.parse_args()

def main():
    """Main execution function for the Kokoro TTS inference script."""
    args = parse_arguments()

    try:
        tts = KokoroTTS(
            model_dir=args.model,
            device=args.device,
            vocoder_type=args.vocoder,
            vocoder_path=args.vocoder_path,
            inference_max_len=args.max_len,
            inference_stop_threshold=args.stop_threshold,
            inference_min_len_ratio=args.min_len_ratio,
            inference_min_len_floor=args.min_len_floor,
            weights=args.weights,
        )
    except Exception as e:
        logger.critical(f"Fatal error during TTS system initialization: {e}")
        exit(1)

    # Optional: Apply torch.compile for potential performance improvement
    # This is an advanced optimization and can sometimes cause issues.
    # It's commented out by default for stability. Uncomment and test if needed.
    # try:
    #     # Compile the model's forward pass if your model structure allows it
    #     # Ensure `forward_inference` method exists in your `KokoroModel`
    #     tts.model.forward = torch.compile(tts.model.forward, mode="reduce-overhead")
    #     logger.info("Model's forward method compiled with torch.compile for optimized inference.")
    # except Exception as e:
    #     logger.warning(f"Failed to compile model with torch.compile: {e}. Inference will proceed without compilation.")


    if args.interactive:
        logger.info("\n--- Interactive Mode ---")
        logger.info("Enter Russian text to convert to speech. Type 'quit' or 'exit' to end.")
        while True:
            try:
                text_input = input("\nEnter Russian text: ").strip()
                if text_input.lower() in ['quit', 'exit', 'q']:
                    logger.info("Exiting interactive mode.")
                    break
                if not text_input:
                    continue

                # Generate a unique output filename for interactive mode
                output_path_interactive = f"interactive_output_{abs(hash(text_input)) % 10000}.wav"
                tts.text_to_speech(text_input, output_path_interactive, stop_threshold=args.stop_threshold)
                print(f"Audio saved to: {output_path_interactive}")

            except KeyboardInterrupt:
                logger.info("Interactive mode interrupted by user (Ctrl+C). Exiting.")
                break
            except ValueError as ve:
                logger.error(f"Input Error: {ve}")
            except RuntimeError as re:
                logger.error(f"Runtime Error during conversion: {re}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during interactive conversion: {e}")

    elif args.text:
        # Single text conversion
        try:
            tts.text_to_speech(args.text, args.output)
            logger.info(f"Successfully converted text to {args.output}")
        except ValueError as ve:
            logger.error(f"Error converting text '{args.text}': {ve}")
        except RuntimeError as re:
            logger.error(f"Runtime Error during conversion of '{args.text}': {re}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during single text conversion: {e}")

    elif args.text_file:
        # Text file conversion
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                texts_from_file = [line.strip() for line in f if line.strip()]

            if not texts_from_file:
                logger.warning(f"Text file '{args.text_file}' is empty or contains no valid lines.")
                return

            # If output is a single file, it will be overwritten in batch_text_to_speech.
            # So, we inform the user or adjust. For now, batch_text_to_speech handles naming.
            output_dir_for_batch = Path(args.output).parent if Path(args.output).suffix else args.output
            if not Path(output_dir_for_batch).is_dir():
                output_dir_for_batch = Path("./batch_outputs") # Default to a directory if not specified properly
                logger.info(f"Output for batch text will be saved to '{output_dir_for_batch}'")

            tts.batch_text_to_speech(texts_from_file, str(output_dir_for_batch))
            logger.info(f"Batch conversion complete. Audio files saved to {output_dir_for_batch}")

        except FileNotFoundError:
            logger.error(f"Error: Text file not found at '{args.text_file}'")
        except Exception as e:
            logger.error(f"An error occurred during text file processing: {e}")

    else:
        logger.error("No input provided. Please use --text, --text-file, or --interactive.")
        parse_arguments().print_help()

if __name__ == "__main__":
    main()
