#!/usr/bin/env python3
"""
Kokoro Russian TTS Inference Script with HiFi-GAN Vocoder
Convert Russian text to speech using trained Kokoro model with neural vocoder
"""

import torch
import argparse
import pickle
from pathlib import Path
from typing import List, Optional
import logging

# Import our training configuration, model and phoneme processor
from model import KokoroModel
from russian_phoneme_processor import RussianPhonemeProcessor
from vocoder_manager import VocoderManager
from audio_utils import AudioUtils, PhonemeProcessorUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KokoroTTS:
    """Main TTS inference class with neural vocoder support."""

    def __init__(self, model_dir: str, device: str = None, vocoder_type: str = "hifigan", vocoder_path: str = None):
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

        # Initialize utility classes
        self.audio_utils = AudioUtils(self.sample_rate)

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

        # Define common model parameters that should match training config
        # These values MUST correspond to the architecture the loaded model was trained with.
        # You might want to load these from a `model_config.json` saved during training.
        EMBED_DIM = 512
        NUM_LAYERS = 6
        NUM_HEADS = 8
        FF_DIM = 2048
        DROPOUT = 0.1
        MAX_DECODER_SEQ_LEN = 4000 # Critical: This must match the positional encoding size used during training

        vocab_size = len(self.phoneme_processor.phoneme_to_id)
        model = KokoroModel(
            vocab_size=vocab_size,
            mel_dim=self.n_mels,
            hidden_dim=EMBED_DIM,
            n_encoder_layers=NUM_LAYERS,
            n_heads=NUM_HEADS,
            encoder_ff_dim=FF_DIM,
            encoder_dropout=DROPOUT,
            n_decoder_layers=NUM_LAYERS,
            decoder_ff_dim=FF_DIM,
            max_decoder_seq_len=MAX_DECODER_SEQ_LEN,
            enable_profiling= getattr(self, 'enable_profiling', False)
        )

        # Filter and load state dict
        filtered_state_dict = {}
        model_keys = set(model.state_dict().keys())

        for k, v in state_dict_to_load.items():
            if k in model_keys:
                if model.state_dict()[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    logger.warning(f"Skipping parameter '{k}' due to shape mismatch: checkpoint shape {v.shape}, model shape {model.state_dict()[k].shape}.")
            else:
                logger.warning(f"Skipping unknown key '{k}' from checkpoint. It might be an optimizer state or a deprecated parameter.")

        try:
            model.load_state_dict(filtered_state_dict, strict=True)
            logger.info("Model state dictionary loaded successfully (strict=True).")
        except RuntimeError as e:
            logger.warning(f"Failed to load model state dict strictly: {e}. Attempting non-strict load. This may indicate a mismatch between the model architecture and the loaded weights.")
            model.load_state_dict(filtered_state_dict, strict=False)
            logger.info("Model state dictionary loaded with strict=False.")

        model.to(self.device)
        model.eval() # Set model to evaluation mode

        logger.info(f"Model '{model_path.name}' loaded successfully with vocab_size={vocab_size}.")
        return model

    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> torch.Tensor:
        """
        Updated text_to_speech with better parameters and error handling.
        """
        if not text:
            logger.warning("Received empty text for conversion. Returning empty audio.")
            return torch.empty(0)

        logger.info(f"Converting text: '{text}'")

        try:
            # Step 1: Process text into phoneme sequence
            raw_processor_output = self.phoneme_processor.process_text(text)
            phoneme_sequence = PhonemeProcessorUtils.flatten_phoneme_output(raw_processor_output)

            if not phoneme_sequence:
                logger.error(f"Phoneme processor produced no phonemes for text: '{text}'. Conversion aborted.")
                raise ValueError("No phonemes generated from the input text.")

            logger.info(f"Phonemes: {' '.join(phoneme_sequence)}")

            # Step 2: Convert phonemes to numerical indices
            phoneme_indices = PhonemeProcessorUtils.phonemes_to_indices(
                phoneme_sequence, self.phoneme_processor.phoneme_to_id
            )
            logger.debug(f"Phoneme indices (first 20): {phoneme_indices[:20]}...")

            # Convert to tensor and add batch dimension
            phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long).unsqueeze(0).to(self.device)

            # Step 3: Generate mel spectrogram with updated parameters
            with torch.no_grad():
                # Use more conservative parameters for broken models
                mel_spec = self.model.forward_inference(
                    phoneme_indices=phoneme_tensor,
                    max_len=400,  # Reduced from potential 10000
                    stop_threshold=0.01,  # Much lower threshold
                    text_padding_mask=None
                )

            # Remove batch dimension and move to CPU for vocoder
            mel_spec = mel_spec.squeeze(0).cpu()

            # Step 4: Convert mel spectrogram to audio using the neural vocoder
            audio = self.vocoder_manager.mel_to_audio(mel_spec)

            # Save audio if an output path is provided
            if output_path:
                self.audio_utils.save_audio(audio, output_path)
                logger.info(f"Audio saved to: {output_path}")

            return audio

        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}")
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

    return parser.parse_args()

def main():
    """Main execution function for the Kokoro TTS inference script."""
    args = parse_arguments()

    try:
        tts = KokoroTTS(
            model_dir=args.model,
            device=args.device,
            vocoder_type=args.vocoder,
            vocoder_path=args.vocoder_path
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
                tts.text_to_speech(text_input, output_path_interactive)
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
