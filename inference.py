#!/usr/bin/env python3
"""
Kokoro Russian TTS Inference Script with HiFi-GAN Vocoder
Convert Russian text to speech using trained Kokoro model with neural vocoder
"""

import os
import torch
import torchaudio
import numpy as np
import argparse
import pickle
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import soundfile as sf
from urllib.parse import urlparse

# Import our training configuration, model and phoneme processor
from config import TrainingConfig
from model import KokoroModel
from russian_phoneme_processor import RussianPhonemeProcessor
from hifigan_vocoder import HiFiGANGenerator, load_hifigan_model, HiFiGANConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VocoderManager:
    """Manages different vocoder backends"""

    HIFIGAN_URLS = {
        # Universal HiFi-GAN models (22kHz)
        "universal_v1": {
            "model": "https://drive.google.com/uc?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW",
            "config": "https://drive.google.com/uc?id=1pAB2kQunkDuv6W5fcJiQ0CY8xcJKB22e"
        },
        # LJ Speech model (good for general purpose)
        "ljspeech": {
            "model": "https://drive.google.com/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx",
            "config": "https://drive.google.com/uc?id=1Jt_imitfckTfM9TPhT4TQKPUgkcGhv5f"
        }
    }

    def __init__(self, vocoder_type: str = "hifigan", vocoder_path: Optional[str] = None, device: str = "cpu"):
        self.vocoder_type = vocoder_type.lower()
        self.device = device
        self.vocoder = None

        if vocoder_type == "hifigan":
            self.vocoder = self._load_hifigan(vocoder_path)
        elif vocoder_type == "griffin_lim":
            self.vocoder = self._setup_griffin_lim()
        else:
            raise ValueError(f"Unsupported vocoder type: {vocoder_type}")

    def _download_file(self, url: str, filepath: Path):
        """Download file with progress"""
        logger.info(f"Downloading {filepath.name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='')
        print()  # New line after progress

    def _load_hifigan(self, vocoder_path: Optional[str] = None) -> torch.nn.Module:
        """Load HiFi-GAN vocoder using the separated module"""
        if vocoder_path and Path(vocoder_path).exists():
            # Load custom HiFi-GAN model
            model_path = Path(vocoder_path)
            if model_path.is_dir():
                generator_path = model_path / "generator.pth"
                config_path = model_path / "config.json"
            else:
                generator_path = model_path
                config_path = model_path.parent / "config.json"

            try:
                generator = load_hifigan_model(generator_path, config_path, self.device)
                logger.info(f"Loaded custom HiFi-GAN from: {vocoder_path}")
                return generator
            except Exception as e:
                logger.warning(f"Failed to load custom HiFi-GAN: {e}")
                logger.info("Falling back to pre-trained model")

        # Try to load pre-trained model or download
        vocoder_dir = Path("./vocoder_models/hifigan")
        vocoder_dir.mkdir(parents=True, exist_ok=True)

        model_name = "universal_v1"  # Default to universal model
        model_file = vocoder_dir / f"generator_{model_name}.pth"
        config_file = vocoder_dir / f"config_{model_name}.json"

        # Download if not exists
        if not model_file.exists() or not config_file.exists():
            logger.info(f"Downloading HiFi-GAN {model_name} model...")
            try:
                if not model_file.exists():
                    self._download_file(self.HIFIGAN_URLS[model_name]["model"], model_file)
                if not config_file.exists():
                    self._download_file(self.HIFIGAN_URLS[model_name]["config"], config_file)
            except Exception as e:
                logger.warning(f"Failed to download HiFi-GAN model: {e}")
                logger.info("Falling back to Griffin-Lim")
                return self._setup_griffin_lim()

        # Load downloaded model
        try:
            generator = load_hifigan_model(model_file, config_file, self.device)
            logger.info(f"Loaded pre-trained HiFi-GAN {model_name}")
            return generator
        except Exception as e:
            logger.warning(f"Failed to load HiFi-GAN: {e}")
            logger.info("Falling back to Griffin-Lim")
            return self._setup_griffin_lim()

    def _setup_griffin_lim(self):
        """Setup Griffin-Lim as fallback with device compatibility"""
        logger.info("Using Griffin-Lim vocoder")

        # For MPS device compatibility, create Griffin-Lim on CPU initially
        # and move to device later if supported
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=2.0,
            n_iter=60  # More iterations for better quality
        )

        # Try to move to target device, fallback to CPU if not supported
        try:
            griffin_lim = griffin_lim.to(self.device)
        except Exception as e:
            logger.warning(f"Griffin-Lim not fully compatible with {self.device}, using CPU fallback for some operations: {e}")
            griffin_lim = griffin_lim.to("cpu")

        return griffin_lim

    def mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to audio"""
        if self.vocoder_type == "hifigan":
            return self._hifigan_inference(mel_spec)
        elif self.vocoder_type == "griffin_lim":
            return self._griffin_lim_inference(mel_spec)
        else:
            raise ValueError(f"Unknown vocoder type: {self.vocoder_type}")

    def _hifigan_inference(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """HiFi-GAN inference"""
        if isinstance(self.vocoder, torchaudio.transforms.GriffinLim):
            # Fallback to Griffin-Lim if HiFi-GAN failed to load
            return self._griffin_lim_inference(mel_spec)

        with torch.no_grad():
            # Ensure mel_spec is on the right device and has the right shape
            mel_spec = mel_spec.to(self.device)

            if len(mel_spec.shape) == 2:  # (n_mels, time)
                mel_spec = mel_spec.unsqueeze(0)  # (1, n_mels, time)
            elif len(mel_spec.shape) == 3 and mel_spec.shape[0] != 1:  # (batch, time, n_mels)
                mel_spec = mel_spec.transpose(1, 2)  # (batch, n_mels, time)

            # Generate audio
            audio = self.vocoder(mel_spec)

            if len(audio.shape) == 3:  # Remove batch dimension if present
                audio = audio.squeeze(0)
            if len(audio.shape) == 2:  # Remove channel dimension if present
                audio = audio.squeeze(0)

        return audio.cpu()

    def _griffin_lim_inference(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Griffin-Lim inference with device compatibility handling"""
        # Convert log mel to linear scale
        mel_spec = torch.exp(mel_spec)

        # Transpose to get correct shape: (time, n_mels) -> (n_mels, time)
        if len(mel_spec.shape) == 2 and mel_spec.shape[1] == 80:
            mel_spec = mel_spec.transpose(0, 1)

        # Handle MPS device incompatibility with InverseMelScale
        device_for_mel_ops = "cpu" if self.device == "mps" else self.device

        # Convert mel spectrogram back to linear magnitude spectrogram
        inverse_mel_scale = torchaudio.transforms.InverseMelScale(
            n_stft=513,
            n_mels=80,
            sample_rate=22050,
            f_min=0.0,
            f_max=8000.0
        ).to(device_for_mel_ops)

        # Move mel_spec to compatible device for inverse transform
        mel_spec_for_inverse = mel_spec.to(device_for_mel_ops)
        linear_spec = inverse_mel_scale(mel_spec_for_inverse)

        # Move back to original device for Griffin-Lim if needed
        if device_for_mel_ops != self.device:
            linear_spec = linear_spec.to(self.device)

        # Convert linear magnitude spectrogram to audio using Griffin-Lim
        audio = self.vocoder(linear_spec)

        return audio.cpu()

class KokoroTTS:
    """Main TTS inference class with neural vocoder support"""

    def __init__(self, model_dir: str, device: str = None, vocoder_type: str = "hifigan", vocoder_path: str = None):
        self.model_dir = Path(model_dir)

        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Audio configuration (should match training config)
        self.sample_rate = 22050
        self.hop_length = 256
        self.win_length = 1024
        self.n_fft = 1024
        self.n_mels = 80
        self.f_min = 0.0
        self.f_max = 8000.0

        # Load phoneme processor
        self.phoneme_processor = self._load_phoneme_processor()

        # Load model
        self.model = self._load_model()

        # Initialize vocoder
        self.vocoder_manager = VocoderManager(vocoder_type, vocoder_path, self.device)

    def _load_phoneme_processor(self) -> RussianPhonemeProcessor:
        """Load the phoneme processor"""
        processor_path = self.model_dir / "phoneme_processor.pkl"
        
        if processor_path.exists():
            with open(processor_path, 'rb') as f:
                processor_data = pickle.load(f)
            processor = RussianPhonemeProcessor.from_dict(processor_data)
            logger.info(f"Loaded phoneme processor from: {processor_path}")
        else:
            logger.warning("Phoneme processor not found, creating new one")
            processor = RussianPhonemeProcessor()
            
        return processor
    
    def _load_model(self) -> KokoroModel:
        """Load the trained model with robust error handling"""
        # Try to find model file
        final_model_path = self.model_dir / "kokoro_russian_final.pth"
        checkpoint_files = list(self.model_dir.glob("checkpoint_epoch_*.pth"))

        model_path = None
        if final_model_path.exists():
            model_path = final_model_path
            logger.info(f"Loading final model: {model_path}")
        elif checkpoint_files:
            # Use latest checkpoint
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            model_path = checkpoint_files[-1]
            logger.info(f"Loading latest checkpoint: {model_path}")
        else:
            raise FileNotFoundError(f"No model files found in {self.model_dir}")

        # Load model with error handling for PyTorch 2.6+ weights_only changes
        checkpoint = None

        # Method 1: Try with safe globals (PyTorch 2.6+)
        try:
            with torch.serialization.safe_globals([TrainingConfig]):
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            logger.info("Loaded checkpoint with safe_globals and weights_only=True")
        except Exception as e:
            logger.warning(f"Failed to load with safe_globals: {e}")

        # Method 2: Try with weights_only=False (less secure but works)
        if checkpoint is None:
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                logger.info("Loaded checkpoint with weights_only=False")
            except Exception as e:
                logger.warning(f"Failed to load with weights_only=False: {e}")

        # Method 3: Try adding safe globals and loading (alternative approach)
        if checkpoint is None:
            try:
                torch.serialization.add_safe_globals([TrainingConfig])
                checkpoint = torch.load(model_path, map_location='cpu')
                logger.info("Loaded checkpoint with add_safe_globals")
            except Exception as e:
                logger.error(f"Failed to load checkpoint with add_safe_globals: {e}")
                raise e

        if checkpoint is None:
            raise RuntimeError("Failed to load checkpoint with any method")

        # Initialize model - FIX: Use the correct vocab size calculation
        vocab_size = len(self.phoneme_processor.phoneme_to_id)  # Fixed from .phonemes to .phoneme_to_id
        model = KokoroModel(vocab_size, self.n_mels)

        # Load state dict - handle different checkpoint formats
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded model from 'model_state_dict' key")
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                logger.info("Loaded model from 'model' key")
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded model directly from checkpoint")
        except Exception as e:
            logger.error(f"Error loading model state dict: {e}")
            # Try to load with strict=False to ignore missing keys
            try:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                logger.warning("Loaded model with strict=False (some parameters may be missing)")
            except Exception as e2:
                logger.error(f"Failed to load model even with strict=False: {e2}")
                raise e

        model.to(self.device)
        model.eval()

        logger.info(f"Model loaded successfully with vocab_size={vocab_size}")
        return model

    def _save_audio(self, audio: torch.Tensor, output_path: str):
        """Save audio with multiple fallback methods"""
        try:
            # Method 1: Try torchaudio with explicit format
            torchaudio.save(output_path, audio.unsqueeze(0), self.sample_rate, format="wav")
            logger.info(f"Audio saved using torchaudio: {output_path}")
            return
        except Exception as e:
            logger.warning(f"torchaudio.save failed: {e}, trying alternative methods...")

        try:
            # Method 2: Try soundfile
            audio_np = audio.numpy()
            sf.write(output_path, audio_np, self.sample_rate)
            logger.info(f"Audio saved using soundfile: {output_path}")
            return
        except Exception as e:
            logger.warning(f"soundfile failed: {e}, trying scipy...")

        try:
            # Method 3: Try scipy.io.wavfile
            from scipy.io import wavfile
            # Convert to 16-bit integer
            audio_np = audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wavfile.write(output_path, self.sample_rate, audio_int16)
            logger.info(f"Audio saved using scipy: {output_path}")
            return
        except Exception as e:
            logger.warning(f"scipy failed: {e}, trying raw numpy save...")

        try:
            # Method 4: Save as numpy array (for debugging)
            audio_np = audio.numpy()
            np.save(output_path.replace('.wav', '.npy'), audio_np)
            logger.info(f"Audio saved as numpy array: {output_path.replace('.wav', '.npy')}")
            logger.info("Note: You can convert the .npy file to WAV using external tools")
            return
        except Exception as e:
            logger.error(f"All audio saving methods failed: {e}")
            raise RuntimeError("Could not save audio file with any method")

    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> torch.Tensor:
        """Convert text to speech using neural vocoder"""
        logger.info(f"Converting text: '{text}'")

        # FIX: Use the correct method names for the phoneme processor
        results = self.phoneme_processor.process_text(text)
        phonemes = []
        for word, word_phonemes, stress_info in results:
            phonemes.extend(word_phonemes)

        # Convert phonemes to indices
        phoneme_indices = self.phoneme_processor.text_to_indices(text)

        logger.info(f"Phonemes: {' '.join(phonemes)}")
        logger.info(f"Phoneme indices: {phoneme_indices[:20]}...")  # Show first 20 indices

        # Convert to tensor and add batch dimension
        phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long).unsqueeze(0)
        phoneme_tensor = phoneme_tensor.to(self.device)

        # Generate mel spectrogram
        with torch.no_grad():
            mel_spec = self.model(phoneme_tensor)

        # Move to CPU for audio processing
        mel_spec = mel_spec.squeeze(0).cpu()  # Remove batch dimension

        # Convert mel spectrogram to audio using neural vocoder
        audio = self.vocoder_manager.mel_to_audio(mel_spec)

        # Normalize audio
        audio = audio / torch.max(torch.abs(audio))

        # Save audio if path provided
        if output_path:
            self._save_audio(audio, output_path)

        return audio

    def batch_text_to_speech(self, texts: List[str], output_dir: str):
        """Convert multiple texts to speech"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        for i, text in enumerate(texts):
            output_path = output_dir / f"output_{i:03d}.wav"
            self.text_to_speech(text, str(output_path))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Kokoro Russian TTS Inference Script with Neural Vocoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with HiFi-GAN (default, best quality)
  python inference.py --model ./kokoro_russian_model --text "Привет, как дела?"

  # Use custom HiFi-GAN model
  python inference.py --model ./kokoro_russian_model --text "Привет мир" --vocoder-path ./my_hifigan_model

  # Use Griffin-Lim (fallback, lower quality)
  python inference.py --model ./kokoro_russian_model --text "Привет мир" --vocoder griffin_lim

  # Convert text from file with HiFi-GAN
  python inference.py --model ./kokoro_russian_model --text-file input.txt --output output.wav

  # Interactive mode with neural vocoder
  python inference.py --model ./kokoro_russian_model --interactive --vocoder hifigan
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to the trained model directory'
    )

    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Text to convert to speech'
    )

    parser.add_argument(
        '--text-file', '-f',
        type=str,
        help='File containing text to convert'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output.wav',
        help='Output audio file path (default: output.wav)'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode - enter text manually'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for inference (auto-detected if not specified)'
    )

    parser.add_argument(
        '--vocoder',
        type=str,
        choices=['hifigan', 'griffin_lim'],
        default='hifigan',
        help='Vocoder type to use (default: hifigan)'
    )

    parser.add_argument(
        '--vocoder-path',
        type=str,
        help='Path to custom vocoder model (for HiFi-GAN)'
    )

    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()

    # Initialize TTS system with vocoder arguments
    try:
        tts = KokoroTTS(
            model_dir=args.model,
            device=args.device,
            vocoder_type=args.vocoder,
            vocoder_path=args.vocoder_path
        )
    except Exception as e:
        logger.error(f"Failed to initialize TTS system: {e}")
        return

    if args.interactive:
        # Interactive mode
        logger.info("Interactive mode - type 'quit' to exit")
        while True:
            try:
                text = input("\nEnter Russian text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue

                output_path = f"interactive_output_{hash(text) % 10000}.wav"
                tts.text_to_speech(text, output_path)
                print(f"Audio saved to: {output_path}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error during conversion: {e}")

    elif args.text:
        # Single text conversion
        tts.text_to_speech(args.text, args.output)

    elif args.text_file:
        # Text file conversion
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            tts.text_to_speech(text, args.output)
        except Exception as e:
            logger.error(f"Error reading text file: {e}")

    else:
        logger.error("Please provide either --text, --text-file, or --interactive")
        return

if __name__ == "__main__":
    main()
