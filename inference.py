#!/usr/bin/env python3
"""
Kokoro Russian TTS Inference Script
Convert Russian text to speech using trained Kokoro model
"""

import os
import torch
import torchaudio
import numpy as np
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import logging
import soundfile as sf  # Alternative audio saving library

# Import our model and phoneme processor
from model import KokoroModel
from russian_phoneme_processor import RussianPhonemeProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Training configuration class - needed for checkpoint loading"""
    def __init__(self):
        self.vocab_size = 50
        self.mel_dim = 80
        self.hidden_dim = 512
        self.learning_rate = 0.001
        self.batch_size = 16
        self.num_epochs = 100
        self.sample_rate = 22050
        self.hop_length = 256
        self.win_length = 1024
        self.n_fft = 1024
        self.n_mels = 80
        self.f_min = 0.0
        self.f_max = 8000.0

class KokoroTTS:
    """Main TTS inference class"""
    
    def __init__(self, model_dir: str, device: str = None):
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
        
        # Audio configuration (should match training config) - MOVED BEFORE MODEL LOADING
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
        
        # Griffin-Lim for mel-to-audio conversion
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=2.0,
            n_iter=32
        )
        
        # Mel scale converter for converting mel back to linear scale
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            n_stft=self.n_fft // 2 + 1
        )
        
        # Inverse mel scale converter
        self.inverse_mel_scale = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max
        )
        
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
        
        # Initialize model
        vocab_size = len(self.phoneme_processor.phonemes)
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
        
        logger.info("Model loaded successfully")
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
        """Convert text to speech"""
        logger.info(f"Converting text: '{text}'")
        
        # Convert text to phonemes
        phonemes = self.phoneme_processor.text_to_phonemes(text)
        phoneme_indices = self.phoneme_processor.phonemes_to_indices(phonemes)
        
        logger.info(f"Phonemes: {' '.join(phonemes)}")
        
        # Convert to tensor and add batch dimension
        phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long).unsqueeze(0)
        phoneme_tensor = phoneme_tensor.to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel_spec = self.model(phoneme_tensor)
        
        # Move to CPU for audio processing
        mel_spec = mel_spec.squeeze(0).cpu()  # Remove batch dimension
        
        # Convert log mel to linear scale
        mel_spec = torch.exp(mel_spec)
        
        # Transpose to get correct shape: (time, n_mels) -> (n_mels, time)
        mel_spec = mel_spec.transpose(0, 1)
        
        # Convert mel spectrogram back to linear magnitude spectrogram
        # mel_spec shape: (n_mels, time) -> (n_fft//2 + 1, time)
        linear_spec = self.inverse_mel_scale(mel_spec)
        
        # Convert linear magnitude spectrogram to audio using Griffin-Lim
        audio = self.griffin_lim(linear_spec)
        
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
        description="Kokoro Russian TTS Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single text to speech
  python inference.py --model ./kokoro_russian_model --text "Привет, как дела?"

  # Convert text and save to specific file
  python inference.py --model ./kokoro_russian_model --text "Привет мир" --output hello.wav

  # Convert text from file
  python inference.py --model ./kokoro_russian_model --text-file input.txt --output output.wav

  # Interactive mode
  python inference.py --model ./kokoro_russian_model --interactive
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
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize TTS system
    try:
        tts = KokoroTTS(args.model, args.device)
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
