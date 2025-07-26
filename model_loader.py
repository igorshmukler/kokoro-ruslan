#!/usr/bin/env python3
"""
ModelLoader - Handles loading of trained Kokoro TTS models
Provides robust loading with PyTorch 2.6+ compatibility
"""

import torch
import pickle
import logging
from pathlib import Path
from typing import Optional

# Import model and training components
from config import TrainingConfig
from model import KokoroModel
from russian_phoneme_processor import RussianPhonemeProcessor

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading of trained models and associated components"""

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device

        # Audio configuration (should match training config)
        self.n_mels = 80

    def load_phoneme_processor(self) -> RussianPhonemeProcessor:
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

    def find_model_file(self) -> Path:
        """Find the best available model file"""
        final_model_path = self.model_dir / "kokoro_russian_final.pth"
        checkpoint_files = list(self.model_dir.glob("checkpoint_epoch_*.pth"))

        if final_model_path.exists():
            logger.info(f"Found final model: {final_model_path}")
            return final_model_path
        elif checkpoint_files:
            # Use latest checkpoint
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            latest_checkpoint = checkpoint_files[-1]
            logger.info(f"Found latest checkpoint: {latest_checkpoint}")
            return latest_checkpoint
        else:
            raise FileNotFoundError(f"No model files found in {self.model_dir}")

    def load_checkpoint(self, model_path: Path) -> dict:
        """Load checkpoint with PyTorch 2.6+ compatibility"""
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

        return checkpoint

    def create_model(self, vocab_size: int) -> KokoroModel:
        """Create model with appropriate architecture parameters"""
        # Define model parameters that should match training configuration
        EMBED_DIM = 512
        NUM_LAYERS = 6
        NUM_HEADS = 8
        FF_DIM = 2048
        DROPOUT = 0.1
        MAX_DECODER_SEQ_LEN = 4000  # Match checkpoint's PE size

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
            max_decoder_seq_len=MAX_DECODER_SEQ_LEN
        )

        logger.info(f"Created model with vocab_size={vocab_size}")
        return model

    def load_state_dict(self, model: KokoroModel, checkpoint: dict) -> KokoroModel:
        """Load state dict with filtering for compatibility"""
        # Extract state dict from checkpoint
        state_dict_to_load = None
        if 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict_to_load = checkpoint['model']
        else:
            state_dict_to_load = checkpoint

        if state_dict_to_load is None:
            raise RuntimeError("Checkpoint does not contain a recognized model state dict.")

        # Filter out problematic keys and ensure size compatibility
        filtered_state_dict = {}
        model_keys = model.state_dict().keys()

        for k, v in state_dict_to_load.items():
            if k == "decoder.positional_encoding.pe":
                logger.warning(f"Skipping unexpected key from checkpoint: {k}")
                continue

            if k in model_keys:
                if model.state_dict()[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    logger.warning(f"Size mismatch for key {k}: checkpoint shape {v.shape}, model shape {model.state_dict()[k].shape}. Skipping this parameter.")
            else:
                logger.warning(f"Skipping unknown key from checkpoint: {k}")

        # Load filtered state dict
        try:
            model.load_state_dict(filtered_state_dict, strict=True)
            logger.info("Model loaded successfully with filtered state_dict (strict=True).")
        except RuntimeError as e:
            logger.error(f"Failed to load model state dict even after filtering: {e}")
            # Fallback to strict=False if anything else causes a problem
            model.load_state_dict(filtered_state_dict, strict=False)
            logger.warning("Loaded model with filtered state_dict (strict=False) as a fallback.")

        return model

    def load_model(self, phoneme_processor: RussianPhonemeProcessor) -> KokoroModel:
        """Load complete model with robust error handling"""
        # Find model file
        model_path = self.find_model_file()

        # Load checkpoint
        checkpoint = self.load_checkpoint(model_path)

        # Create model
        vocab_size = len(phoneme_processor.phoneme_to_id)
        model = self.create_model(vocab_size)

        # Load state dict
        model = self.load_state_dict(model, checkpoint)

        # Move to device and set to eval mode
        model.to(self.device)
        model.eval()

        logger.info("Model loaded successfully and ready for inference")
        return model
