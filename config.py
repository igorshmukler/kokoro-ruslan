#!/usr/bin/env python3
"""
Configuration classes for Kokoro Language Model Training
"""

import torch
from dataclasses import dataclass, field # Import field for default_factory
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration for Kokoro model"""
    data_dir: str = "./ruslan_corpus"
    output_dir: str = "./kokoro_russian_model"
    batch_size: int = 12  # Increased for MPS
    learning_rate: float = 1e-4
    num_epochs: int = 100
    max_seq_length: int = 1024
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    save_every: int = 2
    use_mixed_precision: bool = True  # Enable for MPS
    num_workers: int = 0  # Set to 0 for MPS compatibility
    pin_memory: bool = False  # Disable for MPS
    resume_checkpoint: Optional[str] = None  # Path to checkpoint to resume from

    # Parameters for CosineAnnealingWarmRestarts
    lr_T_0: int = 20        # Number of epochs for the first restart cycle
    lr_T_mult: int = 2      # Multiplier for subsequent cycle lengths
    lr_eta_min: float = 1e-6 # Minimum learning rate

    # Parameters for Duration Modeling and End-of-Speech Prediction ---
    hidden_dim: int = 512 # Hidden dimension for internal model layers (matches KokoroModel)
    duration_loss_weight: float = 0.1 # Weight for the duration prediction loss
    stop_token_loss_weight: float = 1.0 # Weight for the stop token prediction loss
