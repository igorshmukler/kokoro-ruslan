#!/usr/bin/env python3
"""
Configuration classes for Kokoro Language Model Training
"""

import torch
from dataclasses import dataclass
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
