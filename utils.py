#!/usr/bin/env python3
"""
Utility functions for Kokoro training
"""

import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def clear_gpu_cache():
    """Clear GPU cache based on available backend"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices"""
    device_info = {
        'mps_available': torch.backends.mps.is_available(),
        'cuda_available': torch.cuda.is_available(),
        'recommended_device': 'cpu'
    }
    
    if device_info['mps_available']:
        device_info['recommended_device'] = 'mps'
    elif device_info['cuda_available']:
        device_info['recommended_device'] = 'cuda'
        device_info['cuda_device_count'] = torch.cuda.device_count()
        device_info['cuda_device_name'] = torch.cuda.get_device_name(0)
    
    return device_info


def log_device_info():
    """Log device information"""
    device_info = get_device_info()
    
    logger.info("=== Device Information ===")
    logger.info(f"MPS Available: {device_info['mps_available']}")
    logger.info(f"CUDA Available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        logger.info(f"CUDA Device Count: {device_info.get('cuda_device_count', 0)}")
        logger.info(f"CUDA Device Name: {device_info.get('cuda_device_name', 'Unknown')}")
    
    logger.info(f"Recommended Device: {device_info['recommended_device']}")
    logger.info("=" * 27)


def setup_training_environment():
    """Setup optimal training environment"""
    log_device_info()
    
    # Set optimal settings for different backends
    if torch.backends.mps.is_available():
        logger.info("Optimizing for MPS backend...")
        # MPS-specific optimizations could go here
    elif torch.cuda.is_available():
        logger.info("Optimizing for CUDA backend...")
        # CUDA-specific optimizations could go here
    else:
        logger.info("Using CPU backend...")


def validate_training_config(config) -> bool:
    """Validate training configuration"""
    errors = []
    
    # Check required parameters
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    
    if config.learning_rate <= 0:
        errors.append("learning_rate must be positive")
    
    if config.num_epochs <= 0:
        errors.append("num_epochs must be positive")
    
    # Check audio parameters
    if config.sample_rate <= 0:
        errors.append("sample_rate must be positive")
    
    if config.n_mels <= 0:
        errors.append("n_mels must be positive")
    
    if config.hop_length <= 0:
        errors.append("hop_length must be positive")
    
    # Log errors if any
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Configuration validation passed")
    return True


def format_model_size(num_params: int) -> str:
    """Format model parameter count in human-readable format"""
    if num_params < 1000:
        return f"{num_params}"
    elif num_params < 1_000_000:
        return f"{num_params/1000:.1f}K"
    elif num_params < 1_000_000_000:
        return f"{num_params/1_000_000:.1f}M"
    else:
        return f"{num_params/1_000_000_000:.1f}B"


def calculate_model_memory(num_params: int, precision: str = "float32") -> float:
    """Calculate approximate model memory usage in MB"""
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1
    }
    
    if precision not in bytes_per_param:
        precision = "float32"
    
    total_bytes = num_params * bytes_per_param[precision]
    return total_bytes / (1024 * 1024)  # Convert to MB


def log_training_progress(epoch: int, total_epochs: int, loss: float, lr: float):
    """Log training progress in a formatted way"""
    progress_percent = (epoch / total_epochs) * 100
    logger.info(f"Epoch {epoch:3d}/{total_epochs} ({progress_percent:5.1f}%) | "
                f"Loss: {loss:.6f} | LR: {lr:.2e}")


def estimate_training_time(samples_per_epoch: int, batch_size: int, 
                          seconds_per_batch: float, num_epochs: int) -> str:
    """Estimate total training time"""
    batches_per_epoch = samples_per_epoch // batch_size
    total_batches = batches_per_epoch * num_epochs
    total_seconds = total_batches * seconds_per_batch
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    if hours > 0:
        return f"~{hours}h {minutes}m"
    else:
        return f"~{minutes}m"
