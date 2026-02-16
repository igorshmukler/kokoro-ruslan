#!/usr/bin/env python3
"""
Enhanced Training Configuration with Gradient Checkpointing Support
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Enhanced training configuration with gradient checkpointing options"""

    # Basic training parameters
    data_dir: str = "data/processed_data"
    output_dir: str = "output_models"
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Gradient accumulation for larger effective batch sizes
    gradient_accumulation_steps: int = 4  # Effective batch size = batch_size * gradient_accumulation_steps

    # Learning rate scheduler (OneCycleLR)
    use_onecycle_lr: bool = True  # Use OneCycleLR instead of CosineAnnealingWarmRestarts
    max_lr_multiplier: float = 10.0  # Max LR = learning_rate * this value
    pct_start: float = 0.3  # Percentage of cycle spent increasing LR (warmup)

    # Linear warmup before OneCycleLR
    use_warmup: bool = True  # Enable linear warmup before OneCycleLR
    warmup_steps: int = 500  # Number of optimizer steps for linear warmup (not batches!)
    warmup_start_lr_ratio: float = 0.01  # Start LR = learning_rate * this value

    # EMA (Exponential Moving Average) of model weights
    use_ema: bool = True  # Enable EMA for better inference quality
    ema_decay: float = 0.9999  # EMA decay rate (higher = slower update, more smoothing)
    ema_update_every: int = 1  # Update EMA every N optimizer steps (1 = every step)

    # Legacy CosineAnnealingWarmRestarts settings (used if use_onecycle_lr=False)
    lr_T_0: int = 20
    lr_T_mult: int = 2
    lr_eta_min: float = 1e-6

    # Model parameters
    n_mels: int = 80
    hidden_dim: int = 512
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    n_heads: int = 8
    encoder_ff_dim: int = 2048
    decoder_ff_dim: int = 2048
    encoder_dropout: float = 0.1
    max_decoder_seq_len: int = 4000

    # Stochastic depth (layer dropout) for regularization
    use_stochastic_depth: bool = True  # Enable layer dropout during training
    stochastic_depth_rate: float = 0.1  # Maximum drop probability for last layer
    # Drop probability increases linearly from 0 (first layer) to stochastic_depth_rate (last layer)

    # Loss weights
    duration_loss_weight: float = 0.1
    stop_token_loss_weight: float = 1.0
    pitch_loss_weight: float = 0.1  # Normalized to [0,1], safe to use
    energy_loss_weight: float = 0.1  # Normalized to [0,1], safe to use

    # Variance predictor settings
    use_variance_predictor: bool = True  # Enabled with normalized [0,1] inputs and auto-reset
    variance_filter_size: int = 256
    variance_kernel_size: int = 3
    variance_dropout: float = 0.1
    n_variance_bins: int = 256
    pitch_min: float = 0.0  # Normalized range (extractors output [0, 1])
    pitch_max: float = 1.0  # Normalized range (extractors output [0, 1])
    energy_min: float = 0.0
    energy_max: float = 1.0  # Normalized range (extractors output [0, 1])

    # Audio processing
    max_seq_length: int = 2500
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    f_min: float = 0.0
    f_max: float = 8000.0

    # Data loading
    num_workers: int = 2
    pin_memory: bool = False

    # Feature caching for faster training
    use_feature_cache: bool = True  # Cache mel spectrograms, pitch, energy to disk
    feature_cache_dir: str = ""  # Will be set to {data_dir}/.feature_cache if empty
    precompute_features: bool = False  # Precompute all features before training starts

    # Dynamic batching (batch by total frames instead of fixed size)
    use_dynamic_batching: bool = True  # Enable frame-based batching
    max_frames_per_batch: int = 30000  # Maximum mel frames per batch (auto-adjusted for MPS)
    min_batch_size: int = 4  # Minimum samples per batch
    max_batch_size: int = 32  # Maximum samples per batch

    # Checkpointing
    save_every: int = 2
    resume_checkpoint: str = 'auto'

    # Validation settings
    validation_split: float = 0.1  # 10% of data for validation
    validation_interval: int = 1  # Run validation every N epochs
    early_stopping_patience: int = 10  # Stop if no improvement for N epochs
    early_stopping_min_delta: float = 0.001  # Minimum improvement to count

    # Montreal Forced Aligner (MFA) settings
    use_mfa: bool = True
    mfa_alignment_dir: str = "./mfa_output/alignments"
    mfa_acoustic_model: str = "russian_mfa"
    mfa_dictionary: str = "russian_mfa"

    # Enable gradient checkpointing by default
    gradient_checkpointing: bool = True

    # Number of segments to divide layers into for checkpointing
    # Higher values = more memory savings but slightly more computation
    checkpoint_segments: int = 2

    # Auto-optimize checkpoint segments based on available GPU memory
    auto_optimize_checkpointing: bool = True

    # Target memory usage percentage (0.0-1.0) for auto-optimization
    target_memory_usage: float = 0.8

    # Run checkpointing benchmark at startup
    benchmark_checkpointing: bool = False

    # Profiling
    enable_profiling: bool = False
    profile_epoch_start: int = 1  # Start profiling from this epoch (0-indexed)
    profile_wait_steps: int = 1   # Number of steps to wait before starting warmup
    profile_warmup_steps: int = 1 # Number of steps to warm up the profiler
    profile_steps: int = 5        # Number of active steps to profile
    run_standalone_profiling: bool = False  # Run standalone profiling before training
    verbose: bool = False  # Enable verbose training diagnostics

    # Interbatch profiling
    enable_interbatch_profiling: bool = False
    interbatch_report_interval: int = 100

    use_mixed_precision: bool = True

    # torch.compile optimization (PyTorch 2.0+)
    use_torch_compile: bool = True
    torch_compile_mode: str = 'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'
    torch_compile_dynamic: bool = True  # Handle dynamic shapes better

    def __post_init__(self):
        """Post-initialization to handle gradient checkpointing optimization"""

        # Set default feature cache directory if not specified
        if not self.feature_cache_dir:
            from pathlib import Path
            self.feature_cache_dir = str(Path(self.data_dir) / ".feature_cache")

        # MPS-specific memory optimizations
        if self.device == 'mps' or (torch.backends.mps.is_available() and self.device != 'cuda'):
            # Balanced settings to avoid MPS dimension overflow while maximizing throughput
            if self.max_frames_per_batch > 30000:
                print(f"MPS detected: Reducing max_frames_per_batch from {self.max_frames_per_batch} to 30000")
                self.max_frames_per_batch = 30000

            if self.max_seq_length > 1800:
                print(f"MPS detected: Reducing max_seq_length from {self.max_seq_length} to 1800")
                self.max_seq_length = 1800

            if self.batch_size > 10:
                print(f"MPS detected: Reducing batch_size from {self.batch_size} to 10")
                self.batch_size = 10

            if self.max_batch_size > 20:
                print(f"MPS detected: Reducing max_batch_size from {self.max_batch_size} to 20")
                self.max_batch_size = 20
        if self.checkpoint_segments < 1:
            self.checkpoint_segments = 1
            print("Warning: checkpoint_segments must be >= 1, setting to 1")

        # Auto-optimize checkpointing if requested
        if self.auto_optimize_checkpointing and self.gradient_checkpointing:
            self._optimize_checkpointing()

        # Log gradient checkpointing configuration
        if self.gradient_checkpointing:
            print(f"Gradient checkpointing enabled with {self.checkpoint_segments} segments")
            estimated_savings = (self.checkpoint_segments - 1) / self.checkpoint_segments * 100
            print(f"Estimated memory savings: {estimated_savings:.1f}%")
        else:
            print("Gradient checkpointing disabled")

    def _optimize_checkpointing(self):
        """Optimize checkpoint segments based on available GPU or MPS memory"""
        device = None
        if torch.cuda.is_available():
            device = "cuda"
            print("CUDA available, optimizing checkpointing for GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("MPS available, optimizing checkpointing for Apple Silicon (MPS)")
        else:
            print("Neither CUDA nor MPS available, skipping checkpointing optimization")
            return

        try:
            total_memory_mb = 0
            if device == "cuda":
                # Get GPU memory info for CUDA
                total_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            elif device == "mps":
                # For MPS, getting exact dedicated memory is not as straightforward as CUDA.
                # We'll use a placeholder or a common value for Apple Silicon devices.
                # A more robust solution might involve checking system memory or
                # making assumptions based on common MPS-enabled device configurations.
                # For demonstration, let's use a common value for M1/M2 chips.
                # This needs to be adjusted based on actual device memory if possible.
                print("MPS memory estimation is a heuristic. Actual available memory might vary.")
                # Assuming unified memory, this value is a rough estimate for available "graphics" memory
                # You might want to adjust this based on specific device models (e.g., 8GB, 16GB, 32GB unified memory)
                # For now, let's use a general heuristic for a common M-series chip
                total_memory_mb = torch.mps.current_allocated_memory() / 1024**2 # This gives current allocated, not total.
                # A better approach for total_memory_mb on MPS is challenging without system-level calls.
                # Let's use a sensible default or classify by typical unified memory sizes.
                # For simplicity in this example, we'll make a broad assumption.
                # In a real application, you might need to query system total memory.

                # Fallback/Heuristic for MPS total memory if specific API isn't easy
                # This is a rough guess based on common unified memory configurations (e.g., 8GB, 16GB, 24GB, 32GB).
                # This part is highly dependent on how precise you need the MPS memory detection to be.
                # For now, let's just make sure we have a non-zero value to proceed with heuristics.
                if total_memory_mb == 0:
                    # If current_allocated_memory is 0 (e.g., at start), provide a sensible default for total.
                    # This is a rough estimation for typical M1/M2 Macs
                    total_memory_mb = 8000 # Assume at least 8GB unified memory for basic MPS devices.
                    print(f"Using a heuristic total memory of {total_memory_mb:.0f} MB for MPS.")

            target_memory_mb = total_memory_mb * self.target_memory_usage

            # Estimate model parameter memory
            total_layers = self.n_encoder_layers + self.n_decoder_layers

            # Rough heuristic for memory usage
            # The 4 here typically represents float32, which is 4 bytes.
            # This estimate is generally applicable regardless of CUDA or MPS as it's about parameter size.
            param_memory_estimate = self.hidden_dim * self.hidden_dim * total_layers * 4 / 1024**2  # MB
            available_for_activations = target_memory_mb - param_memory_estimate - 2000  # Reserve 2GB

            if available_for_activations <= 0:
                print(f"Warning: Very limited {device.upper()} memory ({total_memory_mb:.0f} MB), using conservative checkpointing")
                self.checkpoint_segments = max(4, total_layers // 2)
            else:
                # More segments for less memory, fewer segments for more memory
                if total_memory_mb < 8000:  # < 8GB
                    self.checkpoint_segments = max(3, total_layers // 2)
                elif total_memory_mb < 16000:  # < 16GB
                    self.checkpoint_segments = max(2, total_layers // 3)
                else:  # >= 16GB
                    self.checkpoint_segments = max(2, total_layers // 4)

            print(f"Auto-optimized checkpoint segments: {self.checkpoint_segments} ({device.upper()}: {total_memory_mb:.0f} MB)")

        except Exception as e:
            print(f"Error in checkpointing optimization ({device.upper()}): {e}, using default segments")
            # Set a sensible default if optimization fails
            self.checkpoint_segments = max(2, (self.n_encoder_layers + self.n_decoder_layers) // 3)

    def get_memory_efficient_batch_size(self) -> int:
        """Suggest memory-efficient batch size based on GPU memory and checkpointing"""
        if not torch.cuda.is_available():
            return self.batch_size

        total_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

        # Base batch size recommendations
        if total_memory_mb < 6000:  # < 6GB (e.g., GTX 1060)
            base_batch_size = 4
        elif total_memory_mb < 8000:  # < 8GB (e.g., GTX 1070)
            base_batch_size = 8
        elif total_memory_mb < 12000:  # < 12GB (e.g., GTX 1080 Ti)
            base_batch_size = 12
        elif total_memory_mb < 16000:  # < 16GB (e.g., RTX 3070)
            base_batch_size = 16
        elif total_memory_mb < 24000:  # < 24GB (e.g., RTX 3090)
            base_batch_size = 24
        else:  # >= 24GB (e.g., RTX 4090, A100)
            base_batch_size = 32

        # Adjust based on gradient checkpointing
        if self.gradient_checkpointing:
            # Can use larger batch size with checkpointing
            multiplier = 1.2 + (self.checkpoint_segments - 1) * 0.1
            suggested_batch_size = int(base_batch_size * multiplier)
        else:
            suggested_batch_size = base_batch_size

        # Don't exceed original batch size by too much
        max_increase = int(self.batch_size * 1.5)
        suggested_batch_size = min(suggested_batch_size, max_increase)

        if suggested_batch_size != self.batch_size:
            print(f"Suggested batch size for memory efficiency: {suggested_batch_size} (current: {self.batch_size})")

        return suggested_batch_size

    def enable_memory_optimization(self):
        """Enable all memory optimization features"""
        self.gradient_checkpointing = True
        self.auto_optimize_checkpointing = True
        self.pin_memory = True if self.device == 'cuda' else False

        # Adjust batch size if beneficial
        optimized_batch_size = self.get_memory_efficient_batch_size()
        if optimized_batch_size > self.batch_size:
            print(f"Increasing batch size from {self.batch_size} to {optimized_batch_size} for better memory efficiency")
            self.batch_size = optimized_batch_size

        print("Memory optimization enabled:")
        print(f"  - Gradient checkpointing: {self.gradient_checkpointing}")
        print(f"  - Checkpoint segments: {self.checkpoint_segments}")
        print(f"  - Pin memory: {self.pin_memory}")
        print(f"  - Batch size: {self.batch_size}")

    def get_config_summary(self) -> str:
        """Get a formatted summary of the configuration"""
        summary = []
        summary.append("=== Training Configuration Summary ===")
        summary.append(f"Model: Hidden dim {self.hidden_dim}, {self.n_encoder_layers}+{self.n_decoder_layers} layers")
        summary.append(f"Training: {self.num_epochs} epochs, batch size {self.batch_size}, LR {self.learning_rate}")
        summary.append(f"Device: {self.device}")

        if torch.cuda.is_available():
            memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            summary.append(f"GPU Memory: {memory_mb:.0f} MB")

        summary.append(f"Gradient Checkpointing: {'Enabled' if self.gradient_checkpointing else 'Disabled'}")
        if self.gradient_checkpointing:
            summary.append(f"  - Segments: {self.checkpoint_segments}")
            estimated_savings = (self.checkpoint_segments - 1) / self.checkpoint_segments * 100
            summary.append(f"  - Est. memory savings: {estimated_savings:.1f}%")

        summary.append(f"Profiling: {'Enabled' if self.enable_profiling else 'Disabled'}")
        summary.append("=" * 38)

        return "\n".join(summary)

    @classmethod
    def create_memory_optimized_config(cls, **kwargs):
        """Create a configuration optimized for memory usage"""
        config = cls(**kwargs)
        config.enable_memory_optimization()
        return config

    @classmethod
    def create_speed_optimized_config(cls, **kwargs):
        """Create a configuration optimized for training speed"""
        config = cls(**kwargs)
        config.gradient_checkpointing = False  # Disable for speed
        config.checkpoint_segments = 1
        config.num_workers = 4  # More workers for faster data loading
        config.pin_memory = True if config.device == 'cuda' else False

        print("Speed optimization enabled:")
        print("  - Gradient checkpointing: Disabled")
        print("  - Increased data loading workers")
        print("  - Pin memory enabled")

        return config


# Example usage and factory functions
def get_default_config() -> TrainingConfig:
    """Get default configuration with gradient checkpointing enabled"""
    return TrainingConfig()


def get_low_memory_config() -> TrainingConfig:
    """Get configuration optimized for low memory GPUs (< 8GB)"""
    return TrainingConfig.create_memory_optimized_config(
        batch_size=4,
        hidden_dim=384,
        checkpoint_segments=4,
        n_encoder_layers=4,
        n_decoder_layers=4
    )


def get_high_performance_config() -> TrainingConfig:
    """Get configuration optimized for high-end GPUs (>= 16GB)"""
    return TrainingConfig.create_memory_optimized_config(
        batch_size=32,
        hidden_dim=768,
        checkpoint_segments=2,
        n_encoder_layers=8,
        n_decoder_layers=8,
        encoder_ff_dim=3072,
        decoder_ff_dim=3072
    )


def get_speed_config() -> TrainingConfig:
    """Get configuration optimized for training speed"""
    return TrainingConfig.create_speed_optimized_config(
        batch_size=24,
        num_workers=4
    )


# Example usage
if __name__ == "__main__":
    # Test different configurations
    print("=== Default Config ===")
    default_config = get_default_config()
    print(default_config.get_config_summary())

    print("\n=== Low Memory Config ===")
    low_mem_config = get_low_memory_config()
    print(low_mem_config.get_config_summary())

    print("\n=== High Performance Config ===")
    high_perf_config = get_high_performance_config()
    print(high_perf_config.get_config_summary())

    print("\n=== Speed Optimized Config ===")
    speed_config = get_speed_config()
    print(speed_config.get_config_summary())