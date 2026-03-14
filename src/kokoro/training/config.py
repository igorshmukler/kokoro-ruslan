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
    num_epochs: int = 26  # 26 × ~339 steps/epoch − 800 warmup ≈ 8,000 OneCycleLR steps
    batch_size: int = 16
    learning_rate: float = 9.09e-5  # peak LR = learning_rate × max_lr_multiplier = 9.09e-5 × 1.1 ≈ 1.0e-4
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Gradient accumulation for larger effective batch sizes
    gradient_accumulation_steps: int = 4  # Effective batch size = batch_size * gradient_accumulation_steps

    # Learning rate scheduler (OneCycleLR)
    use_onecycle_lr: bool = True  # Use OneCycleLR instead of CosineAnnealingWarmRestarts
    max_lr_multiplier: float = 1.1   # Peak decoder LR = learning_rate × this = 9.09e-5 × 1.1 ≈ 1.0e-4 (hard ceiling)
    pct_start: float = 0.06  # Fraction of OneCycleLR cycle spent ascending to peak; with 8000 steps → peak at step ~480
    # Per-group LR multiplier for encoder params (text_embedding, positional_encoding,
    # transformer_encoder_layers). Encoder receives encoder_lr_multiplier × base LR so
    # that the encoder layers get proportionally more gradient signal vs the decoder.
    # peak encoder = 9.09e-5 × 1.1 (max_lr) × 1.1 ≈ 1.1e-4
    encoder_lr_multiplier: float = 1.1
    # LR multiplier for the stop-token head's dedicated optimizer param group.
    # The stop head is a single Linear(hidden_dim→1) with a heavily skewed target
    # distribution (~137:1 neg/pos).  Running it at the same LR as the decoder
    # lets OneCycleLR peak-phase spikes destabilize it.  0.2× should keep the
    # effective step size well below the adaptive clip threshold
    # while still giving the head meaningful updates every step.
    stop_head_lr_multiplier: float = 0.2

    # Linear warmup before OneCycleLR
    use_warmup: bool = True  # Enable linear warmup before OneCycleLR
    warmup_steps: int = 800  # Number of optimizer steps for linear warmup (not batches!)
    warmup_start_lr_ratio: float = 0.01  # Start LR = learning_rate * this value

    # EMA (Exponential Moving Average) of model weights
    use_ema: bool = True  # Enable EMA for better inference quality
    ema_decay: Optional[float] = None  # EMA decay rate; if None, trainer will compute a recommended value
    ema_half_life_epochs: float = 1.0  # Half-life in epochs used to compute recommended EMA when ema_decay is None
    ema_update_every: int = 1  # Update EMA every N optimizer steps (1 = every step)

    # Legacy CosineAnnealingWarmRestarts settings (used if use_onecycle_lr=False)
    lr_T_0: int = 20
    lr_T_mult: int = 2
    lr_eta_min: float = 1e-6

    # Model parameters
    n_mels: int = 80
    hidden_dim: int = 768
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    n_heads: int = 8
    encoder_ff_dim: int = 3072
    decoder_ff_dim: int = 3072
    encoder_dropout: float = 0.1
    max_decoder_seq_len: int = 4000

    # Stochastic depth (layer dropout) for regularization
    use_stochastic_depth: bool = True  # Enable layer dropout during training
    stochastic_depth_rate: float = 0.05  # Maximum drop probability for last layer
    # Drop probability increases linearly from 0 (first layer) to stochastic_depth_rate (last layer)

    # Loss weights
    duration_loss_weight: float = 0.35
    # Global scaling of the stop-token loss in the total loss sum.
    # This parameter controls only how much stop learning contributes to the
    # total gradient — it is intentionally decoupled from stop_token_pos_weight.
    # Rule of thumb: stop should contribute ~1–3% of the mel gradient budget.
    # Set independently of pos_weight; do NOT re-derive from pos_weight.
    stop_token_loss_weight: float = 0.010
    pitch_loss_weight: float = 1.0  # Normalized to [0,1]; 1.0 gives pitch predictor adequate gradient signal vs mel loss
    energy_loss_weight: float = 1.0  # Normalized to [0,1]; matched to pitch_loss_weight

    # SpecAugment (Park et al. 2019) — applied to teacher-forced mel decoder input only.
    # The unmasked original mel is still used as the loss target, so gradients for
    # unmasked frames are unaffected.  Masking forces the decoder to rely on encoder
    # context rather than memorising the previous mel frame.
    use_spec_augment: bool = True
    spec_augment_time_mask_max: int = 30   # Max consecutive frames masked per mask
    spec_augment_freq_mask_max: int = 10   # Max consecutive mel bins masked per mask
    spec_augment_num_time_masks: int = 2   # Number of independent time masks per batch
    spec_augment_num_freq_masks: int = 2   # Number of independent frequency masks per batch
    # Epoch gate: SpecAugment is too noisy while the LR is still ramping.
    # With pct_start=0.06 and 8014 OneCycleLR steps (~339 steps/epoch) the LR
    # peaks at OneCycleLR step ~480, i.e. absolute epoch ~4 (800 warmup + 480
    # onecycle = 1280 steps / 339 ≈ epoch 3.8).  Start 2 epochs after peak so
    # the schedule is solidly descending before augmentation noise is introduced.
    spec_augment_start_epoch: int = 6
    # Class-imbalance correction for stop-token BCE.
    # Sole purpose: re-weight positive (stop) frames vs negative (non-stop) frames
    # so the model cannot collapse to always-predict-no-stop.
    # Set this to approximate the actual neg/pos frame ratio in the corpus
    # (500-sample RUSLAN: mean mel length ≈ 138 frames → ratio ≈ 137:1).
    # This parameter is intentionally decoupled from stop_token_loss_weight:
    #   • stop_token_pos_weight  → fixes class balance (sampling concern)
    #   • stop_token_loss_weight → controls global loss contribution (scale concern)
    # Do NOT scale pos_weight up/down to compensate for the global weight; adjust
    # stop_token_loss_weight independently for that purpose.
    # History: 150 → spikes at high LR; 100 → grad_norm to 39.8; 50 → 35 chosen as
    # stable partial correction (~25%) that kept the stop predictor learning.
    # Reduced to 25.0: stop head bias drifted steadily negative (−0.006 → −0.016
    # across ep1-5) at 35.0 even with pos_weight, indicating the head is still
    # treating most frames as non-stop.  With gradient isolation via detach now in
    # place, 25.0 gives sufficient class-balance correction without over-amplifying
    # the already-isolated stop gradient at the peak LR phase.
    stop_token_pos_weight: float = 25.0
    # Temporal smoothing of stop-token targets.
    # Instead of a single hard 1.0 at the last frame, a short exponentially
    # decaying tail is added to the frames immediately before it:
    #   frame[T-1]          = 1.0          (the actual stop boundary)
    #   frame[T-1-k]        = decay^k       for k = 1 … stop_token_smooth_tail
    # This spreads the positive gradient over several frames, eliminating the
    # single-frame spike that can cause grad-norm bursts when pos_weight is large.
    # Set stop_token_smooth_tail=0 to disable (recover the original hard target).
    stop_token_smooth_tail: int = 6
    stop_token_smooth_decay: float = 0.5

    # Variance predictor settings
    use_variance_predictor: bool = True  # Enabled with normalized [0,1] inputs and auto-reset
    variance_filter_size: int = 256
    variance_kernel_size: int = 3
    variance_dropout: float = 0.1
    n_variance_bins: int = 256
    # Pitch extraction range in Hz (used to extract F0 targets before normalization)
    pitch_extract_fmin: float = 50.0
    pitch_extract_fmax: float = 800.0
    pitch_min: float = 0.0  # Normalized range (extractors output [0, 1])
    pitch_max: float = 1.0  # Normalized range (extractors output [0, 1])
    energy_min: float = 0.0
    energy_max: float = 1.0  # Normalized range (extractors output [0, 1])

    # Audio processing
    max_seq_length: int = 1800
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    f_min: float = 0.0
    f_max: float = 8000.0

    # Data loading
    num_workers: int = 0
    pin_memory: bool = False

    # Feature caching for faster training
    use_feature_cache: bool = True  # Cache mel spectrograms, pitch, energy to disk
    feature_cache_dir: str = ""  # Will be set to {data_dir}/.feature_cache if empty
    precompute_features: bool = False  # Precompute all features before training starts
    # In-memory feature cache (separate from on-disk feature cache)
    use_memory_cache: bool = True  # Keep features in RAM for faster access; disable to reduce GPU/host memory

    # Dynamic batching (batch by total frames instead of fixed size)
    use_dynamic_batching: bool = True  # Enable frame-based batching
    max_frames_per_batch: int = 15000  # Maximum mel frames per batch (auto-adjusted for MPS)
    min_batch_size: int = 4  # Minimum samples per batch
    max_batch_size: int = 8  # Maximum samples per batch

    # Gradient clipping
    # This is the base ceiling for the adaptive gradient clip norm used during the
    # normal (non-outlier) training step.  The adaptive stabilizer may lower it
    # further for batches with extreme mel lengths or durations.
    max_grad_norm: float = 1.5 # Global gradient clip ceiling

    # Gradient stability safeguards
    projection_spike_clip_norm: float = 20.0
    attention_spike_clip_norm: float = 8.0
    # Per-layer clip norm for decoder FFN linear1/linear2 (consistent regression driver)
    ffn_spike_clip_norm: float = 5.0
    # Encoder FFN per-layer pre-clip. Previously 10.0 (too tight, zeroed encoder grads);
    # raised to 100.0 while encoder was passive (ep1-ep5 grads ~1e-7).
    # By ep6 encoder FFN avg_grad reached 40-100 per tensor (active learning phase),
    # so 100.0 provided no real protection and encoder attention was also fully uncapped.
    # Set to 12.0: ~20% above the decoder FFN cap (8.0), proportional to the higher
    # encoder LR multiplier (1.3×). Encoder attention is now also clipped at
    # attention_spike_clip_norm (20.0) via the extended pre-clip in trainer.
    encoder_ffn_spike_clip_norm: float = 8.0
    # Per-parameter clip norm applied exclusively to the stop-token head
    # (stop_token_predictor.weight and .bias) before the global clip.
    # The stop head is a single Linear(hidden_dim→1); its gradient can spike disproportionately
    # when pos_weight is large or the smoothing tail has a frame error near the boundary.
    # A tight ceiling here prevents the stop head from corrupting the mel decoder's gradient
    # budget while still giving the head a meaningful update signal.
    # Set ≤ 0 to disable.  With gradient isolation now provided by detaching
    # decoder_outputs before the stop head (model.py _project_decoder_outputs),
    # this clip defends only the stop head's own weight/bias against LR-phase
    # spikes.  Reduced to 0.5: the stop head weight norm at ep5 is ~1.6 with
    # a 1.0 clip already present; halving the ceiling throttles the peak-LR
    # update magnitude proportionally without starving the head.
    stop_head_spike_clip_norm: float = 0.5
    # Post-step max weight-norm clamp for decoder.layers.0.ff.linear1.weight.
    # After every successful optimizer step the L2 norm of each decoder FF weight
    # matrix is projected back to this ceiling, preventing unconstrained growth
    # across all decoder FFN layers while leaving gradients untouched.
    # Covers all decoder.layers.{i}.ff.linear1 and linear2 weights (12 matrices
    # for a 6-layer decoder).  Set ≤ 0.0 to disable.
    # dec_ff0_linear1_max_weight_norm is the legacy single-layer key and is kept
    # for backward compat; dec_ffn_max_weight_norm takes precedence when present.
    dec_ffn_max_weight_norm: float = 28.0
    dec_ff0_linear1_max_weight_norm: float = 60.0  # legacy — superseded by dec_ffn_max_weight_norm
    grad_explosion_warmup_steps: int = 400
    grad_explosion_warmup_floor: float = 8000.0
    grad_explosion_min_ema_steps: int = 100

    # Checkpointing
    save_every: int = 5  # Save every X epochs (even if it not the best result); best model is always saved separately
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
    auto_optimize_checkpointing: bool = False

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

    # Disabled for MPS, for now, due to multiple issues, still unresolved
    use_mixed_precision: bool = False

    # Optimizer behavior
    # AdamW regularization and numerical stability
    weight_decay: float = 0.01   # L2 penalty applied to decoder/rest param group; encoder group always uses 0.0
    adam_eps: float = 1e-8       # AdamW epsilon for numerical stability
    adam_betas: tuple = (0.9, 0.999)  # AdamW beta coefficients (momentum, RMS)
    # None = auto (enabled on CUDA, disabled otherwise)
    use_fused_adamw: Optional[bool] = None
    # Try fused AdamW on MPS by default (may fall back)
    try_fused_adamw_on_mps: bool = True

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

        MPS_MAX_FRAMES_PER_BATCH = 12000
        MPS_MAX_BATCH_SIZE = 16

        # MPS-specific memory optimizations
        if self.device == 'mps' or (torch.backends.mps.is_available() and self.device != 'cuda'):
            # Balanced settings to avoid MPS dimension overflow while maximizing throughput
            if self.max_frames_per_batch > MPS_MAX_FRAMES_PER_BATCH:
                print(f"MPS detected: Reducing max_frames_per_batch from {self.max_frames_per_batch} to {MPS_MAX_FRAMES_PER_BATCH}")
                self.max_frames_per_batch = MPS_MAX_FRAMES_PER_BATCH

            if self.max_seq_length > 1800:
                print(f"MPS detected: Reducing max_seq_length from {self.max_seq_length} to 1800")
                self.max_seq_length = 1800

            if self.batch_size > 10:
                print(f"MPS detected: Reducing batch_size from {self.batch_size} to 10")
                self.batch_size = 10

            if self.max_batch_size > MPS_MAX_BATCH_SIZE:
                print(f"MPS detected: Reducing max_batch_size from {self.max_batch_size} to {MPS_MAX_BATCH_SIZE}")
                self.max_batch_size = MPS_MAX_BATCH_SIZE

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