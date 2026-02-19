#!/usr/bin/env python3
"""
Variance Predictor Module for Pitch and Energy Prediction
Based on FastSpeech 2 architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Default architecture constants
DEFAULT_HIDDEN_DIM = 512
DEFAULT_FILTER_SIZE = 256
DEFAULT_N_BINS = 256
DEFAULT_HOP_LENGTH = 256


class VariancePredictor(nn.Module):
    """
    Variance predictor for pitch/energy prediction
    Uses 1D convolutions with layer normalization
    """

    def __init__(self,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 filter_size: int = DEFAULT_FILTER_SIZE,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 num_layers: int = 2):
        """
        Initialize variance predictor

        Args:
            hidden_dim: Hidden dimension from encoder
            filter_size: Filter size for conv layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            num_layers: Number of conv layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.filter_size = filter_size

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else filter_size

            conv = nn.Conv1d(
                in_channels,
                filter_size,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2
            )
            self.conv_layers.append(conv)

            # Layer normalization
            self.layer_norms.append(nn.LayerNorm(filter_size))

        # Output projection
        self.linear = nn.Linear(filter_size, 1)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for conv in self.conv_layers:
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            mask: Padding mask (batch, seq_len) - True for padding

        Returns:
            Predicted variance (batch, seq_len)
        """
        # Transpose for Conv1d: (batch, hidden_dim, seq_len)
        # Ensure contiguous for torch.compile compatibility
        x = x.transpose(1, 2).contiguous()

        # Apply conv layers
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            # Conv1d - ensure input is contiguous
            x = conv(x)

            # Transpose for layer norm: (batch, seq_len, filter_size)
            x = x.transpose(1, 2).contiguous()

            # Layer norm
            x = norm(x)

            # Activation and dropout
            x = self.activation(x)
            x = self.dropout(x)

            # Transpose back for next conv: (batch, filter_size, seq_len)
            # Ensure contiguous before next convolution
            x = x.transpose(1, 2).contiguous()

        # Final transpose for linear layer
        x = x.transpose(1, 2).contiguous()  # (batch, seq_len, filter_size)

        # Project to single value per timestep
        output = self.linear(x).squeeze(-1)  # (batch, seq_len)

        # Apply mask if provided
        if mask is not None:
            output = output.masked_fill(mask, 0.0)

        return output


class VarianceAdaptor(nn.Module):
    """
    Variance Adaptor combining duration, pitch, and energy predictors
    Based on FastSpeech 2
    """

    def __init__(self,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 filter_size: int = DEFAULT_FILTER_SIZE,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 n_bins: int = DEFAULT_N_BINS,
                 pitch_min: float = 50.0,
                 pitch_max: float = 800.0,
                 energy_min: float = 0.0,
                 energy_max: float = 100.0):
        """
        Initialize variance adaptor

        Args:
            hidden_dim: Hidden dimension
            filter_size: Filter size for variance predictors
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            n_bins: Number of bins for pitch/energy quantization
            pitch_min: Minimum pitch value (Hz)
            pitch_max: Maximum pitch value (Hz)
            energy_min: Minimum energy value
            energy_max: Maximum energy value
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_bins = n_bins
        # Store pitch and energy ranges for normalization helpers
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.energy_min = energy_min
        self.energy_max = energy_max

        # Duration predictor (already exists in model, but we can add here for completeness)
        self.duration_predictor = VariancePredictor(
            hidden_dim, filter_size, kernel_size, dropout, num_layers=2
        )

        # Pitch predictor
        self.pitch_predictor = VariancePredictor(
            hidden_dim, filter_size, kernel_size, dropout, num_layers=5
        )

        # Energy predictor
        self.energy_predictor = VariancePredictor(
            hidden_dim, filter_size, kernel_size, dropout, num_layers=2
        )

        # Pitch quantization bins (normalized 0..1)
        # The pitch extractor and any inputs to quantize_pitch must provide
        # pitch values normalized into the [0, 1] range.
        self.register_buffer(
            'pitch_bins',
            torch.linspace(0.0, 1.0, n_bins - 1)
        )

        # Energy quantization bins (normalized 0..1)
        # Energy inputs should be normalized to [0,1] before quantization.
        self.register_buffer(
            'energy_bins',
            torch.linspace(0.0, 1.0, n_bins - 1)
        )

        # Pitch embedding
        self.pitch_embedding = nn.Embedding(n_bins, hidden_dim)

        # Energy embedding
        self.energy_embedding = nn.Embedding(n_bins, hidden_dim)

        logger.info(f"VarianceAdaptor initialized with {n_bins} bins for pitch/energy")

    def quantize_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        """
        Quantize normalized pitch values to bin indices.

        Notes:
        - Expects `pitch` to be normalized to the [0, 1] range (matching
          `self.pitch_bins`). If your extractor returns Hz, convert to
          normalized form before calling this method.

        Args:
            pitch: Normalized pitch values in [0, 1] (batch, seq_len)

        Returns:
            Quantized pitch indices (batch, seq_len)
        """
        # Use torch.bucketize for efficient quantization
        pitch_quantized = torch.bucketize(pitch, self.pitch_bins)
        return pitch_quantized.long()

    def _hz_to_normalized(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Convert Hz-valued F0 to normalized [0, 1] using adaptor pitch_min/pitch_max.

        Args:
            f0: Tensor of frequencies in Hz

        Returns:
            Tensor of normalized values in [0,1]
        """
        return torch.clamp((f0 - self.pitch_min) / (self.pitch_max - self.pitch_min + 1e-8), 0.0, 1.0)

    def _maybe_normalize_pitch(self, pitch: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Heuristic: if `pitch` appears to be in Hz (values outside [0,1]),
        convert to normalized [0,1]. Otherwise, assume already normalized.
        """
        if pitch is None:
            return pitch

        # Work on a tensor copy to avoid in-place surprises
        if not torch.is_tensor(pitch):
            pitch = torch.tensor(pitch)

        if device is not None:
            pitch = pitch.to(device)

        # If any value outside [0,1], treat as Hz and convert
        if pitch.numel() > 0 and (torch.max(pitch) > 1.0 or torch.min(pitch) < 0.0):
            return self._hz_to_normalized(pitch)

        return pitch

    def quantize_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous energy values to bins

        Args:
            energy: Continuous energy values (batch, seq_len)

        Returns:
            Quantized energy indices (batch, seq_len)
        """
        # Use torch.bucketize for efficient quantization
        energy_quantized = torch.bucketize(energy, self.energy_bins)
        return energy_quantized.long()

    def _energy_to_normalized(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Convert energy to normalized [0, 1] using adaptor energy_min/energy_max.
        """
        return torch.clamp((energy - self.energy_min) / (self.energy_max - self.energy_min + 1e-8), 0.0, 1.0)

    def _maybe_normalize_energy(self, energy: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Heuristic: if `energy` appears to be outside [0,1], convert to normalized.
        """
        if energy is None:
            return energy

        if not torch.is_tensor(energy):
            energy = torch.tensor(energy)

        if device is not None:
            energy = energy.to(device)

        if energy.numel() > 0 and (torch.max(energy) > 1.0 or torch.min(energy) < 0.0):
            return self._energy_to_normalized(energy)

        return energy

    def forward(self,
                encoder_output: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pitch_target: Optional[torch.Tensor] = None,
                energy_target: Optional[torch.Tensor] = None,
                duration_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through variance adaptor

        Args:
            encoder_output: Encoder output (batch, seq_len, hidden_dim)
            mask: Padding mask (batch, seq_len) - True for padding
            pitch_target: Target pitch values for training (batch, seq_len)
            energy_target: Target energy values for training (batch, seq_len)
            duration_target: Target duration values for training (batch, seq_len)

        Returns:
            Tuple of (adapted_output, duration_pred, pitch_pred, energy_pred)
        """
        # Predict durations
        duration_pred = self.duration_predictor(encoder_output, mask)

        # Predict pitch
        pitch_pred = self.pitch_predictor(encoder_output, mask)

        # Predict energy
        energy_pred = self.energy_predictor(encoder_output, mask)

        # Normalize targets/predictions to [0,1] before quantization.
        # Inference naturally falls back to predictions when targets are None.
        pitch_to_quantize = pitch_target if pitch_target is not None else pitch_pred
        pitch_to_quantize = self._maybe_normalize_pitch(pitch_to_quantize, device=encoder_output.device)
        pitch_quantized = self.quantize_pitch(pitch_to_quantize)

        energy_to_quantize = energy_target if energy_target is not None else energy_pred
        energy_to_quantize = self._maybe_normalize_energy(energy_to_quantize, device=encoder_output.device)
        energy_quantized = self.quantize_energy(energy_to_quantize)

        # Get embeddings
        pitch_embed = self.pitch_embedding(pitch_quantized)
        energy_embed = self.energy_embedding(energy_quantized)

        # Zero out embeddings at padded positions to avoid leakage
        if mask is not None:
            mask_unsq = mask.unsqueeze(-1)
            pitch_embed = pitch_embed.masked_fill(mask_unsq, 0.0)
            energy_embed = energy_embed.masked_fill(mask_unsq, 0.0)

        # Add variance embeddings to encoder output
        adapted_output = encoder_output + pitch_embed + energy_embed

        return adapted_output, duration_pred, pitch_pred, energy_pred


class PitchExtractor:
    """
    Extract pitch (F0) from audio waveform
    Uses PyTorch-compatible implementation
    """

    @staticmethod
    def extract_pitch(waveform: torch.Tensor,
                     sample_rate: int = 22050,
                     hop_length: int = DEFAULT_HOP_LENGTH,
                     fmin: float = 50.0,
                     fmax: float = 800.0,
                     win_length: Optional[int] = None) -> torch.Tensor:
        """
        Extract pitch (F0) from waveform using YIN-style CMND algorithm.

        Improvements over previous version:
        - Cumulative Mean Normalized Difference (CMND) reduces octave errors
        - Parabolic interpolation for sub-sample lag accuracy
        - Adaptive per-utterance voicing threshold
        - Median filtering to remove frame-to-frame jitter
        - Linear interpolation across short unvoiced gaps
        - win_length parameter is now respected

        Args:
            waveform: Audio waveform (batch, samples) or (samples,)
            sample_rate: Sample rate
            hop_length: Hop length for framing
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            win_length: Analysis window length (defaults to max(2048, hop*8))

        Returns:
            Normalized pitch contour in [0, 1] — (batch, frames) or (frames,)
            Unvoiced frames are 0.0.
        """
        try:
            # --- Setup ---
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False

            device = waveform.device
            dtype = waveform.dtype
            hop = int(hop_length)

            # Respect caller's win_length, fall back to adaptive default
            if win_length is not None:
                win_len = int(win_length)
            else:
                win_len = max(2048, hop * 8)

            # Pad signal to at least one full window
            if waveform.size(1) < win_len:
                waveform = F.pad(waveform, (0, win_len - waveform.size(1)))

            # --- Pre-emphasis ---
            pre_emphasis = 0.97
            waveform = torch.cat([
                waveform[..., :1],
                waveform[..., 1:] - pre_emphasis * waveform[..., :-1]
            ], dim=-1)

            # --- Framing & Windowing ---
            frames = waveform.unfold(1, win_len, hop).contiguous()
            batch_size, n_frames, _ = frames.shape

            window = torch.hann_window(win_len, device=device, dtype=dtype)
            frames = frames * window.unsqueeze(0).unsqueeze(0)

            # --- Autocorrelation via Wiener-Khinchin ---
            nfft = win_len * 2
            spec = torch.fft.rfft(frames, n=nfft)
            acf = torch.fft.irfft(spec.abs() ** 2, n=nfft)[..., :win_len]

            # --- CMND (Cumulative Mean Normalized Difference) ---
            # Difference function: d(tau) = acf[0] + acf[0] - 2*acf[tau]
            zero_lag = acf[..., 0:1]  # (B, frames, 1)
            diff = 2 * zero_lag - 2 * acf  # (B, frames, win_len)

            # CMND: d'(tau) = d(tau) / [(1/tau) * sum_{j=1}^{tau} d(j)]
            # d'[0] = 1 by convention
            cmnd = torch.zeros_like(diff)
            cmnd[..., 0] = 1.0
            cumsum = torch.cumsum(diff[..., 1:], dim=-1)  # (B, frames, win_len-1)
            tau_range = torch.arange(1, win_len, device=device, dtype=dtype)
            cmnd[..., 1:] = diff[..., 1:] / (cumsum / tau_range + 1e-8)

            # --- Lag Search Range ---
            lag_min = max(2, int(sample_rate / fmax))
            lag_max = min(win_len - 2, max(lag_min + 1, int(sample_rate / fmin)))
            # -2 upper bound to leave room for parabolic interpolation

            lags = torch.arange(lag_min, lag_max + 1, device=device)
            n_lags = len(lags)

            # CMND values in search range: (B, frames, n_lags)
            # Lower CMND = better pitch candidate (unlike ACF where higher = better)
            cmnd_lags = cmnd[..., lag_min:lag_max + 1]

            # Also gather ACF in search range for voicing confidence
            acf_norm = acf / zero_lag.clamp(min=1e-8)
            ac_lags = acf_norm[..., lag_min:lag_max + 1]
            ac_max_vals, _ = ac_lags.max(dim=-1)  # (B, frames)

            # Best lag = first dip below threshold in CMND, else global minimum
            # We use argmin on CMND (equivalent to YIN threshold search)
            best_idx = torch.argmin(cmnd_lags, dim=-1)  # (B, frames)

            # --- Parabolic Interpolation for Sub-Sample Accuracy ---
            # Clamp neighbors to valid range
            idx_prev = (best_idx - 1).clamp(min=0)
            idx_next = (best_idx + 1).clamp(max=n_lags - 1)

            alpha = cmnd_lags.gather(-1, idx_prev.unsqueeze(-1)).squeeze(-1)
            beta  = cmnd_lags.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
            gamma = cmnd_lags.gather(-1, idx_next.unsqueeze(-1)).squeeze(-1)

            denom = (alpha - 2 * beta + gamma).clamp(min=1e-8)

            offset = 0.5 * (alpha - gamma) / denom
            offset = offset.clamp(-1.0, 1.0)  # Sanity clamp

            # Map best_idx to actual lag, apply sub-sample offset
            best_lags = (lags[best_idx].float() + offset).clamp(min=1.0)
            freqs = sample_rate / best_lags  # (B, frames)

            # --- Adaptive Voicing Threshold ---
            # Per-utterance: base threshold on 25th percentile of ACF peaks
            # This adapts to recording conditions rather than using a fixed value
            ac_25th = torch.quantile(ac_max_vals, 0.25, dim=-1, keepdim=True)
            voicing_thresh = torch.clamp(ac_25th * 0.8, min=0.15, max=0.35)

            frame_energy = frames.pow(2).mean(dim=-1)
            energy_thresh = torch.clamp(torch.median(frame_energy, dim=-1, keepdim=True).values * 0.05, min=1e-9)

            unvoiced_mask = (ac_max_vals < voicing_thresh) | (frame_energy < energy_thresh)
            freqs = freqs.masked_fill(unvoiced_mask, 0.0)

            # Clip to valid frequency range before normalization
            freqs = torch.where(
                (freqs < fmin) | (freqs > fmax),
                torch.zeros_like(freqs),
                freqs
            )

            # --- Interpolate Short Unvoiced Gaps ---
            # Fill gaps of up to MAX_GAP_FRAMES with linear interpolation
            # This smooths over consonants mid-word without filling long silences
            MAX_GAP_FRAMES = 5
            freqs_np = freqs.cpu().numpy()

            for b in range(batch_size):
                f = freqs_np[b]
                voiced = f > 0
                if not voiced.any() or voiced.all():
                    continue

                indices = np.arange(len(f), dtype=np.float32)
                voiced_idx = indices[voiced]
                voiced_vals = f[voiced]

                # Interpolate across the full contour
                f_interp = np.interp(indices, voiced_idx, voiced_vals)

                # Only fill gaps shorter than MAX_GAP_FRAMES; leave long silences as 0
                in_gap = False
                gap_start = 0
                for i in range(len(f)):
                    if not voiced[i]:
                        if not in_gap:
                            in_gap = True
                            gap_start = i
                    else:
                        if in_gap:
                            gap_len = i - gap_start
                            if gap_len <= MAX_GAP_FRAMES:
                                f[gap_start:i] = f_interp[gap_start:i]
                            in_gap = False
                # Don't fill trailing silence

                freqs_np[b] = f

            freqs = torch.from_numpy(freqs_np).to(device=device, dtype=dtype)

            # --- Median Filter (removes jitter) ---
            MEDIAN_K = 5
            pad = MEDIAN_K // 2
            freqs_padded = F.pad(freqs, (pad, pad), mode='reflect')
            freqs = freqs_padded.unfold(-1, MEDIAN_K, 1).median(dim=-1).values

            # --- Normalize to [0, 1] ---
            freqs_norm = torch.clamp((freqs - fmin) / (fmax - fmin + 1e-8), min=0.0, max=1.0)
            # Preserve zero (unvoiced) frames — normalization may shift them slightly
            freqs_norm = freqs_norm.masked_fill(freqs == 0.0, 0.0)

            return freqs_norm.squeeze(0) if squeeze_output else freqs_norm

        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}, using zeros")
            n_frames = max(1, waveform.shape[-1] // hop_length)
            if squeeze_output:
                return torch.zeros(n_frames, device=waveform.device)
            return torch.zeros(waveform.shape[0], n_frames, device=waveform.device)

class EnergyExtractor:
    """
    Extract energy (RMS) from mel spectrogram or waveform
    """

    @staticmethod
    def extract_energy_from_mel(mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Extract energy from mel spectrogram, normalized to [0, 1]

        Args:
            mel_spec: Mel spectrogram (batch, n_mels, frames) or (n_mels, frames)

        Returns:
            Energy contour normalized to [0, 1] (batch, frames) or (frames,)
        """
        # Convert to linear domain if input appears to be log-mel
        if mel_spec.min() < 0:
            mel_linear = torch.exp(mel_spec)
        else:
            mel_linear = mel_spec

        # Energy is frame-level mean across mel bins (works for batched or unbatched inputs)
        energy = torch.mean(mel_linear, dim=-2)

        # Dynamic range compression and robust per-utterance normalization
        energy = torch.log1p(torch.clamp(energy, min=0.0))

        # Compute robust per-utterance floor/ceil along the frames axis.
        # Use keepdim=True so broadcasting works for both (frames,) and (batch, frames).
        floor = torch.quantile(energy, 0.05, dim=-1, keepdim=True)
        ceil = torch.quantile(energy, 0.95, dim=-1, keepdim=True)
        denom = torch.clamp(ceil - floor, min=1e-8)
        energy = (energy - floor) / denom

        energy = torch.clamp(energy, 0.0, 1.0)

        return energy

    @staticmethod
    def extract_energy_from_waveform(waveform: torch.Tensor,
                                    hop_length: int = DEFAULT_HOP_LENGTH,
                                    win_length: int = 1024) -> torch.Tensor:
        """
        Extract RMS energy from waveform

        Args:
            waveform: Audio waveform (batch, samples) or (samples,)
            hop_length: Hop length
            win_length: Window length for RMS calculation

        Returns:
            Energy contour (batch, frames) or (frames,)
        """
        # Ensure 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Unfold to create overlapping windows
        frames = waveform.unfold(1, win_length, hop_length)

        # Calculate RMS energy per frame
        energy = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-8)

        if squeeze_output:
            energy = energy.squeeze(0)

        return energy


def normalize_variance(values: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       mean: Optional[float] = None,
                       std: Optional[float] = None) -> Tuple[torch.Tensor, float, float]:
    """
    Normalize variance values (pitch/energy) to zero mean and unit variance

    Args:
        values: Variance values (batch, seq_len)
        mask: Padding mask (batch, seq_len) - True for padding
        mean: Pre-computed mean (if None, compute from values)
        std: Pre-computed std (if None, compute from values)

    Returns:
        Tuple of (normalized_values, mean, std)
    """
    if mask is not None:
        # Compute statistics only on non-masked values
        non_masked_values = values[~mask]
        if mean is None:
            mean = non_masked_values.mean().item()
        if std is None:
            std = non_masked_values.std().item() + 1e-8
    else:
        if mean is None:
            mean = values.mean().item()
        if std is None:
            std = values.std().item() + 1e-8

    # Normalize
    normalized = (values - mean) / std

    # Zero out masked values
    if mask is not None:
        normalized = normalized.masked_fill(mask, 0.0)

    return normalized, mean, std


if __name__ == "__main__":
    """Test variance predictor"""
    logging.basicConfig(level=logging.INFO)

    # Test variance predictor
    predictor = VariancePredictor(hidden_dim=DEFAULT_HIDDEN_DIM, filter_size=DEFAULT_FILTER_SIZE)
    x = torch.randn(4, 100, DEFAULT_HIDDEN_DIM)  # (batch, seq_len, hidden_dim)
    mask = torch.zeros(4, 100).bool()

    output = predictor(x, mask)
    print(f"VariancePredictor output shape: {output.shape}")  # Should be (4, 100)

    # Test variance adaptor
    adaptor = VarianceAdaptor(hidden_dim=DEFAULT_HIDDEN_DIM, n_bins=DEFAULT_N_BINS)
    pitch_target = torch.randn(4, 100) * 100 + 200  # Pitch in Hz
    energy_target = torch.randn(4, 100) * 10 + 50   # Energy
    duration_target = torch.randint(1, 20, (4, 100)).float()

    adapted, dur_pred, pitch_pred, energy_pred = adaptor(
        x, mask, pitch_target, energy_target, duration_target
    )

    print(f"Adapted output shape: {adapted.shape}")
    print(f"Duration prediction shape: {dur_pred.shape}")
    print(f"Pitch prediction shape: {pitch_pred.shape}")
    print(f"Energy prediction shape: {energy_pred.shape}")

    # Test pitch extractor
    waveform = torch.randn(1, 22050)  # 1 second of audio
    pitch = PitchExtractor.extract_pitch(waveform, sample_rate=22050, hop_length=DEFAULT_HOP_LENGTH)
    print(f"Pitch shape: {pitch.shape}")

    # Test energy extractor
    mel_spec = torch.randn(80, 100)  # (n_mels, frames)
    energy = EnergyExtractor.extract_energy_from_mel(mel_spec)
    print(f"Energy shape: {energy.shape}")

    logger.info("All variance predictor tests passed!")
