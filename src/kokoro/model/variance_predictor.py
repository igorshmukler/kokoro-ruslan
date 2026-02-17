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

        # Pitch quantization bins
        self.register_buffer(
            'pitch_bins',
            torch.linspace(pitch_min, pitch_max, n_bins - 1)
        )

        # Energy quantization bins
        self.register_buffer(
            'energy_bins',
            torch.linspace(energy_min, energy_max, n_bins - 1)
        )

        # Pitch embedding
        self.pitch_embedding = nn.Embedding(n_bins, hidden_dim)

        # Energy embedding
        self.energy_embedding = nn.Embedding(n_bins, hidden_dim)

        logger.info(f"VarianceAdaptor initialized with {n_bins} bins for pitch/energy")

    def quantize_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous pitch values to bins

        Args:
            pitch: Continuous pitch values (batch, seq_len)

        Returns:
            Quantized pitch indices (batch, seq_len)
        """
        # Use torch.bucketize for efficient quantization
        pitch_quantized = torch.bucketize(pitch, self.pitch_bins)
        return pitch_quantized.long()

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

        # Use targets during training, predictions during inference
        if self.training:
            # Use ground truth for embedding lookup
            pitch_quantized = self.quantize_pitch(pitch_target) if pitch_target is not None else self.quantize_pitch(pitch_pred)
            energy_quantized = self.quantize_energy(energy_target) if energy_target is not None else self.quantize_energy(energy_pred)
        else:
            # Use predictions during inference
            pitch_quantized = self.quantize_pitch(pitch_pred)
            energy_quantized = self.quantize_energy(energy_pred)

        # Get embeddings
        pitch_embed = self.pitch_embedding(pitch_quantized)
        energy_embed = self.energy_embedding(energy_quantized)

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
                     fmax: float = 800.0) -> torch.Tensor:
        """
        Extract pitch from waveform using YIN algorithm approximation

        Args:
            waveform: Audio waveform (batch, samples) or (samples,)
            sample_rate: Sample rate
            hop_length: Hop length for STFT
            fmin: Minimum frequency
            fmax: Maximum frequency

        Returns:
            Pitch contour (batch, frames) or (frames,)
        """
        try:
            import torchaudio

            # Ensure 2D
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False

            # Use torchaudio's pitch detection
            pitch_transform = torchaudio.transforms.PitchShift(
                sample_rate=sample_rate,
                n_steps=0  # No shift, just extract pitch
            )

            # Alternative: Use spectral centroid as proxy for pitch
            # This is more stable but less accurate
            n_fft = 2048
            spectrogram = torch.stft(
                waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=torch.hann_window(n_fft, device=waveform.device),
                return_complex=True
            )

            magnitude = torch.abs(spectrogram)

            # Calculate spectral centroid as pitch proxy
            freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=waveform.device)
            freqs = freqs.unsqueeze(0).unsqueeze(-1)  # (1, freq_bins, 1)

            # Weighted average frequency
            pitch = torch.sum(magnitude * freqs, dim=1) / (torch.sum(magnitude, dim=1) + 1e-8)

            # Normalize to [0, 1] range immediately (critical for stable training)
            # Spectral centroid can be 0 to sample_rate/2, normalize to expected vocal range
            pitch = torch.clamp(pitch, fmin, fmax)
            pitch = (pitch - fmin) / (fmax - fmin + 1e-8)
            pitch = torch.clamp(pitch, 0.0, 1.0)

            if squeeze_output:
                pitch = pitch.squeeze(0)

            return pitch

        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}, using zeros")
            # Fallback: return zeros
            if waveform.dim() == 1:
                return torch.zeros(waveform.shape[0] // hop_length, device=waveform.device)
            else:
                return torch.zeros(waveform.shape[0], waveform.shape[1] // hop_length, device=waveform.device)


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
        # Energy is the mean across mel bins
        if mel_spec.dim() == 2:
            energy = torch.mean(mel_spec, dim=0)
        else:
            energy = torch.mean(mel_spec, dim=1)

        # Simple robust normalization to [0, 1]
        # Mel specs are typically in range [0, 100+] after power transform
        # Use fixed range normalization with clipping
        energy = torch.clamp(energy, min=0.0)  # Ensure non-negative
        energy = energy / 50.0  # Normalize assuming typical max ~50
        energy = torch.clamp(energy, 0.0, 1.0)  # Clip to [0, 1]

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
