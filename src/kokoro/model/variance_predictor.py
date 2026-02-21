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


class LengthRegulator(nn.Module):
    """
    Expands phoneme-level hidden states to frame-level based on durations.
    Essential for bridging the gap between text and audio length.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, durations, max_len=None):
        durations = torch.round(durations.clamp(min=0)).long()
        device = x.device
        batch_size = x.shape[0]
        # Vectorized expansion strategy:
        # 1) Move the per-token tensors to CPU to avoid MPS repeat_interleave bugs
        # 2) Flatten (B, L, D) -> (B*L, D) and repeat_interleave with flattened repeats
        # 3) Scatter the single expanded flat tensor into a preallocated (B, max_len, D)
        #    buffer using computed offsets — this uses one big allocation for the output
        # Fallback: if vectorized path fails, fall back to safe per-example expansion
        x_cpu = x.to('cpu')
        durations_cpu = durations.to('cpu')

        try:
            B, L, D = x_cpu.shape

            # Flatten tokens and durations
            x_flat = x_cpu.reshape(B * L, D)
            durations_flat = durations_cpu.reshape(B * L)

            # Total expanded frames across batch
            lengths = durations_cpu.sum(dim=1)
            total_expanded = int(lengths.sum().item())

            if total_expanded == 0:
                # No expansion; return minimal padded tensor
                out_len = 1 if max_len is None else max(1, int(max_len))
                out = torch.zeros((B, out_len, D), dtype=x_cpu.dtype, device=device)
                return out

            # Repeat-interleave flattened tokens -> (total_expanded, D)
            expanded_flat = torch.repeat_interleave(x_flat, durations_flat, dim=0)

            # Compute per-batch offsets
            lengths_list = lengths.tolist()
            max_expanded_len = max(lengths_list) if max_len is None else min(int(max(lengths_list)), int(max_len))

            # Preallocate output on CPU then move to device once filled
            out_cpu = x_cpu.new_zeros((B, max_expanded_len, D))

            # Fill per-batch slices from expanded_flat using cumulative offsets
            offsets = [0]
            for ln in lengths_list:
                offsets.append(offsets[-1] + int(ln))

            for i in range(B):
                start = offsets[i]
                end = offsets[i + 1]
                ln = end - start
                if ln == 0:
                    continue
                take = expanded_flat[start:end]
                if take.size(0) > max_expanded_len:
                    take = take[:max_expanded_len]
                    ln = max_expanded_len
                out_cpu[i, :ln] = take

            # Apply max_len cropping if requested
            if max_len is not None:
                if out_cpu.size(1) < max_len:
                    # pad on the right
                    pad_size = int(max_len) - out_cpu.size(1)
                    pad_tensor = x_cpu.new_zeros((B, pad_size, D))
                    out_cpu = torch.cat([out_cpu, pad_tensor], dim=1)
                else:
                    out_cpu = out_cpu[:, :int(max_len), :]

            output = out_cpu.to(device)
            return output
        except Exception:
            # Fallback to safe per-example expansion if anything goes wrong
            expanded = []
            for i in range(batch_size):
                if durations_cpu[i].sum() == 0:
                    expanded.append(torch.zeros((1, x_cpu.size(-1)), dtype=x_cpu.dtype))
                else:
                    expanded.append(torch.repeat_interleave(x_cpu[i], durations_cpu[i], dim=0))

            output = torch.nn.utils.rnn.pad_sequence(expanded, batch_first=True).to(device)

            if max_len is not None:
                if output.size(1) < max_len:
                    output = F.pad(output, (0, 0, 0, max_len - output.size(1)))
                else:
                    output = output[:, :max_len, :]

            return output


class VariancePredictor(nn.Module):
    """
    Variance Predictor using GroupNorm to maintain (B, C, L) format.
    """
    def __init__(self,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 filter_size: int = DEFAULT_FILTER_SIZE,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 num_layers: int = 2):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else filter_size

            # Standard 1D Conv: Expects (Batch, Channels, Length)
            self.conv_layers.append(nn.Conv1d(
                in_channels,
                filter_size,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2
            ))

            # GroupNorm(1, ...) is equivalent to LayerNorm over the channel dim
            # but natively supports the (B, C, L) format of Conv1d.
            self.norms.append(nn.GroupNorm(num_groups=1, num_channels=filter_size))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

        # Output projection back to (Batch, Length, 1)
        self.linear = nn.Linear(filter_size, 1)
        self._init_weights()

    def _init_weights(self):
        for conv in self.conv_layers:
            nn.init.xavier_uniform_(conv.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optimized dimensionality handling.
        Input: (B, L, H)
        Output: (B, L)
        """
        batch_size, seq_len, _ = x.shape
        chunk_size = 512

        if seq_len <= chunk_size:
            return self._forward_chunk(x, mask)

        outputs = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            outputs.append(self._forward_chunk(x[:, start:end, :],
                                               mask[:, start:end] if mask is not None else None))
        return torch.cat(outputs, dim=1)

    def _forward_chunk(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Flip to (B, C, L) ONCE
        x = x.transpose(1, 2).contiguous()

        # 2. Sequential processing without dimension swapping
        for conv, norm in zip(self.conv_layers, self.norms):
            x = conv(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        # 3. Flip back to (B, L, C) for the final linear projection
        x = x.transpose(1, 2).contiguous()
        output = self.linear(x).squeeze(-1)

        if mask is not None:
            output = output.masked_fill(mask, 0.0)

        return output


class VarianceAdaptor(nn.Module):
    """
    Variance Adaptor combining duration, pitch, and energy predictors.
    Based on FastSpeech 2.

    Processing order:
      1. Predict durations at the TOKEN level (phoneme-level encoder output).
      2. Expand encoder output to FRAME level via LengthRegulator.
      3. Predict pitch and energy at the FRAME level for finer intonation detail.
      4. Add pitch/energy embeddings to the frame-level representation.
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
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.energy_min = energy_min
        self.energy_max = energy_max

        # Duration predictor operates at token (phoneme) level
        self.duration_predictor = VariancePredictor(
            hidden_dim, filter_size, kernel_size, dropout, num_layers=2
        )

        # Pitch and energy predictors operate at frame level (post-expansion)
        self.pitch_predictor = VariancePredictor(
            hidden_dim, filter_size, kernel_size, dropout, num_layers=2
        )
        self.energy_predictor = VariancePredictor(
            hidden_dim, filter_size, kernel_size, dropout, num_layers=2
        )

        # Quantization bins in normalized [0, 1] space
        self.register_buffer('pitch_bins', torch.linspace(0.0, 1.0, n_bins - 1))
        self.register_buffer('energy_bins', torch.linspace(0.0, 1.0, n_bins - 1))

        self.pitch_embedding = nn.Embedding(n_bins, hidden_dim)
        self.energy_embedding = nn.Embedding(n_bins, hidden_dim)

        # Length regulator for expanding token-level to frame-level
        self.length_regulator = LengthRegulator()

        logger.info(f"VarianceAdaptor initialized with {n_bins} bins for pitch/energy")

    # ------------------------------------------------------------------
    # Quantization helpers
    # ------------------------------------------------------------------

    def quantize_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        """
        Quantize normalized pitch values (in [0, 1]) to bin indices.

        Args:
            pitch: Normalized pitch values in [0, 1] (batch, seq_len)

        Returns:
            Quantized pitch indices (batch, seq_len)
        """
        return torch.bucketize(pitch, self.pitch_bins).long()

    def quantize_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Quantize normalized energy values (in [0, 1]) to bin indices.

        Args:
            energy: Normalized energy values in [0, 1] (batch, seq_len)

        Returns:
            Quantized energy indices (batch, seq_len)
        """
        return torch.bucketize(energy, self.energy_bins).long()

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _hz_to_normalized(self, f0: torch.Tensor) -> torch.Tensor:
        """Convert Hz-valued F0 to normalized [0, 1]."""
        return torch.clamp(
            (f0 - self.pitch_min) / (self.pitch_max - self.pitch_min + 1e-8),
            0.0, 1.0
        )

    def _maybe_normalize_pitch(self,
                                pitch: torch.Tensor,
                                device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Heuristic normalization for pitch targets only (never for model predictions).
        If values are outside [0, 1] they are assumed to be in Hz and converted.
        """
        if pitch is None:
            return pitch
        if not torch.is_tensor(pitch):
            pitch = torch.tensor(pitch)
        if device is not None:
            pitch = pitch.to(device)
        if pitch.numel() > 0 and (torch.max(pitch) > 1.0 or torch.min(pitch) < 0.0):
            return self._hz_to_normalized(pitch)
        return pitch

    def _energy_to_normalized(self, energy: torch.Tensor) -> torch.Tensor:
        """Convert energy to normalized [0, 1]."""
        return torch.clamp(
            (energy - self.energy_min) / (self.energy_max - self.energy_min + 1e-8),
            0.0, 1.0
        )

    def _maybe_normalize_energy(self,
                                 energy: torch.Tensor,
                                 device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Heuristic normalization for energy targets only (never for model predictions).
        If values are outside [0, 1] they are assumed to be raw energy and converted.
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

    # ------------------------------------------------------------------
    # Frame-level target expansion
    # ------------------------------------------------------------------
    def _expand_targets_to_frame_level(self,
                                    targets: torch.Tensor,
                                    durations: torch.Tensor,
                                    max_len: Optional[int] = None) -> torch.Tensor:
        durations = torch.round(durations.clamp(min=0)).long()
        device = targets.device
        batch_size = targets.shape[0]
        # Vectorized expansion similar to LengthRegulator: flatten tokens and durations,
        # repeat_interleave once, then scatter into a preallocated (B, max_len) buffer.
        targets_cpu = targets.detach().to('cpu')
        durations_cpu = durations.detach().to('cpu')

        try:
            B, L = targets_cpu.shape

            t_flat = targets_cpu.reshape(B * L)
            d_flat = durations_cpu.reshape(B * L)

            lengths = durations_cpu.sum(dim=1)
            total_expanded = int(lengths.sum().item())

            if total_expanded == 0:
                out_len = 1 if max_len is None else max(1, int(max_len))
                return targets_cpu.new_zeros((B, out_len)).to(device)

            expanded_flat = torch.repeat_interleave(t_flat, d_flat, dim=0)

            lengths_list = lengths.tolist()
            max_expanded_len = max(lengths_list) if max_len is None else min(int(max(lengths_list)), int(max_len))

            out_cpu = targets_cpu.new_zeros((B, max_expanded_len))

            offsets = [0]
            for ln in lengths_list:
                offsets.append(offsets[-1] + int(ln))

            for i in range(B):
                start = offsets[i]
                end = offsets[i + 1]
                ln = end - start
                if ln == 0:
                    continue
                take = expanded_flat[start:end]
                if take.size(0) > max_expanded_len:
                    take = take[:max_expanded_len]
                    ln = max_expanded_len
                out_cpu[i, :ln] = take

            if max_len is not None:
                if out_cpu.size(1) < max_len:
                    pad_size = int(max_len) - out_cpu.size(1)
                    pad_tensor = targets_cpu.new_zeros((B, pad_size))
                    out_cpu = torch.cat([out_cpu, pad_tensor], dim=1)
                else:
                    out_cpu = out_cpu[:, :int(max_len)]

            return out_cpu.to(device)
        except Exception:
            # Fallback to per-sample expand if vectorized approach fails
            expanded = []
            for i in range(batch_size):
                if durations_cpu[i].sum() == 0:
                    expanded.append(torch.zeros((1,), dtype=targets_cpu.dtype))
                else:
                    expanded.append(torch.repeat_interleave(targets_cpu[i], durations_cpu[i], dim=0))

            output = torch.nn.utils.rnn.pad_sequence(expanded, batch_first=True).to(device)

            if max_len is not None:
                if output.size(1) < max_len:
                    output = F.pad(output, (0, max_len - output.size(1)))
                else:
                    output = output[:, :max_len]

            return output

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self,
                encoder_output: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pitch_target: Optional[torch.Tensor] = None,
                energy_target: Optional[torch.Tensor] = None,
                duration_target: Optional[torch.Tensor] = None,
                mel_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through variance adaptor.

        During training, ground-truth duration/pitch/energy targets are supplied
        so the model conditions on correct values (teacher forcing).  During
        inference all three are None and the model uses its own predictions.

        Args:
            encoder_output: Phoneme-level encoder output (batch, n_phonemes, hidden_dim)
            mask: Phoneme-level padding mask (batch, n_phonemes) — True for padding
            pitch_target: Token-level normalized pitch targets (batch, n_phonemes).
                          May be in Hz; will be auto-converted to [0, 1] if needed.
            energy_target: Token-level normalized energy targets (batch, n_phonemes).
                           Will be auto-converted to [0, 1] if needed.
            duration_target: Ground-truth frame counts per phoneme (batch, n_phonemes).
                             If None, model predictions are used (inference mode).
                             NOTE: if your duration predictor is trained in log-domain,
                             pass raw (non-log) integer frame counts here and apply
                             exp() to duration_pred before using it for inference.
            mel_target: Ground-truth mel spectrogram (batch, n_frames, n_mels) used
                        only to determine the target frame length for alignment.

        Returns:
            Tuple of:
                adapted_output  — Frame-level hidden states (batch, n_frames, hidden_dim)
                duration_pred   — Token-level duration logits  (batch, n_phonemes)
                pitch_pred      — Frame-level pitch predictions (batch, n_frames)
                energy_pred     — Frame-level energy predictions (batch, n_frames)
                frame_mask      — Frame-level padding mask (batch, n_frames), True=padding
        """
        device = encoder_output.device

        # 1. Predict durations at the TOKEN level
        duration_pred = self.duration_predictor(encoder_output, mask)

        # 2. Determine which durations to use for expansion
        if duration_target is not None:
            durations_to_use = duration_target
        else:
            # Inference: use predicted durations (assumed to be raw frame counts,
            # not log-domain — apply exp() upstream if your predictor uses log targets)
            durations_to_use = torch.clamp(torch.round(duration_pred), min=0)

        durations_to_use = durations_to_use.to(device)

        # 3. Expand encoder hidden states to frame level
        max_len = mel_target.size(1) if mel_target is not None else None
        x = self.length_regulator(encoder_output, durations_to_use, max_len=max_len)

        # Guard: expanded sequence must be at least as long as the conv kernel so
        # that VariancePredictor doesn't raise "kernel size > input size".  This can
        # happen during early training when predicted durations are near zero.
        min_frames = 3  # matches default kernel_size in VariancePredictor
        if x.size(1) < min_frames:
            x = F.pad(x, (0, 0, 0, min_frames - x.size(1)))

        # 4. Build a frame-level padding mask from the expanded durations
        lengths = durations_to_use.long().sum(dim=1)
        if max_len is not None:
            lengths = lengths.clamp(max=max_len)
        frame_mask = (
            torch.arange(x.size(1), device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        )

        # 5. Predict pitch and energy at the FRAME level
        pitch_pred = self.pitch_predictor(x, frame_mask)
        energy_pred = self.energy_predictor(x, frame_mask)

        # 6. Determine values to embed
        #    Training:  expand token-level targets to frame-level, then normalize.
        #    Inference: clamp model predictions into [0, 1] (no Hz heuristic applied
        #               to raw logits — that would silently corrupt predictions).
        if pitch_target is not None:
            # Expand token-level targets → frame-level before normalization
            pitch_frame = self._expand_targets_to_frame_level(
                pitch_target, durations_to_use, max_len=max_len
            )
            p_val = self._maybe_normalize_pitch(pitch_frame, device=device)
        else:
            p_val = pitch_pred.clamp(0.0, 1.0)

        if energy_target is not None:
            energy_frame = self._expand_targets_to_frame_level(
                energy_target, durations_to_use, max_len=max_len
            )
            e_val = self._maybe_normalize_energy(energy_frame, device=device)
        else:
            e_val = energy_pred.clamp(0.0, 1.0)

        # 7. Look up embeddings and add to frame-level hidden states
        pitch_embed = self.pitch_embedding(self.quantize_pitch(p_val))
        energy_embed = self.energy_embedding(self.quantize_energy(e_val))

        adapted_output = x + pitch_embed + energy_embed

        # Zero out padded positions
        if frame_mask is not None:
            adapted_output = adapted_output.masked_fill(frame_mask.unsqueeze(-1), 0.0)

        return adapted_output, duration_pred, pitch_pred, energy_pred, frame_mask


class PitchExtractor:
    """
    Extract pitch (F0) from audio waveform.
    Uses a PyTorch-native YIN-style CMND algorithm.
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

        Features:
        - Cumulative Mean Normalized Difference (CMND) reduces octave errors
        - Parabolic interpolation for sub-sample lag accuracy
        - Adaptive per-utterance voicing threshold
        - Median filtering to remove frame-to-frame jitter
        - Linear interpolation across short unvoiced gaps

        Args:
            waveform: Audio waveform (batch, samples) or (samples,)
            sample_rate: Sample rate in Hz
            hop_length: Hop length for framing
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            win_length: Analysis window length (defaults to max(2048, hop*8))

        Returns:
            Normalized pitch contour in [0, 1] — (batch, frames) or (frames,).
            Unvoiced frames are 0.0.
        """
        try:
            # --- Setup ---
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False

            # Capture shapes before any modification so the exception handler
            # can reconstruct the correct output shape.
            orig_num_samples = waveform.shape[-1]
            orig_batch_size = waveform.shape[0]

            device = waveform.device
            dtype = waveform.dtype
            hop = int(hop_length)

            win_len = int(win_length) if win_length is not None else max(2048, hop * 8)

            # Pad signal to at least one full window
            if waveform.size(1) < win_len:
                waveform = F.pad(waveform, (0, win_len - waveform.size(1)))

            # --- Pre-emphasis ---
            pre_emphasis = 0.97
            waveform = torch.cat([
                waveform[..., :1],
                waveform[..., 1:] - pre_emphasis * waveform[..., :-1]
            ], dim=-1)

            pad_val = win_len // 2
            waveform = F.pad(waveform, (pad_val, pad_val), mode='reflect')

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
            zero_lag = acf[..., 0:1]
            diff = 2 * zero_lag - 2 * acf

            cmnd = torch.zeros_like(diff)
            cmnd[..., 0] = 1.0
            cumsum = torch.cumsum(diff[..., 1:], dim=-1)
            tau_range = torch.arange(1, win_len, device=device, dtype=dtype)
            cmnd[..., 1:] = diff[..., 1:] / (cumsum / tau_range + 1e-8)

            # --- Lag Search Range ---
            lag_min = max(2, int(sample_rate / fmax))
            lag_max = min(win_len - 2, max(lag_min + 1, int(sample_rate / fmin)))

            lags = torch.arange(lag_min, lag_max + 1, device=device)
            n_lags = len(lags)
            cmnd_lags = cmnd[..., lag_min:lag_max + 1]

            acf_norm = acf / zero_lag.clamp(min=1e-8)
            ac_lags = acf_norm[..., lag_min:lag_max + 1]
            ac_max_vals, _ = ac_lags.max(dim=-1)

            # --- CMND Threshold + Argmin Fallback ---
            threshold = 0.15
            below_thresh = cmnd_lags < threshold
            first_dip = (below_thresh.cumsum(-1) == 1) & below_thresh
            has_dip = below_thresh.any(dim=-1)
            first_dip_idx = first_dip.long().argmax(dim=-1)
            argmin_idx = torch.argmin(cmnd_lags, dim=-1)
            best_idx = torch.where(has_dip, first_dip_idx, argmin_idx)

            # --- Parabolic Interpolation ---
            idx_prev = (best_idx - 1).clamp(min=0)
            idx_next = (best_idx + 1).clamp(max=n_lags - 1)

            alpha = cmnd_lags.gather(-1, idx_prev.unsqueeze(-1)).squeeze(-1)
            beta  = cmnd_lags.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
            gamma = cmnd_lags.gather(-1, idx_next.unsqueeze(-1)).squeeze(-1)

            denom = (alpha - 2 * beta + gamma).clamp(min=1e-8)
            offset = (0.5 * (alpha - gamma) / denom).clamp(-1.0, 1.0)

            best_lags = (lags[best_idx].float() + offset).clamp(min=1.0)
            freqs = sample_rate / best_lags

            # --- Adaptive Voicing Threshold ---
            ac_25th = torch.quantile(ac_max_vals, 0.25, dim=-1, keepdim=True)
            voicing_thresh = torch.clamp(ac_25th * 0.8, min=0.15, max=0.35)

            frame_energy = frames.pow(2).mean(dim=-1)
            energy_thresh = torch.clamp(
                torch.median(frame_energy, dim=-1, keepdim=True).values * 0.05, min=1e-9
            )

            unvoiced_mask = (ac_max_vals < voicing_thresh) | (frame_energy < energy_thresh)
            freqs = freqs.masked_fill(unvoiced_mask, 0.0)

            freqs = torch.where(
                (freqs < fmin) | (freqs > fmax), torch.zeros_like(freqs), freqs
            )

            # --- Interpolate Short Unvoiced Gaps (vectorized, torch-only) ---
            MAX_GAP_FRAMES = 5
            Bf, Tf = freqs.shape
            idx = torch.arange(Tf, device=device, dtype=torch.long).unsqueeze(0).expand(Bf, Tf)
            voiced_mask = freqs > 0.0

            if voiced_mask.any():
                prev_candidates = torch.where(voiced_mask, idx, torch.full_like(idx, -1))
                prev_idx = torch.cummax(prev_candidates, dim=1)[0]

                next_candidates = torch.where(voiced_mask, idx, torch.full_like(idx, Tf))
                next_idx = torch.cummin(next_candidates.flip(dims=[1]), dim=1)[0].flip(dims=[1])

                gap_len = next_idx - prev_idx - 1
                fill_mask = (
                    (~voiced_mask) & (prev_idx >= 0) & (next_idx < Tf) & (gap_len <= MAX_GAP_FRAMES)
                )

                if fill_mask.any():
                    prev_vals = freqs.gather(1, prev_idx.clamp(min=0))
                    next_vals = freqs.gather(1, next_idx.clamp(max=Tf - 1))
                    denom = (next_idx.float() - prev_idx.float()).clamp(min=1.0)
                    t = (idx.float() - prev_idx.float()) / denom
                    interp = prev_vals * (1.0 - t) + next_vals * t
                    freqs[fill_mask] = interp[fill_mask]

            # --- Median Filter ---
            MEDIAN_K = 5
            pad = MEDIAN_K // 2
            freqs = F.pad(freqs, (pad, pad), mode='reflect').unfold(-1, MEDIAN_K, 1).median(dim=-1).values

            # --- Normalize to [0, 1], preserving unvoiced zeros ---
            freqs_norm = torch.clamp((freqs - fmin) / (fmax - fmin + 1e-8), 0.0, 1.0)
            freqs_norm = freqs_norm.masked_fill(freqs == 0.0, 0.0)

            return freqs_norm.squeeze(0) if squeeze_output else freqs_norm

        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}, using zeros")
            n_frames = max(1, orig_num_samples // hop_length)
            device = waveform.device if 'waveform' in locals() else torch.device('cpu')
            zeros = torch.zeros((orig_batch_size, n_frames), device=device)
            return zeros.squeeze(0) if squeeze_output else zeros


class EnergyExtractor:
    """
    Extract energy (RMS) from mel spectrogram or waveform.
    """

    @staticmethod
    def extract_energy_from_mel(mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Extract energy from mel spectrogram, normalized to [0, 1].

        Args:
            mel_spec: Mel spectrogram (batch, frames, n_mels) or (frames, n_mels)

        Returns:
            Energy contour normalized to [0, 1] (batch, frames) or (frames,)
        """
        # Convert to linear domain if input appears to be log-mel
        mel_linear = torch.exp(mel_spec) if mel_spec.min() < 0 else mel_spec

        # average over mel bins to get a single energy value per frame, then apply log compression
        energy = torch.mean(mel_linear, dim=-1)
        energy = torch.log1p(torch.clamp(energy, min=0.0))

        floor = torch.quantile(energy, 0.05, dim=-1, keepdim=True)
        ceil  = torch.quantile(energy, 0.95, dim=-1, keepdim=True)
        energy = (energy - floor) / torch.clamp(ceil - floor, min=1e-8)

        return torch.clamp(energy, 0.0, 1.0)

    @staticmethod
    def extract_energy_from_waveform(waveform: torch.Tensor,
                                     hop_length: int = DEFAULT_HOP_LENGTH,
                                     win_length: int = 1024) -> torch.Tensor:
        """
        Extract RMS energy from waveform.

        Args:
            waveform: Audio waveform (batch, samples) or (samples,)
            hop_length: Hop length
            win_length: Window length for RMS calculation

        Returns:
            Energy contour (batch, frames) or (frames,)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        hop = int(hop_length)
        win_len = int(win_length)
        pad_val = win_len // 2

        waveform = F.pad(waveform, (pad_val, pad_val), mode='reflect')
        if waveform.size(1) < win_len:
            waveform = F.pad(waveform, (0, win_len - waveform.size(1)))

        frames = waveform.unfold(1, win_len, hop).contiguous()

        device, dtype = waveform.device, waveform.dtype
        window = torch.hann_window(win_len, device=device, dtype=dtype)
        frames = frames * window.unsqueeze(0).unsqueeze(0)

        energy = torch.sqrt(torch.mean(frames.pow(2), dim=-1) + 1e-8)

        return energy.squeeze(0) if squeeze_output else energy


def normalize_variance(values: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       mean: Optional[float] = None,
                       std: Optional[float] = None) -> Tuple[torch.Tensor, float, float]:
    """
    Normalize variance values (pitch/energy) to zero mean and unit variance.

    Args:
        values: Variance values (batch, seq_len)
        mask: Padding mask (batch, seq_len) - True for padding
        mean: Pre-computed mean (if None, compute from values)
        std: Pre-computed std (if None, compute from values)

    Returns:
        Tuple of (normalized_values, mean, std)
    """
    if mask is not None:
        non_masked = values[~mask]
        if mean is None:
            mean = non_masked.mean().item()
        if std is None:
            std = non_masked.std().item() + 1e-8
    else:
        if mean is None:
            mean = values.mean().item()
        if std is None:
            std = values.std().item() + 1e-8

    normalized = (values - mean) / std

    if mask is not None:
        normalized = normalized.masked_fill(mask, 0.0)

    return normalized, mean, std


if __name__ == "__main__":
    """Smoke tests for all components."""
    logging.basicConfig(level=logging.INFO)

    # VariancePredictor
    predictor = VariancePredictor(hidden_dim=DEFAULT_HIDDEN_DIM, filter_size=DEFAULT_FILTER_SIZE)
    x = torch.randn(4, 100, DEFAULT_HIDDEN_DIM)
    mask = torch.zeros(4, 100).bool()
    output = predictor(x, mask)
    print(f"VariancePredictor output shape: {output.shape}")  # (4, 100)

    # VarianceAdaptor — training mode with token-level targets
    adaptor = VarianceAdaptor(hidden_dim=DEFAULT_HIDDEN_DIM, n_bins=DEFAULT_N_BINS)
    pitch_target = torch.randn(4, 100) * 100 + 200   # Hz-scale (will be auto-normalized)
    energy_target = torch.randn(4, 100) * 10 + 50    # raw energy (will be auto-normalized)
    duration_target = torch.randint(1, 5, (4, 100)).float()  # token-level frame counts

    adapted, dur_pred, pitch_pred, energy_pred, frame_mask = adaptor(
        x, mask, pitch_target, energy_target, duration_target
    )

    print(f"Adapted output shape:    {adapted.shape}")
    print(f"Duration pred shape:     {dur_pred.shape}")
    print(f"Pitch pred shape:        {pitch_pred.shape}")
    print(f"Energy pred shape:       {energy_pred.shape}")
    print(f"Frame mask shape:        {frame_mask.shape}")

    # PitchExtractor
    waveform = torch.randn(1, 22050)
    pitch = PitchExtractor.extract_pitch(waveform, sample_rate=22050, hop_length=DEFAULT_HOP_LENGTH)
    print(f"Pitch shape: {pitch.shape}")

    # EnergyExtractor
    mel_spec = torch.randn(80, 100)
    energy = EnergyExtractor.extract_energy_from_mel(mel_spec)
    print(f"Energy shape: {energy.shape}")

    logger.info("All variance predictor tests passed!")
