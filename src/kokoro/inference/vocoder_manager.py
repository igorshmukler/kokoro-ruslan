#!/usr/bin/env python3
"""
VocoderManager - Handles different vocoder backends for TTS inference
Supports HiFi-GAN and Griffin-Lim vocoders
"""

import torch
import torchaudio
import requests
import logging
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import urlparse
import numpy as np

# Import vocoder modules
from kokoro.inference.hifigan_vocoder import load_hifigan_model

logger = logging.getLogger(__name__)


class VocoderManager:
    """Manages different vocoder backends"""

    HIFIGAN_URLS = {
        # Universal HiFi-GAN models (22kHz)
        "universal_v1": {
            "model": "https://drive.google.com/uc?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW",
            "config": "https://drive.google.com/uc?id=1pAB2kQunkDuv6W5fcJiQ0CY8xcJKB22e"
        },
        # LJ Speech model (good for general purpose)
        "ljspeech": {
            "model": "https://drive.google.com/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx",
            "config": "https://drive.google.com/uc?id=1Jt_imitfckTfM9TPhT4TQKPUgkcGhv5f"
        }
    }

    def __init__(self, vocoder_type: str = "hifigan", vocoder_path: Optional[str] = None, device: str = "cpu"):
        self.vocoder_type = vocoder_type.lower()
        self.device = device
        self.vocoder = None

        if vocoder_type == "hifigan":
            self.vocoder = self._load_hifigan(vocoder_path)
        elif vocoder_type == "griffin_lim":
            self.vocoder = self._setup_griffin_lim()
        else:
            raise ValueError(f"Unsupported vocoder type: {vocoder_type}")

    def _download_file(self, url: str, filepath: Path):
        """Download file with progress"""
        logger.info(f"Downloading {filepath.name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='')
        print()  # New line after progress

    def _load_hifigan(self, vocoder_path: Optional[str] = None) -> torch.nn.Module:
        """Load HiFi-GAN vocoder using the separated module"""
        if vocoder_path and Path(vocoder_path).exists():
            # Load custom HiFi-GAN model
            model_path = Path(vocoder_path)
            if model_path.is_dir():
                generator_path = model_path / "generator.pth"
                config_path = model_path / "config.json"
            else:
                generator_path = model_path
                config_path = model_path.parent / "config.json"

            try:
                generator = load_hifigan_model(generator_path, config_path, self.device)
                logger.info(f"Loaded custom HiFi-GAN from: {vocoder_path}")
                return generator
            except Exception as e:
                logger.warning(f"Failed to load custom HiFi-GAN: {e}")
                logger.info("Falling back to pre-trained model")

        # Try to load pre-trained model or download
        vocoder_dir = Path("./vocoder_models/hifigan")
        vocoder_dir.mkdir(parents=True, exist_ok=True)

        model_name = "universal_v1"  # Default to universal model
        model_file = vocoder_dir / f"generator_{model_name}.pth"
        config_file = vocoder_dir / f"config_{model_name}.json"

        # Download if not exists
        if not model_file.exists() or not config_file.exists():
            logger.info(f"Downloading HiFi-GAN {model_name} model...")
            try:
                if not model_file.exists():
                    self._download_file(self.HIFIGAN_URLS[model_name]["model"], model_file)
                if not config_file.exists():
                    self._download_file(self.HIFIGAN_URLS[model_name]["config"], config_file)
            except Exception as e:
                logger.warning(f"Failed to download HiFi-GAN model: {e}")
                logger.info("Falling back to Griffin-Lim")
                return self._setup_griffin_lim()

        # Load downloaded model
        try:
            generator = load_hifigan_model(model_file, config_file, self.device)
            # Diagnostic: inspect parameter statistics to detect accidental random init
            try:
                total_params = sum(p.numel() for p in generator.parameters())
                means = [p.data.mean().item() for p in generator.parameters() if p.data.numel() > 0]
                stds = [p.data.std().item() for p in generator.parameters() if p.data.numel() > 0]
                overall_mean = float(sum(means) / len(means)) if means else 0.0
                overall_std = float(sum(stds) / len(stds)) if stds else 0.0
                logger.info(f"Loaded pre-trained HiFi-GAN {model_name} (params={total_params:,}, mean={overall_mean:.6f}, std={overall_std:.6f})")
                if overall_std < 1e-6:
                    logger.warning("HiFi-GAN parameters have near-zero std — checkpoint may be empty or improperly loaded.")
            except Exception as diag_e:
                logger.debug(f"Vocoder parameter diagnostics failed: {diag_e}")

            return generator
        except Exception as e:
            logger.warning(f"Failed to load HiFi-GAN: {e}")
            logger.info("Falling back to Griffin-Lim")
            return self._setup_griffin_lim()

    def _setup_griffin_lim(self):
        """Setup Griffin-Lim as fallback with device compatibility"""
        logger.info("Using Griffin-Lim vocoder")

        # For MPS device compatibility, create Griffin-Lim on CPU initially
        # and move to device later if supported
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=2.0,
            n_iter=60  # More iterations for better quality
        )

        # Try to move to target device, fallback to CPU if not supported
        try:
            griffin_lim = griffin_lim.to(self.device)
        except Exception as e:
            logger.warning(f"Griffin-Lim not fully compatible with {self.device}, using CPU fallback for some operations: {e}")
            griffin_lim = griffin_lim.to("cpu")

        return griffin_lim

    def mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to audio"""
        if self.vocoder_type == "hifigan":
            return self._hifigan_inference(mel_spec)
        elif self.vocoder_type == "griffin_lim":
            return self._griffin_lim_inference(mel_spec)
        else:
            raise ValueError(f"Unknown vocoder type: {self.vocoder_type}")

    def _hifigan_inference(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """HiFi-GAN inference"""
        if isinstance(self.vocoder, torchaudio.transforms.GriffinLim):
            # Fallback to Griffin-Lim if HiFi-GAN failed to load
            return self._griffin_lim_inference(mel_spec)

        with torch.no_grad():
            # Ensure mel_spec is on the right device and has the right shape
            # Log mel diagnostics (shape and basic stats) to help trace noisy outputs
            try:
                m_min, m_max = float(mel_spec.min().item()), float(mel_spec.max().item())
                m_mean, m_std = float(mel_spec.mean().item()), float(mel_spec.std().item())
                logger.info(f"Vocoder received mel stats (pre-device): shape={tuple(mel_spec.shape)}, min={m_min:.4f}, max={m_max:.4f}, mean={m_mean:.4f}, std={m_std:.6f}")
            except Exception:
                logger.debug("Failed to compute mel stats pre-device")

            mel_spec = mel_spec.to(self.device)

            if len(mel_spec.shape) == 2:  # (n_mels, time)
                mel_spec = mel_spec.unsqueeze(0)  # (1, n_mels, time)
            elif len(mel_spec.shape) == 3 and mel_spec.shape[0] != 1:  # (batch, time, n_mels)
                mel_spec = mel_spec.transpose(1, 2)  # (batch, n_mels, time)

            # Generate audio
            audio = self.vocoder(mel_spec)

            # Diagnostics on generated audio
            try:
                audio_cpu = audio.cpu()
                a_min, a_max = float(audio_cpu.min().item()), float(audio_cpu.max().item())
                a_mean, a_std = float(audio_cpu.mean().item()), float(audio_cpu.std().item())
                logger.info(f"Raw vocoder output stats: shape={tuple(audio_cpu.shape)}, min={a_min:.6f}, max={a_max:.6f}, mean={a_mean:.6f}, std={a_std:.6f}")
                if a_std < 1e-4:
                    logger.warning("Vocoder output has near-zero variance (silence or collapsed output).")
                if a_std > 1.0:
                    logger.warning("Vocoder output has unexpectedly large variance — may be noisy/random weights.")
            except Exception:
                logger.debug("Failed to compute vocoder output diagnostics")
            if len(audio.shape) == 3:  # Remove batch dimension if present
                audio = audio.squeeze(0)
            if len(audio.shape) == 2:  # Remove channel dimension if present
                audio = audio.squeeze(0)

        return audio.cpu()

    def _griffin_lim_inference(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Griffin-Lim inference with device compatibility handling"""
        # Convert log mel to linear scale
        mel_spec = torch.exp(mel_spec)

        # Transpose to get correct shape: (time, n_mels) -> (n_mels, time)
        if len(mel_spec.shape) == 2 and mel_spec.shape[1] == 80:
            mel_spec = mel_spec.transpose(0, 1)

        # Handle MPS device incompatibility with InverseMelScale
        device_for_mel_ops = "cpu" if self.device == "mps" else self.device

        # Convert mel spectrogram back to linear magnitude spectrogram
        inverse_mel_scale = torchaudio.transforms.InverseMelScale(
            n_stft=513,
            n_mels=80,
            sample_rate=22050,
            f_min=0.0,
            f_max=8000.0,
            mel_scale='htk'
        ).to(device_for_mel_ops)

        # Move mel_spec to compatible device for inverse transform
        mel_spec_for_inverse = mel_spec.to(device_for_mel_ops)
        linear_spec = inverse_mel_scale(mel_spec_for_inverse)

        # The dataset's MelSpectrogram uses power=2.0 and Griffin-Lim was
        # constructed with `power=2.0`, so pass the power spectrogram through
        # unchanged. Do not convert to magnitude here.

        # Move back to original device for Griffin-Lim if needed
        if device_for_mel_ops != self.device:
            linear_spec = linear_spec.to(self.device)

        # Diagnostic: re-project linear_spec back to mel using different mel scales
        try:
            # linear_spec shape: (n_freq, time) or (1, n_freq, time)
            ls = linear_spec.clone().to(device_for_mel_ops)
            # Ensure shape is (..., n_stft, time)
            if ls.dim() == 2:
                ls_unsq = ls.unsqueeze(0)
            else:
                ls_unsq = ls

            mel_scale_htk = torchaudio.transforms.MelScale(n_mels=80, sample_rate=22050, n_stft=513, f_min=0.0, f_max=8000.0, mel_scale='htk').to(device_for_mel_ops)
            mel_scale_slaney = torchaudio.transforms.MelScale(n_mels=80, sample_rate=22050, n_stft=513, f_min=0.0, f_max=8000.0, mel_scale='slaney').to(device_for_mel_ops)

            mel_htk = mel_scale_htk(ls_unsq)
            mel_slaney = mel_scale_slaney(ls_unsq)

            # original mel was mel_spec_for_inverse shape (n_mels, time) -> unsqueeze to (..., n_mels, time)
            orig_mel = mel_spec_for_inverse.clone().unsqueeze(0).to(device_for_mel_ops)

            mse_htk = float(((mel_htk - orig_mel) ** 2).mean().item())
            mse_slaney = float(((mel_slaney - orig_mel) ** 2).mean().item())

            logger.info(f"Mel reconstruction MSE: htk={mse_htk:.6e}, slaney={mse_slaney:.6e} (lower is better)")
            if mse_htk > 1e-3 and mse_slaney > 1e-3:
                logger.warning("High mel reconstruction error — mel basis mismatch likely. Check MelSpectrogram/InvereMelScale mel_scale and normalization settings.")
                # Dump diagnostics for offline inspection
                try:
                    debug_dir = Path("vocoder_models/hifigan")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(debug_dir / "debug_mel_reconstruction.npz",
                             orig_mel=orig_mel.cpu().numpy(),
                             mel_htk=mel_htk.cpu().numpy(),
                             mel_slaney=mel_slaney.cpu().numpy(),
                             linear_spec=linear_spec.cpu().numpy())
                    logger.info(f"Wrote mel reconstruction diagnostics to {debug_dir / 'debug_mel_reconstruction.npz'}")
                except Exception as e:
                    logger.debug(f"Failed to dump mel reconstruction diagnostics: {e}")
        except Exception as e:
            logger.debug(f"Mel reconstruction diagnostics failed: {e}")
        # Convert linear magnitude spectrogram to audio using Griffin-Lim
        # Run Griffin-Lim on CPU to avoid possible MPS numerical/implementation issues
        try:
            if str(self.device).startswith('mps') or str(self.device).startswith('cuda'):
                # Ensure vocoder module is on CPU and input tensor is on CPU
                try:
                    self.vocoder = self.vocoder.to('cpu')
                except Exception:
                    logger.debug("Could not move Griffin-Lim module to CPU; proceeding with existing device")
                audio = self.vocoder(linear_spec.to('cpu'))
            else:
                audio = self.vocoder(linear_spec)
        except Exception as e:
            logger.warning(f"Griffin-Lim inference failed on target device: {e}. Retrying on CPU.")
            audio = self.vocoder(linear_spec.to('cpu'))

        # Diagnostic: compute STFT of reconstructed audio and compare power spectrogram
        try:
            # Move audio to device_for_mel_ops for STFT comparison
            audio_dev = audio.to(device_for_mel_ops)

            # If audio has batch dim, squeeze
            if audio_dev.dim() > 1:
                audio_for_stft = audio_dev.squeeze(0)
            else:
                audio_for_stft = audio_dev

            # Compute complex STFT
            stft = torch.stft(
                audio_for_stft,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                return_complex=True,
                window=torch.hann_window(1024).to(device_for_mel_ops)
            )

            # Compute power spectrogram (magnitude squared)
            recon_power = (stft.abs() ** 2)

            # Align shapes: recon_power -> (n_freq, time)
            if recon_power.dim() == 3:
                recon_power = recon_power.squeeze(0)

            # Ensure linear_spec is on same device
            ref_spec = linear_spec.clone().to(device_for_mel_ops)

            # If shapes mismatch in time/freq, crop to min
            min_freq = min(recon_power.size(0), ref_spec.size(0))
            min_time = min(recon_power.size(1), ref_spec.size(1))
            recon_crop = recon_power[:min_freq, :min_time]
            ref_crop = ref_spec[:min_freq, :min_time]

            mse_recon = float(((recon_crop - ref_crop) ** 2).mean().item())
            logger.info(f"Griffin-Lim reconstruction spectrogram MSE vs target: {mse_recon:.6e}")
            if mse_recon > 1e-2:
                logger.warning("High Griffin-Lim spectrogram MSE — Griffin-Lim did not converge to target spectrogram.")
                # Dump diagnostic tensors for offline analysis
                try:
                    debug_dir = Path("vocoder_models/hifigan")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(debug_dir / "debug_griffin_recon.npz",
                             target_spec=ref_crop.cpu().numpy(),
                             recon_spec=recon_crop.cpu().numpy())
                    logger.info(f"Wrote Griffin-Lim diagnostics to {debug_dir / 'debug_griffin_recon.npz'}")
                except Exception as e:
                    logger.debug(f"Failed to dump Griffin-Lim diagnostics: {e}")
        except Exception as e:
            logger.debug(f"Griffin-Lim post-reconstruction diagnostics failed: {e}")

        return audio.cpu()
