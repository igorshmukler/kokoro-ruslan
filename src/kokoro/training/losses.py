import logging
from typing import Any, Callable, Optional, Tuple

import torch

from kokoro.utils.lengths import vectorized_expand_tokens


def calculate_training_losses(
    *,
    device: torch.device,
    config: Any,
    model: torch.nn.Module,
    criterion_mel: torch.nn.Module,
    criterion_duration: torch.nn.Module,
    criterion_stop_token: torch.nn.Module,
    criterion_pitch: Optional[torch.nn.Module],
    criterion_energy: Optional[torch.nn.Module],
    average_by_duration: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    logger: logging.Logger,
    predicted_mel: torch.Tensor,
    predicted_log_durations: torch.Tensor,
    predicted_stop_logits: torch.Tensor,
    mel_specs: torch.Tensor,
    phoneme_durations: torch.Tensor,
    stop_token_targets: torch.Tensor,
    mel_lengths: torch.Tensor,
    phoneme_lengths: torch.Tensor,
    predicted_pitch: Optional[torch.Tensor] = None,
    predicted_energy: Optional[torch.Tensor] = None,
    pitch_targets: Optional[torch.Tensor] = None,
    energy_targets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_mel_len = mel_specs.size(1)
    max_phoneme_len = phoneme_durations.size(1)

    mel_mask_2d = torch.arange(max_mel_len, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)
    phoneme_mask_2d = torch.arange(max_phoneme_len, device=device).unsqueeze(0) < phoneme_lengths.unsqueeze(1)

    loss_mel_unreduced = criterion_mel(predicted_mel, mel_specs)
    mel_mask_3d = mel_mask_2d.unsqueeze(-1).expand_as(loss_mel_unreduced)
    mel_valid = mel_mask_3d & torch.isfinite(loss_mel_unreduced)
    if mel_valid.any():
        loss_mel = loss_mel_unreduced[mel_valid].mean()
    else:
        loss_mel = torch.tensor(0.0, device=device)

    target_log_durations = torch.log(phoneme_durations.float() + 1.0)

    if getattr(config, 'verbose', False):
        try:
            pred = predicted_log_durations.detach()
            targ = target_log_durations.detach()
            pred_valid_mask = torch.isfinite(pred)
            targ_valid_mask = torch.isfinite(targ)

            if pred_valid_mask.any():
                p = pred[pred_valid_mask]
                p_mean = float(p.mean().item())
                p_std = float(p.std().item())
                p_min = float(p.min().item())
                p_max = float(p.max().item())
            else:
                p_mean = p_std = p_min = p_max = float('nan')

            if targ_valid_mask.any():
                t = targ[targ_valid_mask]
                t_mean = float(t.mean().item())
                t_std = float(t.std().item())
                t_min = float(t.min().item())
                t_max = float(t.max().item())
            else:
                t_mean = t_std = t_min = t_max = float('nan')

            logger.info(
                f"Duration pred: mean={p_mean:.4f} std={p_std:.4f} min={p_min:.4f} max={p_max:.4f} | "
                f"target: mean={t_mean:.4f} std={t_std:.4f} min={t_min:.4f} max={t_max:.4f}"
            )
        except Exception as error:
            logger.debug(f"Failed to log duration stats: {error}")

    loss_duration_unreduced = criterion_duration(predicted_log_durations, target_log_durations)

    duration_valid = phoneme_mask_2d & (phoneme_durations > 0)

    if getattr(config, 'verbose', False):
        try:
            phoneme_mask_count = int(phoneme_mask_2d.sum().item())
            duration_valid_count = int(duration_valid.sum().item())
            logger.info(f"Phoneme mask total positions={phoneme_mask_count}, duration_valid positions={duration_valid_count}")
        except Exception:
            logger.debug("Could not compute phoneme/duration mask counts for logging")

    if duration_valid.any():
        loss_duration = loss_duration_unreduced[duration_valid].mean()
    else:
        logger.warning("Duration loss: no valid phoneme positions found for this batch; using 0.0 as loss")
        loss_duration = torch.tensor(0.0, device=device)

    loss_stop_token_unreduced = criterion_stop_token(predicted_stop_logits, stop_token_targets)
    stop_valid = mel_mask_2d & torch.isfinite(loss_stop_token_unreduced)
    if stop_valid.any():
        loss_stop_token = loss_stop_token_unreduced[stop_valid].mean()
    else:
        loss_stop_token = torch.tensor(0.0, device=device)

    loss_pitch = torch.tensor(0.0, device=device)
    if predicted_pitch is not None and pitch_targets is not None and criterion_pitch is not None:
        if predicted_pitch.dim() == 2 and predicted_pitch.size(1) != phoneme_durations.size(1):
            # predicted_pitch is frame-level; pitch_targets is phoneme-level.
            # Expand phoneme-level targets to frame level so that *every* predicted
            # frame receives an independent gradient signal.  Previously the code
            # averaged predictions down to phoneme level first, which collapsed all
            # intra-phoneme variation to a single gradient value and allowed the
            # predictor to converge to a constant per phoneme (zero variance).
            max_frame_len = mel_mask_2d.size(1)
            # The model may expand to more frames than the target batch was padded
            # to (e.g. predicted 258 frames vs target 148).  Truncate to the
            # shorter of the two so shapes agree for element-wise MSE.
            pred_pitch_aligned = predicted_pitch[:, :max_frame_len]
            pitch_targets_frame = vectorized_expand_tokens(
                pitch_targets, phoneme_durations, max_len=max_frame_len
            )
            loss_pitch_unreduced = criterion_pitch(pred_pitch_aligned, pitch_targets_frame)
            pitch_valid = mel_mask_2d & torch.isfinite(loss_pitch_unreduced)
        else:
            # Predictions already at phoneme level — compare directly.
            loss_pitch_unreduced = criterion_pitch(predicted_pitch, pitch_targets)
            pitch_valid = phoneme_mask_2d & torch.isfinite(loss_pitch_unreduced)
        if pitch_valid.any():
            loss_pitch = loss_pitch_unreduced[pitch_valid].mean()

    loss_energy = torch.tensor(0.0, device=device)
    if predicted_energy is not None and energy_targets is not None and criterion_energy is not None:
        if predicted_energy.dim() == 2 and predicted_energy.size(1) != phoneme_durations.size(1):
            # Same frame-level expansion for energy — mirrors the pitch above.
            max_frame_len = mel_mask_2d.size(1)
            # Truncate in case the model expanded to more frames than the target.
            pred_energy_aligned = predicted_energy[:, :max_frame_len]
            energy_targets_frame = vectorized_expand_tokens(
                energy_targets, phoneme_durations, max_len=max_frame_len
            )
            loss_energy_unreduced = criterion_energy(pred_energy_aligned, energy_targets_frame)
            energy_valid = mel_mask_2d & torch.isfinite(loss_energy_unreduced)
        else:
            # Predictions already at phoneme level — compare directly.
            loss_energy_unreduced = criterion_energy(predicted_energy, energy_targets)
            energy_valid = phoneme_mask_2d & torch.isfinite(loss_energy_unreduced)
        if energy_valid.any():
            loss_energy = loss_energy_unreduced[energy_valid].mean()

    variance_diverged = False
    if loss_pitch > 10.0 or loss_energy > 10.0:
        logger.warning(f"⚠️  Variance predictor divergence detected - pitch: {loss_pitch:.2f}, energy: {loss_energy:.2f}")

        if predicted_pitch is not None and pitch_targets is not None:
            pred_min, pred_max = predicted_pitch.min().item(), predicted_pitch.max().item()
            targ_min, targ_max = pitch_targets.min().item(), pitch_targets.max().item()
            logger.warning(f"   Pitch predictions: [{pred_min:.3f}, {pred_max:.3f}]")
            logger.warning(f"   Pitch targets: [{targ_min:.3f}, {targ_max:.3f}]")

        if predicted_energy is not None and energy_targets is not None:
            pred_min, pred_max = predicted_energy.min().item(), predicted_energy.max().item()
            targ_min, targ_max = energy_targets.min().item(), energy_targets.max().item()
            logger.warning(f"   Energy predictions: [{pred_min:.3f}, {pred_max:.3f}]")
            logger.warning(f"   Energy targets: [{targ_min:.3f}, {targ_max:.3f}]")

        logger.warning("   Auto-recovery: Resetting variance predictor weights and reducing loss contribution")

        va = getattr(model, 'variance_adaptor', None)
        if va is not None:
            if hasattr(va, 'pitch_predictor') and va.pitch_predictor is not None:
                va.pitch_predictor._init_weights()
            if hasattr(va, 'energy_predictor') and va.energy_predictor is not None:
                va.energy_predictor._init_weights()

        loss_pitch = torch.tensor(0.0, device=device)
        loss_energy = torch.tensor(0.0, device=device)
        variance_diverged = True

    loss_mel = torch.clamp(loss_mel, max=100.0)
    loss_duration = torch.clamp(loss_duration, max=100.0)
    loss_stop_token = torch.clamp(loss_stop_token, max=100.0)
    if not variance_diverged:
        loss_pitch = torch.clamp(loss_pitch, max=10.0)
        loss_energy = torch.clamp(loss_energy, max=10.0)

    total_loss = (
        loss_mel
        + loss_duration * config.duration_loss_weight
        + loss_stop_token * config.stop_token_loss_weight
        + loss_pitch * getattr(config, 'pitch_loss_weight', 0.1)
        + loss_energy * getattr(config, 'energy_loss_weight', 0.1)
    )

    if not torch.isfinite(total_loss):
        logger.warning(
            f"Non-finite total_loss detected! "
            f"mel={loss_mel:.2f}, dur={loss_duration:.2f}, stop={loss_stop_token:.2f}, "
            f"pitch={loss_pitch:.2f}, energy={loss_energy:.2f}"
        )

    return total_loss, loss_mel, loss_duration, loss_stop_token, loss_pitch, loss_energy
