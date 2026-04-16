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
            # predicted_pitch is frame-level.
            max_frame_len = mel_mask_2d.size(1)
            pred_pitch_aligned = predicted_pitch[:, :max_frame_len]
            if pitch_targets.size(1) == phoneme_durations.size(1):
                # targets are phoneme-level (legacy path) → expand to frame-level.
                # This path is kept for backward compatibility only; the normal
                # training path now passes frame-level targets directly.
                pitch_targets_frame = vectorized_expand_tokens(
                    pitch_targets, phoneme_durations, max_len=max_frame_len
                )
            else:
                # targets are already frame-level → truncate to align lengths.
                # This is the normal path: frame-level targets passed directly from
                # the dataset, avoiding the phoneme-averaging that caused f0_RMSE to
                # freeze (frame→phoneme mean→re-expand = phoneme-constant targets).
                pitch_targets_frame = pitch_targets[:, :max_frame_len]
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
            # predicted_energy is frame-level.
            max_frame_len = mel_mask_2d.size(1)
            pred_energy_aligned = predicted_energy[:, :max_frame_len]
            if energy_targets.size(1) == phoneme_durations.size(1):
                # targets are phoneme-level (legacy path) → expand to frame-level.
                energy_targets_frame = vectorized_expand_tokens(
                    energy_targets, phoneme_durations, max_len=max_frame_len
                )
            else:
                # targets are already frame-level → truncate to align lengths.
                energy_targets_frame = energy_targets[:, :max_frame_len]
            loss_energy_unreduced = criterion_energy(pred_energy_aligned, energy_targets_frame)
            energy_valid = mel_mask_2d & torch.isfinite(loss_energy_unreduced)
        else:
            # Predictions already at phoneme level — compare directly.
            loss_energy_unreduced = criterion_energy(predicted_energy, energy_targets)
            energy_valid = phoneme_mask_2d & torch.isfinite(loss_energy_unreduced)
        if energy_valid.any():
            loss_energy = loss_energy_unreduced[energy_valid].mean()

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

        # Clamp losses to a safe ceiling rather than zeroing them out or
        # re-initialising weights mid-forward-pass.
        #
        # The previous approach called `_init_weights()` here — after the forward
        # pass but before `.backward()` — which:
        #   1. Randomly re-initialised the predictor weights while the corrupt
        #      activations from those weights were still in the autograd graph,
        #      so the computation graph referred to the OLD weights but .backward()
        #      would update the NEW (reset) weights — a silent state mismatch.
        #   2. Zeroed the pitch/energy loss, giving the variance predictor zero
        #      gradient signal for that batch, preventing it from learning to
        #      recover from the diverged state.
        #   3. If triggered on consecutive batches the predictor oscillated
        #      between random reinit and divergence with no chance to stabilise.
        #
        # Instead: hard-clamp the loss contribution (same ceiling as the normal
        # path, already applied below) so the gradient magnitude is bounded but
        # non-zero.  The predictor receives a corrective signal and can recover
        # on its own.  If it does not recover within a few batches the finite-loss
        # guard in _execute_training_step will skip the batch entirely.
        logger.warning("   Clamping variance loss contribution to 10.0 (no weight reset)")

    loss_mel = torch.clamp(loss_mel, max=100.0)
    loss_duration = torch.clamp(loss_duration, max=100.0)
    loss_stop_token = torch.clamp(loss_stop_token, max=100.0)
    loss_pitch = torch.clamp(loss_pitch, max=10.0)
    loss_energy = torch.clamp(loss_energy, max=10.0)

    total_loss = (
        loss_mel
        + loss_duration * config.duration_loss_weight
        + loss_stop_token * config.stop_token_loss_weight
        + loss_pitch * getattr(config, 'pitch_loss_weight', 1.0)
        + loss_energy * getattr(config, 'energy_loss_weight', 1.0)
    )

    if not torch.isfinite(total_loss):
        logger.warning(
            f"Non-finite total_loss detected! "
            f"mel={loss_mel:.2f}, dur={loss_duration:.2f}, stop={loss_stop_token:.2f}, "
            f"pitch={loss_pitch:.2f}, energy={loss_energy:.2f}"
        )

    return total_loss, loss_mel, loss_duration, loss_stop_token, loss_pitch, loss_energy
