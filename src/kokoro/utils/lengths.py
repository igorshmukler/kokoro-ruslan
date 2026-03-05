"""Length regulation and duration/energy averaging utilities.

This module centralizes token->frame expansion and frame->token averaging
to avoid duplicated implementations across the codebase.
"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

logger = logging.getLogger(__name__)


def vectorized_expand_tokens(
    tokens: torch.Tensor,
    durations: torch.Tensor,
    max_len: Optional[int] = None
) -> torch.Tensor:
    """Vectorized token expansion. CPU round-trip retained for MPS stability.

    Mirrors the previous implementations in `variance_predictor.py` and
    `model.py` but extracted for reuse.
    """
    device = tokens.device
    is_3d = tokens.dim() == 3
    B, L = tokens.shape[0], tokens.shape[1]

    tokens_cpu = tokens.detach().to('cpu')
    durations_cpu = durations.detach().to('cpu').long().clamp(min=0)

    lengths = durations_cpu.sum(dim=1)                # (B,)
    max_expanded = int(lengths.max().item())
    if max_len is not None:
        max_expanded = min(max_expanded, int(max_len))

    if max_expanded == 0:
        out_len = max(1, int(max_len)) if max_len is not None else 1
        if is_3d:
            return tokens_cpu.new_zeros(B, out_len, tokens_cpu.size(2)).to(device)
        return tokens_cpu.new_zeros(B, out_len).to(device)

    try:
        d_flat = durations_cpu.reshape(B * L)

        if is_3d:
            D = tokens_cpu.size(2)
            flat = tokens_cpu.reshape(B * L, D)
            expanded_flat = torch.repeat_interleave(flat, d_flat, dim=0)  # (total, D)
        else:
            flat = tokens_cpu.reshape(B * L)
            expanded_flat = torch.repeat_interleave(flat, d_flat, dim=0)  # (total,)

        # Build batch index for each expanded frame — no Python loop
        batch_ids = torch.arange(B).repeat_interleave(lengths)            # (total,)

        # Build within-batch frame index by subtracting each batch item's start offset
        batch_start_offsets = torch.zeros(B, dtype=torch.long)
        batch_start_offsets[1:] = lengths.cumsum(0)[:-1]
        per_frame_offset = batch_start_offsets.repeat_interleave(lengths)  # (total,)
        frame_ids = torch.arange(expanded_flat.size(0)) - per_frame_offset # (total,)

        # Mask frames that exceed max_expanded (happens when max_len clips)
        valid = frame_ids < max_expanded

        if is_3d:
            out_cpu = tokens_cpu.new_zeros(B, max_expanded, D)
            out_cpu[batch_ids[valid], frame_ids[valid]] = expanded_flat[valid]
        else:
            out_cpu = tokens_cpu.new_zeros(B, max_expanded)
            out_cpu[batch_ids[valid], frame_ids[valid]] = expanded_flat[valid]

        # Pad to exact max_len if needed
        if max_len is not None and out_cpu.size(1) < max_len:
            pad = max_len - out_cpu.size(1)
            out_cpu = F.pad(out_cpu, (0, 0, 0, pad) if is_3d else (0, pad))

        return out_cpu.to(device)

    except Exception as e:
        logger.warning(f"Vectorized expansion failed, using fallback: {e}")
        expanded = []
        for i in range(B):
            if durations_cpu[i].sum() == 0:
                z_shape = (1, tokens_cpu.size(2)) if is_3d else (1,)
                expanded.append(torch.zeros(z_shape, dtype=tokens_cpu.dtype))
            else:
                expanded.append(torch.repeat_interleave(tokens_cpu[i], durations_cpu[i], dim=0))
        output = torch.nn.utils.rnn.pad_sequence(expanded, batch_first=True).to(device)
        if max_len is not None:
            if output.size(1) < max_len:
                output = F.pad(output, (0, 0, 0, max_len - output.size(1)) if is_3d else (0, max_len - output.size(1)))
            else:
                output = output[:, :max_len] if not is_3d else output[:, :max_len, :]
        return output


class LengthRegulator(nn.Module):
    """Simple wrapper exposing the vectorized expansion as an nn.Module."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, durations: torch.Tensor, max_len: Optional[int] = None):
        return vectorized_expand_tokens(x, durations, max_len=max_len)


def length_regulate(encoder_outputs: torch.Tensor,
                    durations: torch.Tensor,
                    text_padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """CPU-friendly length regulation that mirrors the previous model logic.

    Returns (expanded_outputs, frame_mask)
    """
    batch_size, max_text_len, hidden_dim = encoder_outputs.shape
    device = encoder_outputs.device

    durations = torch.clamp(durations, min=1.0)

    lengths = []
    non_padded_masks = []
    for i in range(batch_size):
        npm = ~text_padding_mask[i].to(torch.bool)
        non_padded_masks.append(npm)
        if torch.any(npm):
            dur_i = durations[i][npm].clamp(min=1).long()
            lengths.append(int(dur_i.sum().item()))
        else:
            lengths.append(0)

    max_expanded_len = max(max(lengths, default=0), 1)

    out = torch.zeros(batch_size, max_expanded_len, hidden_dim, device=device, dtype=encoder_outputs.dtype)
    mask = torch.ones(batch_size, max_expanded_len, dtype=torch.bool, device=device)

    for i in range(batch_size):
        if lengths[i] == 0:
            logger.warning(f"Batch {i}: All tokens are padding, leaving row zeroed")
            continue

        filtered_enc = encoder_outputs[i][non_padded_masks[i]]
        filtered_dur = durations[i][non_padded_masks[i]].clamp(min=1).long()

        try:
            expanded = torch.repeat_interleave(filtered_enc, filtered_dur, dim=0)
            ln = expanded.shape[0]
            out[i, :ln] = expanded
            mask[i, :ln] = False
            del expanded
        except Exception as e:
            logger.error(f"Error in repeat_interleave for batch {i}: {e}")

    return out, mask


def average_by_duration(values: torch.Tensor,
                        durations: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Vectorized average of frame-level values to token-level using durations."""
    batch_size, num_phonemes = durations.shape
    device = durations.device

    durations_clamped = durations.long().clamp(min=0)

    ends = durations_clamped.cumsum(dim=1)                          # (B, P)
    starts = ends - durations_clamped                               # (B, P)
    max_frames = values.size(1)

    starts = starts.clamp(max=max_frames - 1)
    ends = ends.clamp(max=max_frames)

    phoneme_range = torch.arange(num_phonemes, device=device)

    frame_labels_float = torch.zeros(batch_size, max_frames + 1, device=device)
    frame_labels_float.scatter_add_(
        1,
        starts.clamp(max=max_frames),
        phoneme_range.unsqueeze(0).expand(batch_size, -1).float()
    )
    frame_labels_float.scatter_add_(
        1,
        ends.clamp(max=max_frames),
        -phoneme_range.unsqueeze(0).expand(batch_size, -1).float()
    )

    frame_to_phoneme = frame_labels_float[:, :-1].cumsum(dim=1).long()

    phoneme_sums = torch.zeros(batch_size, num_phonemes, device=device, dtype=values.dtype)
    phoneme_counts = torch.zeros(batch_size, num_phonemes, device=device, dtype=values.dtype)

    valid_frame_mask = frame_to_phoneme < num_phonemes
    safe_labels = frame_to_phoneme.clamp(max=num_phonemes - 1)

    phoneme_sums.scatter_add_(1, safe_labels, values * valid_frame_mask.to(values.dtype))
    phoneme_counts.scatter_add_(1, safe_labels, valid_frame_mask.to(values.dtype))

    output = phoneme_sums / phoneme_counts.clamp(min=1.0)

    if mask is not None:
        output = output.masked_fill(mask.bool(), 0.0)
    else:
        output = output.masked_fill(durations_clamped == 0, 0.0)

    return output
