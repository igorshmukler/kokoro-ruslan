import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable

from kokoro.utils.lengths import length_regulate


class BaseDurationAdaptor(nn.Module):
    """Unified interface for duration/pitch/energy adaptors.

    All implementations must return a tuple:
      (adapted_encoder_output, predicted_log_durations, predicted_pitch, predicted_energy, frame_mask)
    """

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        raise NotImplementedError()


class VarianceAdaptorWrapper(BaseDurationAdaptor):
    """Wrap an existing `VarianceAdaptor` so it conforms to the unified interface.

    The wrapped adaptor is expected to accept the same keyword args used below.
    """

    def __init__(self, variance_adaptor: nn.Module):
        super().__init__()
        self.variance_adaptor = variance_adaptor

    def forward(self,
                text_encoded: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pitch_target: Optional[torch.Tensor] = None,
                energy_target: Optional[torch.Tensor] = None,
                duration_target: Optional[torch.Tensor] = None,
                inference: bool = False,
                pitch_target_is_frame_level: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Pass through to underlying variance adaptor, forwarding the
        # frame-level flag so it reaches VarianceAdaptor.forward.
        return self.variance_adaptor(
            text_encoded,
            mask=mask,
            pitch_target=pitch_target,
            energy_target=energy_target,
            duration_target=duration_target,
            pitch_target_is_frame_level=pitch_target_is_frame_level,
        )


class SimpleDurationAdaptor(BaseDurationAdaptor):
    """Fallback adaptor that uses a duration predictor and `length_regulate`.

    This adaptor keeps the same return signature as the `VarianceAdaptor` but
    produces `predicted_pitch` and `predicted_energy` as `None`.

    Parameters:
      duration_predictor_fn: a callable that accepts `text_encoded` and returns
        `predicted_log_durations` (tensor shape (..., 1) or (...,) )
      length_regulate_fn: function to expand encoder outputs given durations
    """

    def __init__(self, duration_predictor_fn: Callable[[torch.Tensor], torch.Tensor],
                 length_regulate_fn: Callable = length_regulate):
        super().__init__()
        self.duration_predictor_fn = duration_predictor_fn
        self.length_regulate_fn = length_regulate_fn

    def forward(self,
                text_encoded: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pitch_target: Optional[torch.Tensor] = None,
                energy_target: Optional[torch.Tensor] = None,
                duration_target: Optional[torch.Tensor] = None,
                inference: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # predicted_log_durations: (batch, phonemes)
        predicted_log_durations = self.duration_predictor_fn(text_encoded)
        # Only collapse an explicit trailing feature-channel of size 1 produced by
        # predictors that output (B, T, 1) — do NOT squeeze (B, 1) which is the
        # legitimate T=1 (single-phoneme) case.
        if predicted_log_durations.dim() == 3 and predicted_log_durations.size(-1) == 1:
            predicted_log_durations = predicted_log_durations.squeeze(-1)

        # Determine durations to use for length regulation
        if duration_target is not None:
            durations_for_length = duration_target.long()
        else:
            # Inference: the duration predictor is trained against log1p targets
            # (losses.py: target = log(d + 1)), so the correct inverse is expm1:
            #   expm1(log1p(d)) = d
            # Using exp() (the previous bug) would yield (d + 1), biasing every
            # phoneme duration upward by one frame.
            durations_for_length = torch.clamp(torch.expm1(predicted_log_durations), min=1.0).round().long()

        # Expand encoder outputs to frame level
        expanded_encoder_outputs, frame_mask = self.length_regulate_fn(text_encoded, durations_for_length, mask)

        # No pitch/energy predictions in the simple adaptor
        predicted_pitch = None
        predicted_energy = None

        return expanded_encoder_outputs, predicted_log_durations, predicted_pitch, predicted_energy, frame_mask
