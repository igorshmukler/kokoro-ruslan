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
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Pass through to underlying variance adaptor
        return self.variance_adaptor(
            text_encoded,
            mask=mask,
            pitch_target=pitch_target,
            energy_target=energy_target,
            duration_target=duration_target,
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
        if predicted_log_durations.dim() > 1 and predicted_log_durations.size(-1) == 1:
            predicted_log_durations = predicted_log_durations.squeeze(-1)

        # Determine durations to use for length regulation
        if duration_target is not None:
            durations_for_length = duration_target.long()
        else:
            # inference path: convert predicted log durations to integer durations
            durations_for_length = torch.clamp(torch.exp(predicted_log_durations), min=1.0).long()

        # Expand encoder outputs to frame level
        expanded_encoder_outputs, frame_mask = self.length_regulate_fn(text_encoded, durations_for_length, mask)

        # No pitch/energy predictions in the simple adaptor
        predicted_pitch = None
        predicted_energy = None

        return expanded_encoder_outputs, predicted_log_durations, predicted_pitch, predicted_energy, frame_mask
