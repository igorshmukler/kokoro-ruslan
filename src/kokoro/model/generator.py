from __future__ import annotations
import time
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kokoro.model.model import KokoroModel

import torch

logger = logging.getLogger(__name__)


class KokoroGenerator:
    """Encapsulates autoregressive generation for `KokoroModel`.

    Handles precompute, stepping, stopping criteria, energy-based fallback,
    and local KV cache management.
    """
    def __init__(self, model: "KokoroModel"):
        self.model = model
        self.decoder = model.decoder

    def generate(self,
                 expanded_encoder_outputs: torch.Tensor,
                 encoder_output_padding_mask: torch.Tensor,
                 expected_length: int,
                 min_expected_length: int,
                 max_expected_length: int,
                 stop_threshold: float,
                 post_expected_stop_threshold: float) -> torch.Tensor:
        batch_size = expanded_encoder_outputs.size(0)
        device = expanded_encoder_outputs.device

        generated_mels = []
        decoder_input_mel = torch.zeros(batch_size, 1, self.model.mel_dim, device=device)

        # Prepare decoder cross-attention caches for fast autoregressive decoding
        self.decoder.precompute_cross_attention_kv(expanded_encoder_outputs)
        self_attn_kv_caches = [() for _ in range(len(self.decoder.layers))]

        generation_start_time = time.time()

        for t in range(max_expected_length):
            step_start_time = time.time()

            with torch.profiler.record_function(f"inference_decode_step_{t}"):
                try:
                    mel_projected_t_with_pe = self.model._project_mel_frame(decoder_input_mel, seq_offset=t)

                    decoder_outputs, self_attn_kv_caches = self.decoder(
                        tgt=mel_projected_t_with_pe,
                        memory=expanded_encoder_outputs,
                        tgt_mask=None,
                        memory_key_padding_mask=encoder_output_padding_mask,
                        tgt_key_padding_mask=None,
                        self_attn_kv_caches=self_attn_kv_caches,
                    )

                    decoder_out_t = decoder_outputs[:, -1:, :].clone()
                    del decoder_outputs

                    mel_pred_t, stop_token_logit_t = self.model._project_decoder_outputs(decoder_out_t)
                    generated_mels.append(mel_pred_t)
                    del decoder_out_t

                    stop_probability = torch.sigmoid(stop_token_logit_t).mean().item()
                    del stop_token_logit_t

                    if t >= min_expected_length:
                        effective_stop_threshold = (
                            stop_threshold if t < expected_length
                            else min(stop_threshold, post_expected_stop_threshold)
                        )
                        if stop_probability > effective_stop_threshold:
                            logger.info(
                                f"Stopping at frame {t} (stop_prob: {stop_probability:.4f}, threshold: {effective_stop_threshold:.4f})"
                            )
                            break

                        if len(generated_mels) >= 30:
                            recent_frames = torch.cat(generated_mels[-30:], dim=1)
                            recent_energy = recent_frames.mean().item()
                            if recent_energy < -9.5:
                                logger.info(
                                    f"Energy-based early stop at frame {t} (recent mean energy: {recent_energy:.3f} < -9.5)"
                                )
                                break

                    decoder_input_mel = mel_pred_t

                    if self.model.enable_profiling and t % 50 == 0:
                        step_time = (time.time() - step_start_time) * 1000
                        logger.debug(
                            f"Generated frame {t}, stop_prob: {stop_probability:.6f}, step_time: {step_time:.2f}ms"
                        )
                        self.model._log_memory(f"inference_step_{t}")

                except Exception as e:
                    logger.error(f"Error at generation step {t}: {e}")
                    if self.model.enable_profiling:
                        logger.error(f"GPU Memory at step {t}: {self.model.profiler.get_memory_summary()}")
                    break

        generation_time = time.time() - generation_start_time

        # Free expanded encoder tensors held for generation
        try:
            del expanded_encoder_outputs, encoder_output_padding_mask
        except Exception:
            pass

        if generated_mels:
            mel_output = torch.cat(generated_mels, dim=1)
            mel_output.clamp_(min=-11.5, max=2.0)

            logger.info(
                f"Final Mel range: {mel_output.min().item():.2f} to {mel_output.max().item():.2f}"
            )
            logger.info(
                f"Generated {mel_output.shape[1]} mel frames in {generation_time:.2f}s ({mel_output.shape[1]/generation_time:.1f} frames/s)"
            )
        else:
            logger.warning("No mel frames were generated.")
            mel_output = torch.empty(batch_size, 0, self.model.mel_dim, device=device)

        return mel_output
