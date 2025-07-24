import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch.utils.checkpoint import checkpoint
import logging
import torch.profiler # Import profiler

from positional_encoding import PositionalEncoding
from transformers import TransformerDecoder, TransformerEncoderBlock

logger = logging.getLogger(__name__)


class KokoroModel(nn.Module):
    """
    Fixed Kokoro-style model architecture with proper tensor dimension handling.
    Optimized for MPS (Metal Performance Shaders) acceleration
    """

    def __init__(self, vocab_size: int, mel_dim: int = 80, hidden_dim: int = 512,
                 n_encoder_layers: int = 6, n_heads: int = 8, encoder_ff_dim: int = 2048,
                 encoder_dropout: float = 0.1, n_decoder_layers: int = 6, decoder_ff_dim: int = 2048,
                 max_decoder_seq_len: int = 5000):
        """
        Initialize the Kokoro model with Transformer encoder and decoder
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        self.max_decoder_seq_len = max_decoder_seq_len

        # Text encoder: Embedding + Positional Encoding + Stack of Transformer Blocks
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_positional_encoding = PositionalEncoding(
            hidden_dim, dropout=encoder_dropout, max_len=max_decoder_seq_len
        )

        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, n_heads, encoder_ff_dim, encoder_dropout)
            for _ in range(n_encoder_layers)
        ])

        # Duration Predictor
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(encoder_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(encoder_dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Mel feature projection to match hidden dimension for decoder input
        self.mel_projection_in = nn.Linear(mel_dim, hidden_dim)

        self.decoder = TransformerDecoder(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=decoder_ff_dim,
            dropout=encoder_dropout,
            num_layers=n_decoder_layers
        )

        # Output projection for Mel Spectrogram
        self.mel_projection_out = nn.Linear(hidden_dim, mel_dim)

        # End-of-Speech (Stop Token) Predictor
        self.stop_token_predictor = nn.Linear(hidden_dim, 1)

        # General dropout
        self.dropout = nn.Dropout(encoder_dropout)

    def encode_text(self, phoneme_indices: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text with proper scaling and positional encoding"""
        with torch.profiler.record_function("encode_text"):
            text_emb = self.text_embedding(phoneme_indices) * (self.hidden_dim ** 0.5)
            text_emb = self.encoder_positional_encoding(text_emb, seq_offset=0)

            x = text_emb
            for i, layer in enumerate(self.transformer_encoder_layers):
                with torch.profiler.record_function(f"encoder_layer_{i}"):
                    if self.training:
                        if mask is not None:
                            # Ensure mask is boolean for checkpoint
                            x = checkpoint(layer, x, src_key_padding_mask=mask.to(torch.bool), use_reentrant=False)
                        else:
                            x = checkpoint(layer, x, use_reentrant=False)
                    else:
                        if mask is not None:
                            x = layer(x, src_key_padding_mask=mask.to(torch.bool))
                        else:
                            x = layer(x)
            return x

    def _predict_durations(self, text_encoded: torch.Tensor) -> torch.Tensor:
        """
        Predicts log durations for each phoneme.
        """
        with torch.profiler.record_function("predict_durations"):
            log_durations = self.duration_predictor(text_encoded).squeeze(-1)
            return log_durations

    def _length_regulate(self, encoder_outputs, durations, text_padding_mask):
        """
        Fixed length regulation with proper tensor dimension handling.

        Args:
            encoder_outputs (Tensor): Encoder outputs (B, L_text, D).
            durations (Tensor): Predicted durations (B, L_text).
            text_padding_mask (Tensor): Boolean mask for text padding (B, L_text).
                                        True for padded positions, False for actual content.
        Returns:
            expanded_encoder_outputs (Tensor): Expanded outputs (B, L_mel, D).
            encoder_output_padding_mask (Tensor): Padding mask for expanded outputs.
        """
        with torch.profiler.record_function("length_regulate"):
            batch_size, max_text_len, hidden_dim = encoder_outputs.shape
            device = encoder_outputs.device

            # Ensure durations are positive and properly clamped
            durations = torch.clamp(durations, min=1.0)

            expanded_encoder_outputs_list = []
            encoder_output_padding_mask_list = []
            max_expanded_len = 0

            for i in range(batch_size):
                current_encoder_output = encoder_outputs[i]  # (L_text, D)
                current_durations = durations[i]             # (L_text,)
                # Ensure text_padding_mask is boolean here as well
                current_text_padding_mask = text_padding_mask[i].to(torch.bool)  # (L_text,)

                # Select non-padded elements
                non_padded_indices = ~current_text_padding_mask

                if not torch.any(non_padded_indices):
                    # Handle case where entire sequence is padding
                    logger.warning(f"Batch {i}: All tokens are padding, creating empty sequence")
                    expanded_encoder_output = torch.empty(0, hidden_dim, device=device)
                    expanded_padding_mask = torch.empty(0, dtype=torch.bool, device=device)
                else:
                    filtered_encoder_output = current_encoder_output[non_padded_indices]
                    filtered_durations = current_durations[non_padded_indices]

                    # Ensure durations are integers and positive
                    filtered_durations = torch.clamp(filtered_durations, min=1).long()

                    # Expand encoder outputs
                    try:
                        expanded_encoder_output = torch.repeat_interleave(
                            filtered_encoder_output, filtered_durations, dim=0
                        )
                        # The mask for the expanded output should initially be all False (not padded)
                        expanded_padding_mask = torch.zeros(
                            expanded_encoder_output.shape[0], dtype=torch.bool, device=device
                        )
                    except Exception as e:
                        logger.error(f"Error in repeat_interleave for batch {i}: {e}")
                        logger.error(f"filtered_encoder_output shape: {filtered_encoder_output.shape}")
                        logger.error(f"filtered_durations: {filtered_durations}")
                        # Fallback: create a minimal valid output
                        expanded_encoder_output = torch.empty(0, hidden_dim, device=device)
                        expanded_padding_mask = torch.empty(0, dtype=torch.bool, device=device)

                expanded_encoder_outputs_list.append(expanded_encoder_output)
                encoder_output_padding_mask_list.append(expanded_padding_mask)
                max_expanded_len = max(max_expanded_len, expanded_encoder_output.shape[0])

            # Handle empty batch case gracefully if all sequences are empty after expansion
            if max_expanded_len == 0:
                logger.warning("All sequences resulted in empty expansion, creating dummy output for stability.")
                max_expanded_len = 1 # Set a minimum length to avoid issues with torch.stack on empty list
                # Create dummy tensors for the batch to prevent crashing
                dummy_output = torch.zeros(1, hidden_dim, device=device, dtype=encoder_outputs.dtype)
                dummy_mask = torch.ones(1, dtype=torch.bool, device=device)
                expanded_encoder_outputs_list = [dummy_output] * batch_size
                encoder_output_padding_mask_list = [dummy_mask] * batch_size


            # Pad all sequences to the same length
            final_expanded_outputs = []
            final_padding_masks = []

            for i in range(batch_size):
                current_output = expanded_encoder_outputs_list[i]
                current_mask = encoder_output_padding_mask_list[i]
                current_len = current_output.shape[0]

                padding_needed = max_expanded_len - current_len

                if padding_needed > 0:
                    # Pad the output with zeros
                    padding_tensor = torch.zeros(
                        padding_needed, hidden_dim, device=device, dtype=current_output.dtype
                    )
                    current_output = torch.cat([current_output, padding_tensor], dim=0)

                    # Pad the mask with True (indicating padded positions)
                    padding_mask_fill = torch.ones(padding_needed, dtype=torch.bool, device=device)
                    current_mask = torch.cat([current_mask, padding_mask_fill], dim=0)

                final_expanded_outputs.append(current_output)
                final_padding_masks.append(current_mask)

            # Stack into batch tensors
            expanded_encoder_outputs = torch.stack(final_expanded_outputs, dim=0)
            encoder_output_padding_mask = torch.stack(final_padding_masks, dim=0)

            logger.debug(f"Length regulation completed: {encoder_outputs.shape} -> {expanded_encoder_outputs.shape}")

            return expanded_encoder_outputs, encoder_output_padding_mask

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, used for masked self-attention."""
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward_training(
        self,
        phoneme_indices: torch.Tensor,
        mel_specs: torch.Tensor,
        phoneme_durations: torch.Tensor,
        stop_token_targets: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        mel_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fixed training forward pass with proper error handling.
        """
        batch_size, mel_seq_len = mel_specs.shape[0], mel_specs.shape[1]
        device = mel_specs.device

        # Ensure text_padding_mask is provided and converted to boolean
        if text_padding_mask is None:
            text_padding_mask = (phoneme_indices == 0).to(torch.bool)
        else:
            text_padding_mask = text_padding_mask.to(torch.bool)

        try:
            # Encode text using Transformer encoder
            text_encoded = self.encode_text(phoneme_indices, mask=text_padding_mask)

            # Predict durations
            predicted_log_durations = self._predict_durations(text_encoded)

            # Length regulate text_encoded using ground-truth durations
            expanded_encoder_outputs, encoder_output_padding_mask = self._length_regulate(
                text_encoded, phoneme_durations.float(), text_padding_mask
            )

            # Ensure expanded_encoder_outputs sequence length matches mel_seq_len
            with torch.profiler.record_function("mel_length_adjust"):
                current_expanded_len = expanded_encoder_outputs.shape[1]
                if current_expanded_len != mel_seq_len:
                    # logger.warning(
                    #     f"Length mismatch: expanded_encoder_outputs length ({current_expanded_len}) "
                    #     f"does not match mel_specs length ({mel_seq_len}). Adjusting."
                    # )
                    if current_expanded_len > mel_seq_len:
                        # Truncate expanded_encoder_outputs
                        expanded_encoder_outputs = expanded_encoder_outputs[:, :mel_seq_len, :]
                        encoder_output_padding_mask = encoder_output_padding_mask[:, :mel_seq_len]
                    else:
                        # Pad expanded_encoder_outputs
                        pad_len = mel_seq_len - current_expanded_len
                        padding_tensor = torch.zeros(
                            batch_size, pad_len, self.hidden_dim,
                            device=device, dtype=expanded_encoder_outputs.dtype
                        )
                        expanded_encoder_outputs = torch.cat(
                            [expanded_encoder_outputs, padding_tensor], dim=1
                        )
                        padding_mask_tensor = torch.ones(
                            batch_size, pad_len, dtype=torch.bool, device=device
                        )
                        encoder_output_padding_mask = torch.cat(
                            [encoder_output_padding_mask, padding_mask_tensor], dim=1
                        )

            with torch.profiler.record_function("decoder_input_prep"):
                # Prepare decoder input (teacher forcing: shift right)
                decoder_input_mels = F.pad(mel_specs[:, :-1, :], (0, 0, 1, 0), "constant", 0.0)
                decoder_input_projected = self.mel_projection_in(decoder_input_mels)

                # Apply positional encoding
                decoder_input_projected_with_pe = self.encoder_positional_encoding(
                    decoder_input_projected, seq_offset=0
                )

                # Generate causal mask for decoder self-attention
                tgt_mask = self._generate_square_subsequent_mask(mel_seq_len, device)

                # Ensure mel_padding_mask is boolean if provided
                if mel_padding_mask is not None:
                    mel_padding_mask = mel_padding_mask.to(torch.bool)

            with torch.profiler.record_function("transformer_decoder_forward"):
                # Pass through Transformer Decoder
                decoder_outputs = self.decoder(
                    tgt=decoder_input_projected_with_pe,
                    memory=expanded_encoder_outputs,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=encoder_output_padding_mask, # This is already boolean
                    tgt_key_padding_mask=mel_padding_mask # This is now boolean
                )

            with torch.profiler.record_function("output_projections"):
                # Project decoder outputs back to mel dimension
                predicted_mel_frames = self.mel_projection_out(decoder_outputs)

                # Predict stop tokens from decoder outputs
                predicted_stop_logits = self.stop_token_predictor(decoder_outputs).squeeze(-1)

            return predicted_mel_frames, predicted_log_durations, predicted_stop_logits

        except Exception as e:
            logger.error(f"Error in forward_training: {e}")
            logger.error(f"Input shapes - phoneme_indices: {phoneme_indices.shape}, mel_specs: {mel_specs.shape}")
            logger.error(f"phoneme_durations: {phoneme_durations.shape}")
            raise e

    def forward_inference(self, phoneme_indices: torch.Tensor, max_len: int = 5000,
                         stop_threshold: float = 0.5,
                         text_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Improved inference with better error handling and stopping conditions.
        """
        if phoneme_indices.size(0) > 1:
            logger.warning("Inference with stop token is most reliable with batch_size=1.")

        batch_size = phoneme_indices.size(0)
        device = phoneme_indices.device

        self.eval()

        with torch.no_grad():
            try:
                # Ensure text_padding_mask is correctly defined and boolean
                if text_padding_mask is None:
                    text_padding_mask = (phoneme_indices == 0).to(torch.bool)
                else:
                    text_padding_mask = text_padding_mask.to(torch.bool)


                with torch.profiler.record_function("inference_encode_text"):
                    # Encode text
                    text_encoded = self.encode_text(phoneme_indices, mask=text_padding_mask)

                with torch.profiler.record_function("inference_predict_durations"):
                    # Predict durations
                    predicted_log_durations = self._predict_durations(text_encoded)
                    durations_for_length_regulate = torch.exp(predicted_log_durations)
                    durations_for_length_regulate = torch.clamp(durations_for_length_regulate, min=1.0).long()

                with torch.profiler.record_function("inference_length_regulate"):
                    # Length regulate
                    expanded_encoder_outputs, encoder_output_padding_mask = self._length_regulate(
                        text_encoded, durations_for_length_regulate, text_padding_mask
                    )

                expected_length = expanded_encoder_outputs.shape[1]
                logger.info(f"Starting inference with expanded encoder outputs shape: {expanded_encoder_outputs.shape}")

                # Calculate reasonable bounds for generation
                min_expected_length = max(10, expected_length // 3)
                max_expected_length = min(max_len, expected_length * 2, 800)

                logger.info(f"Generation bounds: min={min_expected_length}, max={max_expected_length}")

                # Initialize generation
                generated_mels = []
                decoder_input_mel = torch.zeros(batch_size, 1, self.mel_dim, device=device)

                # Generation loop with improved stopping
                for t in range(max_expected_length):
                    with torch.profiler.record_function(f"inference_decode_step_{t}"):
                        try:
                            # Project current mel frame
                            mel_projected_t = self.mel_projection_in(decoder_input_mel)

                            # Create input sequence up to current position
                            if t == 0:
                                decoder_input_seq = mel_projected_t
                            else:
                                previous_mels = torch.cat(generated_mels, dim=1)
                                previous_projected = self.mel_projection_in(previous_mels)
                                decoder_input_seq = torch.cat([previous_projected, mel_projected_t], dim=1)

                            # Apply positional encoding
                            decoder_input_seq_with_pe = self.encoder_positional_encoding(
                                decoder_input_seq, seq_offset=0
                            )

                            # Generate causal mask
                            current_seq_len = decoder_input_seq.shape[1]
                            tgt_mask = self._generate_square_subsequent_mask(current_seq_len, device)

                            # Decoder forward pass
                            decoder_outputs = self.decoder(
                                tgt=decoder_input_seq_with_pe,
                                memory=expanded_encoder_outputs,
                                tgt_mask=tgt_mask,
                                memory_key_padding_mask=encoder_output_padding_mask,
                                tgt_key_padding_mask=None
                            )

                            # Take the last frame's output
                            decoder_out_t = decoder_outputs[:, -1:, :]

                            # Generate mel frame
                            mel_pred_t = self.mel_projection_out(decoder_out_t)
                            generated_mels.append(mel_pred_t)

                            # Check stop condition
                            stop_token_logit_t = self.stop_token_predictor(decoder_out_t)
                            stop_probability = torch.sigmoid(stop_token_logit_t).item()

                            # Multiple stopping conditions
                            if t >= min_expected_length:
                                if stop_probability > stop_threshold:
                                    logger.info(f"Stopping at frame {t} (stop_prob: {stop_probability:.4f})")
                                    break

                                # Stop if we've reached expected length and have reasonable stop probability
                                if t >= expected_length and stop_probability > 0.1:
                                    logger.info(f"Stopping at expected length {t} (stop_prob: {stop_probability:.4f})")
                                    break

                            # Use prediction as next input
                            decoder_input_mel = mel_pred_t

                            if t % 50 == 0:
                                logger.debug(f"Generated frame {t}, stop_prob: {stop_probability:.6f}")

                        except Exception as e:
                            logger.error(f"Error at generation step {t}: {e}")
                            break

                # Concatenate all generated frames
                if generated_mels:
                    mel_output = torch.cat(generated_mels, dim=1)
                    logger.info(f"Generated {mel_output.shape[1]} mel frames")
                else:
                    logger.warning("No mel frames were generated.")
                    mel_output = torch.empty(batch_size, 0, self.mel_dim, device=device)

                return mel_output

            except Exception as e:
                logger.error(f"Error in forward_inference: {e}")
                return torch.empty(batch_size, 0, self.mel_dim, device=device)

    def forward(
        self,
        phoneme_indices: torch.Tensor,
        mel_specs: Optional[torch.Tensor] = None,
        phoneme_durations: Optional[torch.Tensor] = None,
        stop_token_targets: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
        mel_padding_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Main forward pass that dispatches to training or inference mode.
        """
        if mel_specs is not None:
            self.train()
            if phoneme_durations is None or stop_token_targets is None:
                raise ValueError("phoneme_durations and stop_token_targets must be provided for training mode.")
            return self.forward_training(
                phoneme_indices, mel_specs, phoneme_durations,
                stop_token_targets, text_padding_mask, mel_padding_mask
            )
        else:
            self.eval()
            return self.forward_inference(
                phoneme_indices, max_len=self.max_decoder_seq_len,
                text_padding_mask=text_padding_mask
            )

    def get_model_info(self) -> dict:
        """Get model information and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'vocab_size': self.vocab_size,
            'mel_dim': self.mel_dim,
            'hidden_dim': self.hidden_dim,
            'n_encoder_layers': len(self.transformer_encoder_layers),
            'n_decoder_layers': len(self.decoder.layers),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }
