import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch.utils.checkpoint import checkpoint # Keep for gradient checkpointing
import logging

from positional_encoding import PositionalEncoding
from transformers import TransformerDecoder, TransformerDecoderBlock, TransformerEncoderBlock

logger = logging.getLogger(__name__)


class KokoroModel(nn.Module):
    """
    Simplified Kokoro-style model architecture.
    Optimized for MPS (Metal Performance Shaders) acceleration
    """

    def __init__(self, vocab_size: int, mel_dim: int = 80, hidden_dim: int = 512,
                 n_encoder_layers: int = 6, n_heads: int = 8, encoder_ff_dim: int = 2048,
                 encoder_dropout: float = 0.1, n_decoder_layers: int = 6, decoder_ff_dim: int = 2048,
                 max_seq_len: int = 5000): # Use a single max_seq_len for both
        """
        Initialize the Kokoro model with Transformer encoder and decoder
        
        Args:
            vocab_size: Size of the phoneme vocabulary
            mel_dim: Dimension of mel spectrogram features
            hidden_dim: Hidden dimension for internal layers (d_model for Transformer)
            n_encoder_layers: Number of Transformer encoder layers
            n_heads: Number of attention heads in MultiheadAttention
            encoder_ff_dim: Dimension of the feed-forward network in Encoder Transformer blocks
            encoder_dropout: Dropout rate for the Transformer encoder
            n_decoder_layers: Number of Transformer decoder layers
            decoder_ff_dim: Dimension of the feed-forward network in Decoder Transformer blocks
            max_seq_len: Maximum sequence length for positional encodings (both encoder and decoder).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim # This will now be d_model
        self.max_seq_len = max_seq_len # Store this for PE initialization

        # Text encoder: Embedding + Positional Encoding + Stack of Transformer Blocks
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_positional_encoding = PositionalEncoding(hidden_dim, dropout=encoder_dropout, max_len=self.max_seq_len)

        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, n_heads, encoder_ff_dim, encoder_dropout)
            for _ in range(n_encoder_layers)
        ])

        # Duration Predictor (remains the same)
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Predicts 1 duration value per token
        )

        # Mel feature projection to match hidden dimension for decoder input
        self.mel_projection_in = nn.Linear(mel_dim, hidden_dim)

        # FIX: Add a dedicated positional encoding for the decoder input
        self.decoder_positional_encoding = PositionalEncoding(hidden_dim, dropout=encoder_dropout, max_len=self.max_seq_len)

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

        # General dropout (can be used in different parts)
        self.dropout = nn.Dropout(encoder_dropout)

        # Postnet (Placeholder - ensure this is defined in your actual model if used)
        # If your model has a Postnet for refining mel spectrograms:
        # self.postnet = Postnet(mel_dim=mel_dim, hidden_dim=hidden_dim, ...)
        # For now, it's not defined, so the call in forward_inference is commented.


    def encode_text(self, phoneme_indices: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        text_emb = self.text_embedding(phoneme_indices) * self.hidden_dim ** 0.5
        # For the encoder, seq_offset is 0 as it's a fixed sequence from the beginning
        text_emb = self.encoder_positional_encoding(text_emb, seq_offset=0) 

        x = text_emb
        for layer in self.transformer_encoder_layers:
            if self.training: # Only use checkpointing during training
                if mask is not None:
                    x = checkpoint(layer, x, src_key_padding_mask=mask, use_reentrant=False)
                else:
                    x = checkpoint(layer, x, use_reentrant=False)
            else: # Inference path, no checkpointing
                if mask is not None:
                    x = layer(x, src_key_padding_mask=mask)
                else:
                    x = layer(x)
        return x

    def _predict_durations(self, text_encoded: torch.Tensor) -> torch.Tensor:
        """
        Predicts log durations for each phoneme.
        Args:
            text_encoded: Output from text encoder (batch_size, text_seq_len, hidden_dim)
        Returns:
            Log durations of shape (batch_size, text_seq_len)
        """
        log_durations = self.duration_predictor(text_encoded).squeeze(-1)
        return log_durations

    def _length_regulate(self, encoder_outputs, durations, text_padding_mask):
        """
        Expands encoder outputs based on predicted durations.
        Args:
            encoder_outputs (Tensor): Encoder outputs (B, L_text, D).
            durations (Tensor): Predicted durations (B, L_text) - these are expected to be positive integers.
            text_padding_mask (Tensor): Boolean mask for text padding (B, L_text).
                                        True for padded positions, False for actual content.
        Returns:
            expanded_encoder_outputs (Tensor): Expanded outputs (B, L_mel, D).
            encoder_output_padding_mask (Tensor): Padding mask for expanded outputs.
        """
        batch_size, max_text_len, hidden_dim = encoder_outputs.shape
        
        expanded_encoder_outputs_list = []
        encoder_output_padding_mask_list = []
        actual_expanded_lengths_per_item = [] # To store the actual length of each expanded item

        for i in range(batch_size):
            current_encoder_output = encoder_outputs[i] # (L_text, D)
            current_durations = durations[i]             # (L_text,)
            current_text_padding_mask = text_padding_mask[i] # (L_text,)

            logger.debug(f"Batch {i}: Original phoneme indices length: {max_text_len}")
            logger.debug(f"Batch {i}: current_durations (full, should have no zeros after clamp in inference): {current_durations}")
            logger.debug(f"Batch {i}: current_text_padding_mask (True for 0s): {current_text_padding_mask}")
            
            # Select non-padded elements. `True` in non_padded_indices means it's actual content.
            non_padded_indices = ~current_text_padding_mask
            
            filtered_encoder_output = current_encoder_output[non_padded_indices]
            filtered_durations = current_durations[non_padded_indices] 

            logger.debug(f"Batch {i}: non_padded_indices (True for content): {non_padded_indices}")
            logger.debug(f"Batch {i}: filtered_durations (sliced durations, expected >0): {filtered_durations}")
            logger.debug(f"Batch {i}: Number of elements in filtered_durations: {filtered_durations.numel()}")


            if filtered_durations.numel() == 0:
                logger.warning(f"Batch {i}: filtered_durations is empty or all elements are 0/padding. "
                               "No valid phonemes to expand. Returning empty tensor for this batch item.")
                expanded_encoder_output = torch.empty(0, hidden_dim, device=encoder_outputs.device)
                expanded_padding_mask = torch.ones(0, dtype=torch.bool, device=encoder_outputs.device)
                actual_expanded_length = 0
            else:
                # This clamp should ideally not be needed here if clamping is done upstream
                # but leaving it as a final failsafe, it won't change values if they are already >= 1
                filtered_durations = torch.clamp(filtered_durations, min=1).long() 
                logger.debug(f"Batch {i}: filtered_durations after final clamp: {filtered_durations}")

                if torch.any(filtered_durations <= 0):
                    logger.error(f"Batch {i}: Found non-positive duration values in filtered_durations: {filtered_durations}")
                    raise ValueError("Non-positive durations found after filtering, which should not happen if clamping is effective upstream and mask is correct.")

                expanded_encoder_output = torch.repeat_interleave(
                    filtered_encoder_output, filtered_durations, dim=0
                )
                expanded_padding_mask = torch.zeros(expanded_encoder_output.shape[0], dtype=torch.bool, device=encoder_outputs.device)
                actual_expanded_length = expanded_encoder_output.shape[0]
            
            expanded_encoder_outputs_list.append(expanded_encoder_output)
            encoder_output_padding_mask_list.append(expanded_padding_mask)
            actual_expanded_lengths_per_item.append(actual_expanded_length) # Store the true length

        if not expanded_encoder_outputs_list:
            logger.warning("All batch items resulted in empty expanded encoder outputs.")
            # Ensure return has correct batch size dimensions, even if length is 0
            return torch.empty(batch_size, 0, hidden_dim, device=encoder_outputs.device), \
                   torch.empty(batch_size, 0, dtype=torch.bool, device=encoder_outputs.device)

        # NOW calculate max_expanded_len based on the *actual* lengths generated
        # This is the FIX: Max length should be derived from what was actually generated, not input durations including padding.
        max_expanded_len = max(actual_expanded_lengths_per_item) 
        logger.debug(f"Calculated max_expanded_len from actual expanded items: {max_expanded_len}")

        # Final padding to the common max_expanded_len
        final_expanded_outputs = []
        final_padding_masks = []
        for i in range(batch_size):
            current_output = expanded_encoder_outputs_list[i]
            current_mask = encoder_output_padding_mask_list[i]
            current_len = actual_expanded_lengths_per_item[i] # Use the stored actual length

            padding_needed = max_expanded_len - current_len

            if padding_needed < 0:
                # This should now ideally NEVER happen. If it does, there's a serious bug.
                logger.error(f"FATAL ERROR: Padding needed is still negative for batch {i}: {padding_needed}. "
                             "max_expanded_len calculation is fundamentally flawed if this occurs.")
                # This implies max_expanded_len was not the true maximum.
                # In this extremely unlikely scenario, we would need to truncate, but it's better to crash and fix.
                raise RuntimeError("Internal inconsistency: current_len exceeded max_expanded_len.")

            if padding_needed > 0:
                current_output = F.pad(current_output, (0, 0, 0, padding_needed))
                current_mask = F.pad(current_mask, (0, padding_needed), value=True)
            
            final_expanded_outputs.append(current_output)
            final_padding_masks.append(current_mask)

        expanded_encoder_outputs = torch.stack(final_expanded_outputs, dim=0)
        encoder_output_padding_mask = torch.stack(final_padding_masks, dim=0)

        logger.debug(f"Final Expanded encoder outputs shape: {expanded_encoder_outputs.shape}")
        logger.debug(f"Final Encoder output padding mask shape: {encoder_output_padding_mask.shape}")

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
        phoneme_durations: torch.Tensor, # Ground truth durations (batch_size, text_seq_len)
        stop_token_targets: torch.Tensor, # Ground truth stop tokens (batch_size, mel_seq_len)
        text_padding_mask: Optional[torch.Tensor] = None, # Mask for text encoder
        mel_padding_mask: Optional[torch.Tensor] = None # New: Mask for mel decoder
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass during training (teacher forcing).

        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            mel_specs: Target mel spectrograms of shape (batch_size, mel_seq_len, mel_dim)
            phoneme_durations: Ground truth durations for each phoneme (batch_size, text_seq_len)
            stop_token_targets: Ground truth stop tokens (0 for all but last frame, which is 1)
                                of shape (batch_size, mel_seq_len)
            text_padding_mask: Optional boolean mask (batch_size, text_seq_len) for the
                               text encoder's padding. True for padded positions.
            mel_padding_mask: Optional boolean mask (batch_size, mel_seq_len) for the
                              decoder's mel sequence padding. True for padded positions.

        Returns:
            - Predicted mel spectrograms of shape (batch_size, mel_seq_len, mel_dim)
            - Predicted log durations of shape (batch_size, text_seq_len) (for duration loss)
            - Predicted stop token logits of shape (batch_size, mel_seq_len) (for stop token loss)
        """
        batch_size, mel_seq_len = mel_specs.shape[0], mel_specs.shape[1]
        device = mel_specs.device

        # Ensure text_padding_mask is provided for correct processing within _length_regulate
        if text_padding_mask is None:
            # Assuming 0 is the padding ID for phoneme_indices if no mask is explicitly provided
            text_padding_mask = (phoneme_indices == 0)

        # Encode text using Transformer encoder
        text_encoded = self.encode_text(phoneme_indices, mask=text_padding_mask)

        # Predict durations
        predicted_log_durations = self._predict_durations(text_encoded)

        # Length regulate text_encoded using ground-truth durations
        # Pass text_padding_mask as the third argument as required by the updated _length_regulate
        expanded_encoder_outputs, encoder_output_padding_mask = self._length_regulate(
            text_encoded, phoneme_durations.float(), text_padding_mask
        )

        # Prepare decoder input (teacher forcing: shift right)
        # Add a zero frame at the beginning
        decoder_input_mels = F.pad(mel_specs[:, :-1, :], (0, 0, 1, 0), "constant", 0.0)
        decoder_input_projected = self.mel_projection_in(decoder_input_mels)

        # FIX: Apply the dedicated decoder positional encoding
        # For training, seq_offset is 0 as we're processing the full sequence from the start
        decoder_input_projected_with_pe = self.decoder_positional_encoding(decoder_input_projected, seq_offset=0)

        # Generate causal mask for decoder self-attention
        tgt_mask = self._generate_square_subsequent_mask(mel_seq_len, device)

        # Pass through Transformer Decoder
        decoder_outputs = self.decoder(
            tgt=decoder_input_projected_with_pe, # Pass the input with its own PE
            memory=expanded_encoder_outputs,
            tgt_mask=tgt_mask, # Causal mask for self-attention
            memory_key_padding_mask=encoder_output_padding_mask, # Mask for cross-attention
            tgt_key_padding_mask=mel_padding_mask # Mask for padded target sequence
        )

        # Project decoder outputs back to mel dimension
        predicted_mel_frames = self.mel_projection_out(decoder_outputs)

        # Predict stop tokens from decoder outputs
        predicted_stop_logits = self.stop_token_predictor(decoder_outputs).squeeze(-1) # (batch_size, mel_seq_len)

        return predicted_mel_frames, predicted_log_durations, predicted_stop_logits


    def forward_inference(self, phoneme_indices: torch.Tensor, max_len: int = 1420, stop_threshold: float = 0.5,
                      text_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fixed forward pass during inference with multiple fallback stopping mechanisms.
        """
        if phoneme_indices.size(0) > 1:
            logger.warning("Inference with stop token is most reliable with batch_size=1.")

        batch_size = phoneme_indices.size(0)
        device = phoneme_indices.device

        self.eval()

        with torch.no_grad():
            # Ensure text_padding_mask is correctly defined
            if text_padding_mask is None:
                text_padding_mask = (phoneme_indices == 0)

            # Encode text
            text_encoded = self.encode_text(phoneme_indices, mask=text_padding_mask)

            # Predict durations
            predicted_log_durations = self._predict_durations(text_encoded)
            durations_for_length_regulate = torch.exp(predicted_log_durations)
            durations_for_length_regulate = torch.clamp(durations_for_length_regulate, min=1.0).long()

            # Length regulate
            expanded_encoder_outputs, encoder_output_padding_mask = self._length_regulate(
                text_encoded, durations_for_length_regulate, text_padding_mask
            )

            expected_length = expanded_encoder_outputs.shape[1]
            logger.info(f"Starting inference with expanded encoder outputs shape: {expanded_encoder_outputs.shape}")
            logger.info(f"Expected output length based on durations: ~{expected_length} frames")

            # Calculate reasonable bounds for generation
            min_expected_length = max(20, expected_length // 2)  # At least 20 frames, or half expected
            max_expected_length = min(max_len, expected_length * 3, self.max_seq_len) # Use self.max_seq_len as upper bound for PE

            logger.info(f"Generation bounds: min={min_expected_length}, max={max_expected_length}")

            # Initialize generation
            generated_mels = []
            decoder_input_mel = torch.zeros(batch_size, 1, self.mel_dim, device=device)

            # Pre-allocate for efficiency
            decoder_input_seq_projected = torch.zeros(
                batch_size, max_expected_length, self.hidden_dim, device=device
            )

            # FIX: Use the dedicated decoder_positional_encoding for caching
            dummy_input = torch.zeros(batch_size, max_expected_length, self.hidden_dim, device=device)
            pe_cache = self.decoder_positional_encoding(dummy_input, seq_offset=0) # Use decoder's PE

            # Cache for masks
            mask_cache = {}

            # Tracking variables for intelligent stopping
            stop_probs_history = []
            mel_stability_buffer = []
            consecutive_low_stop_count = 0

            for t in range(max_expected_length):
                # Project current mel frame
                mel_projected_t = self.mel_projection_in(decoder_input_mel)

                # Use pre-computed positional encoding
                mel_projected_t_with_pe = mel_projected_t + pe_cache[:, t:t+1, :]

                # Store in pre-allocated tensor
                decoder_input_seq_projected[:, t:t+1, :] = mel_projected_t_with_pe

                # Get current sequence length
                current_seq_len = t + 1

                # Use cached mask or create new one
                if current_seq_len not in mask_cache:
                    mask_cache[current_seq_len] = self._generate_square_subsequent_mask(current_seq_len, device)
                tgt_mask = mask_cache[current_seq_len]

                # Process only the current sequence
                current_input = decoder_input_seq_projected[:, :current_seq_len, :]

                # Decoder forward pass
                try:
                    decoder_outputs = self.decoder(
                        tgt=current_input,
                        memory=expanded_encoder_outputs,
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=encoder_output_padding_mask,
                        tgt_key_padding_mask=None
                    )
                except Exception as e:
                    logger.error(f"Decoder failed at frame {t}: {e}")
                    break

                # Take the last frame's output
                decoder_out_t = decoder_outputs[:, -1:, :]

                # Generate mel frame
                mel_pred_t = self.mel_projection_out(decoder_out_t)
                generated_mels.append(mel_pred_t)

                # Check stop condition
                stop_token_logit_t = self.stop_token_predictor(decoder_out_t)
                stop_probability = torch.sigmoid(stop_token_logit_t).item()
                stop_probs_history.append(stop_probability)

                # Track mel frame for stability analysis
                if len(mel_stability_buffer) >= 5:
                    mel_stability_buffer.pop(0)
                mel_stability_buffer.append(mel_pred_t.clone())

                # Progress logging
                if t % 50 == 0 or t < 10:
                    logger.debug(f"Generated frame {t}, stop_prob: {stop_probability:.6f}")

                # MULTIPLE STOPPING CONDITIONS

                # 1. Original stop token threshold (if it works)
                if batch_size == 1 and stop_probability > stop_threshold:
                    logger.info(f"Stopping generation at frame {t} (stop_prob: {stop_probability:.4f})")
                    break

                # 2. Adaptive threshold based on history
                if len(stop_probs_history) >= 10:
                    recent_avg = sum(stop_probs_history[-10:]) / 10
                    if recent_avg > 0.1 and stop_probability > recent_avg * 2:
                        logger.info(f"Stopping due to adaptive threshold at frame {t} (stop_prob: {stop_probability:.4f}, avg: {recent_avg:.4f})")
                        break

                # 3. Count consecutive very low probabilities
                if stop_probability < 0.001:
                    consecutive_low_stop_count += 1
                else:
                    consecutive_low_stop_count = 0

                # 4. Force stop if probabilities are consistently too low
                if consecutive_low_stop_count > 20 and t > min_expected_length:
                    logger.warning(f"Forcing stop at frame {t} due to consistently low stop probabilities")
                    break

                # 5. Mel frame stability check
                if len(mel_stability_buffer) >= 5 and t > min_expected_length:
                    # Calculate variance in recent mel frames
                    stacked_mels = torch.cat(mel_stability_buffer, dim=1)  # (B, 5, mel_dim)
                    mel_variance = torch.var(stacked_mels, dim=1).mean().item()

                    if mel_variance < 0.001:  # Very stable/repetitive output
                        logger.info(f"Stopping due to stable mel output at frame {t} (variance: {mel_variance:.6f})")
                        break

                # 6. Length-based intelligent stopping
                if t >= expected_length and stop_probability > 0.01:
                    logger.info(f"Stopping at expected length {t} with reasonable stop_prob: {stop_probability:.4f}")
                    break

                # 7. Hard safety limit
                if t >= max_expected_length - 1:
                    logger.warning(f"Reached maximum generation length ({max_expected_length})")
                    break

                # 8. Emergency brake for completely broken models
                if t > 100 and max(stop_probs_history[-50:]) < 0.01:
                    logger.error(f"Emergency stop: stop token predictor appears completely broken (max prob in last 50 frames: {max(stop_probs_history[-50:]):.6f})")
                    break

                # Use prediction as next input
                decoder_input_mel = mel_pred_t

            # Concatenate all generated frames
            if generated_mels:
                mel_output = torch.cat(generated_mels, dim=1)
                logger.info(f"Generated {mel_output.shape[1]} mel frames (expected ~{expected_length})")

                # Log statistics
                if stop_probs_history:
                    avg_stop_prob = sum(stop_probs_history) / len(stop_probs_history)
                    max_stop_prob = max(stop_probs_history)
                    logger.info(f"Stop probability stats - avg: {avg_stop_prob:.6f}, max: {max_stop_prob:.6f}")
            else:
                logger.warning("No mel frames were generated.")
                mel_output = torch.empty(batch_size, 0, self.mel_dim, device=device)

            return mel_output


    def forward(
        self,
        phoneme_indices: torch.Tensor,
        mel_specs: Optional[torch.Tensor] = None,
        phoneme_durations: Optional[torch.Tensor] = None, # Added as optional
        stop_token_targets: Optional[torch.Tensor] = None, # Added as optional
        text_padding_mask: Optional[torch.Tensor] = None,
        mel_padding_mask: Optional[torch.Tensor] = None # Added as optional
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Main forward pass that dispatches to training or inference mode.

        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len).
            mel_specs: Optional; Target mel spectrograms for training, shape (batch_size, mel_seq_len, mel_dim).
            phoneme_durations: Optional; Ground truth durations for training, shape (batch_size, text_seq_len).
            stop_token_targets: Optional; Ground truth stop tokens for training, shape (batch_size, mel_seq_len).
            text_padding_mask: Optional boolean mask (batch_size, text_seq_len) for encoder. True for padded positions.
            mel_padding_mask: Optional boolean mask (batch_size, mel_seq_len) for decoder. True for padded positions.

        Returns:
            - During training: Tuple of predicted mel spectrograms, predicted log durations, predicted stop token logits.
            - During inference: Generated mel spectrograms.
        """
        if mel_specs is not None:
            self.train() # Set to training mode
            if phoneme_durations is None or stop_token_targets is None:
                raise ValueError("phoneme_durations and stop_token_targets must be provided for training mode.")
            return self.forward_training(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets, text_padding_mask, mel_padding_mask)
        else:
            self.eval() # Set to evaluation mode for inference
            # Pass max_len to forward_inference, ensure it's clamped by self.max_seq_len
            return self.forward_inference(phoneme_indices, max_len=self.max_seq_len, text_padding_mask=text_padding_mask)


    def get_model_info(self) -> dict:
        """
        Get model information and parameter count.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'vocab_size': self.vocab_size,
            'mel_dim': self.mel_dim,
            'hidden_dim': self.hidden_dim,
            'n_encoder_layers': len(self.transformer_encoder_layers),
            'n_decoder_layers': len(self.decoder.layers),
            'n_heads': self.transformer_encoder_layers[0].self_attn.num_heads if self.transformer_encoder_layers else 8, # Assuming encoder has at least one layer
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }