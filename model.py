import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch.utils.checkpoint import checkpoint # Keep for gradient checkpointing
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as a buffer, not a parameter

    def forward(self, x: torch.Tensor, seq_offset: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        # Ensure pe has enough length, though max_len in __init__ should handle this
        # This check might still cause issues with torch.compile if seq_offset + seq_len is symbolic
        # For torch.compile, it's often better to let it infer shapes or use fixed max_len
        # or handle variable lengths via padding and masking later.
        # For now, we'll keep it but be aware it could be a source of future graph breaks.
        if torch.jit.is_tracing() or not torch.is_grad_enabled(): # Only do this check when not tracing/compiling
            if seq_offset + seq_len > self.pe.size(1):
                # This could trigger if dynamic length goes beyond `max_len`
                print(f"Warning: Positional encoding max_len ({self.pe.size(1)}) exceeded "
                      f"for sequence offset {seq_offset} and length {seq_len}. "
                      f"Consider increasing max_len in PositionalEncoding.")
        
        # Slice `pe` based on actual sequence length and offset
        pe_slice = self.pe[:, seq_offset : seq_offset + seq_len]
        
        # Handle broadcasting: if x has batch_size > 1, pe_slice needs to match
        x = x + pe_slice
        return self.dropout(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: the sequence to the encoder (batch_size, seq_len, d_model).
            src_mask: the mask for the src sequence (seq_len, seq_len).
            src_key_padding_mask: the mask for the src keys per batch (batch_size, seq_len).
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: the sequence to the decoder (batch_size, tgt_seq_len, d_model).
            memory: the sequence from the last layer of the encoder (batch_size, src_seq_len, d_model).
            tgt_mask: the mask for the tgt sequence (tgt_seq_len, tgt_seq_len). For causal attention.
            memory_mask: the mask for the memory sequence (src_seq_len, src_seq_len). (Typically not used for cross-attention).
            tgt_key_padding_mask: the mask for the tgt keys per batch (batch_size, tgt_seq_len).
            memory_key_padding_mask: the mask for the memory keys per batch (batch_size, src_seq_len).
        """
        # Self-attention part (masked for causality)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention part
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward part
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt # Expects PE already applied for inference
        for layer in self.layers:
            if self.training: # Check if in training mode
                output = checkpoint(layer, output, memory, tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask, use_reentrant=False)
            else: # Inference mode
                output = layer(output, memory, tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask)
        return self.norm(output)

class KokoroModel(nn.Module):
    """
    Simplified Kokoro-style model architecture.
    Optimized for MPS (Metal Performance Shaders) acceleration
    """

    def __init__(self, vocab_size: int, mel_dim: int = 80, hidden_dim: int = 512,
                 n_encoder_layers: int = 6, n_heads: int = 8, encoder_ff_dim: int = 2048,
                 encoder_dropout: float = 0.1, n_decoder_layers: int = 6, decoder_ff_dim: int = 2048,
                 max_decoder_seq_len: int = 800): # Added max_decoder_seq_len
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
            max_decoder_seq_len: Maximum sequence length the decoder might generate, for PE.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim # This will now be d_model
        self.max_decoder_seq_len = max_decoder_seq_len # Store this

        # Text encoder: Embedding + Positional Encoding + Stack of Transformer Blocks
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_positional_encoding = PositionalEncoding(hidden_dim, dropout=encoder_dropout, max_len=max_decoder_seq_len) # PE shared for encoder & decoder

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
        # The max_len argument is no longer needed/accepted by _length_regulate
        expanded_encoder_outputs, encoder_output_padding_mask = self._length_regulate(
            text_encoded, phoneme_durations.float(), text_padding_mask
        )

        # Prepare decoder input (teacher forcing: shift right)
        # Add a zero frame at the beginning
        decoder_input_mels = F.pad(mel_specs[:, :-1, :], (0, 0, 1, 0), "constant", 0.0)
        decoder_input_projected = self.mel_projection_in(decoder_input_mels)

        # Apply positional encoding to the full decoder input sequence
        # For training, seq_offset is 0 as we're processing the full sequence from the start
        decoder_input_projected_with_pe = self.encoder_positional_encoding(decoder_input_projected, seq_offset=0)

        # Generate causal mask for decoder self-attention
        tgt_mask = self._generate_square_subsequent_mask(mel_seq_len, device)

        # Pass through Transformer Decoder
        decoder_outputs = self.decoder(
            tgt=decoder_input_projected_with_pe, # Pass the input with PE
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

    def forward_inference(self, phoneme_indices: torch.Tensor, max_len: int = 800, stop_threshold: float = 0.5,
                          text_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass during inference (autoregressive).

        Args:
            phoneme_indices: Tensor of shape (batch_size, text_seq_len)
            max_len: Maximum length for generated sequence (used as hard limit for PE and output).
            stop_threshold: Probability threshold for stopping generation.
            text_padding_mask: Optional boolean mask (batch_size, text_seq_len) for the
                               text encoder's padding. True for padded positions.

        Returns:
            Generated mel spectrograms of shape (batch_size, generated_mel_seq_len, mel_dim)
        """
        if phoneme_indices.size(0) > 1:
            print("Warning: Inference with stop token is most reliable with batch_size=1.")

        batch_size = phoneme_indices.size(0)
        device = phoneme_indices.device

        self.eval() # Ensure model is in evaluation mode

        with torch.no_grad():
            # Ensure text_padding_mask is correctly defined for the encoder
            if text_padding_mask is None:
                # Assuming 0 is the padding ID. Adjust if different.
                text_padding_mask = (phoneme_indices == 0)

            # Encode text (no checkpointing for inference)
            text_encoded = self.encode_text(phoneme_indices, mask=text_padding_mask)

            # Predict durations
            predicted_log_durations = self._predict_durations(text_encoded)

            # Length regulate text_encoded using predicted durations
            # IMPORTANT FIX: Pass text_padding_mask to _length_regulate
            # IMPORTANT FIX: Remove max_len argument, as _length_regulate no longer accepts it.


            # Ensure text_padding_mask is correctly defined for the encoder
            if text_padding_mask is None:
                # Assuming 0 is the padding ID. Adjust if different.
                text_padding_mask = (phoneme_indices == 0)

            text_encoded = self.encode_text(phoneme_indices, mask=text_padding_mask)

            # Predict durations (these are log durations)
            predicted_log_durations = self._predict_durations(text_encoded)

            # Convert log durations to actual counts and ENSURE THEY ARE CLAMPED HERE
            durations_for_length_regulate = torch.exp(predicted_log_durations)

            # This is where the problematic 0s should be caught and turned to 1.
            durations_for_length_regulate = torch.clamp(durations_for_length_regulate, min=1.0)

            # THEN convert to long
            durations_for_length_regulate = durations_for_length_regulate.long()

            logger.debug(f"Durations predicted (exp and clamped) before _length_regulate: {durations_for_length_regulate}")

            # Length regulate text_encoded using predicted durations
            expanded_encoder_outputs, encoder_output_padding_mask = self._length_regulate(
                text_encoded, durations_for_length_regulate, text_padding_mask # Pass the now clamped durations
            )

            # The actual generated length for each batch item will be `expanded_lengths`
            # from _length_regulate's internal calculation, clamped by max_decoder_seq_len.
            # We don't return `expanded_lengths` here, but it's used internally for padding.
            
            generated_mels = []
            decoder_input_mel = torch.zeros(batch_size, 1, self.mel_dim, device=device)
            # The `current_decoder_input_seq_projected` will accumulate the projected (embedding + PE)
            # frames. It must be initialized with the correct hidden_dim.
            current_decoder_input_seq_projected = torch.empty(batch_size, 0, self.hidden_dim, device=device)


            # Re-use the encoder's positional encoding for the decoder's inference loop.
            # This is likely a simplification, assuming positional encoding is shared.
            # If your model has a separate `decoder_positional_encoding` you should use that.
            # Based on your `__init__`, `encoder_positional_encoding` is what's available.
            decoder_pe_layer = self.encoder_positional_encoding 

            for t in range(max_len): # max_len is the hard limit for sequence generation
                mel_projected_t = self.mel_projection_in(decoder_input_mel)

                # Apply positional encoding for the *current* token at its absolute position `t`
                mel_projected_t_with_pe = decoder_pe_layer(mel_projected_t, seq_offset=t)

                current_decoder_input_seq_projected = torch.cat(
                    [current_decoder_input_seq_projected, mel_projected_t_with_pe], dim=1
                )

                # Generate causal mask for the *currently accumulated sequence length*
                # The size of the mask grows with `t+1` (current_decoder_input_seq_projected.size(1))
                tgt_mask = self._generate_square_subsequent_mask(
                    current_decoder_input_seq_projected.size(1), device
                )

                decoder_outputs_full_seq = self.decoder(
                    tgt=current_decoder_input_seq_projected, # Pass the growing sequence with PE
                    memory=expanded_encoder_outputs,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=encoder_output_padding_mask,
                    tgt_key_padding_mask=None # No padding mask needed for a single generated sequence element
                )

                # Take the output of the *last* frame for prediction
                decoder_out_t = decoder_outputs_full_seq[:, -1:, :]

                mel_pred_t = self.mel_projection_out(decoder_out_t)
                generated_mels.append(mel_pred_t)

                stop_token_logit_t = self.stop_token_predictor(decoder_out_t)
                stop_probability = torch.sigmoid(stop_token_logit_t)

                # Use prediction as next input
                decoder_input_mel = mel_pred_t

                # Stop condition: check stop probability for batch_size 1
                if batch_size == 1 and stop_probability.item() > stop_threshold:
                    print(f"Stopping generation at frame {t} (stop_prob: {stop_probability.item():.4f})")
                    break

            # Concatenate all generated mel frames
            if generated_mels:
                mel_output = torch.cat(generated_mels, dim=1) # (B, L_mel, mel_dim)
            else:
                # Handle case where no mels were generated (e.g., max_len=0 or immediate stop)
                print("Warning: No mel frames were generated.")
                mel_output = torch.empty(batch_size, 0, self.mel_dim, device=device)


            # Apply Postnet
            # Ensure mel_output is not empty before applying postnet
            if mel_output.size(1) > 0:
                mel_output = mel_output + self.postnet(mel_output.transpose(1, 2)).transpose(1, 2)
            else:
                # If no mels were generated, postnet cannot be applied, return empty tensor
                pass # mel_output is already empty from above

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
            return self.forward_inference(phoneme_indices, text_padding_mask=text_padding_mask)


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
