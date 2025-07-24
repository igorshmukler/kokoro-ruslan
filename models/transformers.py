import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint # Keep for gradient checkpointing

from typing import Optional


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
                # Debugging print statement:
                # logger.debug(f"Decoder layer input tgt shape: {output.shape}, tgt_mask shape: {tgt_mask.shape if tgt_mask is not None else 'None'}")
                output = layer(output, memory, tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask)
        return self.norm(output)
