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
