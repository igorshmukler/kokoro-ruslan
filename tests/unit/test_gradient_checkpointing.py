"""Test with gradient checkpointing to reproduce batch 281 crash"""
import torch
import sys
sys.path.insert(0, 'src')

from torch.utils.checkpoint import checkpoint
from kokoro.model.transformers import MultiHeadAttentionImproved, ImprovedTransformerDecoderBlock

device = torch.device('mps')
print(f"Testing on {device}\n")

# Batch 281 parameters
batch_size = 20
text_len = 258
mel_len = 1800
d_model = 512
num_heads = 8
d_ff = 2048

print("=" * 80)
print("TEST: Gradient Checkpointing with batch 281 dimensions")
print("=" * 80)
print(f"Batch size: {batch_size}")
print(f"Text length: {text_len}")
print(f"Mel length: {mel_len}")
print()

print("=" * 80)
print("TEST 1: Decoder with gradient checkpointing")
print("=" * 80)

try:
    # Create decoder block
    decoder_block = ImprovedTransformerDecoderBlock(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=d_ff,
        dropout=0.1
    ).to(device)

    decoder_block.train()  # Training mode

    # Create inputs
    decoder_input = torch.randn(batch_size, mel_len, d_model, device=device, requires_grad=True)
    encoder_output = torch.randn(batch_size, text_len, d_model, device=device, requires_grad=True)

    # Create causal mask
    tgt_mask = torch.triu(torch.ones(mel_len, mel_len, device=device) * float('-inf'), diagonal=1)

    print(f"Decoder input: {decoder_input.shape}")
    print(f"Encoder output (memory): {encoder_output.shape}")
    print(f"Causal mask: {tgt_mask.shape}")
    print()

    print("Forward pass WITHOUT checkpointing...")
    output_no_ckpt = decoder_block(decoder_input, encoder_output, tgt_mask=tgt_mask)
    loss_no_ckpt = output_no_ckpt.sum()
    print(f"  Output shape: {output_no_ckpt.shape}")
    print(f"  Loss: {loss_no_ckpt.item():.4f}")
    print("  ✓ No checkpoint forward OK")

    # Reset gradients
    decoder_block.zero_grad()
    if decoder_input.grad is not None:
        decoder_input.grad.zero_()
    if encoder_output.grad is not None:
        encoder_output.grad.zero_()

    print()
    print("Forward pass WITH gradient checkpointing...")

    # Define checkpointed forward function
    def checkpoint_forward(tgt, memory, mask):
        return decoder_block(tgt, memory, tgt_mask=mask)

    # Create new inputs (gradient checkpointing needs fresh tensors)
    decoder_input2 = torch.randn(batch_size, mel_len, d_model, device=device, requires_grad=True)
    encoder_output2 = torch.randn(batch_size, text_len, d_model, device=device, requires_grad=True)

    output_ckpt = checkpoint(checkpoint_forward, decoder_input2, encoder_output2, tgt_mask, use_reentrant=False)
    loss_ckpt = output_ckpt.sum()

    print(f"  Output shape: {output_ckpt.shape}")
    print(f"  Loss: {loss_ckpt.item():.4f}")
    print("  ✓ Checkpoint forward OK")

    print()
    print("Backward pass WITH gradient checkpointing...")
    loss_ckpt.backward()

    print(f"  Decoder input gradient: {decoder_input2.grad.shape}")
    print(f"  Encoder output gradient: {encoder_output2.grad.shape}")
    print("  ✓ Checkpoint backward OK")

    print()
    print("✅ Gradient checkpointing test PASSED\n")

except Exception as e:
    print(f"\n❌ Gradient checkpointing test FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("TEST 2: Multiple decoder layers with checkpointing")
print("=" * 80)

try:
    # Create 4 decoder layers (like in the actual model)
    num_layers = 4
    decoder_layers = torch.nn.ModuleList([
        ImprovedTransformerDecoderBlock(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.1
        ).to(device)
        for _ in range(num_layers)
    ])

    for layer in decoder_layers:
        layer.train()

    # Create inputs
    decoder_input = torch.randn(batch_size, mel_len, d_model, device=device, requires_grad=True)
    encoder_output = torch.randn(batch_size, text_len, d_model, device=device, requires_grad=True)
    tgt_mask = torch.triu(torch.ones(mel_len, mel_len, device=device) * float('-inf'), diagonal=1)

    print(f"Number of decoder layers: {num_layers}")
    print(f"Decoder input: {decoder_input.shape}")
    print(f"Encoder output: {encoder_output.shape}")
    print()

    print("Processing layers with checkpointing...")

    x = decoder_input
    for i, layer in enumerate(decoder_layers):
        print(f"  Layer {i}...")

        def create_layer_forward(layer_module):
            def layer_forward(tgt, memory, mask):
                return layer_module(tgt, memory, tgt_mask=mask)
            return layer_forward

        layer_fn = create_layer_forward(layer)
        x = checkpoint(layer_fn, x, encoder_output, tgt_mask, use_reentrant=False)

        print(f"    Output shape: {x.shape}")

    print()
    print("Computing loss and backward...")
    loss = x.sum()
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"Decoder input gradient: {decoder_input.grad.shape}")
    print()
    print("✅ Multi-layer checkpointing PASSED\n")

except Exception as e:
    print(f"\n❌ Multi-layer checkpointing FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("TEST 3: Segmented checkpointing (like actual model)")
print("=" * 80)

try:
    # Simulate the actual model's segmented checkpointing
    num_layers = 4
    checkpoint_segments = 4  # From training log
    segment_size = max(1, num_layers // checkpoint_segments)

    print(f"Number of layers: {num_layers}")
    print(f"Checkpoint segments: {checkpoint_segments}")
    print(f"Segment size: {segment_size}")
    print()

    decoder_layers = torch.nn.ModuleList([
        ImprovedTransformerDecoderBlock(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.1
        ).to(device)
        for _ in range(num_layers)
    ])

    for layer in decoder_layers:
        layer.train()

    # Create inputs
    decoder_input = torch.randn(batch_size, mel_len, d_model, device=device, requires_grad=True)
    encoder_output = torch.randn(batch_size, text_len, d_model, device=device, requires_grad=True)
    tgt_mask = torch.triu(torch.ones(mel_len, mel_len, device=device) * float('-inf'), diagonal=1)

    print("Processing with segmented checkpointing...")
    x = decoder_input

    for segment_idx in range(0, num_layers, segment_size):
        segment_end = min(segment_idx + segment_size, num_layers)
        segment_layers = decoder_layers[segment_idx:segment_end]

        print(f"  Segment {segment_idx//segment_size} (layers {segment_idx}-{segment_end-1})...")

        def create_segment_forward(layers_list):
            def segment_forward(tgt, memory, mask):
                result = tgt
                for layer in layers_list:
                    result = layer(result, memory, tgt_mask=mask)
                return result
            return segment_forward

        segment_fn = create_segment_forward(segment_layers)
        x = checkpoint(segment_fn, x, encoder_output, tgt_mask, use_reentrant=False)

        print(f"    Output shape: {x.shape}")

    print()
    print("Computing loss and backward...")
    loss = x.sum()
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"Decoder input gradient: {decoder_input.grad.shape}")
    print()
    print("✅ Segmented checkpointing PASSED\n")

except Exception as e:
    print(f"\n❌ Segmented checkpointing FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("CRITICAL FINDINGS")
print("=" * 80)
print()
print("If all tests PASS:")
print("  - Gradient checkpointing itself is not the issue")
print("  - The crash must be in a different part of the model")
print("  - Or it's a cumulative memory issue after 280 batches")
print()
print("Key observations from training log:")
print("  - Batch 281 is when the crash happens consistently")
print("  - Error: 'NDArray dimension length > INT_MAX'")
print("  - Memory usage was 'low' at batch 250")
print("  - This suggests it's not a gradual OOM")
print()
print("Possible explanations:")
print("  1. Batch 281 has an unusually long sequence (>1800)")
print("  2. MPS backend bug with specific tensor dimensions")
print("  3. Interaction with variance predictors or other components")
print("  4. Cumulative MPS internal state corruption")
print()
print("Next step: Check actual batch 281 data dimensions")
