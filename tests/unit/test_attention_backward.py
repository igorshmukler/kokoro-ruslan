"""Test backward pass to find INT_MAX crash"""
import torch
import sys
sys.path.insert(0, 'src')

from kokoro.model.transformers import MultiHeadAttentionImproved

device = torch.device('mps')
print(f"Testing on {device}\n")

# Batch 281 parameters
batch_size = 20
text_len = 258
mel_len = 1800
d_model = 512
num_heads = 8

print("=" * 80)
print("All forward pass tests PASSED - testing BACKWARD PASS")
print("=" * 80)
print()

print("=" * 80)
print("TEST 1: Attention forward + backward")
print("=" * 80)

try:
    attn = MultiHeadAttentionImproved(d_model, num_heads).to(device)

    # Enable gradients
    x = torch.randn(batch_size, mel_len, d_model, device=device, requires_grad=True)
    causal_mask = torch.triu(torch.ones(mel_len, mel_len, device=device) * float('-inf'), diagonal=1)

    print(f"Input shape: {x.shape}")
    print(f"requires_grad: {x.requires_grad}")
    print("Forward pass...")

    output, _ = attn(x, x, x, attn_mask=causal_mask)

    print(f"Output shape: {output.shape}")
    print("Creating loss and computing gradients...")

    # Create a dummy loss
    loss = output.sum()

    print(f"Loss: {loss.item()}")
    print("Calling backward()...")

    loss.backward()

    print(f"Input gradient shape: {x.grad.shape}")
    print("✅ Attention backward PASSED\n")

except Exception as e:
    print(f"❌ Attention backward FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("TEST 2: Large matmul forward + backward")
print("=" * 80)

try:
    A = torch.randn(batch_size, mel_len, d_model, device=device, requires_grad=True)
    B = torch.randn(batch_size, d_model, mel_len, device=device, requires_grad=True)

    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print("Forward: C = A @ B...")

    C = torch.bmm(A, B)

    print(f"C shape: {C.shape}")
    print(f"C elements: {C.numel():,}")
    print("Computing gradients...")

    loss = C.sum()
    loss.backward()

    print(f"A gradient shape: {A.grad.shape}")
    print(f"B gradient shape: {B.grad.shape}")
    print("✅ Large matmul backward PASSED\n")

except Exception as e:
    print(f"❌ Large matmul backward FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("TEST 3: Softmax of large tensor + backward")
print("=" * 80)

try:
    # This is what happens in attention: softmax over (batch*heads, seq_len, seq_len)
    batch_heads = batch_size * num_heads
    scores = torch.randn(batch_heads, mel_len, mel_len, device=device, requires_grad=True)

    print(f"Scores shape: {scores.shape}")
    print(f"Scores elements: {scores.numel():,}")
    print(f"Scores bytes (fp32): {scores.numel() * 4 / (1024**3):.3f} GB")
    print()

    if scores.numel() > 2147483647:
        print(f"⚠️  WARNING: Tensor has > INT_MAX elements!")
        print(f"This is likely the crash point!")
    else:
        print(f"✓ Tensor has < INT_MAX elements")

    print()
    print("Forward: applying softmax...")

    attn_weights = torch.softmax(scores, dim=-1)

    print(f"Attention weights shape: {attn_weights.shape}")
    print("Computing gradients...")

    loss = attn_weights.sum()
    loss.backward()

    print(f"Scores gradient shape: {scores.grad.shape}")
    print("✅ Softmax backward PASSED\n")

except Exception as e:
    print(f"❌ Softmax backward FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("TEST 4: Cumulative gradient accumulation")
print("=" * 80)

try:
    # Simulate gradient accumulation (4 steps as in training)
    gradient_accumulation_steps = 4

    attn = MultiHeadAttentionImproved(d_model, num_heads).to(device)

    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print("Simulating training loop...")
    print()

    for step in range(gradient_accumulation_steps):
        print(f"  Step {step + 1}/{gradient_accumulation_steps}...")

        x = torch.randn(batch_size, mel_len, d_model, device=device, requires_grad=True)
        causal_mask = torch.triu(torch.ones(mel_len, mel_len, device=device) * float('-inf'), diagonal=1)

        output, _ = attn(x, x, x, attn_mask=causal_mask)
        loss = output.sum() / gradient_accumulation_steps

        loss.backward()

        print(f"    Loss: {loss.item():.4f}")

        # Don't zero gradients until the last step (as in real training)

    # Zero gradients (as would happen in optimizer.step())
    attn.zero_grad()

    print()
    print("✅ Gradient accumulation PASSED\n")

except Exception as e:
    print(f"❌ Gradient accumulation FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("TEST 5: Full training step simulation")
print("=" * 80)

try:
    print("Creating attention module with optimizer...")
    attn = MultiHeadAttentionImproved(d_model, num_heads).to(device)
    optimizer = torch.optim.Adam(attn.parameters(), lr=1e-4)

    print("Running one full training step...")
    print()

    # Forward pass
    x = torch.randn(batch_size, mel_len, d_model, device=device, requires_grad=True)
    causal_mask = torch.triu(torch.ones(mel_len, mel_len, device=device) * float('-inf'), diagonal=1)

    print("  Forward...")
    output, _ = attn(x, x, x, attn_mask=causal_mask)
    loss = output.sum()

    print("  Backward...")
    optimizer.zero_grad()
    loss.backward()

    print("  Optimizer step...")
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print()
    print("✅ Full training step PASSED\n")

except Exception as e:
    print(f"❌ Full training step FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("TEST 6: Check tensor allocation limits")
print("=" * 80)
print()

# Calculate actual tensor sizes
print("Tensor size analysis:")
print()

batch_heads = batch_size * num_heads
attn_scores_size = batch_heads * mel_len * mel_len
attn_weights_size = batch_heads * mel_len * mel_len  # Same as scores
attn_output_size = batch_size * mel_len * d_model

print(f"Attention scores: ({batch_heads}, {mel_len}, {mel_len})")
print(f"  Elements: {attn_scores_size:,}")
print(f"  Bytes (fp32): {attn_scores_size * 4 / (1024**3):.3f} GB")
print(f"  < INT_MAX: {attn_scores_size < 2147483647}")
print()

print(f"Attention weights: ({batch_heads}, {mel_len}, {mel_len})")
print(f"  Elements: {attn_weights_size:,}")
print(f"  Bytes (fp32): {attn_weights_size * 4 / (1024**3):.3f} GB")
print(f"  < INT_MAX: {attn_weights_size < 2147483647}")
print()

print(f"Attention output: ({batch_size}, {mel_len}, {d_model})")
print(f"  Elements: {attn_output_size:,}")
print(f"  Bytes (fp32): {attn_output_size * 4 / (1024**3):.3f} GB")
print(f"  < INT_MAX: {attn_output_size < 2147483647}")
print()

# Check for any potential overflow in intermediate calculations
print("Checking for dimension overflow in computations:")
print()

d_k = d_model // num_heads
print(f"d_k (per head): {d_k}")
print(f"Q @ K^T dimensions: ({batch_heads}, {mel_len}, {d_k}) @ ({batch_heads}, {d_k}, {mel_len})")
print(f"  Result: ({batch_heads}, {mel_len}, {mel_len})")
print(f"  Total elements: {batch_heads * mel_len * mel_len:,}")
print()

print(f"Attention @ V dimensions: ({batch_heads}, {mel_len}, {mel_len}) @ ({batch_heads}, {mel_len}, {d_k})")
print(f"  Result: ({batch_heads}, {mel_len}, {d_k})")
print(f"  Total elements: {batch_heads * mel_len * d_k:,}")
print()

print("=" * 80)
print("CRITICAL ANALYSIS")
print("=" * 80)
print()
print("If all tests PASS, the crash is NOT in these operations during")
print("normal forward/backward passes.")
print()
print("Possible causes:")
print("  1. MPS backend bug triggered by specific operation sequence")
print("  2. Memory allocation issue in MPS during buffer creation")
print("  3. Interaction with other model components (encoder, variance predictors)")
print("  4. Mixed precision issues (though it's disabled)")
print("  5. Gradient checkpointing creating temporary tensors")
print("  6. Cumulative memory pressure causing MPS to fail on large allocation")
print()
print("Next steps:")
print("  - Test with full model forward + backward")
print("  - Test with gradient checkpointing enabled")
print("  - Monitor actual MPS memory usage")
print("  - Check if crash happens at specific batch index (281)")
