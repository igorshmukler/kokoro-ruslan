#!/usr/bin/env python3
"""
Test multi-layer attention to reproduce INT_MAX crash
"""
import sys
sys.path.insert(0, 'src')

import pytest
import torch
import torch.nn as nn
from kokoro.model.transformers import MultiHeadAttentionImproved, ImprovedTransformerDecoderBlock


def _test_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')


def _maybe_clear_mps_cache(device):
    if device.type == 'mps' and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def test_single_attention_layer():
    """Test single attention layer"""
    print("\n" + "="*80)
    print("TEST 1: Single MultiHeadAttentionImproved Layer")
    print("="*80)

    device = _test_device()
    batch_size = 20
    seq_len = 304
    d_model = 512
    num_heads = 8

    attn = MultiHeadAttentionImproved(d_model, num_heads, use_relative_pos=False).to(device)

    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)

    try:
        print(f"Forward...")
        output, _, _ = attn(x, x, x, attn_mask=causal_mask)
        print(f"  ✅ Forward: {output.shape}")

        print(f"Backward...")
        loss = output.sum()
        loss.backward()
        print(f"  ✅ Backward: {x.grad.shape}")
    except RuntimeError as e:
        if "INT_MAX" in str(e):
            print(f"  ❌ INT_MAX CRASH in single attention layer")
        print(f"  Error: {e}")
        pytest.fail(f"RuntimeError in single attention layer test: {e}")


def test_decoder_block():
    """Test full decoder block (self-attention + cross-attention + FFN)"""
    print("\n" + "="*80)
    print("TEST 2: Full Decoder Block")
    print("="*80)

    device = _test_device()
    batch_size = 20
    tgt_len = 304
    src_len = 56  # encoder length from batch 281
    d_model = 512
    num_heads = 8
    d_ff = 2048

    decoder_block = ImprovedTransformerDecoderBlock(
        d_model, num_heads, d_ff, dropout=0.1
    ).to(device)

    tgt = torch.randn(batch_size, tgt_len, d_model, device=device, requires_grad=True)
    memory = torch.randn(batch_size, src_len, d_model, device=device, requires_grad=True)
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device) * float('-inf'), diagonal=1)

    try:
        print(f"Forward...")
        output, _ = decoder_block(tgt, memory, tgt_mask=tgt_mask)
        print(f"  ✅ Forward: {output.shape}")

        print(f"Backward...")
        loss = output.sum()
        loss.backward()
        print(f"  ✅ Backward: tgt.grad={tgt.grad.shape}, memory.grad={memory.grad.shape}")
    except RuntimeError as e:
        if "INT_MAX" in str(e):
            print(f"  ❌ INT_MAX CRASH in decoder block")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"RuntimeError in decoder block test: {e}")


def test_multiple_decoder_blocks():
    """Test stacked decoder blocks"""
    print("\n" + "="*80)
    print("TEST 3: Multiple Decoder Blocks (6 layers)")
    print("="*80)

    device = _test_device()
    batch_size = 20
    tgt_len = 304
    src_len = 56
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6

    decoder_blocks = nn.ModuleList([
        ImprovedTransformerDecoderBlock(d_model, num_heads, d_ff, dropout=0.1)
        for _ in range(num_layers)
    ]).to(device)

    tgt = torch.randn(batch_size, tgt_len, d_model, device=device, requires_grad=True)
    memory = torch.randn(batch_size, src_len, d_model, device=device, requires_grad=True)
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device) * float('-inf'), diagonal=1)

    try:
        print(f"Forward through {num_layers} layers...")
        x = tgt
        for i, block in enumerate(decoder_blocks):
            x, _ = block(x, memory, tgt_mask=tgt_mask)
            print(f"  Layer {i+1}: {x.shape}")

        print(f"\nBackward through {num_layers} layers...")
        loss = x.sum()
        loss.backward()
        print(f"  ✅ Backward: tgt.grad={tgt.grad.shape}")
    except RuntimeError as e:
        if "INT_MAX" in str(e):
            print(f"  ❌ INT_MAX CRASH in multi-layer decoder")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"RuntimeError in multi-layer decoder test: {e}")


def test_with_gradient_checkpointing():
    """Test with gradient checkpointing (as used in training)"""
    print("\n" + "="*80)
    print("TEST 4: With Gradient Checkpointing")
    print("="*80)

    device = _test_device()
    batch_size = 20
    tgt_len = 304
    src_len = 56
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6

    decoder_blocks = nn.ModuleList([
        ImprovedTransformerDecoderBlock(d_model, num_heads, d_ff, dropout=0.1)
        for _ in range(num_layers)
    ]).to(device)

    tgt = torch.randn(batch_size, tgt_len, d_model, device=device, requires_grad=True)
    memory = torch.randn(batch_size, src_len, d_model, device=device, requires_grad=True)
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device) * float('-inf'), diagonal=1)

    def run_segment(start_idx, end_idx, x):
        for i in range(start_idx, end_idx):
            x, _ = decoder_blocks[i](x, memory, tgt_mask=tgt_mask)
        return x

    try:
        print(f"Forward with checkpointing...")
        from torch.utils.checkpoint import checkpoint

        # Checkpoint in 2 segments (layers 0-2, 3-5)
        x = checkpoint(lambda x: run_segment(0, 3, x), tgt, use_reentrant=False)
        x = checkpoint(lambda x: run_segment(3, 6, x), x, use_reentrant=False)

        print(f"  ✅ Forward: {x.shape}")

        print(f"Backward with checkpointing...")
        loss = x.sum()
        loss.backward()
        print(f"  ✅ Backward: tgt.grad={tgt.grad.shape}")
    except RuntimeError as e:
        if "INT_MAX" in str(e):
            print(f"  ❌ INT_MAX CRASH with gradient checkpointing")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"RuntimeError in checkpointing test: {e}")


def test_accumulated_gradients():
    """Test with gradient accumulation (multiple backward passes)"""
    print("\n" + "="*80)
    print("TEST 5: Gradient Accumulation (4 steps)")
    print("="*80)

    device = _test_device()
    batch_size = 20
    seq_len = 304
    d_model = 512
    num_heads = 8

    attn = MultiHeadAttentionImproved(d_model, num_heads, use_relative_pos=False).to(device)

    try:
        for step in range(4):
            print(f"\nAccumulation step {step+1}/4...")
            x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)

            output, _, _ = attn(x, x, x, attn_mask=causal_mask)
            loss = output.sum() / 4  # Scale for accumulation
            loss.backward()

            print(f"  ✅ Step {step+1} complete")

            del x, output, loss
            _maybe_clear_mps_cache(device)
    except RuntimeError as e:
        if "INT_MAX" in str(e):
            print(f"  ❌ INT_MAX CRASH during gradient accumulation step {step+1}")
        print(f"  Error: {e}")
        pytest.fail(f"RuntimeError during gradient accumulation test: {e}")


def main():
    print("\n" + "#"*80)
    print("# MULTI-LAYER ATTENTION TESTS - INT_MAX CRASH INVESTIGATION")
    print("#"*80)

    tests = [
        ("Single Attention Layer", test_single_attention_layer),
        ("Full Decoder Block", test_decoder_block),
        ("Multiple Decoder Blocks", test_multiple_decoder_blocks),
        ("Gradient Checkpointing", test_with_gradient_checkpointing),
        ("Gradient Accumulation", test_accumulated_gradients),
    ]

    results = {}
    for test_name, test_fn in tests:
        try:
            success = test_fn()
            results[test_name] = success
            if not success:
                print(f"\n⚠️  CRASH DETECTED in: {test_name}")
                break
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
            break

    print("\n" + "#"*80)
    print("# TEST RESULTS SUMMARY")
    print("#"*80)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    if all(results.values()):
        print("\n🎉 All tests passed")
        print("   INT_MAX crash not reproduced in isolated tests")
        print("   Crash may be specific to full training pipeline")
    else:
        failed_test = [name for name, success in results.items() if not success][0]
        print(f"\n🎯 CRASH REPRODUCED in: {failed_test}")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
