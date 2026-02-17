#!/usr/bin/env python3
"""
Targeted unit tests for attention operations to find INT_MAX crash
"""
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F


def _test_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')


def _maybe_clear_mps_cache(device):
    if device.type == 'mps' and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def test_matmul_operations():
    """Test matmul operations with batch 281 dimensions"""
    print("\n" + "="*80)
    print("TEST 1: Matrix Multiplication Operations")
    print("="*80)

    device = _test_device()
    batch_size = 20
    num_heads = 8
    seq_len = 304
    d_k = 64

    print(f"\nDimensions: batch={batch_size}, heads={num_heads}, seq={seq_len}, d_k={d_k}")

    # Test 1a: Q @ K^T (this creates the attention scores matrix)
    print(f"\nTest 1a: Q @ K^T")
    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)

    print(f"  Q shape: {Q.shape}")
    print(f"  K^T shape: {K.transpose(-2, -1).shape}")

    try:
        scores = torch.matmul(Q, K.transpose(-2, -1))
        print(f"  ✅ Forward: scores shape = {scores.shape}")

        # Test backward
        loss = scores.sum()
        loss.backward()
        print(f"  ✅ Backward: Q.grad shape = {Q.grad.shape}")

    except RuntimeError as e:
        print(f"  ❌ CRASH: {e}")
        if "INT_MAX" in str(e):
            print(f"  🎯 INT_MAX crash in Q @ K^T operation")
            return False

    # Test 1b: Attention @ V
    print(f"\nTest 1b: Attention @ V")
    attn = torch.randn(batch_size, num_heads, seq_len, seq_len, device=device, requires_grad=True)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)

    try:
        output = torch.matmul(attn, V)
        print(f"  ✅ Forward: output shape = {output.shape}")

        loss = output.sum()
        loss.backward()
        print(f"  ✅ Backward: attn.grad shape = {attn.grad.shape}")

    except RuntimeError as e:
        print(f"  ❌ CRASH: {e}")
        if "INT_MAX" in str(e):
            print(f"  🎯 INT_MAX crash in Attention @ V operation")
            return False

    return True


def test_softmax_operation():
    """Test softmax on large attention scores"""
    print("\n" + "="*80)
    print("TEST 2: Softmax Operation")
    print("="*80)

    device = _test_device()
    batch_size = 20
    num_heads = 8
    seq_len = 304

    scores = torch.randn(batch_size, num_heads, seq_len, seq_len, device=device, requires_grad=True)
    print(f"\nScores shape: {scores.shape}")
    print(f"Total elements: {scores.numel():,}")

    try:
        print(f"\nTest 2a: Softmax forward...")
        attn_weights = F.softmax(scores, dim=-1)
        print(f"  ✅ Forward: attn_weights shape = {attn_weights.shape}")

        print(f"\nTest 2b: Softmax backward...")
        loss = attn_weights.sum()
        loss.backward()
        print(f"  ✅ Backward: scores.grad shape = {scores.grad.shape}")

    except RuntimeError as e:
        print(f"  ❌ CRASH: {e}")
        if "INT_MAX" in str(e):
            print(f"  🎯 INT_MAX crash in softmax operation")
            return False

    return True


def test_scaled_attention():
    """Test full scaled dot-product attention"""
    print("\n" + "="*80)
    print("TEST 3: Scaled Dot-Product Attention")
    print("="*80)

    device = _test_device()
    batch_size = 20
    num_heads = 8
    seq_len = 304
    d_k = 64

    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)

    scale = (d_k ** 0.5)

    try:
        print(f"\nTest 3a: Forward pass...")
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        print(f"  ✅ Forward: output shape = {output.shape}")

        print(f"\nTest 3b: Backward pass...")
        loss = output.sum()
        loss.backward()
        print(f"  ✅ Backward: Q.grad shape = {Q.grad.shape}")

    except RuntimeError as e:
        print(f"  ❌ CRASH: {e}")
        if "INT_MAX" in str(e):
            print(f"  🎯 INT_MAX crash in scaled attention")
            return False

    return True


def test_with_causal_mask():
    """Test attention with causal mask (decoder)"""
    print("\n" + "="*80)
    print("TEST 4: Attention with Causal Mask")
    print("="*80)

    device = _test_device()
    batch_size = 20
    num_heads = 8
    seq_len = 304
    d_k = 64

    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)

    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
    scale = (d_k ** 0.5)

    try:
        print(f"\nTest 4a: Forward with mask...")
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        print(f"  ✅ Forward: output shape = {output.shape}")

        print(f"\nTest 4b: Backward with mask...")
        loss = output.sum()
        loss.backward()
        print(f"  ✅ Backward: Q.grad shape = {Q.grad.shape}")

    except RuntimeError as e:
        print(f"  ❌ CRASH: {e}")
        if "INT_MAX" in str(e):
            print(f"  🎯 INT_MAX crash with causal mask")
            return False

    return True


def test_gradient_checkpointing():
    """Test with gradient checkpointing"""
    print("\n" + "="*80)
    print("TEST 5: With Gradient Checkpointing")
    print("="*80)

    device = _test_device()
    batch_size = 20
    num_heads = 8
    seq_len = 304
    d_k = 64

    def attention_fn(Q, K, V):
        scale = (d_k ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)

    try:
        print(f"\nTest 5a: Forward with checkpointing...")
        from torch.utils.checkpoint import checkpoint
        output = checkpoint(attention_fn, Q, K, V, use_reentrant=False)
        print(f"  ✅ Forward: output shape = {output.shape}")

        print(f"\nTest 5b: Backward with checkpointing...")
        loss = output.sum()
        loss.backward()
        print(f"  ✅ Backward: Q.grad shape = {Q.grad.shape}")

    except RuntimeError as e:
        print(f"  ❌ CRASH: {e}")
        if "INT_MAX" in str(e):
            print(f"  🎯 INT_MAX crash with gradient checkpointing")
            return False

    return True


def test_different_batch_sizes():
    """Test with different batch sizes to find threshold"""
    print("\n" + "="*80)
    print("TEST 6: Different Batch Sizes")
    print("="*80)

    device = _test_device()
    num_heads = 8
    seq_len = 304
    d_k = 64
    scale = (d_k ** 0.5)

    for batch_size in [1, 5, 10, 15, 20, 25]:
        print(f"\nTest 6.{batch_size}: batch_size={batch_size}")

        try:
            Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)
            K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)
            V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, requires_grad=True)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)

            loss = output.sum()
            loss.backward()

            total_elements = batch_size * num_heads * seq_len * seq_len
            print(f"  ✅ Success: {total_elements:,} elements in attention matrix")

            del Q, K, V, scores, attn_weights, output, loss
            _maybe_clear_mps_cache(device)

        except RuntimeError as e:
            if "INT_MAX" in str(e):
                print(f"  ❌ CRASH at batch_size={batch_size}")
                print(f"  🎯 Threshold found: crashes when batch_size >= {batch_size}")
                return False
            else:
                print(f"  ❌ Different error: {e}")
                return False

    return True


def main():
    print("\n" + "#"*80)
    print("# TRANSFORMER ATTENTION UNIT TESTS - INT_MAX CRASH INVESTIGATION")
    print("#"*80)

    tests = [
        ("Matrix Multiplication", test_matmul_operations),
        ("Softmax Operation", test_softmax_operation),
        ("Scaled Attention", test_scaled_attention),
        ("Causal Mask", test_with_causal_mask),
        ("Gradient Checkpointing", test_gradient_checkpointing),
        ("Batch Size Threshold", test_different_batch_sizes),
    ]

    results = {}
    for test_name, test_fn in tests:
        try:
            success = test_fn()
            results[test_name] = success
            if not success:
                print(f"\n⚠️  CRASH DETECTED in: {test_name}")
                print(f"⚠️  Stopping tests to report findings")
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
        print("\n🎉 All tests passed - crash not reproduced")
        print("   The crash may require specific training conditions or has been fixed")
        return True
    else:
        failed_test = [name for name, success in results.items() if not success][0]
        print(f"\n🎯 CRASH REPRODUCED in: {failed_test}")
        print(f"   This identifies the specific operation causing INT_MAX overflow")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
