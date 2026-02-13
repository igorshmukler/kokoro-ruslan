#!/usr/bin/env python3
"""
MPS Memory Optimization Helper

This script helps configure optimal settings for training on Apple Silicon (MPS).
Run this before training to set environment variables and check your system.
"""

import os
import sys
import subprocess
import torch

def get_system_memory():
    """Get total system memory in GB"""
    try:
        # macOS: use sysctl to get hardware memory
        result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
        if result.returncode == 0:
            memory_bytes = int(result.stdout.split(':')[1].strip())
            memory_gb = memory_bytes / (1024**3)
            return memory_gb
    except:
        pass
    return None

def check_mps_available():
    """Check if MPS is available"""
    if not torch.backends.mps.is_available():
        print("❌ MPS is not available on this system")
        return False

    print("✅ MPS is available")
    return True

def get_recommended_settings(memory_gb):
    """Get recommended training settings based on available memory"""
    if memory_gb is None:
        return None

    if memory_gb >= 32:
        return {
            'max_frames_per_batch': 8000,
            'max_seq_length': 1200,
            'batch_size': 6,
            'max_batch_size': 12,
            'gradient_accumulation_steps': 4,
            'watermark_ratio': 0.75
        }
    elif memory_gb >= 16:
        return {
            'max_frames_per_batch': 6000,
            'max_seq_length': 1000,
            'batch_size': 4,
            'max_batch_size': 8,
            'gradient_accumulation_steps': 6,
            'watermark_ratio': 0.70
        }
    else:
        return {
            'max_frames_per_batch': 4000,
            'max_seq_length': 800,
            'batch_size': 2,
            'max_batch_size': 6,
            'gradient_accumulation_steps': 8,
            'watermark_ratio': 0.65
        }

def print_recommendations(settings):
    """Print recommended settings"""
    if settings is None:
        print("\n⚠️  Could not determine system memory")
        print("Using conservative defaults")
        return

    print("\n" + "="*60)
    print("RECOMMENDED TRAINING SETTINGS FOR YOUR SYSTEM")
    print("="*60)
    print(f"\nTraining Configuration:")
    print(f"  max_frames_per_batch: {settings['max_frames_per_batch']}")
    print(f"  max_seq_length: {settings['max_seq_length']}")
    print(f"  batch_size: {settings['batch_size']}")
    print(f"  max_batch_size: {settings['max_batch_size']}")
    print(f"  gradient_accumulation_steps: {settings['gradient_accumulation_steps']}")

    effective_batch = settings['batch_size'] * settings['gradient_accumulation_steps']
    print(f"\nEffective batch size: {effective_batch}")

    print(f"\nEnvironment Variable:")
    print(f"  PYTORCH_MPS_HIGH_WATERMARK_RATIO={settings['watermark_ratio']}")

    print("\n" + "="*60)
    print("TO USE THESE SETTINGS:")
    print("="*60)

    # Create a training command
    print(f"\nexport PYTORCH_MPS_HIGH_WATERMARK_RATIO={settings['watermark_ratio']}")
    print("kokoro-train \\")
    print("  --corpus ./ruslan_corpus \\")
    print(f"  --batch-size {settings['batch_size']} \\")
    print(f"  --gradient-accumulation {settings['gradient_accumulation_steps']}")

    print("\nOr create a config file:")
    print(f"""
config = TrainingConfig(
    data_dir="./ruslan_corpus",
    batch_size={settings['batch_size']},
    max_frames_per_batch={settings['max_frames_per_batch']},
    max_seq_length={settings['max_seq_length']},
    max_batch_size={settings['max_batch_size']},
    gradient_accumulation_steps={settings['gradient_accumulation_steps']},
)
""")

def set_environment_variable(ratio):
    """Set the MPS watermark ratio"""
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = str(ratio)
    print(f"\n✅ Set PYTORCH_MPS_HIGH_WATERMARK_RATIO={ratio}")
    print("This setting will apply to training jobs started from this session")

def main():
    print("\n" + "="*60)
    print("MPS MEMORY OPTIMIZATION HELPER")
    print("="*60)

    # Check MPS availability
    if not check_mps_available():
        sys.exit(1)

    # Get system memory
    memory_gb = get_system_memory()
    if memory_gb:
        print(f"✅ System memory: {memory_gb:.1f} GB")
    else:
        print("⚠️  Could not detect system memory")

    # Check PyTorch version
    print(f"✅ PyTorch version: {torch.__version__}")

    # Test MPS
    try:
        test_tensor = torch.randn(100, 100, device='mps')
        result = test_tensor @ test_tensor.T
        print("✅ MPS computation test passed")
    except Exception as e:
        print(f"❌ MPS computation test failed: {e}")
        sys.exit(1)

    # Get and print recommendations
    settings = get_recommended_settings(memory_gb)
    print_recommendations(settings)

    # Ask if user wants to set the environment variable
    print("\n" + "="*60)
    response = input("\nSet PYTORCH_MPS_HIGH_WATERMARK_RATIO now? (y/n): ")
    if response.lower() == 'y' and settings:
        set_environment_variable(settings['watermark_ratio'])
        print("\n⚠️  Note: This only affects the current shell session.")
        print("Add to your ~/.zshrc or ~/.bashrc to make permanent:")
        print(f"\nexport PYTORCH_MPS_HIGH_WATERMARK_RATIO={settings['watermark_ratio']}")

    print("\n" + "="*60)
    print("TIPS TO PREVENT OOM:")
    print("="*60)
    print("""
1. Use feature caching to speed up training:
   kokoro-precompute --corpus ./ruslan_corpus

2. Start with conservative settings and increase gradually

3. Monitor memory usage during training:
   - Check the 'mem' indicator in the progress bar
   - If it shows 'high' or 'cri', reduce batch size

4. If you get OOM errors:
   - Reduce max_frames_per_batch by 2000
   - Reduce max_seq_length by 200
   - Increase gradient_accumulation_steps by 2

5. Close other applications to free up memory

6. Consider training with --no-variance to disable pitch/energy
   (saves ~20% memory)
""")

    print("="*60)
    print("\n✅ Ready to train! Run:")
    print("   kokoro-train --corpus ./ruslan_corpus\n")

if __name__ == "__main__":
    main()
