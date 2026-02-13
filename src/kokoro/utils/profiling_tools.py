#!/usr/bin/env python3
"""
Profiling utilities for Kokoro Language Model
Provides standalone profiling functions for training and inference analysis
"""

import os
import time
import torch
import logging
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def profile_model_training(model, dataloader: DataLoader, num_steps: int = 10,
                          output_dir: str = "./training_profile") -> Dict[str, Any]:
    """
    Profile model training for a specified number of steps

    Args:
        model: The Kokoro model instance
        dataloader: Training dataloader
        num_steps: Number of training steps to profile
        output_dir: Directory to save profiling results

    Returns:
        Dictionary containing profiling report
    """
    logger.info(f"Starting training profiling for {num_steps} steps")

    # Reset profiling stats if available
    if hasattr(model, 'reset_profiling_stats'):
        model.reset_profiling_stats()

    # Start profiler
    os.makedirs(output_dir, exist_ok=True)
    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=num_steps-2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        with_stack=True,
        profile_memory=True,
        record_shapes=True,
        with_flops=True
    )

    model.train()
    total_time = 0
    step_times = []

    with profiler:
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            start_time = time.time()

            try:
                # Extract batch data (adjust according to your dataloader format)
                with torch.profiler.record_function("Data_Preparation"):
                    phoneme_indices = batch['phoneme_indices'].to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
                    mel_specs = batch['mel_specs'].to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
                    phoneme_durations = batch['phoneme_durations'].to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
                    stop_token_targets = batch['stop_token_targets'].to(model.device if hasattr(model, 'device') else next(model.parameters()).device)

                # Forward pass
                with torch.profiler.record_function("Forward_Pass"):
                    outputs = model(
                        phoneme_indices=phoneme_indices,
                        mel_specs=mel_specs,
                        phoneme_durations=phoneme_durations,
                        stop_token_targets=stop_token_targets
                    )

                # Simulate loss computation and backward pass
                with torch.profiler.record_function("Loss_Computation"):
                    if isinstance(outputs, tuple) and len(outputs) >= 3:
                        predicted_mel, predicted_durations, predicted_stop = outputs[:3]
                        # Simple loss simulation for profiling
                        loss = torch.mean(predicted_mel) + torch.mean(predicted_durations) + torch.mean(predicted_stop)
                    else:
                        # Fallback for different output formats
                        loss = torch.mean(torch.stack([torch.mean(o) for o in outputs if torch.is_tensor(o)]))

                with torch.profiler.record_function("Backward_Pass"):
                    loss.backward()

                step_time = time.time() - start_time
                total_time += step_time
                step_times.append(step_time)

                # Step the profiler
                profiler.step()

                # Log memory stats if available
                if hasattr(model, 'log_memory_stats'):
                    model.log_memory_stats(f"training_step_{step}")

                if step % 2 == 0:
                    logger.info(f"Step {step}, Time: {step_time:.3f}s")

            except Exception as e:
                logger.error(f"Error in profiling step {step}: {e}")
                continue

    # Generate profiling report
    report = {
        'total_time': total_time,
        'average_step_time': total_time / len(step_times) if step_times else 0,
        'step_times': step_times,
        'num_steps': len(step_times),
        'profiling_type': 'training'
    }

    # Add model-specific profiling report if available
    if hasattr(model, 'get_profiling_report'):
        model_report = model.get_profiling_report()
        report.update(model_report)

    logger.info(f"Training profiling completed. Total time: {total_time:.2f}s, "
               f"Avg time per step: {report['average_step_time']:.3f}s")

    return report


def profile_model_inference(model, phoneme_indices: torch.Tensor, max_samples: int = 5,
                           output_dir: str = "./inference_profile") -> tuple[Dict[str, Any], List[float]]:
    """
    Profile model inference for a specified number of samples

    Args:
        model: The Kokoro model instance
        phoneme_indices: Input phoneme indices tensor
        max_samples: Number of inference samples to profile
        output_dir: Directory to save profiling results

    Returns:
        Tuple of (profiling report dict, list of inference times)
    """
    logger.info(f"Starting inference profiling for {max_samples} samples")

    # Reset profiling stats if available
    if hasattr(model, 'reset_profiling_stats'):
        model.reset_profiling_stats()

    # Start profiler
    os.makedirs(output_dir, exist_ok=True)
    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=max_samples-2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        with_stack=True,
        profile_memory=True,
        record_shapes=True,
        with_flops=True
    )

    model.eval()
    inference_times = []
    device = next(model.parameters()).device

    with profiler:
        with torch.no_grad():
            for i in range(max_samples):
                start_time = time.time()

                try:
                    with torch.profiler.record_function(f"Inference_Sample_{i}"):
                        # Ensure input is on correct device
                        input_indices = phoneme_indices[i:i+1].to(device) if phoneme_indices.dim() > 1 else phoneme_indices.unsqueeze(0).to(device)

                        # Run inference
                        output = model(phoneme_indices=input_indices)

                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)

                    # Step the profiler
                    profiler.step()

                    # Log memory stats if available
                    if hasattr(model, 'log_memory_stats'):
                        model.log_memory_stats(f"inference_sample_{i}")

                    # Determine output shape for logging
                    if isinstance(output, tuple):
                        output_shape = output[0].shape if len(output) > 0 else "Unknown"
                    else:
                        output_shape = output.shape

                    logger.info(f"Inference {i+1}: {inference_time:.3f}s, "
                               f"Output shape: {output_shape}")

                except Exception as e:
                    logger.error(f"Error in inference sample {i}: {e}")
                    continue

    # Generate profiling report
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

    report = {
        'average_inference_time': avg_inference_time,
        'inference_times': inference_times,
        'num_samples': len(inference_times),
        'total_inference_time': sum(inference_times),
        'profiling_type': 'inference'
    }

    # Add model-specific profiling report if available
    if hasattr(model, 'get_profiling_report'):
        model_report = model.get_profiling_report()
        report.update(model_report)

    logger.info(f"Inference profiling completed. Avg inference time: {avg_inference_time:.3f}s")

    return report, inference_times


def analyze_profiling_results(profiling_report: Dict[str, Any], print_detailed: bool = True):
    """
    Analyze and print profiling results in a readable format

    Args:
        profiling_report: Dictionary containing profiling results
        print_detailed: Whether to print detailed analysis
    """
    if not print_detailed:
        return

    print("\n" + "="*60)
    print("PROFILING ANALYSIS REPORT")
    print("="*60)

    # Basic profiling info
    profiling_type = profiling_report.get('profiling_type', 'unknown')
    print(f"\nProfiling Type: {profiling_type.upper()}")

    if profiling_type == 'training':
        print(f"Total Steps: {profiling_report.get('num_steps', 0)}")
        print(f"Total Time: {profiling_report.get('total_time', 0):.2f}s")
        print(f"Average Step Time: {profiling_report.get('average_step_time', 0):.3f}s")
    elif profiling_type == 'inference':
        print(f"Total Samples: {profiling_report.get('num_samples', 0)}")
        print(f"Total Time: {profiling_report.get('total_inference_time', 0):.2f}s")
        print(f"Average Inference Time: {profiling_report.get('average_inference_time', 0):.3f}s")

    # Device information
    device_info = profiling_report.get('device_info', {})
    if device_info:
        print(f"\nDevice: {device_info.get('device_name', 'Unknown')}")
        print(f"CUDA Available: {device_info.get('cuda_available', False)}")
        print(f"Device Type: {device_info.get('device_type', 'Unknown')}")

    # Memory analysis
    memory_summary = profiling_report.get('memory_summary', {})
    if memory_summary:
        print(f"\nMemory Usage:")
        print(f"  Current: {memory_summary.get('current_memory_mb', 0):.1f} MB")
        print(f"  Peak: {memory_summary.get('peak_memory_mb', 0):.1f} MB")
        print(f"  Reserved: {memory_summary.get('reserved_memory_mb', 0):.1f} MB")
        print(f"  Total GPU: {memory_summary.get('total_memory_mb', 0):.1f} MB")

        # Memory efficiency
        total_memory = memory_summary.get('total_memory_mb', 1)
        peak_memory = memory_summary.get('peak_memory_mb', 0)
        if total_memory > 0:
            memory_efficiency = (peak_memory / total_memory) * 100
            print(f"  Memory Efficiency: {memory_efficiency:.1f}%")

    # Stage-wise analysis
    stage_stats = memory_summary.get('stage_stats', {}) if memory_summary else profiling_report.get('stage_stats', {})
    if stage_stats:
        print(f"\nStage-wise Performance:")
        print(f"{'Stage':<30} {'Memory (MB)':<12} {'Calls':<8}")
        print("-" * 55)

        # Sort by memory usage
        sorted_stages = sorted(stage_stats.items(),
                             key=lambda x: x[1].get('memory_used_mb', 0), reverse=True)

        for stage_name, stats in sorted_stages[:10]:  # Top 10 memory users
            memory_mb = stats.get('memory_used_mb', 0)
            call_count = stats.get('call_count', 0)
            print(f"{stage_name:<30} {memory_mb:>8.1f} {call_count:>8}")

    # Memory analysis summary
    memory_analysis = profiling_report.get('memory_analysis', {})
    if memory_analysis:
        print(f"\nMemory Analysis:")
        print(f"  Most Memory Intensive: {memory_analysis.get('most_memory_intensive_stage', 'N/A')}")
        print(f"  Total Memory Used: {memory_analysis.get('total_memory_used_mb', 0):.1f} MB")

    # Model information
    model_info = profiling_report.get('model_info', {})
    if model_info:
        print(f"\nModel Information:")
        print(f"  Parameters: {model_info.get('total_parameters', 0):,}")
        print(f"  Model Size: {model_info.get('model_size_mb', 0):.1f} MB")
        if 'hidden_dim' in model_info:
            print(f"  Hidden Dim: {model_info.get('hidden_dim', 0)}")
        if 'n_encoder_layers' in model_info:
            print(f"  Encoder Layers: {model_info.get('n_encoder_layers', 0)}")
        if 'n_decoder_layers' in model_info:
            print(f"  Decoder Layers: {model_info.get('n_decoder_layers', 0)}")

    print("\n" + "="*60)
    print("Recommendations:")

    # Generate recommendations based on profiling data
    recommendations = []

    # Memory-based recommendations
    peak_memory = memory_summary.get('peak_memory_mb', 0) if memory_summary else 0
    if peak_memory > 8000:  # > 8GB
        recommendations.append("• Consider reducing batch size or sequence length")
        recommendations.append("• Enable gradient checkpointing for training")

    total_memory = memory_summary.get('total_memory_mb', 1) if memory_summary else 1
    if total_memory > 0 and peak_memory > 0:
        memory_efficiency = (peak_memory / total_memory) * 100
        if memory_efficiency < 50:
            recommendations.append("• GPU memory utilization is low, consider increasing batch size")
        elif memory_efficiency > 90:
            recommendations.append("• High memory usage detected, monitor for OOM errors")

    # Stage-specific recommendations
    most_intensive = memory_analysis.get('most_memory_intensive_stage', '') if memory_analysis else ''
    if most_intensive:
        if 'decoder' in most_intensive.lower():
            recommendations.append("• Decoder is memory intensive, consider optimizing attention computation")
        elif 'encoder' in most_intensive.lower():
            recommendations.append("• Encoder layers using significant memory, consider layer-wise gradient checkpointing")
        elif 'forward' in most_intensive.lower():
            recommendations.append("• Forward pass is memory intensive, consider mixed precision training")
        elif 'backward' in most_intensive.lower():
            recommendations.append("• Backward pass is memory intensive, consider gradient accumulation")

    # Performance-based recommendations
    if profiling_type == 'training':
        avg_step_time = profiling_report.get('average_step_time', 0)
        if avg_step_time > 1.0:  # > 1 second per step
            recommendations.append("• Training steps are slow, consider optimizing data loading or model architecture")
        elif avg_step_time < 0.1:  # < 0.1 second per step
            recommendations.append("• Training steps are fast, consider increasing batch size for better GPU utilization")

    elif profiling_type == 'inference':
        avg_inference_time = profiling_report.get('average_inference_time', 0)
        if avg_inference_time > 0.5:  # > 0.5 second per inference
            recommendations.append("• Inference is slow, consider model quantization or pruning")
            recommendations.append("• Consider using torch.jit.script() for faster inference")

    if not recommendations:
        recommendations.append("• Profiling results look good, no immediate optimizations needed")

    for rec in recommendations:
        print(rec)

    print("="*60)


def save_profiling_report(report: Dict[str, Any], filepath: str):
    """Save profiling report to a file"""
    import json

    # Convert non-serializable objects to serializable format
    serializable_report = {}
    for key, value in report.items():
        if isinstance(value, torch.Tensor):
            serializable_report[key] = value.tolist()
        elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
            serializable_report[key] = value
        else:
            serializable_report[key] = str(value)

    with open(filepath, 'w') as f:
        json.dump(serializable_report, f, indent=2)

    logger.info(f"Profiling report saved to {filepath}")


def load_profiling_report(filepath: str) -> Dict[str, Any]:
    """Load profiling report from a file"""
    import json

    with open(filepath, 'r') as f:
        report = json.load(f)

    logger.info(f"Profiling report loaded from {filepath}")
    return report


def compare_profiling_reports(report1: Dict[str, Any], report2: Dict[str, Any],
                            labels: Optional[List[str]] = None):
    """
    Compare two profiling reports and print the differences

    Args:
        report1: First profiling report
        report2: Second profiling report
        labels: Optional labels for the reports (default: ["Report 1", "Report 2"])
    """
    if labels is None:
        labels = ["Report 1", "Report 2"]

    print("\n" + "="*60)
    print("PROFILING COMPARISON REPORT")
    print("="*60)

    print(f"\nComparing: {labels[0]} vs {labels[1]}")

    # Compare basic metrics
    metrics_to_compare = [
        ('average_step_time', 'Average Step Time (s)'),
        ('average_inference_time', 'Average Inference Time (s)'),
        ('total_time', 'Total Time (s)'),
        ('total_inference_time', 'Total Inference Time (s)')
    ]

    print(f"\n{'Metric':<30} {labels[0]:<15} {labels[1]:<15} {'Difference':<15}")
    print("-" * 80)

    for metric_key, metric_name in metrics_to_compare:
        val1 = report1.get(metric_key, 0)
        val2 = report2.get(metric_key, 0)

        if val1 > 0 or val2 > 0:
            diff = val2 - val1
            diff_pct = (diff / val1 * 100) if val1 > 0 else float('inf')
            diff_str = f"{diff:+.3f} ({diff_pct:+.1f}%)" if abs(diff_pct) != float('inf') else f"{diff:+.3f}"

            print(f"{metric_name:<30} {val1:<15.3f} {val2:<15.3f} {diff_str:<15}")

    # Compare memory usage
    mem1 = report1.get('memory_summary', {})
    mem2 = report2.get('memory_summary', {})

    if mem1 or mem2:
        print(f"\nMemory Comparison:")
        memory_metrics = [
            ('peak_memory_mb', 'Peak Memory (MB)'),
            ('current_memory_mb', 'Current Memory (MB)')
        ]

        for metric_key, metric_name in memory_metrics:
            val1 = mem1.get(metric_key, 0)
            val2 = mem2.get(metric_key, 0)

            if val1 > 0 or val2 > 0:
                diff = val2 - val1
                diff_pct = (diff / val1 * 100) if val1 > 0 else float('inf')
                diff_str = f"{diff:+.1f} ({diff_pct:+.1f}%)" if abs(diff_pct) != float('inf') else f"{diff:+.1f}"

                print(f"  {metric_name:<28} {val1:<15.1f} {val2:<15.1f} {diff_str:<15}")

    print("="*60)


# Example usage functions
def run_training_profiling_example(model, dataloader, steps: int = 10):
    """Run training profiling example"""
    report = profile_model_training(model, dataloader, num_steps=steps)
    analyze_profiling_results(report)
    return report


def run_inference_profiling_example(model, phoneme_indices, samples: int = 5):
    """Run inference profiling example"""
    report, times = profile_model_inference(model, phoneme_indices, max_samples=samples)
    analyze_profiling_results(report)
    return report, times
