import gc
import logging
from typing import Any, Callable, Dict, Tuple

import torch


class RuntimeStepPolicy:
    """Policy service for optimizer/scaler/scheduler/EMA stepping."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def optimizer_step_with_clipping(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        use_mixed_precision: bool,
        device_type: str,
        scaler: Any,
        mixed_precision_stats: Dict[str, int],
        clip_norm: float,
        step_scheduler: bool,
        scheduler_per_batch: bool,
        step_scheduler_fn: Callable[[], None],
        update_ema: bool,
        update_ema_fn: Callable[[], None],
    ) -> Tuple[bool, float]:
        step_successful = False
        post_clip_norm = 0.0

        if use_mixed_precision:
            if device_type == 'cuda':
                scaler.unscale_(optimizer)
                pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                post_clip_norm = min(float(pre_clip), clip_norm)

                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()

                # CUDA GradScaler skips optimizer update on overflow and lowers scale.
                step_successful = not (new_scale < old_scale)

                if new_scale != old_scale:
                    mixed_precision_stats['scale_updates'] += 1
                    if new_scale < old_scale:
                        mixed_precision_stats['scale_decreases'] += 1
                        mixed_precision_stats['overflow_count'] += 1
                    else:
                        mixed_precision_stats['successful_steps'] += 1
                else:
                    mixed_precision_stats['successful_steps'] += 1
            else:
                pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                post_clip_norm = min(float(pre_clip), clip_norm)

                old_scale = scaler.get_scale()
                step_successful = scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()

                if step_successful:
                    mixed_precision_stats['successful_steps'] += 1
                else:
                    mixed_precision_stats['skipped_steps'] += 1
                    mixed_precision_stats['overflow_count'] += 1

                if new_scale != old_scale:
                    mixed_precision_stats['scale_updates'] += 1
                    if new_scale < old_scale:
                        mixed_precision_stats['scale_decreases'] += 1
        else:
            pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            post_clip_norm = min(float(pre_clip), clip_norm)
            optimizer.step()
            step_successful = True

        if step_successful and step_scheduler and scheduler_per_batch:
            step_scheduler_fn()

        if step_successful and update_ema:
            update_ema_fn()

        return step_successful, post_clip_norm


class MemoryOOMPolicy:
    """Policy service for adaptive memory cleanup and OOM recovery decisions."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def adaptive_cleanup(
        self,
        *,
        enable_adaptive_memory: bool,
        memory_manager: Any,
        batch_idx: int,
        force: bool,
        clear_device_cache_fn: Callable[[], None],
    ) -> Dict[str, Any]:
        if enable_adaptive_memory:
            return memory_manager.adaptive_cleanup(batch_idx, force)

        if batch_idx % 200 == 0 and batch_idx > 0:
            clear_device_cache_fn()
        return {'cleaned': False, 'pressure_level': 'disabled'}

    def handle_oom(
        self,
        *,
        enable_adaptive_memory: bool,
        memory_manager: Any,
        batch_idx: int,
        error: Exception,
        device_type: str,
        clear_device_cache_fn: Callable[[], None],
        gc_collect_fn: Callable[[], int] = gc.collect,
    ) -> bool:
        self.logger.error(f"OOM error at batch {batch_idx} on {device_type}: {error}")

        if enable_adaptive_memory:
            cleanup_result = memory_manager.emergency_cleanup()
            if cleanup_result['success']:
                self.logger.info(f"Emergency cleanup freed {cleanup_result['memory_freed_mb']:.1f}MB")
                return True

            self.logger.error("Emergency cleanup failed to free significant memory")
            return False

        clear_device_cache_fn()
        gc_collect_fn()
        return True
