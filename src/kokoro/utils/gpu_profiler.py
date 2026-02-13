import torch
import contextlib
import time
import logging
import platform
import subprocess
import os

# Optional psutil import with fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create a mock psutil for fallback functionality
    class MockPsutil:
        @staticmethod
        def virtual_memory():
            class VirtualMemory:
                def __init__(self):
                    # Try to get system memory from different sources
                    try:
                        # Linux/macOS
                        if platform.system() == "Darwin":  # macOS
                            result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                                  capture_output=True, text=True)
                            if result.returncode == 0:
                                self.total = int(result.stdout.strip())
                                self.used = self.total // 2  # Rough estimate
                                self.percent = 50.0
                            else:
                                self._set_defaults()
                        elif platform.system() == "Linux":
                            with open('/proc/meminfo', 'r') as f:
                                lines = f.readlines()
                                total_kb = int([line for line in lines if 'MemTotal' in line][0].split()[1])
                                available_kb = int([line for line in lines if 'MemAvailable' in line][0].split()[1])
                                self.total = total_kb * 1024
                                self.used = (total_kb - available_kb) * 1024
                                self.percent = (self.used / self.total) * 100
                        else:
                            self._set_defaults()
                    except:
                        self._set_defaults()

                def _set_defaults(self):
                    # Fallback defaults
                    self.total = 8 * 1024 * 1024 * 1024  # 8GB default
                    self.used = self.total // 2
                    self.percent = 50.0

            return VirtualMemory()

        @staticmethod
        def cpu_count():
            return os.cpu_count() or 4

        @staticmethod
        def cpu_percent(interval=None):
            return 50.0  # Default estimate

        @staticmethod
        def cpu_freq():
            return None

    psutil = MockPsutil()

logger = logging.getLogger(__name__)

class GPUProfiler:
    """GPU profiling utility class with support for CUDA, MPS, and CPU"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.device_type = self._detect_device_type()
        self.device_name = self._get_device_name()
        self.memory_stats = {}
        self.supports_memory_profiling = self._check_memory_profiling_support()

        if self.enabled:
            logger.info(f"GPU Profiler initialized for device: {self.device_name} ({self.device_type})")
            if not self.supports_memory_profiling:
                logger.warning(f"Memory profiling not fully supported on {self.device_type}")

    def _detect_device_type(self) -> str:
        """Detect the available device type"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_device_name(self) -> str:
        """Get the device name based on device type"""
        if self.device_type == "cuda":
            return torch.cuda.get_device_name()
        elif self.device_type == "mps":
            # Get Apple Silicon chip info
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return f"Apple Silicon ({result.stdout.strip()})"
                else:
                    return "Apple Silicon MPS"
            except:
                return "Apple Silicon MPS"
        else:
            return f"CPU ({platform.processor()})"

    def _check_memory_profiling_support(self) -> bool:
        """Check if memory profiling is supported on current device"""
        if self.device_type == "cuda":
            return True
        elif self.device_type == "mps":
            # MPS has limited memory profiling capabilities
            return hasattr(torch.mps, 'current_allocated_memory')
        else:
            return False

    def _get_cuda_memory_info(self):
        """Get CUDA memory information"""
        return {
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated(),
            'total': torch.cuda.get_device_properties(0).total_memory
        }

    def _get_mps_memory_info(self):
        """Get MPS memory information"""
        memory_info = {}

        try:
            # MPS memory tracking (available in PyTorch 2.0+)
            if hasattr(torch.mps, 'current_allocated_memory'):
                memory_info['allocated'] = torch.mps.current_allocated_memory()
            else:
                memory_info['allocated'] = 0

            if hasattr(torch.mps, 'driver_allocated_memory'):
                memory_info['reserved'] = torch.mps.driver_allocated_memory()
            else:
                memory_info['reserved'] = 0

            # Get system memory as approximation for total
            if PSUTIL_AVAILABLE:
                memory_info['total'] = psutil.virtual_memory().total
            else:
                # Fallback: try to get system memory
                try:
                    if platform.system() == "Darwin":  # macOS
                        result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            memory_info['total'] = int(result.stdout.strip())
                        else:
                            memory_info['total'] = 16 * 1024 * 1024 * 1024  # 16GB default
                    else:
                        memory_info['total'] = 16 * 1024 * 1024 * 1024  # 16GB default
                except:
                    memory_info['total'] = 16 * 1024 * 1024 * 1024  # 16GB default

            # MPS doesn't have max_allocated tracking, use current as approximation
            memory_info['max_allocated'] = memory_info['allocated']

        except Exception as e:
            logger.debug(f"Could not get MPS memory info: {e}")
            memory_info = {
                'allocated': 0,
                'reserved': 0,
                'max_allocated': 0,
                'total': 16 * 1024 * 1024 * 1024  # 16GB default
            }

        return memory_info

    def _get_cpu_memory_info(self):
        """Get CPU memory information"""
        try:
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                return {
                    'allocated': vm.used,
                    'reserved': vm.used,  # CPU doesn't distinguish reserved vs allocated
                    'max_allocated': vm.used,
                    'total': vm.total
                }
            else:
                # Fallback memory info
                vm = psutil.virtual_memory()  # Uses our mock psutil
                return {
                    'allocated': vm.used,
                    'reserved': vm.used,
                    'max_allocated': vm.used,
                    'total': vm.total
                }
        except Exception as e:
            logger.debug(f"Could not get CPU memory info: {e}")
            # Ultimate fallback
            default_total = 8 * 1024 * 1024 * 1024  # 8GB
            return {
                'allocated': default_total // 2,
                'reserved': default_total // 2,
                'max_allocated': default_total // 2,
                'total': default_total
            }

    def _get_memory_info(self):
        """Get memory information based on device type"""
        if self.device_type == "cuda":
            return self._get_cuda_memory_info()
        elif self.device_type == "mps":
            return self._get_mps_memory_info()
        else:
            return self._get_cpu_memory_info()

    def _synchronize_device(self):
        """Synchronize device operations"""
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        elif self.device_type == "mps":
            # MPS synchronization
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            else:
                # Fallback: create a small tensor operation to force sync
                torch.mps.empty_cache()
        # CPU doesn't need synchronization

    def _empty_cache(self):
        """Empty device cache"""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        elif self.device_type == "mps":
            torch.mps.empty_cache()
        # CPU doesn't have cache to empty

    @contextlib.contextmanager
    def profile_memory(self, stage_name: str):
        """Context manager for profiling GPU/MPS memory usage"""
        if not self.enabled:
            yield
            return

        # Only profile memory if device supports it
        if not self.supports_memory_profiling:
            start_time = time.time()
            try:
                yield
            finally:
                end_time = time.time()
                self.memory_stats[stage_name] = {
                    'memory_used_mb': 0,
                    'memory_reserved_mb': 0,
                    'peak_memory_mb': 0,
                    'time_ms': (end_time - start_time) * 1000,
                    'memory_after_mb': 0,
                    'memory_reserved_after_mb': 0,
                    'device_type': self.device_type
                }
            return

        self._empty_cache()
        self._synchronize_device()

        # Memory before
        memory_before_info = self._get_memory_info()
        memory_before = memory_before_info['allocated']
        memory_reserved_before = memory_before_info['reserved']

        start_time = time.time()

        try:
            yield
        finally:
            self._synchronize_device()
            end_time = time.time()

            # Memory after
            memory_after_info = self._get_memory_info()
            memory_after = memory_after_info['allocated']
            memory_reserved_after = memory_after_info['reserved']

            # Calculate differences
            memory_used = memory_after - memory_before
            memory_reserved_used = memory_reserved_after - memory_reserved_before

            self.memory_stats[stage_name] = {
                'memory_used_mb': memory_used / (1024 ** 2),
                'memory_reserved_mb': memory_reserved_used / (1024 ** 2),
                'peak_memory_mb': memory_after_info['max_allocated'] / (1024 ** 2),
                'time_ms': (end_time - start_time) * 1000,
                'memory_after_mb': memory_after / (1024 ** 2),
                'memory_reserved_after_mb': memory_reserved_after / (1024 ** 2),
                'device_type': self.device_type
            }

    def log_memory_stats(self, stage_name: str = None):
        """Log current GPU/MPS/CPU memory statistics"""
        if not self.enabled:
            return

        try:
            memory_info = self._get_memory_info()
            current_memory = memory_info['allocated'] / (1024 ** 2)
            peak_memory = memory_info['max_allocated'] / (1024 ** 2)
            reserved_memory = memory_info['reserved'] / (1024 ** 2)

            prefix = f"[{stage_name}] " if stage_name else ""
            device_prefix = f"{self.device_type.upper()}"

            logger.info(f"{prefix}{device_prefix} Memory - Current: {current_memory:.1f}MB, "
                       f"Peak: {peak_memory:.1f}MB, Reserved: {reserved_memory:.1f}MB")

        except Exception as e:
            logger.debug(f"Could not log memory stats: {e}")

    def get_memory_summary(self) -> dict:
        """Get comprehensive memory usage summary"""
        try:
            memory_info = self._get_memory_info()

            summary = {
                'device_type': self.device_type,
                'device_name': self.device_name,
                'supports_memory_profiling': self.supports_memory_profiling,
                'current_memory_mb': memory_info['allocated'] / (1024 ** 2),
                'peak_memory_mb': memory_info['max_allocated'] / (1024 ** 2),
                'reserved_memory_mb': memory_info['reserved'] / (1024 ** 2),
                'total_memory_mb': memory_info['total'] / (1024 ** 2),
                'stage_stats': self.memory_stats
            }

            # Add device-specific information
            if self.device_type == "cuda":
                summary['cuda_device_count'] = torch.cuda.device_count()
                summary['cuda_version'] = torch.version.cuda
            elif self.device_type == "mps":
                summary['mps_available'] = torch.backends.mps.is_available()
                summary['mps_built'] = torch.backends.mps.is_built()
            else:
                if PSUTIL_AVAILABLE:
                    summary['cpu_count'] = psutil.cpu_count()
                    cpu_freq = psutil.cpu_freq()
                    summary['cpu_freq'] = cpu_freq._asdict() if cpu_freq else None
                else:
                    summary['cpu_count'] = os.cpu_count() or 4
                    summary['cpu_freq'] = None
                    summary['psutil_available'] = False

            return summary

        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {
                "error": f"Could not get memory summary: {e}",
                "device_type": self.device_type,
                "device_name": self.device_name
            }

    def reset_peak_memory_stats(self):
        """Reset peak memory statistics"""
        try:
            if self.device_type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            elif self.device_type == "mps":
                # MPS doesn't have peak memory reset, just empty cache
                torch.mps.empty_cache()
            # CPU doesn't have peak memory stats to reset

            logger.debug(f"Reset peak memory stats for {self.device_type}")
        except Exception as e:
            logger.debug(f"Could not reset peak memory stats: {e}")

    def get_device_utilization(self) -> dict:
        """Get device utilization information"""
        utilization = {
            'device_type': self.device_type,
            'timestamp': time.time()
        }

        try:
            if self.device_type == "cuda":
                # CUDA utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    utilization.update({
                        'gpu_utilization_percent': gpu_util.gpu,
                        'memory_utilization_percent': gpu_util.memory,
                        'memory_used_mb': memory_info.used / (1024 ** 2),
                        'memory_total_mb': memory_info.total / (1024 ** 2)
                    })
                except ImportError:
                    logger.debug("pynvml not available for CUDA utilization")
                    utilization['gpu_utilization_percent'] = None

            elif self.device_type == "mps":
                # MPS utilization (approximate using system metrics)
                memory_info = self._get_mps_memory_info()
                utilization.update({
                    'gpu_utilization_percent': None,  # Not available for MPS
                    'memory_utilization_percent': None,
                    'memory_used_mb': memory_info['allocated'] / (1024 ** 2),
                    'memory_total_mb': memory_info['total'] / (1024 ** 2)
                })

            else:
                # CPU utilization
                if PSUTIL_AVAILABLE:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()

                    utilization.update({
                        'cpu_utilization_percent': cpu_percent,
                        'memory_utilization_percent': memory.percent,
                        'memory_used_mb': memory.used / (1024 ** 2),
                        'memory_total_mb': memory.total / (1024 ** 2)
                    })
                else:
                    # Fallback CPU utilization (less accurate)
                    memory = psutil.virtual_memory()  # Uses our mock
                    utilization.update({
                        'cpu_utilization_percent': 50.0,  # Estimate
                        'memory_utilization_percent': memory.percent,
                        'memory_used_mb': memory.used / (1024 ** 2),
                        'memory_total_mb': memory.total / (1024 ** 2),
                        'note': 'CPU utilization estimated (psutil not available)'
                    })

        except Exception as e:
            logger.debug(f"Could not get device utilization: {e}")
            utilization['error'] = str(e)

        return utilization

    def profile_model_size(self, model) -> dict:
        """Profile model size and parameter distribution"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Get parameter sizes by layer type
        layer_stats = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    layer_type = type(module).__name__
                    if layer_type not in layer_stats:
                        layer_stats[layer_type] = {'count': 0, 'params': 0}
                    layer_stats[layer_type]['count'] += 1
                    layer_stats[layer_type]['params'] += module_params

        # Estimate memory footprint
        param_memory_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32 parameter

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_mb': param_memory_mb,
            'layer_statistics': layer_stats,
            'device_type': self.device_type
        }


# Example usage and testing functions
def test_profiler():
    """Test the profiler with different device types"""
    profiler = GPUProfiler(enabled=True)

    print(f"Device: {profiler.device_name} ({profiler.device_type})")
    print(f"Memory profiling supported: {profiler.supports_memory_profiling}")

    # Test memory profiling
    with profiler.profile_memory("test_operation"):
        # Create some tensors to use memory
        device = torch.device(profiler.device_type if profiler.device_type != "cpu" else "cpu")
        x = torch.randn(1000, 1000, device=device)
        y = torch.mm(x, x.t())
        del x, y

    # Log memory stats
    profiler.log_memory_stats("after_test")

    # Get memory summary
    summary = profiler.get_memory_summary()
    print("\nMemory Summary:")
    for key, value in summary.items():
        if key != 'stage_stats':
            print(f"  {key}: {value}")

    # Get utilization
    utilization = profiler.get_device_utilization()
    print("\nDevice Utilization:")
    for key, value in utilization.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_profiler()
