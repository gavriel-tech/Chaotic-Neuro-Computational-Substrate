"""
Performance Optimization Configuration for GMCS + THRML

Centralized module for all performance settings:
- XLA compiler flags
- JAX JIT configuration
- Memory management
- Device settings
- Profiling utilities
"""

import os
import logging
from typing import Optional, Dict, Any
import jax
import jax.numpy as jnp
from jax import config

logger = logging.getLogger(__name__)


# ============================================================================
# XLA Optimization Flags
# ============================================================================

XLA_FLAGS = {
    # Core optimizations
    'xla_gpu_enable_async_collectives': 'true',
    'xla_gpu_enable_latency_hiding_scheduler': 'true',
    'xla_gpu_enable_triton_gemm': 'true',
    
    # Memory optimizations
    'xla_gpu_enable_cub_radix_sort': 'true',
    'xla_gpu_graph_level': '3',
    
    # Compilation
    'xla_gpu_autotune_level': '4',
    'xla_gpu_enable_command_buffer': 'true',
}

JAX_CONFIG = {
    # Memory
    'jax_platform_name': None,  # 'gpu', 'cpu', or None for auto
    'jax_enable_x64': False,  # Use float32 for 2x speedup
    'jax_default_matmul_precision': 'high',  # 'default', 'high', 'highest'
    
    # JIT
    'jax_disable_jit': False,  # Set True for debugging
    'jax_check_tracer_leaks': False,  # Expensive checks
    
    # Memory management
    'jax_platform_memory_fraction': 0.9,  # Use 90% of GPU memory
}


class PerformanceConfig:
    """
    Centralized performance configuration manager.
    
    Handles:
    - XLA flags
    - JAX configuration
    - Memory management
    - Device selection
    - Profiling
    """
    
    def __init__(self):
        self.profile_enabled = False
        self.profile_dir = "./profiles"
        self._original_flags = {}
        
    def apply_optimal_settings(self, mode: str = 'gpu_high_performance'):
        """
        Apply pre-configured optimal settings.
        
        Args:
            mode: Performance mode
                - 'gpu_high_performance': Maximum GPU speed (default)
                - 'gpu_low_memory': Minimize GPU memory usage
                - 'cpu_optimized': Optimized for CPU-only
                - 'debug': Debug mode with checks enabled
        """
        logger.info(f"[Performance] Applying mode: {mode}")
        
        if mode == 'gpu_high_performance':
            self._apply_gpu_high_performance()
        elif mode == 'gpu_low_memory':
            self._apply_gpu_low_memory()
        elif mode == 'cpu_optimized':
            self._apply_cpu_optimized()
        elif mode == 'debug':
            self._apply_debug_mode()
        else:
            raise ValueError(f"Unknown performance mode: {mode}")
        
        logger.info(f"[Performance] Mode '{mode}' applied successfully")
    
    def _apply_gpu_high_performance(self):
        """GPU: Maximum performance, high memory usage."""
        # XLA flags
        for key, value in XLA_FLAGS.items():
            os.environ[f'{key.upper()}'] = str(value)
        
        # JAX config
        config.update('jax_enable_x64', False)  # float32 for speed
        config.update('jax_default_matmul_precision', 'high')
        
        # Memory
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
        
        # Force GPU
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        
        logger.info("[Performance] GPU high-performance mode active")
    
    def _apply_gpu_low_memory(self):
        """GPU: Lower memory usage, slight performance trade-off."""
        # XLA flags (subset)
        os.environ['XLA_GPU_ENABLE_ASYNC_COLLECTIVES'] = 'true'
        
        # JAX config
        config.update('jax_enable_x64', False)
        config.update('jax_default_matmul_precision', 'default')  # Lower precision
        
        # Memory - conservative
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
        
        # GPU
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        
        logger.info("[Performance] GPU low-memory mode active")
    
    def _apply_cpu_optimized(self):
        """CPU: Optimized for CPU-only execution."""
        # JAX config
        config.update('jax_enable_x64', False)
        config.update('jax_default_matmul_precision', 'default')
        
        # Force CPU
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        
        # Threading
        os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true'
        
        logger.info("[Performance] CPU-optimized mode active")
    
    def _apply_debug_mode(self):
        """Debug: Enable all checks (slow)."""
        # Disable JIT for better error messages
        config.update('jax_disable_jit', True)
        config.update('jax_check_tracer_leaks', True)
        config.update('jax_debug_nans', True)
        
        # Conservative memory
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        
        logger.info("[Performance] Debug mode active (JIT disabled)")
    
    def set_memory_fraction(self, fraction: float):
        """
        Set GPU memory fraction.
        
        Args:
            fraction: Fraction of GPU memory to use (0.0 to 1.0)
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Memory fraction must be in (0, 1], got {fraction}")
        
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(fraction)
        logger.info(f"[Performance] GPU memory fraction set to {fraction:.1%}")
    
    def enable_memory_profiling(self):
        """Enable JAX memory profiling."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        os.environ['JAX_LOG_COMPILES'] = '1'
        logger.info("[Performance] Memory profiling enabled")
    
    def disable_memory_profiling(self):
        """Disable JAX memory profiling."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['JAX_LOG_COMPILES'] = '0'
        logger.info("[Performance] Memory profiling disabled")
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get current device configuration.
        
        Returns:
            Dict with device info
        """
        devices = jax.devices()
        default_backend = jax.default_backend()
        
        return {
            'n_devices': len(devices),
            'default_backend': default_backend,
            'devices': [
                {
                    'id': i,
                    'platform': d.platform,
                    'device_kind': d.device_kind,
                }
                for i, d in enumerate(devices)
            ],
            'jax_version': jax.__version__,
            'x64_enabled': config.jax_enable_x64,
            'jit_disabled': config.jax_disable_jit,
        }
    
    def print_device_info(self):
        """Print device information."""
        info = self.get_device_info()
        
        print("\n" + "=" * 60)
        print("JAX Device Configuration")
        print("=" * 60)
        print(f"JAX Version: {info['jax_version']}")
        print(f"Default Backend: {info['default_backend']}")
        print(f"Devices: {info['n_devices']}")
        for dev in info['devices']:
            print(f"  {dev['id']}: {dev['platform']} - {dev['device_kind']}")
        print(f"\nConfiguration:")
        print(f"  x64 enabled: {info['x64_enabled']}")
        print(f"  JIT disabled: {info['jit_disabled']}")
        print("=" * 60 + "\n")
    
    def start_profiler(self, output_dir: Optional[str] = None):
        """
        Start JAX profiler.
        
        Args:
            output_dir: Directory to save traces (default: ./profiles)
        """
        if output_dir:
            self.profile_dir = output_dir
        
        os.makedirs(self.profile_dir, exist_ok=True)
        self.profile_enabled = True
        
        logger.info(f"[Performance] Profiler started, traces in {self.profile_dir}")
    
    def profile_context(self, name: str):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name for the profile trace
            
        Example:
            with perf_config.profile_context('my_function'):
                result = my_function()
        """
        if not self.profile_enabled:
            # No-op if profiling disabled
            from contextlib import nullcontext
            return nullcontext()
        
        import jax.profiler
        trace_path = os.path.join(self.profile_dir, name)
        return jax.profiler.trace(trace_path)
    
    def clear_caches(self):
        """Clear JAX compilation caches."""
        jax.clear_backends()
        logger.info("[Performance] JAX caches cleared")


# ============================================================================
# Global Instance
# ============================================================================

# Global performance config instance
_global_perf_config = PerformanceConfig()


def get_perf_config() -> PerformanceConfig:
    """Get global performance configuration instance."""
    return _global_perf_config


def apply_optimal_settings(mode: str = 'gpu_high_performance'):
    """
    Convenience function to apply optimal settings.
    
    Args:
        mode: Performance mode (see PerformanceConfig.apply_optimal_settings)
    """
    _global_perf_config.apply_optimal_settings(mode)


def print_device_info():
    """Print device information."""
    _global_perf_config.print_device_info()


# ============================================================================
# JIT Utilities
# ============================================================================

def jit_with_cache(
    fn,
    static_argnums=None,
    donate_argnums=None,
    device=None
):
    """
    JIT compile with optimal settings.
    
    Args:
        fn: Function to compile
        static_argnums: Static arguments
        donate_argnums: Arguments to donate (save memory)
        device: Target device
        
    Returns:
        JIT-compiled function
    """
    return jax.jit(
        fn,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        device=device
    )


def vmap_efficient(fn, in_axes=0, out_axes=0):
    """
    Efficient vmap for parallel operations.
    
    Args:
        fn: Function to vectorize
        in_axes: Input axes
        out_axes: Output axes
        
    Returns:
        Vmapped function
    """
    return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)


def pmap_multi_device(fn, axis_name='devices', devices=None):
    """
    Pmap for multi-device execution.
    
    Args:
        fn: Function to parallelize
        axis_name: Name of parallel axis
        devices: Target devices (None = all)
        
    Returns:
        Pmapped function
    """
    return jax.pmap(fn, axis_name=axis_name, devices=devices)


# ============================================================================
# Memory Management
# ============================================================================

def estimate_memory_usage(n_nodes: int, n_samples: int, dtype=jnp.float32) -> Dict[str, float]:
    """
    Estimate memory usage for THRML operations.
    
    Args:
        n_nodes: Number of nodes
        n_samples: Number of samples
        dtype: Data type
        
    Returns:
        Dict with memory estimates (in MB)
    """
    bytes_per_element = jnp.dtype(dtype).itemsize
    
    # States: (n_samples, n_nodes)
    states_mb = (n_samples * n_nodes * bytes_per_element) / (1024**2)
    
    # Weights: (n_nodes, n_nodes)
    weights_mb = (n_nodes * n_nodes * bytes_per_element) / (1024**2)
    
    # Biases: (n_nodes,)
    biases_mb = (n_nodes * bytes_per_element) / (1024**2)
    
    # Energy computation (intermediate): (n_samples,)
    energy_mb = (n_samples * bytes_per_element) / (1024**2)
    
    # Total (with 2x overhead for JAX internals)
    total_mb = (states_mb + weights_mb + biases_mb + energy_mb) * 2.0
    
    return {
        'states_mb': states_mb,
        'weights_mb': weights_mb,
        'biases_mb': biases_mb,
        'energy_mb': energy_mb,
        'total_mb': total_mb,
        'total_gb': total_mb / 1024,
    }


def check_memory_available(required_mb: float) -> bool:
    """
    Check if sufficient GPU memory available.
    
    Args:
        required_mb: Required memory in MB
        
    Returns:
        True if sufficient memory available
    """
    devices = jax.devices()
    
    if not devices:
        return False
    
    # This is a simplified check
    # In practice, you'd query actual GPU memory
    # For now, assume at least 8GB available on GPU
    available_mb = 8 * 1024  # 8GB
    
    return required_mb < available_mb


# ============================================================================
# Benchmarking Utilities
# ============================================================================

def benchmark_function(fn, *args, n_warmup=3, n_runs=10, **kwargs):
    """
    Benchmark a function.
    
    Args:
        fn: Function to benchmark
        *args: Function arguments
        n_warmup: Number of warmup runs
        n_runs: Number of benchmark runs
        **kwargs: Function keyword arguments
        
    Returns:
        Dict with timing statistics
    """
    import time
    
    # Warmup
    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
        if isinstance(result, jnp.ndarray):
            result.block_until_ready()
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        t_start = time.time()
        result = fn(*args, **kwargs)
        if isinstance(result, jnp.ndarray):
            result.block_until_ready()
        elapsed = time.time() - t_start
        times.append(elapsed)
    
    import numpy as np
    times = np.array(times)
    
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'median': float(np.median(times)),
        'n_runs': n_runs,
    }


# ============================================================================
# Auto-initialization
# ============================================================================

def auto_configure():
    """
    Automatically configure performance based on available hardware.
    """
    devices = jax.devices()
    
    if not devices:
        logger.warning("[Performance] No JAX devices detected!")
        return
    
    default_backend = jax.default_backend()
    
    if 'gpu' in default_backend.lower():
        # GPU available
        n_gpus = len([d for d in devices if 'gpu' in d.platform.lower()])
        logger.info(f"[Performance] Detected {n_gpus} GPU(s), applying high-performance mode")
        apply_optimal_settings('gpu_high_performance')
    else:
        # CPU only
        logger.info("[Performance] No GPU detected, applying CPU-optimized mode")
        apply_optimal_settings('cpu_optimized')


# Initialize on import
if __name__ != '__main__':
    # Only auto-configure if not running as main script
    # (allows user to configure manually if needed)
    pass  # Don't auto-configure, let user control

