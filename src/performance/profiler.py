"""
Core performance profiler for GMCS.

Tracks execution timing, memory usage, and GPU utilization for all operations.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict
import jax
import jax.numpy as jnp

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class ProfileEntry:
    """Single profiling entry."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: int = 0
    memory_after: int = 0
    memory_delta: int = 0
    gpu_memory_before: int = 0
    gpu_memory_after: int = 0
    gpu_memory_delta: int = 0
    gpu_utilization: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self, end_time: float, memory_after: int, gpu_memory_after: int, gpu_util: float):
        """Finalize the profile entry."""
        self.end_time = end_time
        self.duration = end_time - self.start_time
        self.memory_after = memory_after
        self.memory_delta = memory_after - self.memory_before
        self.gpu_memory_after = gpu_memory_after
        self.gpu_memory_delta = gpu_memory_after - self.gpu_memory_before
        self.gpu_utilization = gpu_util
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "memory_before_mb": self.memory_before / (1024 * 1024),
            "memory_after_mb": self.memory_after / (1024 * 1024),
            "memory_delta_mb": self.memory_delta / (1024 * 1024),
            "gpu_memory_before_mb": self.gpu_memory_before / (1024 * 1024),
            "gpu_memory_after_mb": self.gpu_memory_after / (1024 * 1024),
            "gpu_memory_delta_mb": self.gpu_memory_delta / (1024 * 1024),
            "gpu_utilization_percent": self.gpu_utilization,
            "metadata": self.metadata,
        }


class PerformanceProfiler:
    """
    Comprehensive performance profiler for GMCS.
    
    Tracks timing, memory, and GPU utilization across all operations.
    Thread-safe for concurrent profiling.
    """
    
    def __init__(self, enabled: bool = True, gpu_device_id: int = 0):
        """
        Initialize profiler.
        
        Args:
            enabled: Whether profiling is enabled
            gpu_device_id: GPU device ID to monitor
        """
        self.enabled = enabled
        self.gpu_device_id = gpu_device_id
        self._entries: List[ProfileEntry] = []
        self._active_entries: Dict[str, ProfileEntry] = {}
        self._lock = threading.Lock()
        self._process = psutil.Process()
        
        # Initialize NVML for GPU monitoring
        self._gpu_available = False
        if NVML_AVAILABLE and self.enabled:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device_id)
                self._gpu_available = True
            except Exception as e:
                print(f"Warning: Could not initialize GPU monitoring: {e}")
    
    def __del__(self):
        """Cleanup NVML."""
        if self._gpu_available and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self._process.memory_info().rss
    
    def _get_gpu_memory_usage(self) -> int:
        """Get current GPU memory usage in bytes."""
        if not self._gpu_available:
            return 0
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            return info.used
        except:
            return 0
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        if not self._gpu_available:
            return 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            return float(util.gpu)
        except:
            return 0.0
    
    def start(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start profiling an operation.
        
        Args:
            name: Name of the operation
            metadata: Optional metadata
            
        Returns:
            Profile entry ID
        """
        if not self.enabled:
            return ""
        
        entry_id = f"{name}_{time.time()}_{id(threading.current_thread())}"
        
        entry = ProfileEntry(
            name=name,
            start_time=time.perf_counter(),
            memory_before=self._get_memory_usage(),
            gpu_memory_before=self._get_gpu_memory_usage(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._active_entries[entry_id] = entry
        
        return entry_id
    
    def end(self, entry_id: str):
        """
        End profiling an operation.
        
        Args:
            entry_id: Profile entry ID from start()
        """
        if not self.enabled or not entry_id:
            return
        
        end_time = time.perf_counter()
        memory_after = self._get_memory_usage()
        gpu_memory_after = self._get_gpu_memory_usage()
        gpu_util = self._get_gpu_utilization()
        
        with self._lock:
            entry = self._active_entries.pop(entry_id, None)
            if entry:
                entry.finalize(end_time, memory_after, gpu_memory_after, gpu_util)
                self._entries.append(entry)
    
    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name of the operation
            metadata: Optional metadata
            
        Example:
            with profiler.profile("node_execution", {"node_id": 5}):
                # code to profile
                pass
        """
        entry_id = self.start(name, metadata)
        try:
            yield entry_id
        finally:
            self.end(entry_id)
    
    def get_entries(self, name_filter: Optional[str] = None) -> List[ProfileEntry]:
        """
        Get all profile entries, optionally filtered by name.
        
        Args:
            name_filter: Optional name filter (substring match)
            
        Returns:
            List of profile entries
        """
        with self._lock:
            entries = list(self._entries)
        
        if name_filter:
            entries = [e for e in entries if name_filter in e.name]
        
        return entries
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            entries = list(self._entries)
        
        if not entries:
            return {
                "total_entries": 0,
                "total_time": 0.0,
                "by_name": {}
            }
        
        # Group by name
        by_name = defaultdict(list)
        for entry in entries:
            by_name[entry.name].append(entry)
        
        # Calculate statistics
        summary = {
            "total_entries": len(entries),
            "total_time": sum(e.duration or 0 for e in entries),
            "by_name": {}
        }
        
        for name, name_entries in by_name.items():
            durations = [e.duration for e in name_entries if e.duration is not None]
            memory_deltas = [e.memory_delta for e in name_entries]
            gpu_memory_deltas = [e.gpu_memory_delta for e in name_entries]
            gpu_utils = [e.gpu_utilization for e in name_entries]
            
            summary["by_name"][name] = {
                "count": len(name_entries),
                "total_time": sum(durations),
                "mean_time": sum(durations) / len(durations) if durations else 0,
                "min_time": min(durations) if durations else 0,
                "max_time": max(durations) if durations else 0,
                "mean_memory_delta_mb": sum(memory_deltas) / len(memory_deltas) / (1024 * 1024) if memory_deltas else 0,
                "mean_gpu_memory_delta_mb": sum(gpu_memory_deltas) / len(gpu_memory_deltas) / (1024 * 1024) if gpu_memory_deltas else 0,
                "mean_gpu_utilization": sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
            }
        
        return summary
    
    def clear(self):
        """Clear all profile entries."""
        with self._lock:
            self._entries.clear()
            self._active_entries.clear()
    
    def enable(self):
        """Enable profiling."""
        self.enabled = True
    
    def disable(self):
        """Disable profiling."""
        self.enabled = False


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_global_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


@contextmanager
def ProfilerContext(name: str, metadata: Optional[Dict[str, Any]] = None, profiler: Optional[PerformanceProfiler] = None):
    """
    Convenience context manager using global profiler.
    
    Args:
        name: Name of the operation
        metadata: Optional metadata
        profiler: Optional custom profiler (uses global if None)
    """
    prof = profiler or get_global_profiler()
    with prof.profile(name, metadata):
        yield

