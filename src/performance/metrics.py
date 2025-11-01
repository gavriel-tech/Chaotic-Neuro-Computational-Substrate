"""
Metrics tracking for GMCS performance monitoring.

Collects and aggregates performance metrics over time.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import statistics


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific operation or component."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    mean_time: float = 0.0
    median_time: float = 0.0
    p95_time: float = 0.0
    p99_time: float = 0.0
    total_memory_delta: int = 0
    max_memory_delta: int = 0
    total_gpu_memory_delta: int = 0
    max_gpu_memory_delta: int = 0
    mean_gpu_utilization: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "total_time_s": self.total_time,
            "min_time_s": self.min_time if self.min_time != float('inf') else 0,
            "max_time_s": self.max_time,
            "mean_time_s": self.mean_time,
            "median_time_s": self.median_time,
            "p95_time_s": self.p95_time,
            "p99_time_s": self.p99_time,
            "total_memory_delta_mb": self.total_memory_delta / (1024 * 1024),
            "max_memory_delta_mb": self.max_memory_delta / (1024 * 1024),
            "total_gpu_memory_delta_mb": self.total_gpu_memory_delta / (1024 * 1024),
            "max_gpu_memory_delta_mb": self.max_gpu_memory_delta / (1024 * 1024),
            "mean_gpu_utilization_percent": self.mean_gpu_utilization,
            "last_updated": self.last_updated,
        }


class MetricsTracker:
    """
    Tracks and aggregates performance metrics over time.
    
    Maintains a sliding window of recent measurements for statistical analysis.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Number of recent measurements to keep for statistics
        """
        self.window_size = window_size
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._recent_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._recent_gpu_utils: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()
    
    def record(
        self,
        name: str,
        duration: float,
        memory_delta: int = 0,
        gpu_memory_delta: int = 0,
        gpu_utilization: float = 0.0
    ):
        """
        Record a measurement.
        
        Args:
            name: Operation name
            duration: Duration in seconds
            memory_delta: Memory change in bytes
            gpu_memory_delta: GPU memory change in bytes
            gpu_utilization: GPU utilization percentage
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = PerformanceMetrics(name=name)
            
            metrics = self._metrics[name]
            
            # Update counts and totals
            metrics.count += 1
            metrics.total_time += duration
            metrics.min_time = min(metrics.min_time, duration)
            metrics.max_time = max(metrics.max_time, duration)
            metrics.total_memory_delta += memory_delta
            metrics.max_memory_delta = max(metrics.max_memory_delta, abs(memory_delta))
            metrics.total_gpu_memory_delta += gpu_memory_delta
            metrics.max_gpu_memory_delta = max(metrics.max_gpu_memory_delta, abs(gpu_memory_delta))
            
            # Store recent values for statistical analysis
            self._recent_times[name].append(duration)
            self._recent_gpu_utils[name].append(gpu_utilization)
            
            # Update statistics
            times = list(self._recent_times[name])
            gpu_utils = list(self._recent_gpu_utils[name])
            
            metrics.mean_time = statistics.mean(times) if times else 0
            metrics.median_time = statistics.median(times) if times else 0
            
            if len(times) >= 20:  # Need enough samples for percentiles
                sorted_times = sorted(times)
                metrics.p95_time = sorted_times[int(len(sorted_times) * 0.95)]
                metrics.p99_time = sorted_times[int(len(sorted_times) * 0.99)]
            
            metrics.mean_gpu_utilization = statistics.mean(gpu_utils) if gpu_utils else 0
            metrics.last_updated = time.time()
    
    def get_metrics(self, name: str) -> Optional[PerformanceMetrics]:
        """
        Get metrics for a specific operation.
        
        Args:
            name: Operation name
            
        Returns:
            PerformanceMetrics or None if not found
        """
        with self._lock:
            return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """
        Get all tracked metrics.
        
        Returns:
            Dictionary mapping names to metrics
        """
        with self._lock:
            return dict(self._metrics)
    
    def get_top_time_consumers(self, n: int = 10) -> List[PerformanceMetrics]:
        """
        Get top N operations by total time consumed.
        
        Args:
            n: Number of operations to return
            
        Returns:
            List of metrics sorted by total time (descending)
        """
        with self._lock:
            metrics = sorted(
                self._metrics.values(),
                key=lambda m: m.total_time,
                reverse=True
            )
            return metrics[:n]
    
    def get_slowest_operations(self, n: int = 10) -> List[PerformanceMetrics]:
        """
        Get top N slowest operations by mean time.
        
        Args:
            n: Number of operations to return
            
        Returns:
            List of metrics sorted by mean time (descending)
        """
        with self._lock:
            metrics = sorted(
                self._metrics.values(),
                key=lambda m: m.mean_time,
                reverse=True
            )
            return metrics[:n]
    
    def get_memory_intensive_operations(self, n: int = 10) -> List[PerformanceMetrics]:
        """
        Get top N memory-intensive operations.
        
        Args:
            n: Number of operations to return
            
        Returns:
            List of metrics sorted by max memory delta (descending)
        """
        with self._lock:
            metrics = sorted(
                self._metrics.values(),
                key=lambda m: m.max_memory_delta,
                reverse=True
            )
            return metrics[:n]
    
    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            self._recent_times.clear()
            self._recent_gpu_utils.clear()
    
    def clear_metric(self, name: str):
        """
        Clear metrics for a specific operation.
        
        Args:
            name: Operation name
        """
        with self._lock:
            self._metrics.pop(name, None)
            self._recent_times.pop(name, None)
            self._recent_gpu_utils.pop(name, None)


# Global metrics tracker instance
_global_tracker: Optional[MetricsTracker] = None


def get_global_tracker() -> MetricsTracker:
    """Get or create global metrics tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MetricsTracker()
    return _global_tracker

