"""
Bottleneck analysis for GMCS performance optimization.

Identifies performance bottlenecks and provides recommendations.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

from .profiler import PerformanceProfiler, ProfileEntry
from .metrics import MetricsTracker, PerformanceMetrics


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    GPU_BOUND = "gpu_bound"
    GPU_MEMORY_BOUND = "gpu_memory_bound"
    IO_BOUND = "io_bound"
    SERIAL_EXECUTION = "serial_execution"
    FREQUENT_CALLS = "frequent_calls"


@dataclass
class Bottleneck:
    """Identified performance bottleneck."""
    type: BottleneckType
    operation: str
    severity: float  # 0.0 to 1.0
    total_time_percent: float
    description: str
    recommendations: List[str]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "operation": self.operation,
            "severity": self.severity,
            "total_time_percent": self.total_time_percent,
            "description": self.description,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
        }


class BottleneckAnalyzer:
    """
    Analyzes profiling data to identify performance bottlenecks.
    
    Provides actionable recommendations for optimization.
    """
    
    def __init__(
        self,
        profiler: Optional[PerformanceProfiler] = None,
        tracker: Optional[MetricsTracker] = None
    ):
        """
        Initialize bottleneck analyzer.
        
        Args:
            profiler: PerformanceProfiler instance
            tracker: MetricsTracker instance
        """
        from .profiler import get_global_profiler
        from .metrics import get_global_tracker
        
        self.profiler = profiler or get_global_profiler()
        self.tracker = tracker or get_global_tracker()
    
    def analyze(self, threshold: float = 0.05) -> List[Bottleneck]:
        """
        Analyze profiling data to identify bottlenecks.
        
        Args:
            threshold: Minimum time percentage to consider (default 5%)
            
        Returns:
            List of identified bottlenecks sorted by severity
        """
        bottlenecks = []
        
        # Get all metrics
        all_metrics = self.tracker.get_all_metrics()
        if not all_metrics:
            return bottlenecks
        
        # Calculate total time
        total_time = sum(m.total_time for m in all_metrics.values())
        if total_time == 0:
            return bottlenecks
        
        # Analyze each operation
        for name, metrics in all_metrics.items():
            time_percent = (metrics.total_time / total_time) * 100
            
            # Skip if below threshold
            if time_percent < threshold * 100:
                continue
            
            # Analyze different bottleneck types
            bottlenecks.extend(self._analyze_operation(metrics, time_percent, total_time))
        
        # Sort by severity
        bottlenecks.sort(key=lambda b: b.severity, reverse=True)
        
        return bottlenecks
    
    def _analyze_operation(
        self,
        metrics: PerformanceMetrics,
        time_percent: float,
        total_time: float
    ) -> List[Bottleneck]:
        """Analyze a single operation for bottlenecks."""
        bottlenecks = []
        
        # Check for frequent calls (high count but low individual time)
        if metrics.count > 1000 and metrics.mean_time < 0.001:
            severity = min(1.0, (metrics.count / 10000) * (time_percent / 100))
            bottlenecks.append(Bottleneck(
                type=BottleneckType.FREQUENT_CALLS,
                operation=metrics.name,
                severity=severity,
                total_time_percent=time_percent,
                description=f"Operation called {metrics.count} times, consuming {time_percent:.1f}% of total time",
                recommendations=[
                    "Consider batching multiple calls into one",
                    "Use caching if results are reusable",
                    "Profile to ensure JIT compilation is working",
                    "Check if operation can be vectorized"
                ],
                metrics={
                    "count": metrics.count,
                    "mean_time_ms": metrics.mean_time * 1000,
                    "total_time_s": metrics.total_time,
                }
            ))
        
        # Check for slow individual operations
        if metrics.mean_time > 0.1:  # 100ms+
            severity = min(1.0, (metrics.mean_time / 1.0) * (time_percent / 100))
            
            # Determine if CPU or GPU bound based on GPU utilization
            if metrics.mean_gpu_utilization > 80:
                bottleneck_type = BottleneckType.GPU_BOUND
                desc = f"GPU-bound operation taking {metrics.mean_time*1000:.1f}ms on average"
                recs = [
                    "Consider reducing computational complexity",
                    "Check if algorithm can be optimized",
                    "Profile GPU kernel performance",
                    "Consider model quantization or pruning"
                ]
            elif metrics.mean_gpu_utilization < 20:
                bottleneck_type = BottleneckType.CPU_BOUND
                desc = f"CPU-bound operation taking {metrics.mean_time*1000:.1f}ms on average"
                recs = [
                    "Optimize CPU-side preprocessing",
                    "Move more computation to GPU",
                    "Check for unnecessary data transfers",
                    "Profile CPU hotspots with cProfile"
                ]
            else:
                bottleneck_type = BottleneckType.IO_BOUND
                desc = f"Potentially I/O-bound operation taking {metrics.mean_time*1000:.1f}ms on average"
                recs = [
                    "Check for data transfer bottlenecks",
                    "Use asynchronous I/O if applicable",
                    "Optimize data pipeline",
                    "Consider memory-mapped files for large data"
                ]
            
            bottlenecks.append(Bottleneck(
                type=bottleneck_type,
                operation=metrics.name,
                severity=severity,
                total_time_percent=time_percent,
                description=desc,
                recommendations=recs,
                metrics={
                    "mean_time_ms": metrics.mean_time * 1000,
                    "p95_time_ms": metrics.p95_time * 1000,
                    "p99_time_ms": metrics.p99_time * 1000,
                    "gpu_utilization": metrics.mean_gpu_utilization,
                }
            ))
        
        # Check for memory-intensive operations
        if metrics.max_memory_delta > 100 * 1024 * 1024:  # 100 MB+
            severity = min(1.0, (metrics.max_memory_delta / (1024**3)) * 0.5)
            bottlenecks.append(Bottleneck(
                type=BottleneckType.MEMORY_BOUND,
                operation=metrics.name,
                severity=severity,
                total_time_percent=time_percent,
                description=f"Memory-intensive operation allocating up to {metrics.max_memory_delta/(1024**2):.1f} MB",
                recommendations=[
                    "Consider processing data in smaller batches",
                    "Use memory-efficient data structures",
                    "Profile memory allocation hotspots",
                    "Check for memory leaks",
                    "Use JAX's memory profiler for GPU memory"
                ],
                metrics={
                    "max_memory_delta_mb": metrics.max_memory_delta / (1024**2),
                    "total_memory_delta_mb": metrics.total_memory_delta / (1024**2),
                }
            ))
        
        # Check for GPU memory issues
        if metrics.max_gpu_memory_delta > 500 * 1024 * 1024:  # 500 MB+
            severity = min(1.0, (metrics.max_gpu_memory_delta / (2 * 1024**3)) * 0.7)
            bottlenecks.append(Bottleneck(
                type=BottleneckType.GPU_MEMORY_BOUND,
                operation=metrics.name,
                severity=severity,
                total_time_percent=time_percent,
                description=f"GPU memory-intensive operation using up to {metrics.max_gpu_memory_delta/(1024**2):.1f} MB",
                recommendations=[
                    "Reduce batch size",
                    "Use gradient checkpointing for training",
                    "Consider model parallelism",
                    "Use mixed precision (FP16/BF16)",
                    "Clear JAX cache periodically"
                ],
                metrics={
                    "max_gpu_memory_delta_mb": metrics.max_gpu_memory_delta / (1024**2),
                    "total_gpu_memory_delta_mb": metrics.total_gpu_memory_delta / (1024**2),
                }
            ))
        
        return bottlenecks
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization summary.
        
        Returns:
            Dictionary with optimization recommendations
        """
        bottlenecks = self.analyze()
        
        # Group by type
        by_type = {}
        for bottleneck in bottlenecks:
            type_name = bottleneck.type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(bottleneck)
        
        # Get top time consumers
        top_consumers = self.tracker.get_top_time_consumers(5)
        
        # Get slowest operations
        slowest = self.tracker.get_slowest_operations(5)
        
        return {
            "bottlenecks_by_type": {
                type_name: [b.to_dict() for b in bottles]
                for type_name, bottles in by_type.items()
            },
            "top_time_consumers": [m.to_dict() for m in top_consumers],
            "slowest_operations": [m.to_dict() for m in slowest],
            "total_bottlenecks": len(bottlenecks),
            "critical_bottlenecks": len([b for b in bottlenecks if b.severity > 0.7]),
            "recommendations": self._get_general_recommendations(bottlenecks),
        }
    
    def _get_general_recommendations(self, bottlenecks: List[Bottleneck]) -> List[str]:
        """Get general optimization recommendations based on bottlenecks."""
        recommendations = []
        
        # Count bottleneck types
        type_counts = {}
        for b in bottlenecks:
            type_counts[b.type] = type_counts.get(b.type, 0) + 1
        
        if type_counts.get(BottleneckType.FREQUENT_CALLS, 0) > 2:
            recommendations.append("Multiple operations are called very frequently. Consider batching or caching.")
        
        if type_counts.get(BottleneckType.GPU_BOUND, 0) > 2:
            recommendations.append("System appears GPU-bound. Consider algorithm optimization or upgrading GPU.")
        
        if type_counts.get(BottleneckType.MEMORY_BOUND, 0) > 1:
            recommendations.append("Memory usage is high. Consider batch size reduction or streaming processing.")
        
        if type_counts.get(BottleneckType.GPU_MEMORY_BOUND, 0) > 1:
            recommendations.append("GPU memory is constrained. Use mixed precision or model parallelism.")
        
        if not recommendations:
            recommendations.append("Overall performance looks good. Monitor for changes as workload scales.")
        
        return recommendations

