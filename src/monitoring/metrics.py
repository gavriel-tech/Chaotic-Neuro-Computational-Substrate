"""
System metrics collection for GMCS.
"""

import time
import psutil
from typing import Dict, Any, Optional
from collections import deque
from dataclasses import dataclass, field
import threading

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class SystemMetrics:
    """System-wide metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_used_mb: float
    memory_percent: float
    gpu_utilization: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_used_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_percent": self.memory_percent,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_percent": self.gpu_memory_percent,
            "disk_used_percent": self.disk_used_percent,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
        }


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    active_nodes: int = 0
    active_connections: int = 0
    presets_loaded: int = 0
    api_requests_total: int = 0
    api_requests_success: int = 0
    api_requests_error: int = 0
    websocket_connections: int = 0
    average_response_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "active_nodes": self.active_nodes,
            "active_connections": self.active_connections,
            "presets_loaded": self.presets_loaded,
            "api_requests_total": self.api_requests_total,
            "api_requests_success": self.api_requests_success,
            "api_requests_error": self.api_requests_error,
            "websocket_connections": self.websocket_connections,
            "average_response_time_ms": self.average_response_time_ms,
        }


class GMCSMetrics:
    """
    Centralized metrics collection for GMCS.
    
    Collects system metrics, application metrics, and custom metrics.
    """
    
    def __init__(self, history_size: int = 1000, gpu_device_id: int = 0):
        """
        Initialize metrics collector.
        
        Args:
            history_size: Number of historical metrics to keep
            gpu_device_id: GPU device ID to monitor
        """
        self.history_size = history_size
        self.gpu_device_id = gpu_device_id
        
        self._system_metrics: deque = deque(maxlen=history_size)
        self._app_metrics = ApplicationMetrics()
        self._custom_metrics: Dict[str, Any] = {}
        
        self._lock = threading.Lock()
        self._process = psutil.Process()
        
        # Initialize GPU monitoring
        self._gpu_available = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device_id)
                self._gpu_available = True
            except Exception:
                pass
        
        # Initialize network counters
        self._last_net_io = psutil.net_io_counters()
    
    def __del__(self):
        """Cleanup NVML."""
        if self._gpu_available and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_util = 0.0
        gpu_mem_used = 0.0
        gpu_mem_percent = 0.0
        
        if self._gpu_available:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util = float(util.gpu)
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_mem_used = mem_info.used / (1024 * 1024)  # Convert to MB
                gpu_mem_percent = (mem_info.used / mem_info.total) * 100
            except Exception:
                pass
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent - self._last_net_io.bytes_sent
        net_recv = net_io.bytes_recv - self._last_net_io.bytes_recv
        self._last_net_io = net_io
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_percent=memory.percent,
            gpu_utilization=gpu_util,
            gpu_memory_used_mb=gpu_mem_used,
            gpu_memory_percent=gpu_mem_percent,
            disk_used_percent=disk.percent,
            network_bytes_sent=net_sent,
            network_bytes_recv=net_recv,
        )
        
        with self._lock:
            self._system_metrics.append(metrics)
        
        return metrics
    
    def update_app_metrics(self, **kwargs):
        """Update application metrics."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._app_metrics, key):
                    setattr(self._app_metrics, key, value)
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a counter metric."""
        with self._lock:
            if hasattr(self._app_metrics, counter_name):
                current = getattr(self._app_metrics, counter_name)
                setattr(self._app_metrics, counter_name, current + value)
    
    def set_custom_metric(self, name: str, value: Any):
        """Set a custom metric value."""
        with self._lock:
            self._custom_metrics[name] = value
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        system_metrics = self.collect_system_metrics()
        
        with self._lock:
            return {
                "system": system_metrics.to_dict(),
                "application": self._app_metrics.to_dict(),
                "custom": dict(self._custom_metrics),
            }
    
    def get_system_metrics_history(self, n: Optional[int] = None) -> list:
        """Get historical system metrics."""
        with self._lock:
            metrics = list(self._system_metrics)
        
        if n is not None:
            metrics = metrics[-n:]
        
        return [m.to_dict() for m in metrics]
    
    def get_app_metrics(self) -> Dict[str, Any]:
        """Get current application metrics."""
        with self._lock:
            return self._app_metrics.to_dict()
    
    def reset_app_metrics(self):
        """Reset application metrics."""
        with self._lock:
            self._app_metrics = ApplicationMetrics()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            if not self._system_metrics:
                return {}
            
            recent_metrics = list(self._system_metrics)[-100:]  # Last 100 samples
            
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_mem = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_gpu = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
            
            return {
                "average_cpu_percent": avg_cpu,
                "average_memory_percent": avg_mem,
                "average_gpu_utilization": avg_gpu,
                "samples": len(recent_metrics),
                "application": self._app_metrics.to_dict(),
            }


# Global metrics instance
_global_metrics: Optional[GMCSMetrics] = None


def get_global_metrics() -> GMCSMetrics:
    """Get or create global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = GMCSMetrics()
    return _global_metrics

