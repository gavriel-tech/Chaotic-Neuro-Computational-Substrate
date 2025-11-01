"""
Prometheus metrics exporter for GMCS.

Exports metrics in Prometheus format for monitoring and alerting.
"""

from typing import Dict, Any, Optional
from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, REGISTRY
import time

from .metrics import get_global_metrics


class PrometheusExporter:
    """
    Exports GMCS metrics to Prometheus.
    
    Provides metrics in Prometheus format for scraping.
    """
    
    def __init__(self):
        """Initialize Prometheus exporter."""
        # System metrics
        self.cpu_usage = Gauge('gmcs_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('gmcs_memory_usage_bytes', 'Memory usage in bytes')
        self.memory_percent = Gauge('gmcs_memory_usage_percent', 'Memory usage percentage')
        self.gpu_utilization = Gauge('gmcs_gpu_utilization_percent', 'GPU utilization percentage')
        self.gpu_memory_usage = Gauge('gmcs_gpu_memory_usage_bytes', 'GPU memory usage in bytes')
        self.disk_usage = Gauge('gmcs_disk_usage_percent', 'Disk usage percentage')
        
        # Network metrics
        self.network_bytes_sent = Counter('gmcs_network_bytes_sent_total', 'Total bytes sent')
        self.network_bytes_recv = Counter('gmcs_network_bytes_recv_total', 'Total bytes received')
        
        # Application metrics
        self.active_nodes = Gauge('gmcs_active_nodes', 'Number of active nodes')
        self.active_connections = Gauge('gmcs_active_connections', 'Number of active node connections')
        self.presets_loaded = Gauge('gmcs_presets_loaded', 'Number of loaded presets')
        
        # API metrics
        self.api_requests_total = Counter(
            'gmcs_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status']
        )
        self.api_request_duration = Histogram(
            'gmcs_api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint']
        )
        self.websocket_connections = Gauge('gmcs_websocket_connections', 'Active WebSocket connections')
        
        # Performance metrics
        self.node_execution_duration = Histogram(
            'gmcs_node_execution_duration_seconds',
            'Node execution duration in seconds',
            ['node_type']
        )
        self.node_execution_count = Counter(
            'gmcs_node_execution_count_total',
            'Total node executions',
            ['node_type']
        )
        
        # System info
        self.system_info = Info('gmcs_system', 'GMCS system information')
        
        # Health status
        self.health_status = Gauge('gmcs_health_status', 'Health status (1=healthy, 0.5=degraded, 0=unhealthy)')
        
        # Initialize system info
        self.system_info.info({
            'version': '0.1',
            'platform': 'gmcs',
        })
    
    def update_metrics(self):
        """Update all metrics from global metrics tracker."""
        metrics = get_global_metrics()
        current = metrics.get_current_metrics()
        
        # Update system metrics
        system = current.get('system', {})
        self.cpu_usage.set(system.get('cpu_percent', 0))
        self.memory_usage.set(system.get('memory_used_mb', 0) * 1024 * 1024)
        self.memory_percent.set(system.get('memory_percent', 0))
        self.gpu_utilization.set(system.get('gpu_utilization', 0))
        self.gpu_memory_usage.set(system.get('gpu_memory_used_mb', 0) * 1024 * 1024)
        self.disk_usage.set(system.get('disk_used_percent', 0))
        
        # Network metrics (use increment since counters are cumulative)
        # Note: These are deltas, so we'd need to track previous values
        # For now, we'll update them as-is
        
        # Update application metrics
        app = current.get('application', {})
        self.active_nodes.set(app.get('active_nodes', 0))
        self.active_connections.set(app.get('active_connections', 0))
        self.presets_loaded.set(app.get('presets_loaded', 0))
        self.websocket_connections.set(app.get('websocket_connections', 0))
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Record an API request."""
        self.api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_node_execution(self, node_type: str, duration: float):
        """Record a node execution."""
        self.node_execution_count.labels(node_type=node_type).inc()
        self.node_execution_duration.labels(node_type=node_type).observe(duration)
    
    def set_health_status(self, status: str):
        """Set health status."""
        status_values = {
            'healthy': 1.0,
            'degraded': 0.5,
            'unhealthy': 0.0,
            'unknown': -1.0,
        }
        self.health_status.set(status_values.get(status, -1.0))
    
    def generate_metrics(self) -> bytes:
        """Generate metrics in Prometheus format."""
        self.update_metrics()
        return generate_latest(REGISTRY)


# Global exporter instance
_global_exporter: Optional[PrometheusExporter] = None


def setup_prometheus() -> PrometheusExporter:
    """Set up and return global Prometheus exporter."""
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = PrometheusExporter()
    return _global_exporter


def get_prometheus_exporter() -> PrometheusExporter:
    """Get global Prometheus exporter."""
    return setup_prometheus()

