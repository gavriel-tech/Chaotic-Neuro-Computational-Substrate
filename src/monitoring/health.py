"""
Health check system for GMCS.

Monitors system health and provides detailed status information.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import time
import psutil


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Single health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """
    Comprehensive health checking for GMCS.
    
    Monitors various system components and provides health status.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, HealthCheck] = {}
        self._start_time = time.time()
    
    def check_cpu(self) -> HealthCheck:
        """Check CPU health."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if cpu_percent < 80:
            status = HealthStatus.HEALTHY
            message = f"CPU usage is normal ({cpu_percent:.1f}%)"
        elif cpu_percent < 95:
            status = HealthStatus.DEGRADED
            message = f"CPU usage is high ({cpu_percent:.1f}%)"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"CPU usage is critical ({cpu_percent:.1f}%)"
        
        return HealthCheck(
            name="cpu",
            status=status,
            message=message,
            details={"cpu_percent": cpu_percent}
        )
    
    def check_memory(self) -> HealthCheck:
        """Check memory health."""
        memory = psutil.virtual_memory()
        
        if memory.percent < 80:
            status = HealthStatus.HEALTHY
            message = f"Memory usage is normal ({memory.percent:.1f}%)"
        elif memory.percent < 90:
            status = HealthStatus.DEGRADED
            message = f"Memory usage is high ({memory.percent:.1f}%)"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Memory usage is critical ({memory.percent:.1f}%)"
        
        return HealthCheck(
            name="memory",
            status=status,
            message=message,
            details={
                "percent": memory.percent,
                "used_mb": memory.used / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
            }
        )
    
    def check_disk(self) -> HealthCheck:
        """Check disk space health."""
        disk = psutil.disk_usage('/')
        
        if disk.percent < 80:
            status = HealthStatus.HEALTHY
            message = f"Disk space is adequate ({disk.percent:.1f}% used)"
        elif disk.percent < 90:
            status = HealthStatus.DEGRADED
            message = f"Disk space is running low ({disk.percent:.1f}% used)"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Disk space is critical ({disk.percent:.1f}% used)"
        
        return HealthCheck(
            name="disk",
            status=status,
            message=message,
            details={
                "percent": disk.percent,
                "free_gb": disk.free / (1024**3),
                "total_gb": disk.total / (1024**3),
            }
        )
    
    def check_gpu(self) -> HealthCheck:
        """Check GPU health."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Check GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Check temperature
            if temp > 85:
                status = HealthStatus.UNHEALTHY
                message = f"GPU temperature is critical ({temp}°C)"
            elif temp > 75:
                status = HealthStatus.DEGRADED
                message = f"GPU temperature is high ({temp}°C)"
            else:
                status = HealthStatus.HEALTHY
                message = f"GPU is operating normally (temp: {temp}°C)"
            
            pynvml.nvmlShutdown()
            
            return HealthCheck(
                name="gpu",
                status=status,
                message=message,
                details={
                    "utilization": util.gpu,
                    "temperature": temp,
                    "memory_used_mb": memory.used / (1024 * 1024),
                    "memory_total_mb": memory.total / (1024 * 1024),
                }
            )
        except Exception as e:
            return HealthCheck(
                name="gpu",
                status=HealthStatus.UNKNOWN,
                message=f"GPU check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_dependencies(self) -> HealthCheck:
        """Check that critical dependencies are available."""
        missing = []
        available = []
        
        # Check critical imports
        critical_deps = {
            "jax": "JAX",
            "numpy": "NumPy",
            "fastapi": "FastAPI",
            "jax.numpy": "JAX NumPy",
        }
        
        for module, name in critical_deps.items():
            try:
                __import__(module)
                available.append(name)
            except ImportError:
                missing.append(name)
        
        if not missing:
            status = HealthStatus.HEALTHY
            message = "All critical dependencies are available"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Missing critical dependencies: {', '.join(missing)}"
        
        return HealthCheck(
            name="dependencies",
            status=status,
            message=message,
            details={
                "available": available,
                "missing": missing,
            }
        )
    
    def check_uptime(self) -> HealthCheck:
        """Check system uptime."""
        uptime = time.time() - self._start_time
        
        return HealthCheck(
            name="uptime",
            status=HealthStatus.HEALTHY,
            message=f"System has been running for {uptime:.0f} seconds",
            details={"uptime_seconds": uptime}
        )
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        checks = {
            "cpu": self.check_cpu(),
            "memory": self.check_memory(),
            "disk": self.check_disk(),
            "gpu": self.check_gpu(),
            "dependencies": self.check_dependencies(),
            "uptime": self.check_uptime(),
        }
        
        self._checks = checks
        return checks
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self._checks:
            self.run_all_checks()
        
        statuses = [check.status for check in self._checks.values()]
        
        # If any check is unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # If any check is degraded, system is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # If any check is unknown, system status is unknown
        if HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        
        return HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        checks = self.run_all_checks()
        overall_status = self.get_overall_status()
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "checks": {name: check.to_dict() for name, check in checks.items()},
            "summary": {
                "healthy": sum(1 for c in checks.values() if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in checks.values() if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in checks.values() if c.status == HealthStatus.UNHEALTHY),
                "unknown": sum(1 for c in checks.values() if c.status == HealthStatus.UNKNOWN),
                "total": len(checks),
            }
        }

