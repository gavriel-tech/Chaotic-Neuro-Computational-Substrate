"""
API endpoints for system monitoring and health checks.
"""

from fastapi import APIRouter, Response
from typing import Dict, Any

from src.monitoring import (
    get_global_metrics,
    HealthChecker,
    HealthStatus,
    get_prometheus_exporter,
)


router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Create health checker instance
health_checker = HealthChecker()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns simple status for load balancer health checks.
    """
    report = health_checker.get_health_report()
    
    return {
        "status": report["status"],
        "timestamp": report["timestamp"],
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with all component statuses.
    """
    return health_checker.get_health_report()


@router.get("/metrics")
async def get_current_metrics() -> Dict[str, Any]:
    """Get current system and application metrics."""
    metrics = get_global_metrics()
    return metrics.get_current_metrics()


@router.get("/metrics/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary with averages."""
    metrics = get_global_metrics()
    return metrics.get_summary()


@router.get("/metrics/history")
async def get_metrics_history(n: int = 100) -> Dict[str, Any]:
    """
    Get historical metrics.
    
    Args:
        n: Number of recent samples to return (default: 100)
    """
    metrics = get_global_metrics()
    history = metrics.get_system_metrics_history(n=n)
    
    return {
        "count": len(history),
        "metrics": history,
    }


@router.get("/metrics/prometheus")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for scraping.
    """
    exporter = get_prometheus_exporter()
    metrics_data = exporter.generate_metrics()
    
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4"
    )


@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status.
    
    Combines health checks, metrics, and application status.
    """
    metrics = get_global_metrics()
    health_report = health_checker.get_health_report()
    current_metrics = metrics.get_current_metrics()
    summary = metrics.get_summary()
    
    return {
        "health": health_report,
        "metrics": {
            "current": current_metrics,
            "summary": summary,
        },
        "timestamp": health_report["timestamp"],
    }


@router.post("/collect")
async def trigger_metrics_collection():
    """
    Manually trigger metrics collection.
    
    Useful for testing or forcing an immediate update.
    """
    metrics = get_global_metrics()
    system_metrics = metrics.collect_system_metrics()
    
    return {
        "status": "collected",
        "metrics": system_metrics.to_dict(),
    }

