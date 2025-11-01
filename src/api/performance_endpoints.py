"""
API endpoints for performance monitoring and profiling.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path

from src.performance import (
    get_global_profiler,
    get_global_tracker,
    PerformanceProfiler,
    MetricsTracker,
    BottleneckAnalyzer,
    ReportGenerator,
    ReportFormat
)


router = APIRouter(prefix="/performance", tags=["performance"])


class ProfilerStatusResponse(BaseModel):
    """Profiler status response."""
    enabled: bool
    total_entries: int
    active_entries: int


class MetricsSummaryResponse(BaseModel):
    """Metrics summary response."""
    total_entries: int
    total_time: float
    unique_operations: int


@router.get("/status", response_model=ProfilerStatusResponse)
async def get_profiler_status():
    """Get current profiler status."""
    profiler = get_global_profiler()
    summary = profiler.get_summary()
    
    return ProfilerStatusResponse(
        enabled=profiler.enabled,
        total_entries=summary["total_entries"],
        active_entries=len(profiler._active_entries)
    )


@router.post("/enable")
async def enable_profiler():
    """Enable performance profiling."""
    profiler = get_global_profiler()
    profiler.enable()
    return {"status": "enabled"}


@router.post("/disable")
async def disable_profiler():
    """Disable performance profiling."""
    profiler = get_global_profiler()
    profiler.disable()
    return {"status": "disabled"}


@router.post("/clear")
async def clear_profiler_data():
    """Clear all profiling data."""
    profiler = get_global_profiler()
    tracker = get_global_tracker()
    
    profiler.clear()
    tracker.clear()
    
    return {"status": "cleared", "message": "All profiling data cleared"}


@router.get("/summary")
async def get_performance_summary():
    """Get performance summary."""
    profiler = get_global_profiler()
    summary = profiler.get_summary()
    return summary


@router.get("/metrics")
async def get_all_metrics():
    """Get all tracked metrics."""
    tracker = get_global_tracker()
    metrics = tracker.get_all_metrics()
    
    return {
        name: metric.to_dict()
        for name, metric in metrics.items()
    }


@router.get("/metrics/{operation_name}")
async def get_operation_metrics(operation_name: str):
    """Get metrics for a specific operation."""
    tracker = get_global_tracker()
    metrics = tracker.get_metrics(operation_name)
    
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No metrics found for operation: {operation_name}")
    
    return metrics.to_dict()


@router.get("/top-consumers")
async def get_top_time_consumers(n: int = Query(default=10, ge=1, le=100)):
    """Get top N time-consuming operations."""
    tracker = get_global_tracker()
    top = tracker.get_top_time_consumers(n)
    
    return {
        "count": len(top),
        "operations": [m.to_dict() for m in top]
    }


@router.get("/slowest")
async def get_slowest_operations(n: int = Query(default=10, ge=1, le=100)):
    """Get slowest operations by mean time."""
    tracker = get_global_tracker()
    slowest = tracker.get_slowest_operations(n)
    
    return {
        "count": len(slowest),
        "operations": [m.to_dict() for m in slowest]
    }


@router.get("/memory-intensive")
async def get_memory_intensive_operations(n: int = Query(default=10, ge=1, le=100)):
    """Get most memory-intensive operations."""
    tracker = get_global_tracker()
    memory_ops = tracker.get_memory_intensive_operations(n)
    
    return {
        "count": len(memory_ops),
        "operations": [m.to_dict() for m in memory_ops]
    }


@router.get("/bottlenecks")
async def analyze_bottlenecks(threshold: float = Query(default=0.05, ge=0.0, le=1.0)):
    """Analyze performance bottlenecks."""
    profiler = get_global_profiler()
    tracker = get_global_tracker()
    analyzer = BottleneckAnalyzer(profiler, tracker)
    
    bottlenecks = analyzer.analyze(threshold=threshold)
    
    return {
        "count": len(bottlenecks),
        "bottlenecks": [b.to_dict() for b in bottlenecks]
    }


@router.get("/optimization-summary")
async def get_optimization_summary():
    """Get comprehensive optimization recommendations."""
    profiler = get_global_profiler()
    tracker = get_global_tracker()
    analyzer = BottleneckAnalyzer(profiler, tracker)
    
    return analyzer.get_optimization_summary()


@router.get("/report/{format}")
async def generate_report(
    format: str,
    include_bottlenecks: bool = Query(default=True),
    include_raw_entries: bool = Query(default=False)
):
    """
    Generate performance report in specified format.
    
    Supported formats: json, csv, html, markdown
    """
    try:
        report_format = ReportFormat(format.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format: {format}. Supported: json, csv, html, markdown"
        )
    
    profiler = get_global_profiler()
    tracker = get_global_tracker()
    analyzer = BottleneckAnalyzer(profiler, tracker)
    generator = ReportGenerator(profiler, tracker, analyzer)
    
    report_content = generator.generate_report(
        format=report_format,
        include_bottlenecks=include_bottlenecks,
        include_raw_entries=include_raw_entries
    )
    
    # Determine content type
    content_types = {
        ReportFormat.JSON: "application/json",
        ReportFormat.CSV: "text/csv",
        ReportFormat.HTML: "text/html",
        ReportFormat.MARKDOWN: "text/markdown"
    }
    
    from fastapi.responses import Response
    return Response(
        content=report_content,
        media_type=content_types[report_format]
    )


@router.post("/report/save/{format}")
async def save_report(
    format: str,
    filename: Optional[str] = Query(default=None),
    include_bottlenecks: bool = Query(default=True)
):
    """Save performance report to file."""
    try:
        report_format = ReportFormat(format.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format: {format}. Supported: json, csv, html, markdown"
        )
    
    if not filename:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extensions = {
            ReportFormat.JSON: "json",
            ReportFormat.CSV: "csv",
            ReportFormat.HTML: "html",
            ReportFormat.MARKDOWN: "md"
        }
        filename = f"performance_report_{timestamp}.{extensions[report_format]}"
    
    output_path = Path("reports") / filename
    
    profiler = get_global_profiler()
    tracker = get_global_tracker()
    analyzer = BottleneckAnalyzer(profiler, tracker)
    generator = ReportGenerator(profiler, tracker, analyzer)
    
    generator.generate_report(
        format=report_format,
        output_path=output_path,
        include_bottlenecks=include_bottlenecks
    )
    
    return {
        "status": "saved",
        "path": str(output_path),
        "format": format
    }

