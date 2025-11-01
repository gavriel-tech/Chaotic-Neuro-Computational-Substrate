"""
Report generation for GMCS performance profiling.

Exports profiling data in various formats (JSON, CSV, HTML).
"""

import json
import csv
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
from io import StringIO

from .profiler import PerformanceProfiler, ProfileEntry
from .metrics import MetricsTracker
from .bottleneck_analyzer import BottleneckAnalyzer


class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"


class ReportGenerator:
    """
    Generates performance reports in various formats.
    
    Supports JSON, CSV, HTML, and Markdown output.
    """
    
    def __init__(
        self,
        profiler: Optional[PerformanceProfiler] = None,
        tracker: Optional[MetricsTracker] = None,
        analyzer: Optional[BottleneckAnalyzer] = None
    ):
        """
        Initialize report generator.
        
        Args:
            profiler: PerformanceProfiler instance
            tracker: MetricsTracker instance
            analyzer: BottleneckAnalyzer instance
        """
        from .profiler import get_global_profiler
        from .metrics import get_global_tracker
        
        self.profiler = profiler or get_global_profiler()
        self.tracker = tracker or get_global_tracker()
        self.analyzer = analyzer or BottleneckAnalyzer(self.profiler, self.tracker)
    
    def generate_report(
        self,
        format: ReportFormat,
        output_path: Optional[Path] = None,
        include_bottlenecks: bool = True,
        include_raw_entries: bool = False
    ) -> str:
        """
        Generate performance report.
        
        Args:
            format: Output format
            output_path: Optional file path to save report
            include_bottlenecks: Whether to include bottleneck analysis
            include_raw_entries: Whether to include raw profile entries
            
        Returns:
            Report content as string
        """
        if format == ReportFormat.JSON:
            content = self._generate_json_report(include_bottlenecks, include_raw_entries)
        elif format == ReportFormat.CSV:
            content = self._generate_csv_report()
        elif format == ReportFormat.HTML:
            content = self._generate_html_report(include_bottlenecks)
        elif format == ReportFormat.MARKDOWN:
            content = self._generate_markdown_report(include_bottlenecks)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)
        
        return content
    
    def _generate_json_report(
        self,
        include_bottlenecks: bool,
        include_raw_entries: bool
    ) -> str:
        """Generate JSON report."""
        report = {
            "generated_at": time.time(),
            "summary": self.profiler.get_summary(),
            "metrics": {
                name: metrics.to_dict()
                for name, metrics in self.tracker.get_all_metrics().items()
            },
        }
        
        if include_bottlenecks:
            report["bottleneck_analysis"] = self.analyzer.get_optimization_summary()
        
        if include_raw_entries:
            report["raw_entries"] = [
                entry.to_dict() for entry in self.profiler.get_entries()
            ]
        
        return json.dumps(report, indent=2)
    
    def _generate_csv_report(self) -> str:
        """Generate CSV report with metrics."""
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Operation",
            "Count",
            "Total Time (s)",
            "Mean Time (s)",
            "Min Time (s)",
            "Max Time (s)",
            "P95 Time (s)",
            "P99 Time (s)",
            "Mean GPU Util (%)",
            "Max Memory Delta (MB)",
            "Max GPU Memory Delta (MB)"
        ])
        
        # Write metrics
        for metrics in self.tracker.get_all_metrics().values():
            writer.writerow([
                metrics.name,
                metrics.count,
                f"{metrics.total_time:.6f}",
                f"{metrics.mean_time:.6f}",
                f"{metrics.min_time:.6f}" if metrics.min_time != float('inf') else "0",
                f"{metrics.max_time:.6f}",
                f"{metrics.p95_time:.6f}",
                f"{metrics.p99_time:.6f}",
                f"{metrics.mean_gpu_utilization:.2f}",
                f"{metrics.max_memory_delta / (1024**2):.2f}",
                f"{metrics.max_gpu_memory_delta / (1024**2):.2f}",
            ])
        
        return output.getvalue()
    
    def _generate_html_report(self, include_bottlenecks: bool) -> str:
        """Generate HTML report with visualizations."""
        summary = self.profiler.get_summary()
        all_metrics = self.tracker.get_all_metrics()
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GMCS Performance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0a0a0a;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #00ff88;
            border-bottom: 2px solid #00ff88;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #00ccff;
            margin-top: 30px;
        }}
        .summary {{
            background: #1a1a1a;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid #333;
        }}
        .summary-item {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .summary-label {{
            color: #888;
            font-size: 0.9em;
        }}
        .summary-value {{
            color: #00ff88;
            font-size: 1.5em;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #1a1a1a;
        }}
        th {{
            background: #222;
            color: #00ff88;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #00ff88;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #333;
        }}
        tr:hover {{
            background: #252525;
        }}
        .severity-high {{
            color: #ff4444;
            font-weight: bold;
        }}
        .severity-medium {{
            color: #ffaa00;
        }}
        .severity-low {{
            color: #00ff88;
        }}
        .bottleneck {{
            background: #1a1a1a;
            border-left: 4px solid #ff4444;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .bottleneck h3 {{
            color: #ff4444;
            margin-top: 0;
        }}
        .recommendations {{
            margin-top: 10px;
        }}
        .recommendations li {{
            margin: 5px 0;
            color: #aaa;
        }}
        .metric-bar {{
            height: 20px;
            background: #00ff88;
            border-radius: 3px;
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GMCS Performance Report</h1>
        <p>Generated at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-item">
                <div class="summary-label">Total Entries</div>
                <div class="summary-value">{summary['total_entries']}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Total Time</div>
                <div class="summary-value">{summary['total_time']:.2f}s</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Unique Operations</div>
                <div class="summary-value">{len(all_metrics)}</div>
            </div>
        </div>
        
        <h2>Performance Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Count</th>
                    <th>Total Time (s)</th>
                    <th>Mean Time (ms)</th>
                    <th>P95 Time (ms)</th>
                    <th>GPU Util (%)</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add metrics rows
        for metrics in sorted(all_metrics.values(), key=lambda m: m.total_time, reverse=True):
            html += f"""
                <tr>
                    <td><strong>{metrics.name}</strong></td>
                    <td>{metrics.count}</td>
                    <td>{metrics.total_time:.3f}</td>
                    <td>{metrics.mean_time * 1000:.2f}</td>
                    <td>{metrics.p95_time * 1000:.2f}</td>
                    <td>{metrics.mean_gpu_utilization:.1f}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
        
        # Add bottleneck analysis if requested
        if include_bottlenecks:
            bottlenecks = self.analyzer.analyze()
            
            html += """
        <h2>Bottleneck Analysis</h2>
"""
            
            if bottlenecks:
                for bottleneck in bottlenecks:
                    severity_class = (
                        "severity-high" if bottleneck.severity > 0.7
                        else "severity-medium" if bottleneck.severity > 0.4
                        else "severity-low"
                    )
                    
                    html += f"""
        <div class="bottleneck">
            <h3>{bottleneck.operation} <span class="{severity_class}">({bottleneck.type.value})</span></h3>
            <p><strong>Severity:</strong> <span class="{severity_class}">{bottleneck.severity:.2f}</span> | 
               <strong>Time Impact:</strong> {bottleneck.total_time_percent:.1f}%</p>
            <p>{bottleneck.description}</p>
            <div class="recommendations">
                <strong>Recommendations:</strong>
                <ul>
"""
                    
                    for rec in bottleneck.recommendations:
                        html += f"                    <li>{rec}</li>\n"
                    
                    html += """
                </ul>
            </div>
        </div>
"""
            else:
                html += "<p>No significant bottlenecks detected.</p>"
        
        html += """
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_markdown_report(self, include_bottlenecks: bool) -> str:
        """Generate Markdown report."""
        summary = self.profiler.get_summary()
        all_metrics = self.tracker.get_all_metrics()
        
        md = f"""# GMCS Performance Report

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}

## Summary

- **Total Entries**: {summary['total_entries']}
- **Total Time**: {summary['total_time']:.2f}s
- **Unique Operations**: {len(all_metrics)}

## Performance Metrics

| Operation | Count | Total Time (s) | Mean Time (ms) | P95 Time (ms) | GPU Util (%) |
|-----------|-------|----------------|----------------|---------------|--------------|
"""
        
        for metrics in sorted(all_metrics.values(), key=lambda m: m.total_time, reverse=True):
            md += f"| {metrics.name} | {metrics.count} | {metrics.total_time:.3f} | {metrics.mean_time * 1000:.2f} | {metrics.p95_time * 1000:.2f} | {metrics.mean_gpu_utilization:.1f} |\n"
        
        if include_bottlenecks:
            bottlenecks = self.analyzer.analyze()
            
            md += "\n## Bottleneck Analysis\n\n"
            
            if bottlenecks:
                for i, bottleneck in enumerate(bottlenecks, 1):
                    md += f"""### {i}. {bottleneck.operation} ({bottleneck.type.value})

- **Severity**: {bottleneck.severity:.2f}
- **Time Impact**: {bottleneck.total_time_percent:.1f}%
- **Description**: {bottleneck.description}

**Recommendations:**
"""
                    for rec in bottleneck.recommendations:
                        md += f"- {rec}\n"
                    
                    md += "\n"
            else:
                md += "No significant bottlenecks detected.\n"
        
        return md

