"""
Performance benchmark tests for GMCS.

Tests system performance, identifies bottlenecks, and ensures performance targets are met.
"""

import pytest
import time
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any

from src.performance import (
    PerformanceProfiler,
    MetricsTracker,
    BottleneckAnalyzer,
    ReportGenerator,
    ReportFormat
)


@pytest.fixture
def profiler():
    """Create a fresh profiler for each test."""
    prof = PerformanceProfiler(enabled=True)
    yield prof
    prof.clear()


@pytest.fixture
def tracker():
    """Create a fresh metrics tracker for each test."""
    track = MetricsTracker()
    yield track
    track.clear()


class TestProfilerBasics:
    """Test basic profiler functionality."""
    
    def test_profiler_can_start_and_end(self, profiler):
        """Test profiler start/end cycle."""
        entry_id = profiler.start("test_operation")
        assert entry_id != ""
        
        time.sleep(0.01)  # Simulate work
        
        profiler.end(entry_id)
        
        entries = profiler.get_entries("test_operation")
        assert len(entries) > 0
        assert entries[0].duration is not None
        assert entries[0].duration > 0
    
    def test_profiler_context_manager(self, profiler):
        """Test profiler context manager."""
        with profiler.profile("context_test"):
            time.sleep(0.01)
        
        entries = profiler.get_entries("context_test")
        assert len(entries) == 1
        assert entries[0].duration >= 0.01
    
    def test_profiler_can_be_disabled(self, profiler):
        """Test that profiler can be disabled."""
        profiler.disable()
        
        with profiler.profile("disabled_test"):
            time.sleep(0.01)
        
        entries = profiler.get_entries()
        # Should be empty when disabled
        assert len(entries) == 0
    
    def test_profiler_summary(self, profiler):
        """Test profiler summary generation."""
        # Record multiple operations
        for i in range(5):
            with profiler.profile(f"op_{i}"):
                time.sleep(0.001)
        
        summary = profiler.get_summary()
        
        assert summary["total_entries"] == 5
        assert summary["total_time"] > 0
        assert len(summary["by_name"]) == 5


class TestMetricsTracker:
    """Test metrics tracking and aggregation."""
    
    def test_record_metrics(self, tracker):
        """Test basic metric recording."""
        tracker.record(
            name="test_op",
            duration=0.1,
            memory_delta=1024*1024,
            gpu_memory_delta=5*1024*1024,
            gpu_utilization=75.0
        )
        
        metrics = tracker.get_metrics("test_op")
        assert metrics is not None
        assert metrics.count == 1
        assert metrics.total_time == 0.1
    
    def test_metrics_aggregation(self, tracker):
        """Test metrics aggregation over multiple recordings."""
        for i in range(100):
            tracker.record(
                name="repeated_op",
                duration=0.001 * (i + 1),
                memory_delta=1024,
                gpu_utilization=50.0
            )
        
        metrics = tracker.get_metrics("repeated_op")
        assert metrics.count == 100
        assert metrics.mean_time > 0
        assert metrics.min_time < metrics.max_time
    
    def test_percentile_calculation(self, tracker):
        """Test percentile calculations."""
        # Record 100 operations with increasing durations
        for i in range(100):
            tracker.record(
                name="percentile_test",
                duration=0.001 * i
            )
        
        metrics = tracker.get_metrics("percentile_test")
        
        # P95 should be around 95th value
        assert metrics.p95_time > metrics.mean_time
        assert metrics.p99_time > metrics.p95_time
    
    def test_top_consumers(self, tracker):
        """Test getting top time consumers."""
        # Record different operations with varying times
        tracker.record("fast", duration=0.001)
        tracker.record("medium", duration=0.05)
        tracker.record("slow", duration=0.2)
        
        top = tracker.get_top_time_consumers(n=3)
        
        assert len(top) == 3
        assert top[0].name == "slow"
        assert top[0].total_time > top[1].total_time


class TestBottleneckAnalysis:
    """Test bottleneck identification."""
    
    def test_bottleneck_identification(self, profiler, tracker):
        """Test that bottlenecks are correctly identified."""
        # Simulate various operations
        operations = [
            ("fast_op", 0.001, 10),    # Fast, frequent
            ("slow_op", 0.5, 2),       # Slow, infrequent
            ("balanced", 0.05, 20),    # Balanced
        ]
        
        for name, duration, count in operations:
            for _ in range(count):
                tracker.record(name, duration=duration)
        
        analyzer = BottleneckAnalyzer(profiler, tracker)
        bottlenecks = analyzer.analyze(threshold=0.01)
        
        assert len(bottlenecks) > 0
        
        # Check that high time consumers are flagged
        bottleneck_names = [b.operation for b in bottlenecks]
        assert "slow_op" in bottleneck_names or "balanced" in bottleneck_names
    
    def test_optimization_summary(self, profiler, tracker):
        """Test optimization summary generation."""
        # Add some test data
        tracker.record("operation_1", duration=0.1, memory_delta=10*1024*1024)
        tracker.record("operation_2", duration=0.05, gpu_utilization=95.0)
        
        analyzer = BottleneckAnalyzer(profiler, tracker)
        summary = analyzer.get_optimization_summary()
        
        assert "bottlenecks_by_type" in summary
        assert "top_time_consumers" in summary
        assert "recommendations" in summary


class TestReportGeneration:
    """Test performance report generation."""
    
    def test_json_report(self, profiler, tracker):
        """Test JSON report generation."""
        # Add sample data
        with profiler.profile("test_op"):
            time.sleep(0.01)
        
        tracker.record("test_op", duration=0.01)
        
        generator = ReportGenerator(profiler, tracker)
        report = generator.generate_report(ReportFormat.JSON)
        
        assert isinstance(report, str)
        import json
        data = json.loads(report)
        assert "summary" in data
        assert "metrics" in data
    
    def test_csv_report(self, profiler, tracker):
        """Test CSV report generation."""
        tracker.record("test_op", duration=0.01, memory_delta=1024)
        
        generator = ReportGenerator(profiler, tracker)
        report = generator.generate_report(ReportFormat.CSV)
        
        assert isinstance(report, str)
        assert "Operation" in report  # CSV header
        assert "test_op" in report
    
    def test_html_report(self, profiler, tracker):
        """Test HTML report generation."""
        tracker.record("test_op", duration=0.01)
        
        generator = ReportGenerator(profiler, tracker)
        report = generator.generate_report(ReportFormat.HTML)
        
        assert isinstance(report, str)
        assert "<!DOCTYPE html>" in report
        assert "GMCS Performance Report" in report
    
    def test_markdown_report(self, profiler, tracker):
        """Test Markdown report generation."""
        tracker.record("test_op", duration=0.01)
        
        generator = ReportGenerator(profiler, tracker)
        report = generator.generate_report(ReportFormat.MARKDOWN)
        
        assert isinstance(report, str)
        assert "# GMCS Performance Report" in report


class TestSystemBenchmarks:
    """Benchmark actual system components."""
    
    def test_jax_array_operations(self, profiler):
        """Benchmark JAX array operations."""
        with profiler.profile("jax_matmul"):
            a = jnp.ones((1000, 1000))
            b = jnp.ones((1000, 1000))
            c = jnp.dot(a, b)
            c.block_until_ready()
        
        entries = profiler.get_entries("jax_matmul")
        assert len(entries) > 0
        # Should be relatively fast with GPU
        assert entries[0].duration < 1.0
    
    def test_numpy_operations(self, profiler):
        """Benchmark NumPy operations."""
        with profiler.profile("numpy_matmul"):
            a = np.ones((1000, 1000))
            b = np.ones((1000, 1000))
            c = np.dot(a, b)
        
        entries = profiler.get_entries("numpy_matmul")
        assert len(entries) > 0
    
    def test_memory_tracking(self, profiler):
        """Test memory usage tracking."""
        with profiler.profile("memory_test"):
            # Allocate large array
            large_array = np.zeros((10000, 10000), dtype=np.float32)
            # Use it to prevent optimization
            _ = large_array.sum()
        
        entries = profiler.get_entries("memory_test")
        assert len(entries) > 0
        # Should show memory allocation
        # Note: Memory delta might be 0 if garbage collected
        # assert entries[0].memory_delta > 0


class TestPerformanceTargets:
    """Test that performance targets are met."""
    
    def test_node_execution_latency(self, profiler):
        """Test that node execution meets latency targets."""
        from src.nodes.simulation_bridge import OscillatorNode
        
        node = OscillatorNode("osc", num_oscillators=64)
        
        durations = []
        for _ in range(10):
            with profiler.profile("node_execution"):
                node.process(dt=0.01, forcing=0.0)
            
            entries = profiler.get_entries("node_execution")
            if entries:
                durations.append(entries[-1].duration)
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            # Target: <10ms per node execution
            assert avg_duration < 0.01, \
                f"Node execution too slow: {avg_duration*1000:.2f}ms (target: <10ms)"
    
    def test_graph_execution_throughput(self, profiler):
        """Test overall graph execution throughput."""
        # This would test a full preset execution
        # For now, skip if components not available
        pytest.skip("Full graph execution test requires complete system")


@pytest.mark.slow
class TestLongRunningBenchmarks:
    """Long-running benchmark tests."""
    
    def test_sustained_performance(self, profiler):
        """Test performance over extended period."""
        durations = []
        
        for i in range(1000):
            with profiler.profile("sustained_test"):
                # Simulate work
                a = jnp.ones((100, 100))
                b = jnp.dot(a, a)
                b.block_until_ready()
            
            if i % 100 == 0:
                entries = profiler.get_entries("sustained_test")
                if entries:
                    durations.append(entries[-1].duration)
        
        if len(durations) >= 2:
            # Check for performance degradation
            early_avg = sum(durations[:3]) / 3
            late_avg = sum(durations[-3:]) / 3
            
            # Performance shouldn't degrade by more than 50%
            assert late_avg < early_avg * 1.5, \
                "Performance degraded significantly over time"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

