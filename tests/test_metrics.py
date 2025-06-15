"""
Tests for the metrics tracking utilities.

This module tests the metrics tracking functionality including:
- MetricsTracker initialization and configuration
- MetricSeries data collection and analysis
- PerformanceProfiler timing and resource monitoring
- Trend analysis and convergence detection
- Data export and visualization utilities
"""

import pytest
import torch
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json

from src.utils.metrics import (
    MetricsTracker, MetricSeries, MetricSnapshot, 
    PerformanceProfiler, create_metrics_tracker
)


class TestMetricSeries:
    """Test MetricSeries functionality."""
    
    def test_initialization(self):
        """Test MetricSeries initialization."""
        series = MetricSeries("test_metric")
        
        assert series.name == "test_metric"
        assert series.values == []
        assert series.timestamps == []
        assert series.steps == []
        assert len(series) == 0
    
    def test_add_value(self):
        """Test adding values to MetricSeries."""
        series = MetricSeries("loss")
        
        # Add single value
        series.add(2.5, step=10)
        
        assert len(series) == 1
        assert series.values[0] == 2.5
        assert series.steps[0] == 10
        assert len(series.timestamps) == 1
        
        # Add multiple values
        series.add(2.0, step=20)
        series.add(1.5, step=30)
        
        assert len(series) == 3
        assert series.values == [2.5, 2.0, 1.5]
        assert series.steps == [10, 20, 30]
    
    def test_get_latest(self):
        """Test getting latest value."""
        series = MetricSeries("accuracy")
        
        # Empty series
        assert series.get_latest() is None
        
        # With values
        series.add(0.8, step=1)
        series.add(0.85, step=2)
        
        assert series.get_latest() == 0.85
    
    def test_get_average(self):
        """Test getting average value."""
        series = MetricSeries("loss")
        
        # Empty series
        assert series.get_average() == 0.0
        
        # With values
        series.add(3.0, step=1)
        series.add(2.0, step=2)
        series.add(1.0, step=3)
        
        assert series.get_average() == 2.0
        
        # With window
        assert series.get_average(window=2) == 1.5  # (2.0 + 1.0) / 2
    
    def test_get_trend(self):
        """Test trend analysis."""
        series = MetricSeries("loss")
        
        # Not enough data
        series.add(2.0, step=1)
        trend = series.get_trend()
        assert trend["direction"] == "stable"
        
        # Decreasing trend
        series.add(1.8, step=2)
        series.add(1.6, step=3)
        series.add(1.4, step=4)
        series.add(1.2, step=5)
        
        trend = series.get_trend()
        assert trend["direction"] == "decreasing"
        assert trend["slope"] < 0
        assert trend["r_squared"] > 0.8  # Should be a strong trend
        
        # Increasing trend
        increasing_series = MetricSeries("accuracy")
        for i, val in enumerate([0.6, 0.7, 0.8, 0.9, 0.95]):
            increasing_series.add(val, step=i+1)
        
        trend = increasing_series.get_trend()
        assert trend["direction"] == "increasing"
        assert trend["slope"] > 0
    
    def test_detect_convergence(self):
        """Test convergence detection."""
        series = MetricSeries("loss")
        
        # Not enough data
        assert not series.detect_convergence()
        
        # Add converged data
        for i in range(20):
            # Values converging to 1.0
            val = 1.0 + 0.1 * (0.9 ** i)
            series.add(val, step=i+1)
        
        assert series.detect_convergence(window=10, threshold=0.05)
        
        # Add non-converged data
        volatile_series = MetricSeries("volatile")
        for i in range(20):
            val = 1.0 + 0.5 * ((-1) ** i)  # Oscillating values
            volatile_series.add(val, step=i+1)
        
        assert not volatile_series.detect_convergence(window=10, threshold=0.05)
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        series = MetricSeries("test")
        series.add(1.0, step=1)
        series.add(2.0, step=2)
        
        data = series.to_dict()
        
        assert data["name"] == "test"
        assert data["values"] == [1.0, 2.0]
        assert data["steps"] == [1, 2]
        assert len(data["timestamps"]) == 2
        assert "statistics" in data
        assert data["statistics"]["count"] == 2
        assert data["statistics"]["mean"] == 1.5


class TestMetricSnapshot:
    """Test MetricSnapshot functionality."""
    
    def test_initialization(self):
        """Test MetricSnapshot initialization."""
        snapshot = MetricSnapshot()
        
        assert snapshot.step == 0
        assert snapshot.timestamp is not None
        assert snapshot.metrics == {}
        assert snapshot.metadata == {}
    
    def test_add_metric(self):
        """Test adding metrics to snapshot."""
        snapshot = MetricSnapshot(step=10)
        
        snapshot.add_metric("loss", 2.5)
        snapshot.add_metric("accuracy", 0.85)
        
        assert snapshot.metrics["loss"] == 2.5
        assert snapshot.metrics["accuracy"] == 0.85
    
    def test_add_metadata(self):
        """Test adding metadata to snapshot."""
        snapshot = MetricSnapshot()
        
        snapshot.add_metadata("learning_rate", 1e-3)
        snapshot.add_metadata("batch_size", 32)
        
        assert snapshot.metadata["learning_rate"] == 1e-3
        assert snapshot.metadata["batch_size"] == 32
    
    def test_to_dict(self):
        """Test converting snapshot to dictionary."""
        snapshot = MetricSnapshot(step=5)
        snapshot.add_metric("loss", 1.5)
        snapshot.add_metadata("epoch", 2)
        
        data = snapshot.to_dict()
        
        assert data["step"] == 5
        assert data["metrics"]["loss"] == 1.5
        assert data["metadata"]["epoch"] == 2
        assert "timestamp" in data


class TestMetricsTracker:
    """Test MetricsTracker functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_initialization(self):
        """Test MetricsTracker initialization."""
        tracker = MetricsTracker()
        
        assert tracker.metrics == {}
        assert tracker.snapshots == []
        assert tracker.config["buffer_size"] == 1000
        assert tracker.config["auto_save"] is False
    
    def test_initialization_with_config(self, temp_dir):
        """Test MetricsTracker initialization with custom config."""
        config = {
            "buffer_size": 500,
            "auto_save": True,
            "save_path": str(temp_dir / "metrics.json"),
            "convergence_window": 20
        }
        
        tracker = MetricsTracker(config=config)
        
        assert tracker.config["buffer_size"] == 500
        assert tracker.config["auto_save"] is True
        assert tracker.config["save_path"] == str(temp_dir / "metrics.json")
        assert tracker.config["convergence_window"] == 20
    
    def test_track_metric(self):
        """Test tracking individual metrics."""
        tracker = MetricsTracker()
        
        # Track single metric
        tracker.track("loss", 2.5, step=1)
        
        assert "loss" in tracker.metrics
        assert len(tracker.metrics["loss"]) == 1
        assert tracker.metrics["loss"].get_latest() == 2.5
        
        # Track multiple values
        tracker.track("loss", 2.0, step=2)
        tracker.track("accuracy", 0.8, step=2)
        
        assert len(tracker.metrics["loss"]) == 2
        assert "accuracy" in tracker.metrics
        assert tracker.metrics["accuracy"].get_latest() == 0.8
    
    def test_track_multiple_metrics(self):
        """Test tracking multiple metrics at once."""
        tracker = MetricsTracker()
        
        metrics_dict = {
            "loss": 2.5,
            "accuracy": 0.85,
            "learning_rate": 1e-3
        }
        
        tracker.track_multiple(metrics_dict, step=10)
        
        assert len(tracker.metrics) == 3
        assert tracker.metrics["loss"].get_latest() == 2.5
        assert tracker.metrics["accuracy"].get_latest() == 0.85
        assert tracker.metrics["learning_rate"].get_latest() == 1e-3
        
        # All should have the same step
        for metric in tracker.metrics.values():
            assert metric.steps[-1] == 10
    
    def test_create_snapshot(self):
        """Test creating metric snapshots."""
        tracker = MetricsTracker()
        
        # Track some metrics
        tracker.track("loss", 2.0, step=5)
        tracker.track("accuracy", 0.9, step=5)
        
        # Create snapshot
        snapshot = tracker.create_snapshot(step=5)
        
        assert snapshot.step == 5
        assert snapshot.metrics["loss"] == 2.0
        assert snapshot.metrics["accuracy"] == 0.9
        
        # Should be added to snapshots list
        assert len(tracker.snapshots) == 1
        assert tracker.snapshots[0] is snapshot
    
    def test_get_metric_summary(self):
        """Test getting metric summary."""
        tracker = MetricsTracker()
        
        # Add some data
        for i in range(10):
            tracker.track("loss", 3.0 - i * 0.2, step=i+1)
            tracker.track("accuracy", 0.5 + i * 0.05, step=i+1)
        
        summary = tracker.get_summary()
        
        assert "loss" in summary
        assert "accuracy" in summary
        
        loss_summary = summary["loss"]
        assert loss_summary["count"] == 10
        assert loss_summary["latest"] == 1.2  # 3.0 - 9 * 0.2
        assert loss_summary["trend"]["direction"] == "decreasing"
        
        accuracy_summary = summary["accuracy"]
        assert accuracy_summary["trend"]["direction"] == "increasing"
    
    def test_detect_convergence(self):
        """Test convergence detection across metrics."""
        tracker = MetricsTracker()
        
        # Add converging loss data
        for i in range(20):
            val = 1.0 + 0.1 * (0.9 ** i)
            tracker.track("loss", val, step=i+1)
        
        # Add non-converging accuracy data
        for i in range(20):
            val = 0.8 + 0.1 * ((-1) ** i)
            tracker.track("accuracy", val, step=i+1)
        
        convergence = tracker.detect_convergence()
        
        assert "loss" in convergence
        assert "accuracy" in convergence
        assert convergence["loss"]["converged"] is True
        assert convergence["accuracy"]["converged"] is False
    
    def test_buffer_management(self):
        """Test buffer size management."""
        tracker = MetricsTracker(config={"buffer_size": 5})
        
        # Add more data than buffer size
        for i in range(10):
            tracker.track("loss", float(i), step=i+1)
        
        # Should only keep the most recent buffer_size values
        assert len(tracker.metrics["loss"]) == 5
        assert tracker.metrics["loss"].values == [5.0, 6.0, 7.0, 8.0, 9.0]
    
    def test_save_and_load(self, temp_dir):
        """Test saving and loading metrics."""
        save_path = temp_dir / "test_metrics.json"
        tracker = MetricsTracker()
        
        # Add some data
        tracker.track("loss", 2.5, step=1)
        tracker.track("accuracy", 0.8, step=1)
        tracker.create_snapshot(step=1)
        
        # Save
        tracker.save(str(save_path))
        assert save_path.exists()
        
        # Load into new tracker
        new_tracker = MetricsTracker()
        new_tracker.load(str(save_path))
        
        assert "loss" in new_tracker.metrics
        assert "accuracy" in new_tracker.metrics
        assert new_tracker.metrics["loss"].get_latest() == 2.5
        assert new_tracker.metrics["accuracy"].get_latest() == 0.8
        assert len(new_tracker.snapshots) == 1
    
    def test_auto_save(self, temp_dir):
        """Test automatic saving functionality."""
        save_path = temp_dir / "auto_save.json"
        config = {
            "auto_save": True,
            "save_path": str(save_path),
            "save_interval": 2
        }
        
        tracker = MetricsTracker(config=config)
        
        # Add data - should trigger auto save after save_interval steps
        tracker.track("loss", 3.0, step=1)
        assert not save_path.exists()  # Not yet
        
        tracker.track("loss", 2.5, step=2)
        assert save_path.exists()  # Should save now
    
    def test_export_data(self):
        """Test exporting data in different formats."""
        tracker = MetricsTracker()
        
        # Add some data
        for i in range(5):
            tracker.track("loss", 3.0 - i * 0.5, step=i+1)
            tracker.track("accuracy", 0.6 + i * 0.1, step=i+1)
        
        # Export as dict
        data = tracker.export_data()
        
        assert "metrics" in data
        assert "snapshots" in data
        assert "summary" in data
        assert "config" in data
        
        assert "loss" in data["metrics"]
        assert "accuracy" in data["metrics"]
        
        # Check data structure
        loss_data = data["metrics"]["loss"]
        assert loss_data["name"] == "loss"
        assert len(loss_data["values"]) == 5
        assert len(loss_data["steps"]) == 5


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality."""
    
    def test_initialization(self):
        """Test PerformanceProfiler initialization."""
        profiler = PerformanceProfiler()
        
        assert profiler.timers == {}
        assert profiler.memory_snapshots == []
        assert profiler.active_timers == {}
    
    def test_timing_context_manager(self):
        """Test timing with context manager."""
        profiler = PerformanceProfiler()
        
        with profiler.time("test_operation"):
            time.sleep(0.01)  # Small delay
        
        assert "test_operation" in profiler.timers
        timer_data = profiler.timers["test_operation"]
        assert len(timer_data["durations"]) == 1
        assert timer_data["durations"][0] >= 0.01
        assert timer_data["total_time"] >= 0.01
        assert timer_data["call_count"] == 1
    
    def test_manual_timing(self):
        """Test manual start/stop timing."""
        profiler = PerformanceProfiler()
        
        profiler.start_timer("manual_test")
        time.sleep(0.01)
        duration = profiler.stop_timer("manual_test")
        
        assert duration >= 0.01
        assert "manual_test" in profiler.timers
        assert profiler.timers["manual_test"]["call_count"] == 1
    
    def test_multiple_timing_calls(self):
        """Test multiple calls to the same timer."""
        profiler = PerformanceProfiler()
        
        # Multiple calls with context manager
        for _ in range(3):
            with profiler.time("repeated_op"):
                time.sleep(0.005)
        
        timer_data = profiler.timers["repeated_op"]
        assert timer_data["call_count"] == 3
        assert len(timer_data["durations"]) == 3
        assert timer_data["total_time"] >= 0.015
        assert timer_data["avg_time"] >= 0.005
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_memory_snapshot_cpu(self, mock_cuda):
        """Test memory snapshot on CPU."""
        profiler = PerformanceProfiler()
        
        snapshot = profiler.take_memory_snapshot("test_point")
        
        assert snapshot["name"] == "test_point"
        assert "timestamp" in snapshot
        assert "cpu_memory" in snapshot
        assert snapshot["gpu_memory"] is None  # No CUDA
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024*1024)  # 1MB
    @patch('torch.cuda.memory_reserved', return_value=2*1024*1024)  # 2MB
    def test_memory_snapshot_gpu(self, mock_reserved, mock_allocated, mock_cuda):
        """Test memory snapshot with GPU."""
        profiler = PerformanceProfiler()
        
        snapshot = profiler.take_memory_snapshot("gpu_test")
        
        assert snapshot["gpu_memory"]["allocated_mb"] == 1.0
        assert snapshot["gpu_memory"]["reserved_mb"] == 2.0
    
    def test_get_timing_summary(self):
        """Test getting timing summary."""
        profiler = PerformanceProfiler()
        
        # Add some timing data
        with profiler.time("fast_op"):
            time.sleep(0.001)
        
        with profiler.time("slow_op"):
            time.sleep(0.01)
        
        # Multiple calls to fast_op
        with profiler.time("fast_op"):
            time.sleep(0.001)
        
        summary = profiler.get_timing_summary()
        
        assert "fast_op" in summary
        assert "slow_op" in summary
        
        fast_summary = summary["fast_op"]
        assert fast_summary["call_count"] == 2
        assert fast_summary["total_time"] >= 0.002
        
        slow_summary = summary["slow_op"]
        assert slow_summary["call_count"] == 1
        assert slow_summary["total_time"] >= 0.01
    
    def test_get_memory_summary(self):
        """Test getting memory summary."""
        profiler = PerformanceProfiler()
        
        # Take some snapshots
        profiler.take_memory_snapshot("start")
        profiler.take_memory_snapshot("middle")
        profiler.take_memory_snapshot("end")
        
        summary = profiler.get_memory_summary()
        
        assert summary["snapshot_count"] == 3
        assert "peak_cpu_memory" in summary
        assert "snapshots" in summary
        assert len(summary["snapshots"]) == 3


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_metrics_tracker(self):
        """Test create_metrics_tracker factory function."""
        # Default tracker
        tracker = create_metrics_tracker()
        assert isinstance(tracker, MetricsTracker)
        assert tracker.config["buffer_size"] == 1000
        
        # Custom config
        tracker = create_metrics_tracker(
            buffer_size=500,
            auto_save=True,
            convergence_window=15
        )
        assert tracker.config["buffer_size"] == 500
        assert tracker.config["auto_save"] is True
        assert tracker.config["convergence_window"] == 15


class TestIntegration:
    """Integration tests for metrics components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_training_simulation(self, temp_dir):
        """Test simulating a training loop with metrics tracking."""
        # Setup
        save_path = temp_dir / "training_metrics.json"
        config = {
            "auto_save": True,
            "save_path": str(save_path),
            "save_interval": 5,
            "buffer_size": 100
        }
        
        tracker = MetricsTracker(config=config)
        profiler = PerformanceProfiler()
        
        # Simulate training loop
        for epoch in range(3):
            for step in range(10):
                global_step = epoch * 10 + step + 1
                
                # Simulate training step with timing
                with profiler.time("train_step"):
                    time.sleep(0.001)  # Simulate computation
                    
                    # Simulate decreasing loss
                    loss = 3.0 * (0.95 ** global_step)
                    accuracy = min(0.95, 0.5 + global_step * 0.01)
                    
                    # Track metrics
                    tracker.track_multiple({
                        "loss": loss,
                        "accuracy": accuracy,
                        "learning_rate": 1e-3 * (0.99 ** global_step)
                    }, step=global_step)
                
                # Take memory snapshot occasionally
                if global_step % 5 == 0:
                    profiler.take_memory_snapshot(f"step_{global_step}")
                    
                    # Create snapshot
                    snapshot = tracker.create_snapshot(step=global_step)
                    snapshot.add_metadata("epoch", epoch)
                    snapshot.add_metadata("step_in_epoch", step)
        
        # Verify results
        assert len(tracker.metrics) == 3
        assert len(tracker.snapshots) > 0
        
        # Check convergence
        convergence = tracker.detect_convergence()
        assert convergence["loss"]["converged"]  # Loss should converge
        
        # Check profiler data
        timing_summary = profiler.get_timing_summary()
        assert "train_step" in timing_summary
        assert timing_summary["train_step"]["call_count"] == 30
        
        memory_summary = profiler.get_memory_summary()
        assert memory_summary["snapshot_count"] == 6  # Every 5 steps
        
        # Check auto-save worked
        assert save_path.exists()
        
        # Load and verify saved data
        new_tracker = MetricsTracker()
        new_tracker.load(str(save_path))
        assert len(new_tracker.metrics) == 3
        assert "loss" in new_tracker.metrics


if __name__ == "__main__":
    pytest.main([__file__]) 