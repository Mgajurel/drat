"""
Tests for Cost Tracking Utilities.
"""

import pytest
import torch
import time
from unittest.mock import patch, MagicMock
from src.training.cost_tracker import (
    MemoryTracker, ComputationTimer, CostTracker,
    MemorySnapshot, ComputationCost, LayerCostMetrics, BatchCostMetrics,
    get_global_cost_tracker, reset_global_cost_tracker,
    start_batch_tracking, end_batch_tracking, track_layer_costs, track_operation_costs
)


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""
    
    def test_memory_snapshot_creation(self):
        """Test creating a memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_mb=100.0,
            reserved_mb=150.0,
            max_allocated_mb=120.0,
            cached_mb=50.0,
            system_memory_mb=8192.0,
            system_memory_percent=75.0
        )
        
        assert snapshot.allocated_mb == 100.0
        assert snapshot.reserved_mb == 150.0
        assert snapshot.max_allocated_mb == 120.0
        assert snapshot.cached_mb == 50.0
        assert snapshot.system_memory_mb == 8192.0
        assert snapshot.system_memory_percent == 75.0


class TestComputationCost:
    """Test ComputationCost dataclass."""
    
    def test_computation_cost_creation(self):
        """Test creating computation cost metrics."""
        cost = ComputationCost(
            operation_name="attention",
            execution_time_ms=15.5,
            memory_delta_mb=25.0,
            flops_estimate=1000000,
            energy_estimate=0.5
        )
        
        assert cost.operation_name == "attention"
        assert cost.execution_time_ms == 15.5
        assert cost.memory_delta_mb == 25.0
        assert cost.flops_estimate == 1000000
        assert cost.energy_estimate == 0.5


class TestMemoryTracker:
    """Test MemoryTracker class."""
    
    def test_memory_tracker_initialization(self):
        """Test memory tracker initialization."""
        device = torch.device('cpu')
        tracker = MemoryTracker(device)
        
        assert tracker.device == device
        assert len(tracker.snapshots) == 0
        assert tracker.baseline_memory == 0.0
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_take_snapshot_cpu(self, mock_virtual_memory, mock_process):
        """Test taking memory snapshot on CPU."""
        # Mock system memory
        mock_virtual_memory.return_value = MagicMock(
            total=8589934592,  # 8GB
            percent=75.0
        )
        
        # Mock process memory
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 104857600  # 100MB
        mock_memory_info.vms = 157286400  # 150MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        tracker = MemoryTracker(torch.device('cpu'))
        snapshot = tracker.take_snapshot()
        
        assert snapshot.allocated_mb == 100.0
        assert snapshot.reserved_mb == 150.0
        assert snapshot.system_memory_mb == 8192.0
        assert snapshot.system_memory_percent == 75.0
        assert len(tracker.snapshots) == 1
    
    def test_get_memory_delta(self):
        """Test memory delta calculation."""
        tracker = MemoryTracker(torch.device('cpu'))
        
        # No snapshots yet
        delta = tracker.get_memory_delta()
        assert delta == 0.0
        
        # Mock snapshots
        snapshot1 = MemorySnapshot(
            timestamp=time.time(),
            allocated_mb=100.0,
            reserved_mb=150.0,
            max_allocated_mb=120.0,
            cached_mb=50.0,
            system_memory_mb=8192.0,
            system_memory_percent=75.0
        )
        
        snapshot2 = MemorySnapshot(
            timestamp=time.time(),
            allocated_mb=125.0,
            reserved_mb=175.0,
            max_allocated_mb=140.0,
            cached_mb=60.0,
            system_memory_mb=8192.0,
            system_memory_percent=80.0
        )
        
        tracker.snapshots = [snapshot1, snapshot2]
        
        # Test delta calculation
        with patch.object(tracker, 'take_snapshot', return_value=snapshot2):
            delta = tracker.get_memory_delta(snapshot1)
            assert delta == 25.0  # 125.0 - 100.0


class TestComputationTimer:
    """Test ComputationTimer class."""
    
    def test_timer_initialization(self):
        """Test timer initialization."""
        device = torch.device('cpu')
        timer = ComputationTimer(device)
        
        assert timer.device == device
        assert timer.start_time == 0.0
        assert timer.end_time == 0.0
    
    def test_timer_measurement(self):
        """Test timing measurement."""
        timer = ComputationTimer(torch.device('cpu'))
        
        timer.start()
        time.sleep(0.01)  # Sleep for 10ms
        elapsed = timer.stop()
        
        # Should be approximately 10ms, allow some tolerance
        assert 8.0 <= elapsed <= 20.0
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        timer = ComputationTimer(torch.device('cpu'))
        
        with timer.measure():
            time.sleep(0.01)  # Sleep for 10ms
        
        # Should be approximately 10ms, allow some tolerance
        assert 8.0 <= timer.elapsed_ms <= 20.0


class TestCostTracker:
    """Test CostTracker class."""
    
    def test_cost_tracker_initialization(self):
        """Test cost tracker initialization."""
        device = torch.device('cpu')
        tracker = CostTracker(device, track_flops=True)
        
        assert tracker.device == device
        assert tracker.track_flops is True
        assert len(tracker.batch_costs) == 0
        assert tracker.current_batch_idx == 0
        assert len(tracker.layer_costs) == 0
    
    def test_batch_tracking(self):
        """Test batch-level cost tracking."""
        tracker = CostTracker(torch.device('cpu'))
        
        # Start batch tracking
        tracker.start_batch(0)
        assert tracker.current_batch_idx == 0
        assert len(tracker.layer_costs) == 0
        
        # End batch tracking
        batch_metrics = tracker.end_batch()
        assert isinstance(batch_metrics, BatchCostMetrics)
        assert batch_metrics.batch_idx == 0
        assert len(tracker.batch_costs) == 1
    
    def test_layer_tracking(self):
        """Test layer-level cost tracking."""
        tracker = CostTracker(torch.device('cpu'))
        tracker.start_batch(0)
        
        with tracker.track_layer(0) as layer_metrics:
            assert isinstance(layer_metrics, LayerCostMetrics)
            assert layer_metrics.layer_idx == 0
            
            # Simulate some work
            time.sleep(0.01)
        
        # Check that layer was recorded
        assert len(tracker.layer_costs) == 1
        assert tracker.layer_costs[0].layer_idx == 0
        assert tracker.layer_costs[0].total_time_ms > 0
    
    def test_operation_tracking(self):
        """Test operation-level cost tracking."""
        tracker = CostTracker(torch.device('cpu'))
        tracker.start_batch(0)
        
        with tracker.track_layer(0) as layer_metrics:
            with tracker.track_operation("attention"):
                time.sleep(0.005)  # 5ms
            
            with tracker.track_operation("feedforward"):
                time.sleep(0.005)  # 5ms
        
        # Check that operations were recorded in layer metrics
        layer = tracker.layer_costs[0]
        assert layer.attention_cost.operation_name == "attention"
        assert layer.feedforward_cost.operation_name == "feedforward"
        assert layer.attention_cost.execution_time_ms > 0
        assert layer.feedforward_cost.execution_time_ms > 0
    
    def test_gate_decisions_tracking(self):
        """Test gate decisions tracking."""
        tracker = CostTracker(torch.device('cpu'))
        tracker.start_batch(0)
        
        with tracker.track_layer(0):
            pass
        
        # Track gate decisions
        gate_stats = {
            'attention_gate_prob': 0.7,
            'ff_gate_prob': 0.3
        }
        tracker.track_gate_decisions(0, gate_stats)
        
        # Check that gate decisions were recorded
        layer = tracker.layer_costs[0]
        assert layer.gate_decisions['attention_prob'] == 0.7
        assert layer.gate_decisions['ff_prob'] == 0.3
    
    def test_cost_summary(self):
        """Test cost summary generation."""
        tracker = CostTracker(torch.device('cpu'))
        
        # Create some mock batch data
        for i in range(3):
            tracker.start_batch(i)
            
            with tracker.track_layer(0):
                time.sleep(0.01)
            
            tracker.end_batch()
        
        # Get cost summary
        summary = tracker.get_cost_summary(num_recent_batches=2)
        
        assert 'avg_time_ms' in summary
        assert 'avg_memory_mb' in summary
        assert 'peak_memory_mb' in summary
        assert 'gate_overhead_ratio' in summary
        assert summary['num_batches'] == 2
        assert summary['total_batches_tracked'] == 3
    
    def test_reset(self):
        """Test tracker reset functionality."""
        tracker = CostTracker(torch.device('cpu'))
        
        # Add some data
        tracker.start_batch(0)
        with tracker.track_layer(0):
            pass
        tracker.end_batch()
        
        assert len(tracker.batch_costs) == 1
        assert len(tracker.layer_costs) == 1
        
        # Reset
        tracker.reset()
        
        assert len(tracker.batch_costs) == 0
        assert len(tracker.layer_costs) == 0
        assert tracker.current_batch_idx == 0


class TestGlobalCostTracker:
    """Test global cost tracker functions."""
    
    def test_global_cost_tracker(self):
        """Test global cost tracker singleton."""
        # Reset first
        reset_global_cost_tracker()
        
        # Get tracker
        tracker1 = get_global_cost_tracker()
        tracker2 = get_global_cost_tracker()
        
        # Should be the same instance
        assert tracker1 is tracker2
        
        # Test convenience functions
        start_batch_tracking(0)
        assert tracker1.current_batch_idx == 0
        
        batch_metrics = end_batch_tracking()
        assert isinstance(batch_metrics, BatchCostMetrics)
        assert batch_metrics.batch_idx == 0
    
    def test_track_layer_costs_context(self):
        """Test layer costs tracking context manager."""
        reset_global_cost_tracker()
        start_batch_tracking(0)
        
        with track_layer_costs(0):
            time.sleep(0.01)
        
        tracker = get_global_cost_tracker()
        assert len(tracker.layer_costs) == 1
        assert tracker.layer_costs[0].layer_idx == 0
    
    def test_track_operation_costs_context(self):
        """Test operation costs tracking context manager."""
        reset_global_cost_tracker()
        start_batch_tracking(0)
        
        with track_layer_costs(0):
            with track_operation_costs("attention"):
                time.sleep(0.005)
        
        tracker = get_global_cost_tracker()
        layer = tracker.layer_costs[0]
        assert layer.attention_cost.operation_name == "attention"
        assert layer.attention_cost.execution_time_ms > 0


class TestIntegration:
    """Integration tests for cost tracking."""
    
    def test_full_training_simulation(self):
        """Test full training loop simulation with cost tracking."""
        reset_global_cost_tracker()
        
        # Simulate training loop
        for batch_idx in range(2):
            start_batch_tracking(batch_idx)
            
            # Simulate forward pass through layers
            for layer_idx in range(3):
                with track_layer_costs(layer_idx):
                    # Simulate attention
                    with track_operation_costs("attention"):
                        time.sleep(0.002)
                    
                    # Simulate feedforward
                    with track_operation_costs("feedforward"):
                        time.sleep(0.003)
                    
                    # Simulate gate overhead
                    with track_operation_costs("gate_overhead"):
                        time.sleep(0.001)
                    
                    # Track gate decisions
                    tracker = get_global_cost_tracker()
                    gate_stats = {
                        'attention_gate_prob': 0.6 + layer_idx * 0.1,
                        'ff_gate_prob': 0.4 + layer_idx * 0.1
                    }
                    tracker.track_gate_decisions(layer_idx, gate_stats)
            
            # End batch
            batch_metrics = end_batch_tracking()
            assert batch_metrics.batch_idx == batch_idx
            assert len(batch_metrics.layer_costs) == 3
        
        # Check final state
        tracker = get_global_cost_tracker()
        assert len(tracker.batch_costs) == 2
        
        # Get summary
        summary = tracker.get_cost_summary()
        assert summary['total_batches_tracked'] == 2
        assert summary['avg_time_ms'] > 0
    
    def test_memory_pressure_simulation(self):
        """Test cost tracking under memory pressure."""
        reset_global_cost_tracker()
        tracker = get_global_cost_tracker()
        
        start_batch_tracking(0)
        
        # Simulate high memory usage layer
        with track_layer_costs(0):
            # Create some tensors to increase memory usage
            tensors = [torch.randn(100, 100) for _ in range(10)]
            
            with track_operation_costs("attention"):
                time.sleep(0.005)
            
            # Clean up tensors
            del tensors
        
        batch_metrics = end_batch_tracking()
        
        # Should have recorded some memory usage
        assert len(batch_metrics.layer_costs) == 1
        layer_cost = batch_metrics.layer_costs[0]
        assert layer_cost.attention_cost.execution_time_ms > 0 