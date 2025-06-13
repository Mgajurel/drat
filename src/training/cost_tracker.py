"""
Cost Tracking Utilities for Resource-Aware Training.

Provides real-time monitoring of memory usage and computation costs
during forward and backward passes of transformer models with gates.
"""

import torch
import time
import psutil
import gc
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    cached_mb: float
    system_memory_mb: float
    system_memory_percent: float


@dataclass
class ComputationCost:
    """Cost metrics for a computation operation."""
    operation_name: str
    execution_time_ms: float
    memory_delta_mb: float
    flops_estimate: Optional[int] = None
    energy_estimate: Optional[float] = None


@dataclass
class LayerCostMetrics:
    """Cost metrics for a single transformer layer."""
    layer_idx: int
    attention_cost: ComputationCost
    feedforward_cost: ComputationCost
    gate_overhead_cost: ComputationCost
    total_memory_mb: float
    total_time_ms: float
    gate_decisions: Dict[str, float] = field(default_factory=dict)


@dataclass
class BatchCostMetrics:
    """Aggregated cost metrics for a training batch."""
    batch_idx: int
    total_time_ms: float
    total_memory_mb: float
    peak_memory_mb: float
    layer_costs: List[LayerCostMetrics] = field(default_factory=list)
    forward_cost: float = 0.0
    backward_cost: float = 0.0
    gate_overhead: float = 0.0


class MemoryTracker:
    """Real-time memory usage tracker."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_memory = 0.0
        self._lock = threading.Lock()
    
    def take_snapshot(self, clear_cache: bool = False) -> MemorySnapshot:
        """Take a snapshot of current memory usage."""
        if clear_cache:
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        timestamp = time.time()
        
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2
            cached = torch.cuda.memory_cached(self.device) / 1024**2
        else:
            # For CPU, use process memory
            process = psutil.Process()
            memory_info = process.memory_info()
            allocated = memory_info.rss / 1024**2
            reserved = memory_info.vms / 1024**2
            max_allocated = allocated  # Approximation
            cached = 0.0
        
        # System memory
        system_memory = psutil.virtual_memory()
        system_memory_mb = system_memory.total / 1024**2
        system_memory_percent = system_memory.percent
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            cached_mb=cached,
            system_memory_mb=system_memory_mb,
            system_memory_percent=system_memory_percent
        )
        
        with self._lock:
            self.snapshots.append(snapshot)
        
        return snapshot
    
    def get_memory_delta(self, baseline_snapshot: Optional[MemorySnapshot] = None) -> float:
        """Get memory delta from baseline or last snapshot."""
        current = self.take_snapshot()
        
        if baseline_snapshot is None:
            if len(self.snapshots) < 2:
                return 0.0
            baseline = self.snapshots[-2]
        else:
            baseline = baseline_snapshot
        
        return current.allocated_mb - baseline.allocated_mb
    
    def reset_peak_memory(self):
        """Reset peak memory tracking."""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage since last reset."""
        if self.device.type == 'cuda':
            return torch.cuda.max_memory_allocated(self.device) / 1024**2
        else:
            # For CPU, return max from snapshots
            if not self.snapshots:
                return 0.0
            return max(s.allocated_mb for s in self.snapshots)


class ComputationTimer:
    """High-precision timer for computation cost measurement."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_time = 0.0
        self.end_time = 0.0
    
    def start(self):
        """Start timing."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time in milliseconds."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.end_time = time.perf_counter()
        return (self.end_time - self.start_time) * 1000.0
    
    @contextmanager
    def measure(self):
        """Context manager for timing operations."""
        self.start()
        try:
            yield self
        finally:
            elapsed = self.stop()
            self.elapsed_ms = elapsed


class CostTracker:
    """Comprehensive cost tracker for resource-aware training."""
    
    def __init__(self, device: Optional[torch.device] = None, track_flops: bool = False):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_tracker = MemoryTracker(device)
        self.timer = ComputationTimer(device)
        self.track_flops = track_flops
        
        # Cost tracking state
        self.batch_costs: List[BatchCostMetrics] = []
        self.current_batch_idx = 0
        self.layer_costs: List[LayerCostMetrics] = []
        
        # Operation tracking
        self._operation_stack: List[Tuple[str, float, MemorySnapshot]] = []
        self._current_layer_metrics: Optional[LayerCostMetrics] = None
    
    def start_batch(self, batch_idx: int):
        """Start tracking costs for a new batch."""
        self.current_batch_idx = batch_idx
        self.layer_costs.clear()
        self.memory_tracker.reset_peak_memory()
        self.memory_tracker.take_snapshot(clear_cache=True)
    
    def end_batch(self) -> BatchCostMetrics:
        """End batch tracking and return cost metrics."""
        final_snapshot = self.memory_tracker.take_snapshot()
        peak_memory = self.memory_tracker.get_peak_memory()
        
        # Calculate total costs
        total_time = sum(layer.total_time_ms for layer in self.layer_costs)
        total_memory = final_snapshot.allocated_mb
        
        # Aggregate forward/backward costs
        forward_cost = sum(
            layer.attention_cost.execution_time_ms + layer.feedforward_cost.execution_time_ms
            for layer in self.layer_costs
        )
        gate_overhead = sum(layer.gate_overhead_cost.execution_time_ms for layer in self.layer_costs)
        
        batch_metrics = BatchCostMetrics(
            batch_idx=self.current_batch_idx,
            total_time_ms=total_time,
            total_memory_mb=total_memory,
            peak_memory_mb=peak_memory,
            layer_costs=self.layer_costs.copy(),
            forward_cost=forward_cost,
            backward_cost=0.0,  # Will be updated during backward pass
            gate_overhead=gate_overhead
        )
        
        self.batch_costs.append(batch_metrics)
        return batch_metrics
    
    @contextmanager
    def track_layer(self, layer_idx: int):
        """Context manager for tracking layer-level costs."""
        layer_start_snapshot = self.memory_tracker.take_snapshot()
        layer_start_time = time.perf_counter()
        
        # Initialize layer metrics
        layer_metrics = LayerCostMetrics(
            layer_idx=layer_idx,
            attention_cost=ComputationCost("attention", 0.0, 0.0),
            feedforward_cost=ComputationCost("feedforward", 0.0, 0.0),
            gate_overhead_cost=ComputationCost("gate_overhead", 0.0, 0.0),
            total_memory_mb=0.0,
            total_time_ms=0.0
        )
        
        # Set current layer context
        self._current_layer_metrics = layer_metrics
        
        try:
            yield layer_metrics
        finally:
            # Finalize layer metrics
            layer_end_time = time.perf_counter()
            layer_end_snapshot = self.memory_tracker.take_snapshot()
            
            layer_metrics.total_time_ms = (layer_end_time - layer_start_time) * 1000.0
            layer_metrics.total_memory_mb = layer_end_snapshot.allocated_mb - layer_start_snapshot.allocated_mb
            
            self.layer_costs.append(layer_metrics)
            self._current_layer_metrics = None
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager for tracking individual operations."""
        start_snapshot = self.memory_tracker.take_snapshot()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_snapshot = self.memory_tracker.take_snapshot()
            
            execution_time_ms = (end_time - start_time) * 1000.0
            memory_delta_mb = end_snapshot.allocated_mb - start_snapshot.allocated_mb
            
            cost = ComputationCost(
                operation_name=operation_name,
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )
            
            # Update current layer metrics if in layer context
            if self._current_layer_metrics is not None:
                if operation_name == "attention":
                    self._current_layer_metrics.attention_cost = cost
                elif operation_name == "feedforward":
                    self._current_layer_metrics.feedforward_cost = cost
                elif operation_name.startswith("gate"):
                    self._current_layer_metrics.gate_overhead_cost = cost
    
    def track_gate_decisions(self, layer_idx: int, gate_statistics: Dict[str, Any]):
        """Track gate decisions for cost analysis."""
        if self.layer_costs and self.layer_costs[-1].layer_idx == layer_idx:
            self.layer_costs[-1].gate_decisions = {
                'attention_prob': gate_statistics.get('attention_gate_prob', 0.0),
                'ff_prob': gate_statistics.get('ff_gate_prob', 0.0)
            }
    
    def get_cost_summary(self, num_recent_batches: int = 10) -> Dict[str, Any]:
        """Get summary of recent cost metrics."""
        if not self.batch_costs:
            return {}
        
        recent_batches = self.batch_costs[-num_recent_batches:]
        
        avg_time = sum(batch.total_time_ms for batch in recent_batches) / len(recent_batches)
        avg_memory = sum(batch.total_memory_mb for batch in recent_batches) / len(recent_batches)
        peak_memory = max(batch.peak_memory_mb for batch in recent_batches)
        
        # Gate efficiency metrics
        gate_overhead_ratio = 0.0
        if recent_batches:
            total_forward = sum(batch.forward_cost for batch in recent_batches)
            total_gate_overhead = sum(batch.gate_overhead for batch in recent_batches)
            if total_forward > 0:
                gate_overhead_ratio = total_gate_overhead / total_forward
        
        return {
            'avg_time_ms': avg_time,
            'avg_memory_mb': avg_memory,
            'peak_memory_mb': peak_memory,
            'gate_overhead_ratio': gate_overhead_ratio,
            'num_batches': len(recent_batches),
            'total_batches_tracked': len(self.batch_costs)
        }
    
    def reset(self):
        """Reset all tracking data."""
        self.batch_costs.clear()
        self.layer_costs.clear()
        self.memory_tracker.snapshots.clear()
        self.current_batch_idx = 0
        self._operation_stack.clear()
        self._current_layer_metrics = None


# Global cost tracker instance
_global_cost_tracker: Optional[CostTracker] = None


def get_global_cost_tracker(device: Optional[torch.device] = None) -> CostTracker:
    """Get or create the global cost tracker instance."""
    global _global_cost_tracker
    if _global_cost_tracker is None:
        _global_cost_tracker = CostTracker(device)
    return _global_cost_tracker


def reset_global_cost_tracker():
    """Reset the global cost tracker."""
    global _global_cost_tracker
    if _global_cost_tracker is not None:
        _global_cost_tracker.reset()


# Convenience functions for easy integration
def start_batch_tracking(batch_idx: int, device: Optional[torch.device] = None):
    """Start tracking costs for a batch."""
    tracker = get_global_cost_tracker(device)
    tracker.start_batch(batch_idx)


def end_batch_tracking() -> BatchCostMetrics:
    """End batch tracking and return metrics."""
    tracker = get_global_cost_tracker()
    return tracker.end_batch()


def track_layer_costs(layer_idx: int):
    """Context manager for tracking layer costs."""
    tracker = get_global_cost_tracker()
    return tracker.track_layer(layer_idx)


def track_operation_costs(operation_name: str):
    """Context manager for tracking operation costs."""
    tracker = get_global_cost_tracker()
    return tracker.track_operation(operation_name) 