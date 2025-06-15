"""
Training utilities for the transformer model.

This module provides comprehensive training infrastructure including:
- Resource-aware loss functions with cost tracking
- Training orchestration with logging and monitoring
- Memory and computation cost tracking
- Metrics collection and analysis
- Performance profiling utilities
"""

from .loss import ResourceAwareLoss, CostMetrics
from .cost_tracker import (
    CostTracker, MemoryTracker, ComputationTimer,
    MemorySnapshot, ComputationCost, LayerCostMetrics, BatchCostMetrics,
    get_global_cost_tracker, reset_global_cost_tracker,
    start_batch_tracking, end_batch_tracking, track_layer_costs, track_operation_costs
)
from .trainer import (
    ResourceAwareTrainer, TrainingConfig, TrainingMetrics
)

# Import metrics utilities from utils if available
try:
    from ..utils.metrics import (
        MetricsTracker, MetricSeries, MetricSnapshot, 
        PerformanceProfiler, create_metrics_tracker
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

__all__ = [
    # Core loss and cost tracking
    'ResourceAwareLoss',
    'CostMetrics',
    'CostTracker',
    'MemoryTracker', 
    'ComputationTimer',
    'MemorySnapshot',
    'ComputationCost',
    'LayerCostMetrics',
    'BatchCostMetrics',
    
    # Global cost tracking utilities
    'get_global_cost_tracker',
    'reset_global_cost_tracker',
    'start_batch_tracking',
    'end_batch_tracking',
    'track_layer_costs',
    'track_operation_costs',
    
    # Training orchestration
    'ResourceAwareTrainer',
    'TrainingConfig', 
    'TrainingMetrics',
]

# Add metrics utilities if available
if _METRICS_AVAILABLE:
    __all__.extend([
        'MetricsTracker',
        'MetricSeries', 
        'MetricSnapshot',
        'PerformanceProfiler',
        'create_metrics_tracker'
    ])


def get_training_components():
    """
    Get a summary of available training components.
    
    Returns:
        dict: Dictionary describing available components and their status
    """
    components = {
        'loss_function': {
            'class': 'ResourceAwareLoss',
            'available': True,
            'description': 'Resource-aware loss function with cost penalties'
        },
        'cost_tracking': {
            'class': 'CostTracker',
            'available': True,
            'description': 'Real-time memory and computation cost tracking'
        },
        'trainer': {
            'class': 'ResourceAwareTrainer',
            'available': True,
            'description': 'Complete training orchestration with logging'
        },
        'metrics': {
            'class': 'MetricsTracker',
            'available': _METRICS_AVAILABLE,
            'description': 'Advanced metrics tracking and analysis'
        }
    }
    
    return components


def create_training_setup(config_dict: dict, **kwargs):
    """
    Factory function to create a complete training setup.
    
    Args:
        config_dict: Training configuration dictionary
        **kwargs: Additional arguments for trainer creation
        
    Returns:
        ResourceAwareTrainer: Configured trainer instance
    """
    from .trainer import TrainingConfig, ResourceAwareTrainer
    
    # Create training config
    config = TrainingConfig(**config_dict)
    
    # Create trainer
    trainer = ResourceAwareTrainer(config=config, **kwargs)
    
    return trainer


# Version and compatibility info
__version__ = "1.0.0"
__author__ = "Resource-Aware Transformer Team"
__description__ = "Comprehensive training infrastructure for resource-aware transformers" 