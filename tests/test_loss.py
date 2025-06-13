"""
Tests for Resource-Aware Loss Function.
"""

import pytest
import torch
import torch.nn as nn
from src.training.loss import ResourceAwareLoss, CostMetrics
from src.training.cost_tracker import (
    reset_global_cost_tracker, start_batch_tracking, end_batch_tracking,
    track_layer_costs, track_operation_costs
)


class TestResourceAwareLoss:
    """Test suite for ResourceAwareLoss."""
    
    def test_initialization(self):
        """Test loss function initialization with different parameters."""
        # Default initialization
        loss_fn = ResourceAwareLoss()
        assert loss_fn.lambda_resource == 0.01
        assert loss_fn.memory_cost_base == 1.0
        assert loss_fn.recomputation_cost_base == 2.0
        assert loss_fn.cost_model == "uniform"
        assert loss_fn.normalize_costs is True
        assert loss_fn.use_real_time_costs is False
        
        # Custom initialization
        custom_loss_fn = nn.MSELoss()
        loss_fn = ResourceAwareLoss(
            task_loss_fn=custom_loss_fn,
            lambda_resource=0.05,
            memory_cost_base=2.0,
            recomputation_cost_base=3.0,
            cost_model="layer_weighted",
            normalize_costs=False,
            use_real_time_costs=True
        )
        
        assert loss_fn.task_loss_fn == custom_loss_fn
        assert loss_fn.lambda_resource == 0.05
        assert loss_fn.memory_cost_base == 2.0
        assert loss_fn.recomputation_cost_base == 3.0
        assert loss_fn.cost_model == "layer_weighted"
        assert loss_fn.normalize_costs is False
        assert loss_fn.use_real_time_costs is True
    
    def test_invalid_cost_model(self):
        """Test initialization with invalid cost model."""
        with pytest.raises(ValueError, match="cost_model must be one of"):
            ResourceAwareLoss(cost_model="invalid_model")
    
    def test_task_loss_computation(self):
        """Test task loss computation."""
        loss_fn = ResourceAwareLoss()
        
        # Test with 2D logits (batch_size, vocab_size)
        logits_2d = torch.randn(4, 100)
        targets_2d = torch.randint(0, 100, (4,))
        
        task_loss = loss_fn.compute_task_loss(logits_2d, targets_2d)
        assert task_loss.dim() == 0  # Scalar loss
        assert task_loss.item() >= 0
        
        # Test with 3D logits (batch_size, seq_len, vocab_size)
        logits_3d = torch.randn(2, 10, 100)
        targets_3d = torch.randint(0, 100, (2, 10))
        
        task_loss = loss_fn.compute_task_loss(logits_3d, targets_3d)
        assert task_loss.dim() == 0  # Scalar loss
        assert task_loss.item() >= 0
    
    def test_resource_cost_computation(self):
        """Test resource cost computation with gate statistics."""
        loss_fn = ResourceAwareLoss()
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.7, 'ff_gate_prob': 0.3, 'layer_idx': 0},
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.8, 'layer_idx': 1}
            ]
        }
        
        cost_metrics = loss_fn.compute_resource_costs(
            gate_statistics, hidden_size=512, seq_len=10, batch_size=2
        )
        
        assert isinstance(cost_metrics, CostMetrics)
        assert cost_metrics.memory_cost > 0
        assert cost_metrics.recomputation_cost > 0
        assert cost_metrics.total_resource_cost > 0
        assert len(cost_metrics.layer_costs) == 2
        
        # Check layer-specific costs
        layer_0 = cost_metrics.layer_costs[0]
        assert layer_0['layer_idx'] == 0
        assert layer_0['attention_gate_prob'] == 0.7
        assert layer_0['ff_gate_prob'] == 0.3
    
    def test_cost_normalization(self):
        """Test cost normalization by number of layers."""
        loss_fn = ResourceAwareLoss(normalize_costs=True)
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 0},
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 1}
            ]
        }
        
        cost_metrics = loss_fn.compute_resource_costs(
            gate_statistics, hidden_size=512, seq_len=10, batch_size=2
        )
        
        # Each layer: memory = 0.5*1.0 + 0.5*1.0 = 1.0, recomp = 0.5*2.0 + 0.5*2.0 = 2.0
        # Total per layer = 3.0, normalized by 2 layers = 1.5
        assert abs(cost_metrics.total_resource_cost - 3.0) < 1e-6
    
    def test_different_cost_models(self):
        """Test different cost calculation models."""
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 0}
            ]
        }
        
        # Uniform model
        loss_fn_uniform = ResourceAwareLoss(cost_model="uniform")
        cost_uniform = loss_fn_uniform.compute_resource_costs(
            gate_statistics, hidden_size=512, seq_len=10, batch_size=2
        )
        
        # Layer weighted model
        loss_fn_weighted = ResourceAwareLoss(cost_model="layer_weighted")
        cost_weighted = loss_fn_weighted.compute_resource_costs(
            gate_statistics, hidden_size=512, seq_len=10, batch_size=2
        )
        
        # Activation size model
        loss_fn_activation = ResourceAwareLoss(cost_model="activation_size")
        cost_activation = loss_fn_activation.compute_resource_costs(
            gate_statistics, hidden_size=512, seq_len=10, batch_size=2
        )
        
        # All should produce valid costs
        assert cost_uniform.total_resource_cost > 0
        assert cost_weighted.total_resource_cost > 0
        assert cost_activation.total_resource_cost > 0
        
        # Activation size model should have higher costs due to tensor sizes
        assert cost_activation.total_resource_cost > cost_uniform.total_resource_cost
    
    def test_forward_pass_basic(self):
        """Test basic forward pass without gate statistics."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.1)
        
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        
        # Without gate statistics, should only compute task loss
        loss = loss_fn(logits, targets)
        
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0
        assert loss.requires_grad
    
    def test_forward_pass_with_gate_statistics(self):
        """Test forward pass with gate statistics."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.1)
        
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.7, 'ff_gate_prob': 0.3, 'layer_idx': 0}
            ]
        }
        
        loss, metrics = loss_fn(
            logits, targets, 
            gate_statistics=gate_statistics,
            hidden_size=512,
            return_metrics=True
        )
        
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert loss.requires_grad
        
        # Check metrics
        assert 'task_loss' in metrics
        assert 'resource_loss' in metrics
        assert 'total_loss' in metrics
        assert 'cost_metrics' in metrics
        assert metrics['lambda_resource'] == 0.1
        assert metrics['cost_model'] == "uniform"
        
        # Resource loss should be non-zero
        assert metrics['resource_loss'] > 0
    
    def test_forward_pass_with_real_time_costs(self):
        """Test forward pass with real-time cost tracking."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.1, use_real_time_costs=True)
        
        # Reset and set up cost tracking
        reset_global_cost_tracker()
        start_batch_tracking(0)
        
        # Simulate some layer operations
        with track_layer_costs(0):
            with track_operation_costs("attention"):
                pass
            with track_operation_costs("feedforward"):
                pass
        
        end_batch_tracking()
        
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        
        loss, metrics = loss_fn(
            logits, targets,
            batch_idx=0,
            return_metrics=True
        )
        
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert loss.requires_grad
        
        # Should have used real-time costs
        assert 'cost_metrics' in metrics
        assert metrics['cost_metrics'] is not None
    
    def test_real_time_cost_model(self):
        """Test using real_time cost model."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.1, cost_model="real_time")
        
        # Reset and set up cost tracking
        reset_global_cost_tracker()
        start_batch_tracking(0)
        
        with track_layer_costs(0):
            pass
        
        end_batch_tracking()
        
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        
        loss, metrics = loss_fn(
            logits, targets,
            batch_idx=0,
            return_metrics=True
        )
        
        assert metrics['cost_model'] == "real_time"
        assert 'cost_metrics' in metrics
    
    def test_compute_real_time_costs_no_data(self):
        """Test real-time cost computation with no tracking data."""
        loss_fn = ResourceAwareLoss()
        
        # Reset tracker to ensure no data
        reset_global_cost_tracker()
        
        cost_metrics = loss_fn.compute_real_time_costs()
        
        assert cost_metrics.memory_cost == 0.0
        assert cost_metrics.recomputation_cost == 0.0
        assert cost_metrics.total_resource_cost == 0.0
        assert len(cost_metrics.layer_costs) == 0
    
    def test_compute_real_time_costs_with_data(self):
        """Test real-time cost computation with tracking data."""
        loss_fn = ResourceAwareLoss()
        
        # Reset and set up cost tracking
        reset_global_cost_tracker()
        start_batch_tracking(0)
        
        with track_layer_costs(0):
            pass
        
        # Add gate decisions
        from src.training.cost_tracker import get_global_cost_tracker
        tracker = get_global_cost_tracker()
        tracker.track_gate_decisions(0, {
            'attention_gate_prob': 0.7,
            'ff_gate_prob': 0.3
        })
        
        end_batch_tracking()
        
        cost_metrics = loss_fn.compute_real_time_costs(batch_idx=0)
        
        assert isinstance(cost_metrics, CostMetrics)
        assert len(cost_metrics.layer_costs) == 1
        
        layer_cost = cost_metrics.layer_costs[0]
        assert layer_cost['layer_idx'] == 0
        assert layer_cost['attention_gate_prob'] == 0.7
        assert layer_cost['ff_gate_prob'] == 0.3
        assert 'measured_time_ms' in layer_cost
        assert 'measured_memory_mb' in layer_cost
    
    def test_gradient_flow(self):
        """Test gradient flow through the loss function."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.1)
        
        # Create tensors that require gradients
        logits = torch.randn(4, 100, requires_grad=True)
        targets = torch.randint(0, 100, (4,))
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.7, 'ff_gate_prob': 0.3, 'layer_idx': 0}
            ]
        }
        
        loss = loss_fn(logits, targets, gate_statistics=gate_statistics, hidden_size=512)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
        assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))
    
    def test_lambda_zero(self):
        """Test behavior when lambda_resource is zero."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.0)
        
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.7, 'ff_gate_prob': 0.3, 'layer_idx': 0}
            ]
        }
        
        loss, metrics = loss_fn(
            logits, targets,
            gate_statistics=gate_statistics,
            hidden_size=512,
            return_metrics=True
        )
        
        # Resource loss should be zero
        assert metrics['resource_loss'] == 0.0
        assert metrics['total_loss'] == metrics['task_loss']
    
    def test_configuration_methods(self):
        """Test configuration getter and setter methods."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.01, cost_model="uniform")
        
        # Test lambda setter
        loss_fn.set_lambda(0.05)
        assert loss_fn.lambda_resource == 0.05
        
        # Test cost model setter
        loss_fn.set_cost_model("layer_weighted")
        assert loss_fn.cost_model == "layer_weighted"
        
        # Test invalid cost model
        with pytest.raises(ValueError):
            loss_fn.set_cost_model("invalid")
        
        # Test real-time costs toggle
        loss_fn.enable_real_time_costs(True)
        assert loss_fn.use_real_time_costs is True
        
        loss_fn.enable_real_time_costs(False)
        assert loss_fn.use_real_time_costs is False
        
        # Test configuration getter
        config = loss_fn.get_config()
        expected_keys = [
            'lambda_resource', 'memory_cost_base', 'recomputation_cost_base',
            'cost_model', 'layer_weights', 'normalize_costs', 'use_real_time_costs'
        ]
        for key in expected_keys:
            assert key in config
    
    def test_empty_gate_statistics(self):
        """Test behavior with empty gate statistics."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.1)
        
        # Empty layer stats
        gate_statistics = {'layer_stats': []}
        
        cost_metrics = loss_fn.compute_resource_costs(
            gate_statistics, hidden_size=512, seq_len=10, batch_size=2
        )
        
        assert cost_metrics.memory_cost == 0.0
        assert cost_metrics.recomputation_cost == 0.0
        assert cost_metrics.total_resource_cost == 0.0
        assert len(cost_metrics.layer_costs) == 0
    
    def test_layer_weights(self):
        """Test layer-weighted cost model with custom weights."""
        layer_weights = [1.0, 1.5, 2.0]
        loss_fn = ResourceAwareLoss(
            cost_model="layer_weighted",
            layer_weights=layer_weights
        )
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 0},
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 1},
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 2}
            ]
        }
        
        cost_metrics = loss_fn.compute_resource_costs(
            gate_statistics, hidden_size=512, seq_len=10, batch_size=2
        )
        
        # Layer 2 should have higher cost due to weight 2.0
        layer_costs = cost_metrics.layer_costs
        assert layer_costs[2]['total_cost'] > layer_costs[1]['total_cost']
        assert layer_costs[1]['total_cost'] > layer_costs[0]['total_cost']


if __name__ == "__main__":
    pytest.main([__file__]) 