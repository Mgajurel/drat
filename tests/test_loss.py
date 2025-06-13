"""
Tests for Resource-Aware Loss Function.
"""

import pytest
import torch
import torch.nn as nn
from src.training.loss import ResourceAwareLoss, CostMetrics


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
        
        # Custom initialization
        custom_loss_fn = nn.MSELoss()
        loss_fn = ResourceAwareLoss(
            task_loss_fn=custom_loss_fn,
            lambda_resource=0.1,
            memory_cost_base=2.0,
            recomputation_cost_base=3.0,
            cost_model="layer_weighted",
            layer_weights=[1.0, 1.5, 2.0],
            normalize_costs=False
        )
        assert loss_fn.task_loss_fn == custom_loss_fn
        assert loss_fn.lambda_resource == 0.1
        assert loss_fn.memory_cost_base == 2.0
        assert loss_fn.recomputation_cost_base == 3.0
        assert loss_fn.cost_model == "layer_weighted"
        assert loss_fn.layer_weights == [1.0, 1.5, 2.0]
        assert loss_fn.normalize_costs is False
    
    def test_invalid_cost_model(self):
        """Test that invalid cost model raises ValueError."""
        with pytest.raises(ValueError, match="cost_model must be one of"):
            ResourceAwareLoss(cost_model="invalid_model")
    
    def test_task_loss_computation(self):
        """Test task loss computation."""
        loss_fn = ResourceAwareLoss()
        
        # Test with 3D logits (batch_size, seq_len, vocab_size)
        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        task_loss = loss_fn.compute_task_loss(logits, targets)
        assert isinstance(task_loss, torch.Tensor)
        assert task_loss.dim() == 0  # Scalar
        assert task_loss.item() >= 0
        
        # Test with 2D logits (batch_size * seq_len, vocab_size)
        logits_2d = logits.view(-1, vocab_size)
        targets_2d = targets.view(-1)
        
        task_loss_2d = loss_fn.compute_task_loss(logits_2d, targets_2d)
        assert torch.allclose(task_loss, task_loss_2d, atol=1e-6)
    
    def test_cost_metrics_no_gates(self):
        """Test cost computation with no gate statistics."""
        loss_fn = ResourceAwareLoss()
        
        gate_statistics = {'layer_stats': []}
        cost_metrics = loss_fn.compute_resource_costs(
            gate_statistics, hidden_size=512, seq_len=10, batch_size=2
        )
        
        assert cost_metrics.memory_cost == 0.0
        assert cost_metrics.recomputation_cost == 0.0
        assert cost_metrics.total_resource_cost == 0.0
        assert cost_metrics.layer_costs == []
    
    def test_cost_metrics_uniform_model(self):
        """Test cost computation with uniform cost model."""
        loss_fn = ResourceAwareLoss(
            cost_model="uniform",
            memory_cost_base=1.0,
            recomputation_cost_base=2.0,
            normalize_costs=False
        )
        
        # Mock gate statistics for 2 layers
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.8, 'ff_gate_prob': 0.6, 'layer_idx': 0},
                {'attention_gate_prob': 0.4, 'ff_gate_prob': 0.7, 'layer_idx': 1}
            ]
        }
        
        cost_metrics = loss_fn.compute_resource_costs(
            gate_statistics, hidden_size=512, seq_len=10, batch_size=2
        )
        
        # Expected costs for layer 0:
        # att_mem = 0.8 * 1.0 = 0.8, att_recomp = 0.2 * 2.0 = 0.4
        # ff_mem = 0.6 * 1.0 = 0.6, ff_recomp = 0.4 * 2.0 = 0.8
        # layer_0_total = 0.8 + 0.4 + 0.6 + 0.8 = 2.6
        
        # Expected costs for layer 1:
        # att_mem = 0.4 * 1.0 = 0.4, att_recomp = 0.6 * 2.0 = 1.2
        # ff_mem = 0.7 * 1.0 = 0.7, ff_recomp = 0.3 * 2.0 = 0.6
        # layer_1_total = 0.4 + 1.2 + 0.7 + 0.6 = 2.9
        
        expected_total_memory = (0.8 + 0.6) + (0.4 + 0.7)  # 2.5
        expected_total_recomp = (0.4 + 0.8) + (1.2 + 0.6)  # 3.0
        expected_total = expected_total_memory + expected_total_recomp  # 5.5
        
        assert abs(cost_metrics.memory_cost - expected_total_memory) < 1e-6
        assert abs(cost_metrics.recomputation_cost - expected_total_recomp) < 1e-6
        assert abs(cost_metrics.total_resource_cost - expected_total) < 1e-6
        assert len(cost_metrics.layer_costs) == 2
    
    def test_cost_metrics_layer_weighted_model(self):
        """Test cost computation with layer-weighted cost model."""
        layer_weights = [1.0, 2.0]
        loss_fn = ResourceAwareLoss(
            cost_model="layer_weighted",
            layer_weights=layer_weights,
            memory_cost_base=1.0,
            recomputation_cost_base=2.0,
            normalize_costs=False
        )
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 0},
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 1}
            ]
        }
        
        cost_metrics = loss_fn.compute_resource_costs(
            gate_statistics, hidden_size=512, seq_len=10, batch_size=2
        )
        
        # Layer 0 (weight=1.0): mem=0.5*1.0*1.0 + 0.5*1.0*1.0 = 1.0, recomp=0.5*1.0*2.0 + 0.5*1.0*2.0 = 2.0
        # Layer 1 (weight=2.0): mem=0.5*2.0*1.0 + 0.5*2.0*1.0 = 2.0, recomp=0.5*2.0*2.0 + 0.5*2.0*2.0 = 4.0
        expected_total_memory = 1.0 + 2.0  # 3.0
        expected_total_recomp = 2.0 + 4.0  # 6.0
        
        assert abs(cost_metrics.memory_cost - expected_total_memory) < 1e-6
        assert abs(cost_metrics.recomputation_cost - expected_total_recomp) < 1e-6
    
    def test_cost_metrics_activation_size_model(self):
        """Test cost computation with activation-size cost model."""
        loss_fn = ResourceAwareLoss(
            cost_model="activation_size",
            memory_cost_base=1e-6,  # Scale down for reasonable numbers
            recomputation_cost_base=2e-6,
            normalize_costs=False
        )
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 0}
            ]
        }
        
        batch_size, seq_len, hidden_size = 2, 10, 512
        activation_size = batch_size * seq_len * hidden_size  # 10240
        
        cost_metrics = loss_fn.compute_resource_costs(
            gate_statistics, hidden_size, seq_len, batch_size
        )
        
        # Expected: att_mem + ff_mem = 0.5 * 10240 * 1e-6 + 0.5 * 10240 * 1e-6 = 0.01024
        # Expected: att_recomp + ff_recomp = 0.5 * 10240 * 2e-6 + 0.5 * 10240 * 2e-6 = 0.02048
        expected_memory = 2 * 0.5 * activation_size * 1e-6
        expected_recomp = 2 * 0.5 * activation_size * 2e-6
        
        assert abs(cost_metrics.memory_cost - expected_memory) < 1e-8
        assert abs(cost_metrics.recomputation_cost - expected_recomp) < 1e-8
    
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
        
        # Each layer: att_mem=0.5*1.0=0.5, att_recomp=0.5*2.0=1.0, ff_mem=0.5*1.0=0.5, ff_recomp=0.5*2.0=1.0
        # Per layer: memory=1.0, recomp=2.0, total=3.0
        # Without normalization: total would be 2*3.0 = 6.0
        # With normalization by 2 layers: 6.0 / 2 = 3.0
        assert abs(cost_metrics.total_resource_cost - 3.0) < 1e-6
    
    def test_forward_without_gates(self):
        """Test forward pass without gate statistics."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.1)
        
        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Without gate statistics
        total_loss = loss_fn(logits, targets)
        task_loss = loss_fn.compute_task_loss(logits, targets)
        
        # Should be equal to task loss only
        assert torch.allclose(total_loss, task_loss, atol=1e-6)
    
    def test_forward_with_gates(self):
        """Test forward pass with gate statistics."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.1, normalize_costs=False)
        
        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 0}
            ]
        }
        
        total_loss, metrics = loss_fn(
            logits, targets, gate_statistics, 
            hidden_size=vocab_size, return_metrics=True
        )
        
        # Check that total loss includes both task and resource components
        assert metrics['task_loss'] > 0
        assert metrics['resource_loss'] > 0
        assert abs(metrics['total_loss'] - (metrics['task_loss'] + 0.1 * metrics['resource_loss'])) < 1e-6
        assert metrics['lambda_resource'] == 0.1
        assert metrics['cost_metrics'] is not None
    
    def test_forward_zero_lambda(self):
        """Test forward pass with zero lambda (no resource penalty)."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.0)
        
        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 0}
            ]
        }
        
        total_loss = loss_fn(logits, targets, gate_statistics, hidden_size=vocab_size)
        task_loss = loss_fn.compute_task_loss(logits, targets)
        
        # Should be equal to task loss only
        assert torch.allclose(total_loss, task_loss, atol=1e-6)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the loss function."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.1)
        
        batch_size, seq_len, vocab_size = 2, 5, 100
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        gate_statistics = {
            'layer_stats': [
                {'attention_gate_prob': 0.5, 'ff_gate_prob': 0.5, 'layer_idx': 0}
            ]
        }
        
        total_loss = loss_fn(logits, targets, gate_statistics, hidden_size=vocab_size)
        total_loss.backward()
        
        # Check that gradients are computed
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
        assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))
    
    def test_set_lambda(self):
        """Test updating lambda parameter."""
        loss_fn = ResourceAwareLoss(lambda_resource=0.01)
        assert loss_fn.lambda_resource == 0.01
        
        loss_fn.set_lambda(0.05)
        assert loss_fn.lambda_resource == 0.05
    
    def test_get_config(self):
        """Test getting loss function configuration."""
        layer_weights = [1.0, 1.5, 2.0]
        loss_fn = ResourceAwareLoss(
            lambda_resource=0.1,
            memory_cost_base=2.0,
            recomputation_cost_base=3.0,
            cost_model="layer_weighted",
            layer_weights=layer_weights,
            normalize_costs=False
        )
        
        config = loss_fn.get_config()
        expected_config = {
            'lambda_resource': 0.1,
            'memory_cost_base': 2.0,
            'recomputation_cost_base': 3.0,
            'cost_model': 'layer_weighted',
            'layer_weights': layer_weights,
            'normalize_costs': False
        }
        
        assert config == expected_config
    
    def test_cost_metrics_dataclass(self):
        """Test CostMetrics dataclass functionality."""
        layer_costs = [
            {'layer_idx': 0, 'memory_cost': 1.0, 'recomputation_cost': 2.0, 'total_cost': 3.0}
        ]
        gate_stats = {'avg_attention_prob': 0.5, 'avg_ff_prob': 0.6}
        
        metrics = CostMetrics(
            memory_cost=5.0,
            recomputation_cost=10.0,
            total_resource_cost=15.0,
            gate_statistics=gate_stats,
            layer_costs=layer_costs
        )
        
        assert metrics.memory_cost == 5.0
        assert metrics.recomputation_cost == 10.0
        assert metrics.total_resource_cost == 15.0
        assert metrics.gate_statistics == gate_stats
        assert metrics.layer_costs == layer_costs


if __name__ == "__main__":
    pytest.main([__file__]) 