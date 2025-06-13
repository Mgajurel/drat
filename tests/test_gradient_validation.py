import torch
import torch.nn as nn
import pytest
import numpy as np
from torch.autograd import gradcheck
from src.training.loss import ResourceAwareLoss, CostMetrics


class TestGradientFlow:
    """Test gradient flow through ResourceAwareLoss."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.batch_size = 4
        self.num_classes = 10
        self.num_layers = 3
        
        # Create sample inputs
        self.logits = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Create sample gate statistics
        self.gate_stats = {
            'layer_stats': [
                {
                    'attention_gate_prob': torch.tensor(0.7, requires_grad=True),
                    'ff_gate_prob': torch.tensor(0.8, requires_grad=True)
                }
                for i in range(self.num_layers)
            ]
        }
        
        # Create loss function
        self.loss_fn = ResourceAwareLoss(
            lambda_resource=0.1,
            cost_model='uniform',
            normalize_costs=True
        )
    
    def test_basic_gradient_computation(self):
        """Test that gradients can be computed through the loss function."""
        loss = self.loss_fn(self.logits, self.targets, self.gate_stats)
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients exist and are finite
        assert self.logits.grad is not None
        assert torch.isfinite(self.logits.grad).all()
        assert not torch.isnan(self.logits.grad).any()
        
        # Check gate gradients
        for layer_stats in self.gate_stats['layer_stats']:
            assert layer_stats['attention_gate_prob'].grad is not None
            assert layer_stats['ff_gate_prob'].grad is not None
            assert torch.isfinite(layer_stats['attention_gate_prob'].grad).all()
            assert torch.isfinite(layer_stats['ff_gate_prob'].grad).all()
    
    def test_gradient_magnitude_stability(self):
        """Test that gradients have reasonable magnitudes."""
        loss = self.loss_fn(self.logits, self.targets, self.gate_stats)
        loss.backward()
        
        # Check logits gradients
        logits_grad_norm = torch.norm(self.logits.grad)
        assert logits_grad_norm > 1e-8  # Not vanishing
        assert logits_grad_norm < 100.0  # Not exploding
        
        # Check gate gradients
        for layer_stats in self.gate_stats['layer_stats']:
            att_grad = layer_stats['attention_gate_prob'].grad
            ff_grad = layer_stats['ff_gate_prob'].grad
            
            assert torch.abs(att_grad) > 1e-8
            assert torch.abs(att_grad) < 10.0
            assert torch.abs(ff_grad) > 1e-8
            assert torch.abs(ff_grad) < 10.0
    
    def test_gradient_consistency_across_lambda(self):
        """Test gradient consistency with different lambda values."""
        lambda_values = [0.0, 0.1, 0.5, 1.0]
        gradients = []
        
        for lambda_val in lambda_values:
            # Reset gradients
            self.logits.grad = None
            for layer_stats in self.gate_stats['layer_stats']:
                layer_stats['attention_gate_prob'].grad = None
                layer_stats['ff_gate_prob'].grad = None
            
            loss_fn = ResourceAwareLoss(lambda_resource=lambda_val)
            loss = loss_fn(self.logits, self.targets, self.gate_stats)
            loss.backward()
            
            gradients.append({
                'logits': self.logits.grad.clone(),
                'gates': [
                    {
                        'att': stats['attention_gate_prob'].grad.clone() if stats['attention_gate_prob'].grad is not None else torch.zeros_like(stats['attention_gate_prob']),
                        'ff': stats['ff_gate_prob'].grad.clone() if stats['ff_gate_prob'].grad is not None else torch.zeros_like(stats['ff_gate_prob'])
                    }
                    for stats in self.gate_stats['layer_stats']
                ]
            })
        
        # Check that gradients change smoothly with lambda
        for i in range(1, len(gradients)):
            # Logits gradients should be similar (task loss dominates when lambda is small)
            if lambda_values[i] < 0.5:
                logits_diff = torch.norm(gradients[i]['logits'] - gradients[0]['logits'])
                assert logits_diff < 1.0  # Should be relatively similar
    
    def test_numerical_gradient_check(self):
        """Test gradients using numerical differentiation."""
        # Use smaller tensors for numerical gradient check
        small_logits = torch.randn(2, 3, requires_grad=True, dtype=torch.double)
        small_targets = torch.randint(0, 3, (2,))
        
        small_gate_stats = {
            'layer_stats': [
                {
                    'attention_gate_prob': torch.tensor(0.7, requires_grad=True, dtype=torch.double),
                    'ff_gate_prob': torch.tensor(0.8, requires_grad=True, dtype=torch.double)
                }
            ]
        }
        
        loss_fn = ResourceAwareLoss(lambda_resource=0.1, cost_model='uniform')
        
        def loss_func(logits, att_gate, ff_gate):
            gate_stats = {
                'layer_stats': [
                    {
                        'attention_gate_prob': att_gate,
                        'ff_gate_prob': ff_gate
                    }
                ]
            }
            return loss_fn(logits, small_targets, gate_stats)
        
        # Test gradient check
        assert gradcheck(
            loss_func,
            (small_logits, 
             small_gate_stats['layer_stats'][0]['attention_gate_prob'],
             small_gate_stats['layer_stats'][0]['ff_gate_prob']),
            eps=1e-6,
            atol=1e-4
        )


class TestNumericalStability:
    """Test numerical stability of ResourceAwareLoss."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.batch_size = 4
        self.num_classes = 10
        self.num_layers = 3
        
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Create loss function
        self.loss_fn = ResourceAwareLoss(
            lambda_resource=0.1,
            cost_model='uniform',
            normalize_costs=True
        )
    
    def test_extreme_logits_stability(self):
        """Test stability with extreme logit values."""
        extreme_cases = [
            torch.full((self.batch_size, self.num_classes), 1e6),  # Very large
            torch.full((self.batch_size, self.num_classes), -1e6),  # Very small
            torch.full((self.batch_size, self.num_classes), 0.0),   # Zero
        ]
        
        gate_stats = {
            'layer_stats': [
                {
                    'attention_gate_prob': torch.tensor(0.5),
                    'ff_gate_prob': torch.tensor(0.5)
                }
                for i in range(self.num_layers)
            ]
        }
        
        for logits in extreme_cases:
            loss = self.loss_fn(logits, self.targets, gate_stats)
            
            # Loss should be finite
            assert torch.isfinite(loss)
            assert not torch.isnan(loss)
            assert loss >= 0.0  # Loss should be non-negative
    
    def test_extreme_gate_values_stability(self):
        """Test stability with extreme gate probability values."""
        logits = torch.randn(self.batch_size, self.num_classes)
        
        extreme_gate_cases = [
            {'prob': 0.0, 'desc': 'zero probability'},
            {'prob': 1.0, 'desc': 'one probability'},
            {'prob': 1e-8, 'desc': 'very small probability'},
            {'prob': 1.0 - 1e-8, 'desc': 'very large probability'},
        ]
        
        for case in extreme_gate_cases:
            gate_stats = {
                'layer_stats': [
                    {
                        'attention_gate_prob': torch.tensor(case['prob']),
                        'ff_gate_prob': torch.tensor(case['prob'])
                    }
                    for i in range(self.num_layers)
                ]
            }
            
            loss = self.loss_fn(logits, self.targets, gate_stats)
            
            # Loss should be finite
            assert torch.isfinite(loss), f"Loss not finite for {case['desc']}"
            assert not torch.isnan(loss), f"Loss is NaN for {case['desc']}"
            assert loss >= 0.0, f"Loss is negative for {case['desc']}"
    
    def test_lambda_resourceeter_stability(self):
        """Test stability across different lambda parameter values."""
        logits = torch.randn(self.batch_size, self.num_classes)
        gate_stats = {
            f'layer_{i}': {
                'attention_gate_prob': torch.tensor(0.7),
                'ff_gate_prob': torch.tensor(0.8)
            }
            for i in range(self.num_layers)
        }
        
        lambda_values = [0.0, 1e-8, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        for lambda_val in lambda_values:
            loss_fn = ResourceAwareLoss(lambda_resource=lambda_val)
            loss = loss_fn(logits, self.targets, gate_stats)
            
            assert torch.isfinite(loss), f"Loss not finite for lambda={lambda_val}"
            assert not torch.isnan(loss), f"Loss is NaN for lambda={lambda_val}"
            assert loss >= 0.0, f"Loss is negative for lambda={lambda_val}"
    
    def test_cost_model_stability(self):
        """Test stability across different cost models."""
        logits = torch.randn(self.batch_size, self.num_classes)
        gate_stats = {
            f'layer_{i}': {
                'attention_gate_prob': torch.tensor(0.7),
                'ff_gate_prob': torch.tensor(0.8)
            }
            for i in range(self.num_layers)
        }
        
        cost_models = ['uniform', 'layer_weighted', 'activation_size']
        
        for model in cost_models:
            loss_fn = ResourceAwareLoss(lambda_resource=0.1, cost_model=model)
            loss = loss_fn(logits, self.targets, gate_stats)
            
            assert torch.isfinite(loss), f"Loss not finite for cost_model={model}"
            assert not torch.isnan(loss), f"Loss is NaN for cost_model={model}"
            assert loss >= 0.0, f"Loss is negative for cost_model={model}"
    
    def test_batch_size_scaling(self):
        """Test that loss scales appropriately with batch size."""
        gate_stats = {
            'layer_stats': [
                {
                    'attention_gate_prob': torch.tensor(0.7),
                    'ff_gate_prob': torch.tensor(0.8)
                }
            ]
        }
        
        batch_sizes = [1, 2, 4, 8, 16]
        losses = []
        
        for bs in batch_sizes:
            logits = torch.randn(bs, self.num_classes)
            targets = torch.randint(0, self.num_classes, (bs,))
            
            loss = self.loss_fn(logits, targets, gate_stats)
            losses.append(loss.item())
            
            assert torch.isfinite(loss)
            assert not torch.isnan(loss)
        
        # Loss should not grow unboundedly with batch size
        # (it's averaged, so should be roughly constant)
        loss_std = np.std(losses)
        loss_mean = np.mean(losses)
        assert loss_std / loss_mean < 0.5  # Coefficient of variation < 50%


class TestGradientClipping:
    """Test gradient clipping functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.batch_size = 4
        self.num_classes = 10
        
        # Create inputs that might cause large gradients
        self.logits = torch.randn(self.batch_size, self.num_classes, requires_grad=True) * 10
        self.logits.retain_grad()  # Ensure gradients are retained for gradient clipping tests
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        self.gate_stats = {
            'layer_stats': [
                {
                    'attention_gate_prob': torch.tensor(0.01, requires_grad=True),  # Extreme value
                    'ff_gate_prob': torch.tensor(0.99, requires_grad=True)          # Extreme value
                }
            ]
        }
    
    def test_gradient_clipping_effectiveness(self):
        """Test that gradient clipping prevents exploding gradients."""
        # High lambda to amplify resource costs
        loss_fn = ResourceAwareLoss(lambda_resource=100.0, cost_model='uniform')
        
        loss = loss_fn(self.logits, self.targets, self.gate_stats)
        loss.backward()
        
        # Collect all parameters that should have gradients
        parameters = [self.logits] + \
                    [stats['attention_gate_prob'] for stats in self.gate_stats['layer_stats']] + \
                    [stats['ff_gate_prob'] for stats in self.gate_stats['layer_stats']]
        
        # Collect gradients that exist (some might be None)
        all_grads = []
        for param in parameters:
            if param.grad is not None:
                all_grads.append(param.grad)
        
        # Apply gradient clipping
        max_norm = 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        
        # Check that clipping was effective if there were large gradients
        if total_norm > max_norm:
            # Verify gradients are now within bounds
            for grad in all_grads:
                if grad is not None:
                    assert torch.norm(grad) <= max_norm * 1.1  # Small tolerance for numerical precision
        
        # Ensure we have some gradients to test
        assert len(all_grads) > 0, "No gradients found - test setup issue"
    
    def test_loss_components_balance(self):
        """Test that task loss and resource loss are balanced."""
        lambda_values = [0.0, 0.1, 1.0, 10.0]
        
        for lambda_val in lambda_values:
            loss_fn = ResourceAwareLoss(lambda_resource=lambda_val, cost_model='uniform')
            
            # Get individual components
            task_loss = loss_fn.compute_task_loss(self.logits, self.targets)
            resource_metrics = loss_fn.compute_resource_costs(self.gate_stats, hidden_size=64, seq_len=10)
            resource_cost = resource_metrics.total_resource_cost
            total_loss = loss_fn(self.logits, self.targets, self.gate_stats)
            
            # Verify composition
            expected_total = task_loss + lambda_val * resource_cost
            assert torch.allclose(total_loss, expected_total, atol=1e-6)
            
            # Check component magnitudes
            assert task_loss >= 0.0
            assert resource_cost >= 0.0
            assert total_loss >= 0.0
            
            # When lambda=0, should be pure task loss
            if lambda_val == 0.0:
                assert torch.allclose(total_loss, task_loss, atol=1e-6)


class TestRealTimeCostIntegration:
    """Test integration with real-time cost tracking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.batch_size = 4
        self.num_classes = 10
        
        self.logits = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Create loss function with real-time costs enabled
        self.loss_fn = ResourceAwareLoss(
            lambda_resource=0.1,
            cost_model='real_time',
            use_real_time_costs=True
        )
    
    def test_real_time_cost_gradient_flow(self):
        """Test gradient flow with real-time cost tracking."""
        from src.training.cost_tracker import start_batch_tracking, end_batch_tracking, track_layer_costs
        
        # Start cost tracking
        start_batch_tracking(0)
        
        # Simulate layer operations with cost tracking
        with track_layer_costs(0):
            # Simulate some computation
            _ = torch.matmul(self.logits, self.logits.T)
        
        # End tracking to get real costs
        batch_metrics = end_batch_tracking()
        
        # Compute loss with real-time costs
        gate_stats = {
            'layer_stats': [
                {
                    'attention_gate_prob': torch.tensor(0.7, requires_grad=True),
                    'ff_gate_prob': torch.tensor(0.8, requires_grad=True)
                }
            ]
        }
        
        loss = self.loss_fn(self.logits, self.targets, gate_stats)
        loss.backward()
        
        # Check gradients exist and are finite
        assert self.logits.grad is not None
        assert torch.isfinite(self.logits.grad).all()
        assert not torch.isnan(self.logits.grad).any()
        
        # Check gate gradients (may be None for real-time costs since gates aren't used)
        for layer_stats in gate_stats['layer_stats']:
            if layer_stats['attention_gate_prob'].grad is not None:
                assert torch.isfinite(layer_stats['attention_gate_prob'].grad).all()
                assert not torch.isnan(layer_stats['attention_gate_prob'].grad).any()
            if layer_stats['ff_gate_prob'].grad is not None:
                assert torch.isfinite(layer_stats['ff_gate_prob'].grad).all()
                assert not torch.isnan(layer_stats['ff_gate_prob'].grad).any()
    
    def test_cost_model_switching_stability(self):
        """Test stability when switching between cost models."""
        gate_stats = {
            'layer_stats': [
                {
                    'attention_gate_prob': torch.tensor(0.7),
                    'ff_gate_prob': torch.tensor(0.8)
                }
            ]
        }
        
        # Test switching between models
        models = ['uniform', 'layer_weighted', 'activation_size']
        
        for model in models:
            self.loss_fn.set_cost_model(model)
            loss = self.loss_fn(self.logits, self.targets, gate_stats)
            
            assert torch.isfinite(loss)
            assert not torch.isnan(loss)
            assert loss >= 0.0
        
        # Test enabling/disabling real-time costs
        self.loss_fn.enable_real_time_costs(True)
        loss_rt = self.loss_fn(self.logits, self.targets, gate_stats)
        
        self.loss_fn.enable_real_time_costs(False)
        loss_static = self.loss_fn(self.logits, self.targets, gate_stats)
        
        # Both should be finite and non-negative
        assert torch.isfinite(loss_rt) and torch.isfinite(loss_static)
        assert loss_rt >= 0.0 and loss_static >= 0.0 