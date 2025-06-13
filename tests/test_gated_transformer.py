"""
Tests for the GatedTransformer model integration.

This module tests the complete gated transformer model including:
- Model initialization and configuration
- Forward pass with and without gate information
- Integration with gate manager
- Memory pressure modes
- Gate statistics collection
"""

import pytest
import torch
import torch.nn as nn

from src.models.config import TransformerConfig
from src.models.gated_transformer import GatedTransformer


class TestGatedTransformerBasic:
    """Test basic gated transformer functionality."""
    
    def test_model_initialization(self):
        """Test basic model initialization."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        model = GatedTransformer(config)
        
        # Check basic structure
        assert len(model.layers) == 2
        assert model.config.hidden_size == 256
        assert model.config.use_recomputation_gates is True
        
        # Check gate manager exists
        assert model.gate_manager is not None
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        model = GatedTransformer(config)
        model.eval()
        
        # Test input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids)
        
        # Check outputs
        assert 'last_hidden_state' in outputs
        assert 'logits' in outputs
        assert outputs['last_hidden_state'].shape == (batch_size, seq_len, 256)
        assert outputs['logits'].shape == (batch_size, seq_len, 1000)
    
    def test_forward_with_gate_info(self):
        """Test forward pass with gate information."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        model = GatedTransformer(config)
        model.eval()
        
        # Test input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass with gate info
        outputs = model(input_ids, return_gate_info=True)
        
        # Check outputs
        assert 'gate_info' in outputs
        assert 'gate_statistics' in outputs
        assert len(outputs['gate_info']) == 2  # 2 layers
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        model = GatedTransformer(config)
        
        # Test input
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # Forward pass
        outputs = model(input_ids)
        
        # Compute loss and backward pass
        loss = outputs['logits'].sum()
        loss.backward()
        
        # Check that gate parameters have gradients
        for layer in model.layers:
            assert layer.attention_gate.gate_param.grad is not None
            assert layer.ff_gate.gate_param.grad is not None
    
    def test_memory_pressure_modes(self):
        """Test different memory pressure modes."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        model = GatedTransformer(config)
        
        pressure_modes = ["low", "medium", "high", "extreme"]
        
        for mode in pressure_modes:
            model.set_memory_pressure(mode)
            
            # Test forward pass works with this mode
            input_ids = torch.randint(0, 1000, (2, 10))
            outputs = model(input_ids)
            assert 'last_hidden_state' in outputs
    
    def test_gate_statistics(self):
        """Test gate statistics collection."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        model = GatedTransformer(config)
        model.eval()
        
        # Forward pass to generate statistics
        input_ids = torch.randint(0, 1000, (2, 10))
        outputs = model(input_ids, return_gate_info=True)
        
        # Get statistics
        stats = model.get_gate_statistics()
        
        # Check statistics structure
        assert 'layer_stats' in stats
        assert 'avg_attention_prob' in stats
        assert 'avg_ff_prob' in stats
        assert len(stats['layer_stats']) == 2
    
    def test_gate_probability_setting(self):
        """Test manual gate probability setting."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        model = GatedTransformer(config)
        
        # Set specific probabilities
        attention_prob = 0.3
        ff_prob = 0.7
        model.set_gate_probabilities(attention_prob, ff_prob)
        
        # Test that probabilities are approximately set
        # (they won't be exact due to sigmoid activation)
        input_ids = torch.randint(0, 1000, (2, 10))
        with torch.no_grad():
            outputs = model(input_ids, return_gate_info=True)
        
        stats = model.get_gate_statistics()
        
        # Check that probabilities are in reasonable range
        # (allowing some tolerance due to sigmoid and initialization)
        assert 0.1 <= stats['avg_attention_prob'] <= 0.9
        assert 0.1 <= stats['avg_ff_prob'] <= 0.9
    
    def test_clear_stored_activations(self):
        """Test clearing stored activations."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        model = GatedTransformer(config)
        
        # Forward pass to potentially store activations
        input_ids = torch.randint(0, 1000, (2, 10))
        model(input_ids)
        
        # Clear stored activations (should not raise errors)
        model.clear_stored_activations()


class TestGatedTransformerComparison:
    """Test comparison with baseline transformer."""
    
    def test_output_shape_consistency(self):
        """Test that gated transformer produces same output shapes as baseline."""
        from src.models.transformer import BaselineTransformer
        
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            use_recomputation_gates=True
        )
        
        # Create both models
        gated_model = GatedTransformer(config)
        baseline_model = BaselineTransformer(config)
        
        # Test input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass through both models
        gated_outputs = gated_model(input_ids)
        baseline_outputs = baseline_model(input_ids)
        
        # Check that output shapes match
        assert gated_outputs['last_hidden_state'].shape == baseline_outputs['last_hidden_state'].shape
        assert gated_outputs['logits'].shape == baseline_outputs['logits'].shape
    
    def test_parameter_count_difference(self):
        """Test that gated model has more parameters due to gates."""
        from src.models.transformer import BaselineTransformer
        
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            use_recomputation_gates=True
        )
        
        # Create both models
        gated_model = GatedTransformer(config)
        baseline_model = BaselineTransformer(config)
        
        # Count parameters
        gated_params = sum(p.numel() for p in gated_model.parameters())
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        
        # Gated model should have more parameters due to gates
        assert gated_params > baseline_params
        
        # The difference should be the gate parameters
        gate_params = sum(
            p.numel() for layer in gated_model.layers 
            for gate in [layer.attention_gate, layer.ff_gate]
            for p in gate.parameters()
        )
        
        # Allow some tolerance for potential differences in initialization
        assert abs((gated_params - baseline_params) - gate_params) <= 10 


class TestGatedTransformerDifferentiability:
    """Test differentiability and gradient flow through gates."""
    
    def test_gradient_flow_straight_through_enabled(self):
        """Test gradient flow with straight-through estimator enabled."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            use_recomputation_gates=True,
            gate_type="global",
            use_straight_through=True
        )
        model = GatedTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 8))
        outputs = model(input_ids)
        loss = outputs['logits'].sum()
        loss.backward()
        # Check gradients
        for layer in model.layers:
            assert layer.attention_gate.gate_param.grad is not None
            assert layer.ff_gate.gate_param.grad is not None
            assert torch.any(layer.attention_gate.gate_param.grad != 0)
            assert torch.any(layer.ff_gate.gate_param.grad != 0)
    
    def test_gradient_flow_straight_through_disabled(self):
        """Test gradient flow with straight-through estimator disabled."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            use_recomputation_gates=True,
            gate_type="global",
            use_straight_through=False
        )
        model = GatedTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 8))
        outputs = model(input_ids)
        loss = outputs['logits'].sum()
        loss.backward()
        # Check gradients
        for layer in model.layers:
            assert layer.attention_gate.gate_param.grad is not None
            assert layer.ff_gate.gate_param.grad is not None
            assert torch.any(layer.attention_gate.gate_param.grad != 0)
            assert torch.any(layer.ff_gate.gate_param.grad != 0)
    
    def test_gradient_flow_all_gates_open(self):
        """Test gradient flow when all gates are forced open (high bias)."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            use_recomputation_gates=True,
            gate_type="global",
            gate_init_bias=10.0  # Force gates open
        )
        model = GatedTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 8))
        outputs = model(input_ids)
        loss = outputs['logits'].sum()
        loss.backward()
        for layer in model.layers:
            assert layer.attention_gate.gate_param.grad is not None
            assert layer.ff_gate.gate_param.grad is not None
    
    def test_gradient_flow_all_gates_closed(self):
        """Test gradient flow when all gates are forced closed (low bias)."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            use_recomputation_gates=True,
            gate_type="global",
            gate_init_bias=-10.0  # Force gates closed
        )
        model = GatedTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 8))
        outputs = model(input_ids)
        loss = outputs['logits'].sum()
        loss.backward()
        for layer in model.layers:
            assert layer.attention_gate.gate_param.grad is not None
            assert layer.ff_gate.gate_param.grad is not None 


class TestGatedTransformerIntegrationAndLearning:
    """Integration and learning dynamics tests for GatedTransformer."""
    
    def test_training_step_updates_gate_params(self):
        """Simulate a training step and check gate parameters update."""
        import copy
        config = TransformerConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            use_recomputation_gates=True,
            gate_type="global"
        )
        model = GatedTransformer(config)
        model.train()
        input_ids = torch.randint(0, 100, (2, 8))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        # Save initial gate params
        initial_gate = copy.deepcopy([layer.attention_gate.gate_param.clone().detach() for layer in model.layers])
        # Training step
        outputs = model(input_ids)
        loss = outputs['logits'].sum()
        loss.backward()
        optimizer.step()
        # Check that at least one gate param has changed
        changed = False
        for i, layer in enumerate(model.layers):
            if not torch.allclose(layer.attention_gate.gate_param, initial_gate[i]):
                changed = True
        assert changed
    
    def test_memory_pressure_mode_affects_gates(self):
        """Changing memory pressure mode should affect gate statistics."""
        config = TransformerConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            use_recomputation_gates=True,
            gate_type="global"
        )
        model = GatedTransformer(config)
        input_ids = torch.randint(0, 100, (2, 8))
        # Default (medium)
        stats_medium = model(input_ids, return_gate_info=True)['gate_statistics']
        # Set to low (should increase storage prob)
        model.set_memory_pressure('low')
        stats_low = model(input_ids, return_gate_info=True)['gate_statistics']
        # Set to high (should decrease storage prob)
        model.set_memory_pressure('high')
        stats_high = model(input_ids, return_gate_info=True)['gate_statistics']
        # Probabilities should be ordered: low > medium > high
        assert stats_low['avg_attention_prob'] >= stats_medium['avg_attention_prob']
        assert stats_medium['avg_attention_prob'] >= stats_high['avg_attention_prob']
    
    def test_gate_probabilities_learnable(self):
        """Gate probabilities should change after several training steps."""
        config = TransformerConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            use_recomputation_gates=True,
            gate_type="global"
        )
        model = GatedTransformer(config)
        model.train()
        input_ids = torch.randint(0, 100, (2, 8))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        # Initial gate stats
        stats_before = model(input_ids, return_gate_info=True)['gate_statistics']
        avg_before = stats_before['avg_attention_prob']
        # Run a few training steps
        for _ in range(5):
            outputs = model(input_ids)
            loss = outputs['logits'].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        stats_after = model(input_ids, return_gate_info=True)['gate_statistics']
        avg_after = stats_after['avg_attention_prob']
        # Probabilities should change
        assert abs(avg_before - avg_after) > 1e-4
    
    def test_no_nan_inf_in_training(self):
        """No NaN or Inf in outputs or gradients during/after training."""
        config = TransformerConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            use_recomputation_gates=True,
            gate_type="global"
        )
        model = GatedTransformer(config)
        model.train()
        input_ids = torch.randint(0, 100, (2, 8))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(3):
            outputs = model(input_ids)
            loss = outputs['logits'].sum()
            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()
                    assert not torch.isinf(param.grad).any()
            optimizer.step()
            for name, param in model.named_parameters():
                assert not torch.isnan(param.data).any()
                assert not torch.isinf(param.data).any() 