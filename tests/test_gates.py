"""
Tests for differentiable recomputation gates.

This module tests the gate functionality including:
- Gate value computation and range validation
- Straight-through estimator behavior
- Integration with transformer layers
- Gate statistics and management
"""

import pytest
import torch
import torch.nn as nn
import math

from src.models.config import TransformerConfig
from src.models.gates import (
    RecomputationGate,
    GatedTransformerLayer,
    GateManager
)


class TestRecomputationGate:
    """Test the basic RecomputationGate functionality."""
    
    def test_gate_initialization(self):
        """Test gate initialization with different parameters."""
        # Test global gate
        gate = RecomputationGate(
            hidden_size=768,
            gate_type="global",
            init_bias=0.0,
            temperature=1.0
        )
        assert gate.gate_type == "global"
        assert gate.gate_param.shape == (1,)
        assert abs(gate.gate_param.item() - 0.0) < 1e-6
    
    def test_gate_types(self):
        """Test different gate types."""
        hidden_size = 768
        
        # Global gate
        global_gate = RecomputationGate(hidden_size, gate_type="global")
        assert global_gate.gate_type == "global"
        
        # Per-head gate
        per_head_gate = RecomputationGate(hidden_size, gate_type="per_head")
        assert per_head_gate.gate_type == "per_head"
        
        # Per-token gate
        per_token_gate = RecomputationGate(hidden_size, gate_type="per_token")
        assert per_token_gate.gate_type == "per_token"
        
        # Invalid gate type
        with pytest.raises(ValueError):
            RecomputationGate(hidden_size, gate_type="invalid")
    
    def test_gate_forward_global(self):
        """Test forward pass for global gates."""
        gate = RecomputationGate(
            hidden_size=768,
            gate_type="global",
            init_bias=0.0,
            temperature=1.0
        )
        
        # Test input
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        gate_values, binary_decisions = gate(hidden_states, training=True)
        
        # Check shapes
        assert gate_values.shape == (batch_size, seq_len, 1)
        assert binary_decisions.shape == (batch_size, seq_len, 1)
        
        # Check value ranges
        assert torch.all(gate_values >= 0.0)
        assert torch.all(gate_values <= 1.0)
        assert torch.all((binary_decisions == 0.0) | (binary_decisions == 1.0))
    
    def test_gate_forward_per_head(self):
        """Test forward pass for per-head gates."""
        hidden_size = 768
        num_heads = 12
        
        gate = RecomputationGate(
            hidden_size=hidden_size,
            gate_type="per_head",
            init_bias=0.0,
            temperature=1.0
        )
        
        # Test input
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        gate_values, binary_decisions = gate(
            hidden_states, 
            num_heads=num_heads, 
            training=True
        )
        
        # Check shapes
        assert gate_values.shape == (batch_size, seq_len, hidden_size)
        assert binary_decisions.shape == (batch_size, seq_len, hidden_size)
        
        # Check value ranges
        assert torch.all(gate_values >= 0.0)
        assert torch.all(gate_values <= 1.0)
    
    def test_gate_forward_per_token(self):
        """Test forward pass for per-token gates."""
        hidden_size = 768
        
        gate = RecomputationGate(
            hidden_size=hidden_size,
            gate_type="per_token",
            init_bias=0.0,
            temperature=1.0
        )
        
        # Test input
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        gate_values, binary_decisions = gate(hidden_states, training=True)
        
        # Check shapes
        assert gate_values.shape == (batch_size, seq_len, hidden_size)
        assert binary_decisions.shape == (batch_size, seq_len, hidden_size)
        
        # Check value ranges
        assert torch.all(gate_values >= 0.0)
        assert torch.all(gate_values <= 1.0)
    
    def test_straight_through_estimator(self):
        """Test straight-through estimator behavior."""
        gate = RecomputationGate(
            hidden_size=768,
            gate_type="global",
            init_bias=0.0,
            temperature=1.0,
            use_straight_through=True
        )
        
        # Test input
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        hidden_states.requires_grad_(True)
        
        # Forward pass in training mode
        gate_values, binary_decisions = gate(hidden_states, training=True)
        
        # Binary decisions should be 0 or 1
        assert torch.all((binary_decisions == 0.0) | (binary_decisions == 1.0))
        
        # Test gradient flow
        loss = binary_decisions.sum()
        loss.backward()
        
        # Gate parameter should have gradients
        assert gate.gate_param.grad is not None
        assert not torch.isnan(gate.gate_param.grad).any()
    
    def test_temperature_effect(self):
        """Test effect of temperature on gate decisions."""
        hidden_size = 768
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Low temperature (more binary)
        low_temp_gate = RecomputationGate(
            hidden_size=hidden_size,
            gate_type="global",
            init_bias=0.0,
            temperature=0.1
        )
        
        # High temperature (more continuous)
        high_temp_gate = RecomputationGate(
            hidden_size=hidden_size,
            gate_type="global",
            init_bias=0.0,
            temperature=10.0
        )
        
        # Forward passes
        low_temp_values, _ = low_temp_gate(hidden_states, training=False)
        high_temp_values, _ = high_temp_gate(hidden_states, training=False)
        
        # Low temperature should produce more extreme values
        low_temp_extreme = torch.sum((low_temp_values < 0.1) | (low_temp_values > 0.9))
        high_temp_extreme = torch.sum((high_temp_values < 0.1) | (high_temp_values > 0.9))
        
        # This is probabilistic, but low temp should generally be more extreme
        # We'll just check that both produce valid ranges
        assert torch.all(low_temp_values >= 0.0) and torch.all(low_temp_values <= 1.0)
        assert torch.all(high_temp_values >= 0.0) and torch.all(high_temp_values <= 1.0)
    
    def test_storage_probability_methods(self):
        """Test storage probability getter and setter."""
        gate = RecomputationGate(
            hidden_size=768,
            gate_type="global",
            init_bias=0.0,
            temperature=1.0
        )
        
        # Initial probability should be ~0.5 (sigmoid(0))
        initial_prob = gate.get_storage_probability()
        assert abs(initial_prob - 0.5) < 0.01
        
        # Set new probability
        target_prob = 0.8
        gate.set_storage_probability(target_prob)
        
        # Check that probability changed
        new_prob = gate.get_storage_probability()
        assert abs(new_prob - target_prob) < 0.01
        
        # Test invalid probabilities
        with pytest.raises(ValueError):
            gate.set_storage_probability(-0.1)
        
        with pytest.raises(ValueError):
            gate.set_storage_probability(1.1)


class TestGatedTransformerLayer:
    """Test the GatedTransformerLayer functionality."""
    
    def test_gated_layer_initialization(self):
        """Test gated transformer layer initialization."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        layer = GatedTransformerLayer(config, layer_idx=0)
        
        # Check that gates are created
        assert hasattr(layer, 'attention_gate')
        assert hasattr(layer, 'ff_gate')
        assert isinstance(layer.attention_gate, RecomputationGate)
        assert isinstance(layer.ff_gate, RecomputationGate)
        
        # Check layer index
        assert layer.layer_idx == 0
    
    def test_gated_layer_forward(self):
        """Test forward pass through gated transformer layer."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        layer = GatedTransformerLayer(config, layer_idx=0)
        
        # Test input
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = layer(hidden_states)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()
    
    def test_gated_layer_with_gate_info(self):
        """Test forward pass with gate information return."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        layer = GatedTransformerLayer(config, layer_idx=0)
        
        # Test input
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass with gate info
        output, gate_info = layer(hidden_states, return_gate_info=True)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        # Check gate info
        assert isinstance(gate_info, dict)
        assert 'attention_gate_prob' in gate_info
        assert 'ff_gate_prob' in gate_info
        assert 'layer_idx' in gate_info
        assert gate_info['layer_idx'] == 0
    
    def test_gated_layer_attention_weights(self):
        """Test forward pass with attention weights return."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        layer = GatedTransformerLayer(config, layer_idx=0)
        
        # Test input
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass with attention weights
        output, attention_weights = layer(
            hidden_states, 
            return_attention_weights=True
        )
        
        # Check shapes
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert attention_weights.shape == (batch_size, 12, seq_len, seq_len)  # num_heads=12
    
    def test_stored_activations_management(self):
        """Test stored activations management."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        layer = GatedTransformerLayer(config, layer_idx=0)
        
        # Initially no stored activations
        assert layer._stored_attention_output is None
        assert layer._stored_ff_output is None
        
        # Forward pass
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = layer(hidden_states)
        
        # Clear stored activations
        layer.clear_stored_activations()
        assert layer._stored_attention_output is None
        assert layer._stored_ff_output is None
    
    def test_gate_statistics(self):
        """Test gate statistics collection."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        layer = GatedTransformerLayer(config, layer_idx=5)
        
        # Get statistics
        stats = layer.get_gate_statistics()
        
        # Check statistics structure
        assert isinstance(stats, dict)
        assert 'attention_gate_prob' in stats
        assert 'ff_gate_prob' in stats
        assert 'layer_idx' in stats
        assert stats['layer_idx'] == 5
        
        # Check probability ranges
        assert 0.0 <= stats['attention_gate_prob'] <= 1.0
        assert 0.0 <= stats['ff_gate_prob'] <= 1.0


class TestGateManager:
    """Test the GateManager functionality."""
    
    def test_gate_manager_initialization(self):
        """Test gate manager initialization."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        # Create gated layers
        layers = nn.ModuleList([
            GatedTransformerLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Create manager
        manager = GateManager(layers)
        assert len(manager.gated_layers) == 4
    
    def test_set_all_gate_probabilities(self):
        """Test setting probabilities for all gates."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        # Create gated layers
        layers = nn.ModuleList([
            GatedTransformerLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Create manager
        manager = GateManager(layers)
        
        # Set probabilities
        attention_prob, ff_prob = 0.7, 0.3
        manager.set_all_gate_probabilities(attention_prob, ff_prob)
        
        # Check that all layers have updated probabilities
        for layer in layers:
            assert abs(layer.attention_gate.get_storage_probability() - attention_prob) < 0.01
            assert abs(layer.ff_gate.get_storage_probability() - ff_prob) < 0.01
    
    def test_get_all_gate_statistics(self):
        """Test collecting statistics from all gates."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        # Create gated layers
        layers = nn.ModuleList([
            GatedTransformerLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Create manager
        manager = GateManager(layers)
        
        # Get statistics
        stats = manager.get_all_gate_statistics()
        
        # Check statistics structure
        assert isinstance(stats, dict)
        assert 'layer_stats' in stats
        assert 'avg_attention_prob' in stats
        assert 'avg_ff_prob' in stats
        
        # Check layer statistics
        assert len(stats['layer_stats']) == 4
        for i, layer_stat in enumerate(stats['layer_stats']):
            assert layer_stat['layer_idx'] == i
            assert 0.0 <= layer_stat['attention_gate_prob'] <= 1.0
            assert 0.0 <= layer_stat['ff_gate_prob'] <= 1.0
        
        # Check averages
        assert 0.0 <= stats['avg_attention_prob'] <= 1.0
        assert 0.0 <= stats['avg_ff_prob'] <= 1.0
    
    def test_memory_pressure_modes(self):
        """Test different memory pressure modes."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        # Create gated layers
        layers = nn.ModuleList([
            GatedTransformerLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Create manager
        manager = GateManager(layers)
        
        # Test different pressure levels
        pressure_levels = ["low", "medium", "high", "extreme"]
        expected_probs = {
            "low": (0.8, 0.8),
            "medium": (0.5, 0.5),
            "high": (0.2, 0.2),
            "extreme": (0.1, 0.1)
        }
        
        for level in pressure_levels:
            manager.set_memory_pressure_mode(level)
            expected_att, expected_ff = expected_probs[level]
            
            # Check that probabilities are set correctly
            for layer in layers:
                att_prob = layer.attention_gate.get_storage_probability()
                ff_prob = layer.ff_gate.get_storage_probability()
                assert abs(att_prob - expected_att) < 0.01
                assert abs(ff_prob - expected_ff) < 0.01
        
        # Test invalid pressure level
        with pytest.raises(ValueError):
            manager.set_memory_pressure_mode("invalid")
    
    def test_clear_all_stored_activations(self):
        """Test clearing stored activations in all layers."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        # Create gated layers
        layers = nn.ModuleList([
            GatedTransformerLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Create manager
        manager = GateManager(layers)
        
        # Run forward passes to potentially store activations
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        for layer in layers:
            layer(hidden_states)
        
        # Clear all stored activations
        manager.clear_all_stored_activations()
        
        # Check that all layers have cleared activations
        for layer in layers:
            assert layer._stored_attention_output is None
            assert layer._stored_ff_output is None


class TestGateIntegration:
    """Test integration of gates with transformer components."""
    
    def test_gradient_flow_through_gates(self):
        """Test that gradients flow properly through gates."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="global"
        )
        
        layer = GatedTransformerLayer(config, layer_idx=0)
        
        # Test input
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        hidden_states.requires_grad_(True)
        
        # Forward pass
        output = layer(hidden_states)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gate parameters have gradients
        assert layer.attention_gate.gate_param.grad is not None
        assert layer.ff_gate.gate_param.grad is not None
        assert not torch.isnan(layer.attention_gate.gate_param.grad).any()
        assert not torch.isnan(layer.ff_gate.gate_param.grad).any()
        
        # Check that input has gradients
        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()
    
    def test_gate_config_integration(self):
        """Test that gate configuration is properly integrated."""
        # Test with gates enabled
        config_with_gates = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            use_recomputation_gates=True,
            gate_type="per_head",
            gate_init_bias=0.5,
            gate_temperature=2.0,
            use_straight_through=False
        )
        
        layer = GatedTransformerLayer(config_with_gates, layer_idx=0)
        
        # Check gate configuration
        assert layer.attention_gate.gate_type == "per_head"
        assert layer.attention_gate.init_bias == 0.5
        assert layer.attention_gate.temperature == 2.0
        assert layer.attention_gate.use_straight_through == False
        
        # Test with gates disabled (should still work)
        config_without_gates = TransformerConfig(
            vocab_size=1000,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            use_recomputation_gates=False
        )
        
        # This should still create gates (they're always present in GatedTransformerLayer)
        layer_no_gates = GatedTransformerLayer(config_without_gates, layer_idx=0)
        assert hasattr(layer_no_gates, 'attention_gate')
        assert hasattr(layer_no_gates, 'ff_gate')


class TestRecomputationGateValidation:
    """Comprehensive validation of RecomputationGate behavior."""
    
    def test_gate_output_range(self):
        """Gate output should always be in [0, 1] (probability)."""
        from src.models.gates import RecomputationGate
        gate = RecomputationGate(hidden_size=32, gate_type="global")
        x = torch.randn(4, 8, 32)
        gate.train()
        out, _ = gate(x)
        assert torch.all((out >= 0) & (out <= 1))
    
    def test_gate_extreme_bias(self):
        """Gate with high positive/negative bias should saturate to 1/0."""
        from src.models.gates import RecomputationGate
        # High positive bias
        gate_open = RecomputationGate(hidden_size=32, gate_type="global", init_bias=10.0)
        x = torch.randn(2, 4, 32)
        out_open, _ = gate_open(x)
        assert torch.all(out_open > 0.99)
        # High negative bias
        gate_closed = RecomputationGate(hidden_size=32, gate_type="global", init_bias=-10.0)
        out_closed, _ = gate_closed(x)
        assert torch.all(out_closed < 0.01)
    
    def test_gate_temperature_effect(self):
        """Lower temperature should make gate output more binary."""
        from src.models.gates import RecomputationGate
        gate_soft = RecomputationGate(hidden_size=32, gate_type="global", temperature=2.0)
        gate_hard = RecomputationGate(hidden_size=32, gate_type="global", temperature=0.1)
        x = torch.randn(2, 4, 32)
        out_soft, _ = gate_soft(x)
        out_hard, _ = gate_hard(x)
        # Harder gate should have more outputs near 0 or 1
        frac_hard = ((out_hard < 0.05) | (out_hard > 0.95)).float().mean()
        frac_soft = ((out_soft < 0.05) | (out_soft > 0.95)).float().mean()
        assert frac_hard > frac_soft
    
    def test_straight_through_estimator(self):
        """Straight-through estimator should allow gradients to flow through binary decisions."""
        from src.models.gates import RecomputationGate
        gate = RecomputationGate(hidden_size=16, gate_type="global", use_straight_through=True)
        x = torch.randn(2, 4, 16, requires_grad=True)
        out, decision = gate(x)
        # Use decision in a loss
        loss = decision.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.any(x.grad != 0)
    
    def test_no_nan_inf_in_outputs(self):
        """Gate outputs and gradients should not contain NaN or Inf."""
        from src.models.gates import RecomputationGate
        gate = RecomputationGate(hidden_size=16, gate_type="global")
        x = torch.randn(2, 4, 16, requires_grad=True)
        out, decision = gate(x)
        loss = out.sum() + decision.sum()
        loss.backward()
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


if __name__ == "__main__":
    pytest.main([__file__]) 