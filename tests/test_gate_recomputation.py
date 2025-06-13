"""
Tests for gate-triggered recomputation logic.

This module tests the integration between gates and recomputation hooks,
ensuring that storage/recomputation decisions are made correctly based
on gate values.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.models.config import TransformerConfig, get_tiny_config
from src.models.gate_recomputation import (
    GateTriggeredRecomputation,
    GatedRecomputationLayer,
    gate_recomputation_context,
    create_gated_recomputation_model
)
from src.models.recomputation_hooks import RecomputationStrategy


class TestGateTriggeredRecomputation:
    """Test the GateTriggeredRecomputation class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.gate_recomputation = GateTriggeredRecomputation(
            strategy=RecomputationStrategy.CUSTOM,
            storage_threshold=0.5,
            memory_threshold_mb=100.0,
            enable_profiling=True
        )
    
    def test_initialization(self):
        """Test proper initialization of GateTriggeredRecomputation."""
        assert self.gate_recomputation.strategy == RecomputationStrategy.CUSTOM
        assert self.gate_recomputation.storage_threshold == 0.5
        assert self.gate_recomputation.memory_threshold_mb == 100.0
        assert self.gate_recomputation.enable_profiling is True
        
        # Check that components are initialized
        assert self.gate_recomputation.hook_manager is not None
        assert self.gate_recomputation.checkpoint_hook is not None
        assert isinstance(self.gate_recomputation.gate_decisions, dict)
        assert isinstance(self.gate_recomputation.activation_cache, dict)
        assert isinstance(self.gate_recomputation.stats, dict)
    
    def test_storage_decision_above_threshold(self):
        """Test storage decision when gate value is above threshold."""
        # Create gate values above threshold
        gate_values = torch.tensor([[0.8, 0.7, 0.9]])  # Mean = 0.8 > 0.5
        gate_decisions = torch.tensor([[1.0, 1.0, 1.0]])
        
        should_store = self.gate_recomputation.make_storage_decision(
            gate_values, gate_decisions, "test_module", 0, "attention"
        )
        
        assert should_store is True
        assert "test_module" in self.gate_recomputation.gate_decisions
        torch.testing.assert_close(
            self.gate_recomputation.gate_decisions["test_module"],
            gate_decisions
        )
    
    def test_storage_decision_below_threshold(self):
        """Test storage decision when gate value is below threshold."""
        # Create gate values below threshold
        gate_values = torch.tensor([[0.2, 0.3, 0.1]])  # Mean = 0.2 < 0.5
        gate_decisions = torch.tensor([[0.0, 0.0, 0.0]])
        
        should_store = self.gate_recomputation.make_storage_decision(
            gate_values, gate_decisions, "test_module", 0, "attention"
        )
        
        assert should_store is False
        assert "test_module" in self.gate_recomputation.gate_decisions
    
    def test_store_activation(self):
        """Test activation storage functionality."""
        # Create test activation
        activation = torch.randn(2, 4, 8)
        
        # Store activation
        self.gate_recomputation.store_activation(
            activation, "test_module", 0, "attention"
        )
        
        # Check that activation is stored
        assert "test_module" in self.gate_recomputation.activation_cache
        stored_activation = self.gate_recomputation.activation_cache["test_module"]
        torch.testing.assert_close(stored_activation, activation)
        
        # Check statistics
        assert self.gate_recomputation.stats['total_activations'] == 1
        assert self.gate_recomputation.stats['stored_activations'] == 1
    
    def test_prepare_recomputation(self):
        """Test recomputation preparation."""
        # Create test activation and recomputation function
        activation = torch.randn(2, 4, 8)
        
        def mock_recompute_fn():
            return activation * 2
        
        # Prepare recomputation
        self.gate_recomputation.prepare_recomputation(
            mock_recompute_fn, "test_module", 0, "attention", activation
        )
        
        # Check that recomputation info is stored
        assert "test_module" in self.gate_recomputation.hook_manager.stored_activations
        activation_info = self.gate_recomputation.hook_manager.stored_activations["test_module"]
        
        assert activation_info.tensor is None  # Not stored
        assert activation_info.recompute_fn is not None
        assert activation_info.is_stored is False
        assert activation_info.layer_idx == 0
        assert activation_info.module_type == "attention"
        
        # Check statistics
        assert self.gate_recomputation.stats['total_activations'] == 1
        assert self.gate_recomputation.stats['recomputed_activations'] == 1
        assert self.gate_recomputation.stats['memory_saved_mb'] > 0
    
    def test_get_statistics(self):
        """Test statistics collection."""
        # Add some test data
        activation = torch.randn(2, 4, 8)
        self.gate_recomputation.store_activation(activation, "stored_module", 0, "attention")
        
        def mock_recompute_fn():
            return activation
        
        self.gate_recomputation.prepare_recomputation(
            mock_recompute_fn, "recompute_module", 1, "feedforward", activation
        )
        
        # Get statistics
        stats = self.gate_recomputation.get_statistics()
        
        # Check basic statistics
        assert stats['total_activations'] == 2
        assert stats['stored_activations'] == 1
        assert stats['recomputed_activations'] == 1
        assert stats['storage_rate'] == 0.5
        assert stats['recomputation_rate'] == 0.5
        
        # Check that hook manager stats are included
        assert 'memory_used_mb' in stats
        assert 'memory_saved_mb' in stats
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Add some test data
        activation = torch.randn(2, 4, 8)
        self.gate_recomputation.store_activation(activation, "test_module", 0, "attention")
        
        # Clear cache
        self.gate_recomputation.clear_cache()
        
        # Check that caches are cleared
        assert len(self.gate_recomputation.activation_cache) == 0
        assert len(self.gate_recomputation.gate_decisions) == 0
        assert len(self.gate_recomputation.hook_manager.stored_activations) == 0
        
        # Check that statistics are reset
        assert self.gate_recomputation.stats['total_activations'] == 0
        assert self.gate_recomputation.stats['stored_activations'] == 0
        assert self.gate_recomputation.stats['recomputed_activations'] == 0


class TestGatedRecomputationLayer:
    """Test the GatedRecomputationLayer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = get_tiny_config()
        self.layer = GatedRecomputationLayer(
            config=self.config,
            layer_idx=0,
            recomputation_strategy=RecomputationStrategy.CUSTOM,
            storage_threshold=0.5
        )
    
    def test_initialization(self):
        """Test proper initialization of GatedRecomputationLayer."""
        assert self.layer.config == self.config
        assert self.layer.layer_idx == 0
        assert self.layer.storage_threshold == 0.5
        
        # Check that components are initialized
        assert self.layer.gated_layer is not None
        assert self.layer.gate_recomputation is not None
    
    def test_forward_pass_basic(self):
        """Test basic forward pass functionality."""
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # Forward pass
        output = self.layer(hidden_states)
        
        # Check output shape
        assert output.shape == hidden_states.shape
        assert output.dtype == hidden_states.dtype
    
    def test_forward_pass_with_attention_weights(self):
        """Test forward pass with attention weights."""
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # Forward pass with attention weights
        output, attention_weights = self.layer(
            hidden_states, return_attention_weights=True
        )
        
        # Check output shape
        assert output.shape == hidden_states.shape
        assert attention_weights is not None
        # Attention weights shape: (batch_size, num_heads, seq_len, seq_len)
        expected_attn_shape = (batch_size, self.config.num_attention_heads, seq_len, seq_len)
        assert attention_weights.shape == expected_attn_shape
    
    def test_forward_pass_with_gate_info(self):
        """Test forward pass with gate information."""
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # Forward pass with gate info
        output, gate_info = self.layer(
            hidden_states, return_gate_info=True
        )
        
        # Check output shape
        assert output.shape == hidden_states.shape
        
        # Check gate info
        assert isinstance(gate_info, dict)
        assert 'attention_gate_prob' in gate_info
        assert 'layer_idx' in gate_info
        assert 'total_activations' in gate_info  # From recomputation stats
    
    def test_forward_pass_with_both_returns(self):
        """Test forward pass with both attention weights and gate info."""
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # Forward pass with both returns
        output, attention_weights, gate_info = self.layer(
            hidden_states, 
            return_attention_weights=True,
            return_gate_info=True
        )
        
        # Check all outputs
        assert output.shape == hidden_states.shape
        assert attention_weights is not None
        assert isinstance(gate_info, dict)
    
    def test_gated_attention_with_recomputation(self):
        """Test the gated attention with recomputation logic."""
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # Test with different gate probabilities
        # Set high storage probability
        self.layer.gated_layer.attention_gate.set_storage_probability(0.8)
        
        output1 = self.layer._gated_attention_with_recomputation(hidden_states)
        
        # Set low storage probability
        self.layer.gated_layer.attention_gate.set_storage_probability(0.2)
        
        output2 = self.layer._gated_attention_with_recomputation(hidden_states)
        
        # Both should produce valid outputs
        assert output1.shape == hidden_states.shape
        assert output2.shape == hidden_states.shape
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # Run forward pass to populate caches
        self.layer(hidden_states)
        
        # Clear cache
        self.layer.clear_cache()
        
        # Check that recomputation cache is cleared
        stats = self.layer.get_recomputation_statistics()
        assert stats['total_activations'] == 0
    
    def test_get_recomputation_statistics(self):
        """Test recomputation statistics collection."""
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # Run forward pass
        self.layer(hidden_states)
        
        # Get statistics
        stats = self.layer.get_recomputation_statistics()
        
        # Check that statistics are returned
        assert isinstance(stats, dict)
        assert 'total_activations' in stats
        assert 'storage_rate' in stats
        assert 'recomputation_rate' in stats


class TestGateRecomputationContext:
    """Test the gate recomputation context manager."""
    
    def test_context_manager_with_recomputation(self):
        """Test context manager with recomputation enabled."""
        config = get_tiny_config()
        layers = [
            GatedRecomputationLayer(config, layer_idx=i)
            for i in range(2)
        ]
        
        with gate_recomputation_context(layers, enable_recomputation=True) as ctx_layers:
            assert ctx_layers == layers
            
            # Run some operations
            hidden_states = torch.randn(2, 4, config.hidden_size)
            for layer in ctx_layers:
                hidden_states = layer(hidden_states)
        
        # After context, caches should be cleared
        for layer in layers:
            stats = layer.get_recomputation_statistics()
            assert stats['total_activations'] == 0
    
    def test_context_manager_without_recomputation(self):
        """Test context manager with recomputation disabled."""
        config = get_tiny_config()
        layers = [
            GatedRecomputationLayer(config, layer_idx=i)
            for i in range(2)
        ]
        
        with gate_recomputation_context(layers, enable_recomputation=False) as ctx_layers:
            assert ctx_layers == layers


class TestCreateGatedRecomputationModel:
    """Test the model creation utility function."""
    
    def test_create_gated_recomputation_model(self):
        """Test creating a gated recomputation model."""
        config = get_tiny_config()
        num_layers = 3
        
        model = create_gated_recomputation_model(
            config=config,
            num_layers=num_layers,
            recomputation_strategy=RecomputationStrategy.CUSTOM,
            storage_threshold=0.6
        )
        
        # Check model structure
        assert isinstance(model, nn.ModuleList)
        assert len(model) == num_layers
        
        # Check each layer
        for i, layer in enumerate(model):
            assert isinstance(layer, GatedRecomputationLayer)
            assert layer.layer_idx == i
            assert layer.storage_threshold == 0.6
            assert layer.config == config
    
    def test_model_forward_pass(self):
        """Test forward pass through the created model."""
        config = get_tiny_config()
        num_layers = 2
        
        model = create_gated_recomputation_model(
            config=config,
            num_layers=num_layers
        )
        
        # Test forward pass
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Pass through all layers
        for layer in model:
            hidden_states = layer(hidden_states)
        
        # Check final output
        assert hidden_states.shape == (batch_size, seq_len, config.hidden_size)


class TestIntegrationWithExistingGates:
    """Test integration with existing gate system."""
    
    def test_gate_decision_integration(self):
        """Test that gate decisions properly influence recomputation."""
        config = get_tiny_config()
        layer = GatedRecomputationLayer(config, layer_idx=0, storage_threshold=0.5)
        
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Test with high storage probability
        layer.gated_layer.attention_gate.set_storage_probability(0.9)
        output1, gate_info1 = layer(hidden_states, return_gate_info=True)
        
        # Test with low storage probability
        layer.gated_layer.attention_gate.set_storage_probability(0.1)
        layer.clear_cache()  # Clear previous cache
        output2, gate_info2 = layer(hidden_states, return_gate_info=True)
        
        # Both should produce valid outputs
        assert output1.shape == output2.shape
        
        # Gate probabilities should be different
        assert gate_info1['attention_gate_prob'] > gate_info2['attention_gate_prob']
    
    def test_memory_efficiency_tracking(self):
        """Test that memory efficiency is properly tracked."""
        config = get_tiny_config()
        layer = GatedRecomputationLayer(config, layer_idx=0, storage_threshold=0.5)
        
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Set low storage probability to trigger recomputation
        layer.gated_layer.attention_gate.set_storage_probability(0.1)
        
        # Run forward pass
        output, gate_info = layer(hidden_states, return_gate_info=True)
        
        # Check that memory savings are tracked
        assert 'memory_saved_mb' in gate_info
        assert gate_info['memory_saved_mb'] >= 0
        assert 'recomputation_rate' in gate_info