"""
Tests for recomputation hook architecture.

This module tests the hook system for intercepting and managing
activation recomputation based on gate decisions.
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.models.recomputation_hooks import (
    RecomputationHookManager,
    CheckpointRecomputationHook,
    RecomputationStrategy,
    ActivationInfo,
    recomputation_context,
    create_recomputation_wrapper
)
from src.models.config import TransformerConfig
from src.models.gates import RecomputationGate


class TestRecomputationHookManager:
    """Test the main hook manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.hook_manager = RecomputationHookManager(
            strategy=RecomputationStrategy.CUSTOM,
            memory_threshold_mb=100.0,
            enable_profiling=True
        )
        
        # Create a simple test module
        self.test_module = nn.Linear(64, 64)
        
        # Create a mock gate function
        def mock_gate_fn(hidden_states):
            batch_size, seq_len, hidden_size = hidden_states.shape
            gate_values = torch.ones(batch_size, seq_len, 1) * 0.7  # 70% storage probability
            gate_decisions = (gate_values > 0.5).float()
            return gate_values, gate_decisions
        
        self.mock_gate_fn = mock_gate_fn
    
    def teardown_method(self):
        """Clean up after tests."""
        self.hook_manager.cleanup()
    
    def test_hook_manager_initialization(self):
        """Test hook manager initialization."""
        assert self.hook_manager.strategy == RecomputationStrategy.CUSTOM
        assert self.hook_manager.memory_threshold_mb == 100.0
        assert self.hook_manager.enable_profiling is True
        assert len(self.hook_manager.stored_activations) == 0
        assert len(self.hook_manager.forward_hooks) == 0
        assert len(self.hook_manager.backward_hooks) == 0
    
    def test_module_registration(self):
        """Test registering a module for hooks."""
        module_name = "test_linear"
        layer_idx = 0
        module_type = "feedforward"
        
        self.hook_manager.register_module(
            self.test_module,
            module_name,
            layer_idx,
            module_type,
            self.mock_gate_fn
        )
        
        # Check that hooks were registered
        assert len(self.hook_manager.forward_hooks) == 1
        assert len(self.hook_manager.backward_hooks) == 1
    
    def test_forward_hook_storage_decision(self):
        """Test forward hook makes correct storage decisions."""
        module_name = "test_linear"
        layer_idx = 0
        module_type = "feedforward"
        
        # Register module
        self.hook_manager.register_module(
            self.test_module,
            module_name,
            layer_idx,
            module_type,
            self.mock_gate_fn
        )
        
        # Create test input
        batch_size, seq_len, hidden_size = 2, 10, 64
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass (this will trigger the hook)
        output = self.test_module(input_tensor)
        
        # Check that activation info was stored
        assert module_name in self.hook_manager.stored_activations
        activation_info = self.hook_manager.stored_activations[module_name]
        assert activation_info.layer_idx == layer_idx
        assert activation_info.module_type == module_type
        assert activation_info.is_stored is True  # 70% probability should result in storage
    
    def test_forward_hook_recomputation_decision(self):
        """Test forward hook decides to recompute when gate probability is low."""
        # Create a gate function that returns low probability
        def low_prob_gate_fn(hidden_states):
            batch_size, seq_len, hidden_size = hidden_states.shape
            gate_values = torch.ones(batch_size, seq_len, 1) * 0.3  # 30% storage probability
            gate_decisions = (gate_values > 0.5).float()
            return gate_values, gate_decisions
        
        module_name = "test_linear"
        layer_idx = 0
        module_type = "feedforward"
        
        # Register module with low probability gate
        self.hook_manager.register_module(
            self.test_module,
            module_name,
            layer_idx,
            module_type,
            low_prob_gate_fn
        )
        
        # Create test input
        batch_size, seq_len, hidden_size = 2, 10, 64
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = self.test_module(input_tensor)
        
        # Check that recomputation was chosen
        assert module_name in self.hook_manager.stored_activations
        activation_info = self.hook_manager.stored_activations[module_name]
        assert activation_info.is_stored is False
        assert activation_info.recompute_fn is not None
        assert activation_info.memory_saved_mb > 0
    
    def test_memory_statistics(self):
        """Test memory statistics tracking."""
        module_name = "test_linear"
        layer_idx = 0
        module_type = "feedforward"
        
        # Register module
        self.hook_manager.register_module(
            self.test_module,
            module_name,
            layer_idx,
            module_type,
            self.mock_gate_fn
        )
        
        # Forward pass
        input_tensor = torch.randn(2, 10, 64)
        output = self.test_module(input_tensor)
        
        # Check memory stats
        stats = self.hook_manager.get_memory_stats()
        assert 'memory_used_mb' in stats
        assert 'memory_saved_mb' in stats
        assert 'total_activations' in stats
        assert stats['total_activations'] == 1
        assert stats['stored_activations'] == 1
        assert stats['recompute_activations'] == 0
    
    def test_profiling_statistics(self):
        """Test profiling statistics collection."""
        stats = self.hook_manager.get_profiling_stats()
        assert 'recomputation_count' in stats
        assert 'avg_recomputation_time_ms' in stats
        assert 'total_memory_saved_mb' in stats
        assert 'memory_efficiency' in stats
    
    def test_clear_stored_activations(self):
        """Test clearing stored activations."""
        module_name = "test_linear"
        layer_idx = 0
        module_type = "feedforward"
        
        # Register and run forward pass
        self.hook_manager.register_module(
            self.test_module,
            module_name,
            layer_idx,
            module_type,
            self.mock_gate_fn
        )
        
        input_tensor = torch.randn(2, 10, 64)
        output = self.test_module(input_tensor)
        
        # Verify activation is stored
        assert len(self.hook_manager.stored_activations) == 1
        
        # Clear activations
        self.hook_manager.clear_stored_activations()
        
        # Verify cleared
        assert len(self.hook_manager.stored_activations) == 0
        assert self.hook_manager.memory_used_mb == 0.0
    
    def test_cleanup(self):
        """Test hook cleanup."""
        module_name = "test_linear"
        layer_idx = 0
        module_type = "feedforward"
        
        # Register module
        self.hook_manager.register_module(
            self.test_module,
            module_name,
            layer_idx,
            module_type,
            self.mock_gate_fn
        )
        
        # Verify hooks are registered
        assert len(self.hook_manager.forward_hooks) == 1
        assert len(self.hook_manager.backward_hooks) == 1
        
        # Cleanup
        self.hook_manager.cleanup()
        
        # Verify cleanup
        assert len(self.hook_manager.forward_hooks) == 0
        assert len(self.hook_manager.backward_hooks) == 0
        assert len(self.hook_manager.stored_activations) == 0


class TestCheckpointRecomputationHook:
    """Test checkpoint-based recomputation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.checkpoint_hook = CheckpointRecomputationHook(preserve_rng_state=True)
    
    def test_checkpoint_hook_initialization(self):
        """Test checkpoint hook initialization."""
        assert self.checkpoint_hook.preserve_rng_state is True
        assert len(self.checkpoint_hook.checkpoint_segments) == 0
    
    def test_checkpoint_function(self):
        """Test checkpointing a function."""
        # Create a simple function to checkpoint
        def test_function(x):
            return x * 2 + 1
        
        # Create input tensor
        input_tensor = torch.randn(4, 8, requires_grad=True)
        
        # Apply checkpointing
        output = self.checkpoint_hook.checkpoint_function(test_function, input_tensor)
        
        # Verify output shape and gradient capability
        assert output.shape == input_tensor.shape
        assert output.requires_grad is True
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        assert input_tensor.grad is not None
    
    def test_checkpoint_sequential(self):
        """Test sequential checkpointing."""
        # Create a list of functions
        functions = [
            lambda x: x + 1,
            lambda x: x * 2,
            lambda x: x - 0.5
        ]
        
        input_tensor = torch.randn(4, 8, requires_grad=True)
        
        # Apply sequential checkpointing
        output = self.checkpoint_hook.checkpoint_sequential(
            functions,
            segments=2,
            input_tensor=input_tensor
        )
        
        # Verify output
        expected = ((input_tensor + 1) * 2) - 0.5
        assert torch.allclose(output, expected, atol=1e-6)
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        assert input_tensor.grad is not None


class TestRecomputationContext:
    """Test recomputation context manager."""
    
    def test_recomputation_context_enabled(self):
        """Test context manager when recomputation is enabled."""
        hook_manager = RecomputationHookManager()
        
        with recomputation_context(hook_manager, enable_recomputation=True) as ctx:
            assert ctx is hook_manager
    
    def test_recomputation_context_disabled(self):
        """Test context manager when recomputation is disabled."""
        hook_manager = RecomputationHookManager()
        
        with recomputation_context(hook_manager, enable_recomputation=False) as ctx:
            assert ctx is None


class TestRecomputationWrapper:
    """Test recomputation wrapper functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_module = nn.Linear(64, 64)
        
        # Create a mock gate function
        def mock_gate_fn(hidden_states):
            batch_size, seq_len, hidden_size = hidden_states.shape
            gate_values = torch.ones(batch_size, seq_len, 1) * 0.3  # Low probability
            gate_decisions = (gate_values > 0.5).float()
            return gate_values, gate_decisions
        
        self.mock_gate_fn = mock_gate_fn
    
    def test_checkpoint_strategy_wrapper(self):
        """Test wrapper with checkpoint strategy."""
        wrapped_fn = create_recomputation_wrapper(
            self.test_module,
            self.mock_gate_fn,
            RecomputationStrategy.CHECKPOINT
        )
        
        input_tensor = torch.randn(2, 10, 64, requires_grad=True)
        output = wrapped_fn(input_tensor)
        
        # Verify output and gradient capability
        assert output.shape == (2, 10, 64)
        assert output.requires_grad is True
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        assert input_tensor.grad is not None
    
    def test_custom_strategy_wrapper(self):
        """Test wrapper with custom strategy."""
        wrapped_fn = create_recomputation_wrapper(
            self.test_module,
            self.mock_gate_fn,
            RecomputationStrategy.CUSTOM
        )
        
        input_tensor = torch.randn(2, 10, 64, requires_grad=True)
        output = wrapped_fn(input_tensor)
        
        # Verify output
        assert output.shape == (2, 10, 64)
        assert output.requires_grad is True
    
    def test_hybrid_strategy_wrapper(self):
        """Test wrapper with hybrid strategy."""
        wrapped_fn = create_recomputation_wrapper(
            self.test_module,
            self.mock_gate_fn,
            RecomputationStrategy.HYBRID
        )
        
        # Test with large tensor (should use checkpointing)
        large_input = torch.randn(100, 100, 64, requires_grad=True)
        output = wrapped_fn(large_input)
        
        assert output.shape == (100, 100, 64)
        assert output.requires_grad is True
        
        # Test with small tensor (should use normal forward)
        small_input = torch.randn(2, 5, 64, requires_grad=True)
        output = wrapped_fn(small_input)
        
        assert output.shape == (2, 5, 64)
        assert output.requires_grad is True


class TestActivationInfo:
    """Test ActivationInfo dataclass."""
    
    def test_activation_info_creation(self):
        """Test creating ActivationInfo objects."""
        tensor = torch.randn(4, 8, 64)
        gate_decision = torch.ones(4, 8, 1)
        
        def dummy_recompute():
            return tensor
        
        # Test stored activation
        stored_info = ActivationInfo(
            tensor=tensor,
            recompute_fn=None,
            gate_decision=gate_decision,
            layer_idx=0,
            module_type="attention",
            is_stored=True,
            memory_saved_mb=0.0
        )
        
        assert stored_info.tensor is not None
        assert stored_info.recompute_fn is None
        assert stored_info.is_stored is True
        assert stored_info.layer_idx == 0
        assert stored_info.module_type == "attention"
        
        # Test recomputation activation
        recompute_info = ActivationInfo(
            tensor=None,
            recompute_fn=dummy_recompute,
            gate_decision=gate_decision,
            layer_idx=1,
            module_type="feedforward",
            is_stored=False,
            memory_saved_mb=2.5
        )
        
        assert recompute_info.tensor is None
        assert recompute_info.recompute_fn is not None
        assert recompute_info.is_stored is False
        assert recompute_info.memory_saved_mb == 2.5


class TestIntegrationWithGates:
    """Test integration with existing gate system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=1000
        )
        
        self.gate = RecomputationGate(
            hidden_size=self.config.hidden_size,
            gate_type="global",
            init_bias=0.0
        )
        
        self.hook_manager = RecomputationHookManager(
            strategy=RecomputationStrategy.CUSTOM,
            enable_profiling=True
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.hook_manager.cleanup()
    
    def test_gate_integration(self):
        """Test integration with RecomputationGate."""
        # Create a simple module
        test_module = nn.Linear(64, 64)
        
        # Create gate function
        def gate_fn(hidden_states):
            return self.gate(hidden_states, training=True)
        
        # Register module with hook manager
        self.hook_manager.register_module(
            test_module,
            "test_module",
            layer_idx=0,
            module_type="feedforward",
            gate_fn=gate_fn
        )
        
        # Test forward pass
        input_tensor = torch.randn(2, 10, 64)
        output = test_module(input_tensor)
        
        # Verify hook was triggered
        assert "test_module" in self.hook_manager.stored_activations
        
        # Get memory stats
        stats = self.hook_manager.get_memory_stats()
        assert stats['total_activations'] == 1
    
    def test_multiple_modules_registration(self):
        """Test registering multiple modules."""
        modules = [
            ("attention", nn.Linear(64, 64)),
            ("feedforward", nn.Linear(64, 64)),
            ("output", nn.Linear(64, 64))
        ]
        
        def gate_fn(hidden_states):
            return self.gate(hidden_states, training=True)
        
        # Register all modules
        for i, (name, module) in enumerate(modules):
            self.hook_manager.register_module(
                module,
                f"layer_0_{name}",
                layer_idx=0,
                module_type=name,
                gate_fn=gate_fn
            )
        
        # Test forward passes
        input_tensor = torch.randn(2, 10, 64)
        for name, module in modules:
            output = module(input_tensor)
        
        # Verify all modules were hooked
        stats = self.hook_manager.get_memory_stats()
        assert stats['total_activations'] == len(modules)
    
    def test_memory_pressure_simulation(self):
        """Test behavior under different memory pressure scenarios."""
        test_module = nn.Linear(64, 64)
        
        # High memory pressure (low storage probability)
        self.gate.set_storage_probability(0.1)
        
        def gate_fn(hidden_states):
            return self.gate(hidden_states, training=True)
        
        self.hook_manager.register_module(
            test_module,
            "test_module",
            layer_idx=0,
            module_type="feedforward",
            gate_fn=gate_fn
        )
        
        # Forward pass
        input_tensor = torch.randn(2, 10, 64)
        output = test_module(input_tensor)
        
        # Should choose recomputation due to low storage probability
        activation_info = self.hook_manager.stored_activations["test_module"]
        assert activation_info.is_stored is False
        assert activation_info.memory_saved_mb > 0 