"""
Tests for the baseline transformer model.

This module tests the transformer model architecture including:
- Model configuration
- Forward pass functionality
- Attention mechanisms
- Feed-forward networks
- Model serialization
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path

from src.models import (
    TransformerConfig,
    BaselineTransformer,
    get_tiny_config,
    get_small_config,
    MultiHeadAttention,
    FeedForwardNetwork,
    LayerNorm,
    create_attention_mask,
    create_causal_mask
)


class TestTransformerConfig:
    """Test transformer configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = TransformerConfig()
        
        assert config.vocab_size == 50000
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.head_dim == 64  # 768 / 12
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid attention heads
        with pytest.raises(ValueError):
            TransformerConfig(hidden_size=768, num_attention_heads=7)  # Not divisible
        
        # Test invalid activation
        with pytest.raises(ValueError):
            TransformerConfig(hidden_act="invalid_activation")
        
        # Test invalid layer norm type
        with pytest.raises(ValueError):
            TransformerConfig(layer_norm_type="invalid_type")
    
    def test_config_serialization(self):
        """Test configuration save/load."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8  # 256 is divisible by 8
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save config
            config.save_pretrained(temp_dir)
            
            # Load config
            loaded_config = TransformerConfig.from_pretrained(temp_dir)
            
            assert loaded_config.vocab_size == config.vocab_size
            assert loaded_config.hidden_size == config.hidden_size
            assert loaded_config.num_hidden_layers == config.num_hidden_layers
    
    def test_predefined_configs(self):
        """Test predefined configuration functions."""
        tiny_config = get_tiny_config()
        assert tiny_config.vocab_size == 1000
        assert tiny_config.hidden_size == 256
        
        small_config = get_small_config()
        assert small_config.vocab_size == 30000
        assert small_config.hidden_size == 768
    
    def test_model_size_info(self):
        """Test model size calculation."""
        config = get_tiny_config()
        size_info = config.get_model_size_info()
        
        assert "total_parameters" in size_info
        assert "memory_estimate_mb" in size_info
        assert size_info["total_parameters"] > 0


class TestAttentionMechanisms:
    """Test attention mechanisms."""
    
    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_tiny_config()
    
    def test_multi_head_attention(self, config):
        """Test multi-head attention mechanism."""
        attention = MultiHeadAttention(config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Test forward pass
        output = attention(hidden_states)
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        
        # Test with attention weights
        output, weights = attention(hidden_states, return_attention_weights=True)
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert weights.shape == (batch_size, config.num_attention_heads, seq_len, seq_len)
    
    def test_attention_masking(self, config):
        """Test attention masking functionality."""
        attention = MultiHeadAttention(config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Create attention mask (mask out last 3 tokens)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -3:] = 0
        
        output = attention(hidden_states, attention_mask=attention_mask)
        assert output.shape == (batch_size, seq_len, config.hidden_size)
    
    def test_causal_masking(self, config):
        """Test causal masking for autoregressive generation."""
        attention = MultiHeadAttention(config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = attention(hidden_states, causal_mask=True)
        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestFeedForwardNetwork:
    """Test feed-forward network components."""
    
    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_tiny_config()
    
    def test_feedforward_network(self, config):
        """Test feed-forward network."""
        ff_network = FeedForwardNetwork(config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = ff_network(hidden_states)
        assert output.shape == (batch_size, seq_len, config.hidden_size)
    
    def test_layer_norm(self):
        """Test layer normalization."""
        hidden_size = 256
        layer_norm = LayerNorm(hidden_size)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        output = layer_norm(hidden_states)
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        # Check normalization properties
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


class TestBaselineTransformer:
    """Test the complete baseline transformer model."""
    
    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_tiny_config()
    
    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return BaselineTransformer(config)
    
    def test_model_initialization(self, model, config):
        """Test model initialization."""
        assert isinstance(model, BaselineTransformer)
        assert model.config == config
        assert len(model.layers) == config.num_hidden_layers
    
    def test_forward_pass(self, model, config):
        """Test model forward pass."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids)
        
        assert "last_hidden_state" in outputs
        assert "logits" in outputs
        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, config.hidden_size)
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    
    def test_forward_with_attention_mask(self, model, config):
        """Test forward pass with attention mask."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -3:] = 0  # Mask last 3 tokens
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, config.hidden_size)
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    
    def test_forward_with_optional_outputs(self, model, config):
        """Test forward pass with optional outputs."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        outputs = model(
            input_ids,
            return_attention_weights=True,
            return_hidden_states=True
        )
        
        assert "attention_weights" in outputs
        assert "hidden_states" in outputs
        assert len(outputs["attention_weights"]) == config.num_hidden_layers
        assert len(outputs["hidden_states"]) == config.num_hidden_layers + 1  # +1 for embeddings
    
    def test_generation(self, model, config):
        """Test text generation."""
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
        
        generated = model.generate(
            input_ids,
            max_length=10,
            do_sample=False,  # Use greedy decoding for deterministic test
            temperature=1.0
        )
        
        assert generated.shape[0] == batch_size
        assert generated.shape[1] >= seq_len
        assert generated.shape[1] <= 10
    
    def test_parameter_count(self, model):
        """Test parameter counting."""
        param_counts = model.get_parameter_count()
        
        assert "total" in param_counts
        assert "embeddings" in param_counts
        assert "transformer_layers" in param_counts
        assert param_counts["total"] > 0
        
        # Verify total matches sum of components
        component_sum = (
            param_counts["embeddings"] +
            param_counts["transformer_layers"] +
            param_counts["output_projection"] +
            param_counts["final_layer_norm"]
        )
        assert param_counts["total"] == component_sum
    
    def test_model_serialization(self, model, config):
        """Test model save/load functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set models to eval mode to disable dropout
            model.eval()
            
            # Save model
            model.save_pretrained(temp_dir)
            
            # Load model
            loaded_model = BaselineTransformer.from_pretrained(temp_dir)
            loaded_model.eval()
            
            # Test that loaded model has same configuration
            assert loaded_model.config.vocab_size == config.vocab_size
            assert loaded_model.config.hidden_size == config.hidden_size
            
            # Test that models produce same output with fixed input
            batch_size, seq_len = 1, 5
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Fixed input for deterministic test
            
            with torch.no_grad():
                original_output = model(input_ids)
                loaded_output = loaded_model(input_ids)
            
            torch.testing.assert_close(
                original_output["logits"],
                loaded_output["logits"],
                atol=1e-6,
                rtol=1e-6
            )
    
    def test_embedding_access(self, model):
        """Test embedding layer access methods."""
        # Test getting embeddings
        input_embeddings = model.get_input_embeddings()
        assert isinstance(input_embeddings, nn.Embedding)
        
        output_embeddings = model.get_output_embeddings()
        assert output_embeddings is not None
        
        # Test setting embeddings
        new_embeddings = nn.Embedding(model.config.vocab_size, model.config.hidden_size)
        model.set_input_embeddings(new_embeddings)
        assert model.get_input_embeddings() is new_embeddings


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_attention_mask(self):
        """Test attention mask creation."""
        batch_size, seq_len = 2, 10
        input_ids = torch.tensor([
            [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],  # 4 valid tokens
            [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]   # 6 valid tokens
        ])
        
        mask = create_attention_mask(input_ids, pad_token_id=0)
        
        expected = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        ])
        
        assert torch.equal(mask, expected)
    
    def test_create_causal_mask(self):
        """Test causal mask creation."""
        seq_len = 4
        device = torch.device('cpu')
        
        mask = create_causal_mask(seq_len, device)
        
        expected = torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True]
        ])
        
        assert torch.equal(mask, expected)


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for the complete model."""
    
    def test_end_to_end_training_step(self):
        """Test a complete training step."""
        config = get_tiny_config()
        model = BaselineTransformer(config)
        
        # Create sample data
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids)
        logits = outputs["logits"]
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, config.vocab_size), target_ids.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_modes(self):
        """Test model training/evaluation modes."""
        config = get_tiny_config()
        model = BaselineTransformer(config)
        
        # Test training mode
        model.train()
        assert model.training
        
        # Test evaluation mode
        model.eval()
        assert not model.training
        
        # Test that dropout behaves differently
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            # Multiple forward passes in eval mode should be identical
            model.eval()
            output1 = model(input_ids)
            output2 = model(input_ids)
            torch.testing.assert_close(output1["logits"], output2["logits"])


if __name__ == "__main__":
    pytest.main([__file__]) 