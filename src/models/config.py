"""
Transformer model configuration.

This module provides configuration classes for transformer models,
including hyperparameters, architecture settings, and training options.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


@dataclass
class TransformerConfig:
    """
    Configuration class for transformer models.
    
    This class defines all hyperparameters and architectural choices
    for the baseline transformer implementation.
    """
    
    # Model architecture
    vocab_size: int = 50000
    max_sequence_length: int = 512
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072  # Feed-forward hidden size (typically 4x hidden_size)
    
    # Attention configuration
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Activation functions
    hidden_act: str = "gelu"  # Options: "gelu", "relu", "swish", "gelu_new"
    
    # Positional encoding
    max_position_embeddings: int = 512
    position_embedding_type: str = "absolute"  # Options: "absolute", "relative"
    
    # Layer normalization
    layer_norm_type: str = "pre"  # Options: "pre", "post"
    
    # Initialization
    initializer_range: float = 0.02
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    # Model type and version
    model_type: str = "baseline_transformer"
    model_version: str = "1.0"
    
    # Training specific (optional)
    use_cache: bool = True
    gradient_checkpointing: bool = False
    
    # Advanced features
    use_bias: bool = True
    tie_word_embeddings: bool = False
    
    # Scaling factors
    attention_scale: Optional[float] = None  # If None, uses 1/sqrt(head_dim)
    
    # Computed fields (set in __post_init__)
    head_dim: int = field(init=False)
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Validate attention heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        # Compute head dimension
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Set default attention scale if not provided
        if self.attention_scale is None:
            self.attention_scale = 1.0 / math.sqrt(self.head_dim)
        
        # Validate sequence length
        if self.max_sequence_length > self.max_position_embeddings:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) cannot be larger than "
                f"max_position_embeddings ({self.max_position_embeddings})"
            )
        
        # Validate activation function
        valid_activations = {"gelu", "relu", "swish", "gelu_new", "silu"}
        if self.hidden_act not in valid_activations:
            raise ValueError(f"hidden_act must be one of {valid_activations}")
        
        # Validate layer norm type
        valid_layer_norm_types = {"pre", "post"}
        if self.layer_norm_type not in valid_layer_norm_types:
            raise ValueError(f"layer_norm_type must be one of {valid_layer_norm_types}")
        
        # Validate position embedding type
        valid_position_types = {"absolute", "relative"}
        if self.position_embedding_type not in valid_position_types:
            raise ValueError(f"position_embedding_type must be one of {valid_position_types}")
    

    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        # Remove computed fields that shouldn't be serialized
        computed_fields = {'head_dim'}
        for field_name in computed_fields:
            config_dict.pop(field_name, None)
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """Create configuration from dictionary."""
        # Remove any computed fields that might be in the dictionary
        config_dict = config_dict.copy()
        computed_fields = {'head_dim'}
        for field_name in computed_fields:
            config_dict.pop(field_name, None)
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory: Union[str, Path]):
        """Save configuration to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        config_file = save_directory / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> 'TransformerConfig':
        """Load configuration from directory."""
        model_path = Path(model_path)
        config_file = model_path / "config.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Re-run validation
        self.__post_init__()
    
    def get_model_size_info(self) -> Dict[str, Any]:
        """Get information about model size and parameters."""
        # Embedding parameters
        token_embeddings = self.vocab_size * self.hidden_size
        position_embeddings = self.max_position_embeddings * self.hidden_size
        
        # Attention parameters per layer
        attention_params_per_layer = (
            # Q, K, V projections
            3 * self.hidden_size * self.hidden_size +
            # Output projection
            self.hidden_size * self.hidden_size +
            # Bias terms (if used)
            (4 * self.hidden_size if self.use_bias else 0)
        )
        
        # Feed-forward parameters per layer
        ff_params_per_layer = (
            # First linear layer
            self.hidden_size * self.intermediate_size +
            # Second linear layer
            self.intermediate_size * self.hidden_size +
            # Bias terms (if used)
            (self.intermediate_size + self.hidden_size if self.use_bias else 0)
        )
        
        # Layer normalization parameters per layer
        ln_params_per_layer = 2 * self.hidden_size * 2  # 2 layer norms per layer
        
        # Total transformer layer parameters
        layer_params = (attention_params_per_layer + ff_params_per_layer + ln_params_per_layer)
        total_layer_params = layer_params * self.num_hidden_layers
        
        # Final layer norm and output projection
        final_ln_params = 2 * self.hidden_size
        output_projection = self.vocab_size * self.hidden_size if not self.tie_word_embeddings else 0
        
        # Total parameters
        total_params = (
            token_embeddings + position_embeddings + 
            total_layer_params + final_ln_params + output_projection
        )
        
        return {
            "total_parameters": total_params,
            "embedding_parameters": token_embeddings + position_embeddings,
            "transformer_parameters": total_layer_params,
            "parameters_per_layer": layer_params,
            "attention_parameters_per_layer": attention_params_per_layer,
            "feedforward_parameters_per_layer": ff_params_per_layer,
            "layernorm_parameters_per_layer": ln_params_per_layer,
            "output_parameters": final_ln_params + output_projection,
            "memory_estimate_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }


# Predefined configurations for common model sizes
def get_small_config() -> TransformerConfig:
    """Get configuration for a small transformer model (similar to BERT-base)."""
    return TransformerConfig(
        vocab_size=30000,
        max_sequence_length=512,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
    )


def get_medium_config() -> TransformerConfig:
    """Get configuration for a medium transformer model."""
    return TransformerConfig(
        vocab_size=50000,
        max_sequence_length=1024,
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=1024,
    )


def get_large_config() -> TransformerConfig:
    """Get configuration for a large transformer model (similar to BERT-large)."""
    return TransformerConfig(
        vocab_size=50000,
        max_sequence_length=512,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=512,
    )


def get_tiny_config() -> TransformerConfig:
    """Get configuration for a tiny transformer model (for testing/debugging)."""
    return TransformerConfig(
        vocab_size=1000,
        max_sequence_length=128,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=128,
    ) 