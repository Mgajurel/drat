"""
Models package for transformer architectures and related components.

This package provides:
- Transformer model configurations
- Baseline transformer implementation
- Attention mechanisms
- Feed-forward networks
- Model utilities and helpers
"""

from .config import (
    TransformerConfig,
    get_small_config,
    get_medium_config,
    get_large_config,
    get_tiny_config
)
from .transformer import BaselineTransformer, TransformerEmbeddings
from .attention import (
    MultiHeadAttention,
    ScaledDotProductAttention,
    PositionalEncoding,
    create_attention_mask,
    create_causal_mask
)
from .feedforward import (
    FeedForwardNetwork,
    LayerNorm,
    TransformerLayer
)

__all__ = [
    # Configuration
    'TransformerConfig',
    'get_small_config',
    'get_medium_config', 
    'get_large_config',
    'get_tiny_config',
    
    # Main model
    'BaselineTransformer',
    'TransformerEmbeddings',
    
    # Attention components
    'MultiHeadAttention',
    'ScaledDotProductAttention',
    'PositionalEncoding',
    'create_attention_mask',
    'create_causal_mask',
    
    # Feed-forward components
    'FeedForwardNetwork',
    'LayerNorm',
    'TransformerLayer',
] 