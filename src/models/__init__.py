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
from .gated_transformer import GatedTransformer
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
from .gates import (
    RecomputationGate,
    GatedTransformerLayer,
    GateManager
)
from .recomputation_hooks import (
    RecomputationHookManager,
    CheckpointRecomputationHook,
    RecomputationStrategy,
    ActivationInfo,
    recomputation_context,
    create_recomputation_wrapper
)
from .gate_recomputation import (
    GateTriggeredRecomputation,
    GatedRecomputationLayer,
    gate_recomputation_context,
    create_gated_recomputation_model
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
    'GatedTransformer',
    
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
    
    # Recomputation gates
    'RecomputationGate',
    'GatedTransformerLayer',
    'GateManager',
    
    # Recomputation hooks
    'RecomputationHookManager',
    'CheckpointRecomputationHook',
    'RecomputationStrategy',
    'ActivationInfo',
    'recomputation_context',
    'create_recomputation_wrapper',
    
    # Gate-triggered recomputation
    'GateTriggeredRecomputation',
    'GatedRecomputationLayer',
    'gate_recomputation_context',
    'create_gated_recomputation_model',
] 