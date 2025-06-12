"""
Feed-forward network layers for transformer models.

This module implements the position-wise feed-forward networks
and layer normalization components used in transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .config import TransformerConfig


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network.
    
    Implements the feed-forward component of transformer layers:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Or with GELU activation:
    FFN(x) = GELU(xW1 + b1)W2 + b2
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # First linear transformation (expand)
        self.linear1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias
        )
        
        # Second linear transformation (contract)
        self.linear2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias
        )
        
        # Activation function
        self.activation = self._get_activation_function(config.hidden_act)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation_function(self, activation_name: str):
        """Get the activation function by name."""
        activation_functions = {
            "relu": F.relu,
            "gelu": F.gelu,
            "gelu_new": lambda x: 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )),
            "swish": lambda x: x * torch.sigmoid(x),
            "silu": F.silu,
        }
        
        if activation_name not in activation_functions:
            raise ValueError(f"Unknown activation function: {activation_name}")
        
        return activation_functions[activation_name]
    
    def _init_weights(self):
        """Initialize weights using the specified initialization range."""
        nn.init.normal_(self.linear1.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=self.config.initializer_range)
        
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # First linear transformation and activation
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Second linear transformation
        hidden_states = self.linear2(hidden_states)
        
        return hidden_states


class LayerNorm(nn.Module):
    """
    Layer normalization with optional bias.
    
    Implements layer normalization as described in:
    "Layer Normalization" (Ba et al., 2016)
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-12, use_bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if use_bias else None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            hidden_states: Input tensor of shape (..., hidden_size)
            
        Returns:
            Normalized tensor of the same shape
        """
        # Compute mean and variance along the last dimension
        mean = hidden_states.mean(dim=-1, keepdim=True)
        variance = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.eps)
        
        # Apply learnable parameters
        hidden_states = hidden_states * self.weight
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
        
        return hidden_states


class TransformerLayer(nn.Module):
    """
    Single transformer layer with multi-head attention and feed-forward network.
    
    Implements the standard transformer layer with:
    - Multi-head self-attention
    - Layer normalization
    - Feed-forward network
    - Residual connections
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Import here to avoid circular imports
        from .attention import MultiHeadAttention
        
        # Multi-head attention
        self.attention = MultiHeadAttention(config)
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(config)
        
        # Layer normalization
        self.attention_layer_norm = LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps, 
            use_bias=config.use_bias
        )
        self.ff_layer_norm = LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps, 
            use_bias=config.use_bias
        )
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def _init_weights(self):
        """Initialize weights for the transformer layer."""
        # Initialize attention weights
        if hasattr(self.attention, '_init_weights'):
            self.attention._init_weights()
        
        # Initialize feed-forward weights
        self.feed_forward._init_weights()
        
        # Initialize layer norm weights
        nn.init.ones_(self.attention_layer_norm.weight)
        if self.attention_layer_norm.bias is not None:
            nn.init.zeros_(self.attention_layer_norm.bias)
        
        nn.init.ones_(self.ff_layer_norm.weight)
        if self.ff_layer_norm.bias is not None:
            nn.init.zeros_(self.ff_layer_norm.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Apply transformer layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            causal_mask: Whether to apply causal masking
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Store input for residual connection
        residual = hidden_states
        
        # Pre-layer norm or post-layer norm
        if self.config.layer_norm_type == "pre":
            # Pre-layer norm: LayerNorm -> Attention -> Residual
            hidden_states = self.attention_layer_norm(hidden_states)
            
            if return_attention_weights:
                attention_output, attention_weights = self.attention(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=True
                )
            else:
                attention_output = self.attention(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=False
                )
            
            # Apply dropout and residual connection
            hidden_states = residual + self.dropout(attention_output)
            
            # Feed-forward with pre-layer norm
            residual = hidden_states
            hidden_states = self.ff_layer_norm(hidden_states)
            ff_output = self.feed_forward(hidden_states)
            hidden_states = residual + self.dropout(ff_output)
            
        else:  # post-layer norm
            # Post-layer norm: Attention -> Residual -> LayerNorm
            if return_attention_weights:
                attention_output, attention_weights = self.attention(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=True
                )
            else:
                attention_output = self.attention(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=False
                )
            
            # Apply dropout, residual connection, and layer norm
            hidden_states = self.attention_layer_norm(residual + self.dropout(attention_output))
            
            # Feed-forward with post-layer norm
            residual = hidden_states
            ff_output = self.feed_forward(hidden_states)
            hidden_states = self.ff_layer_norm(residual + self.dropout(ff_output))
        
        if return_attention_weights:
            return hidden_states, attention_weights
        return hidden_states


# Import math for gelu_new activation
import math 