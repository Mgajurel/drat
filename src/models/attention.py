"""
Multi-head attention mechanism for transformer models.

This module implements the core attention components including:
- Scaled dot-product attention
- Multi-head attention
- Positional encoding
- Attention masking utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from .config import TransformerConfig


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.
    
    Implements the attention function: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.attention_dropout)
        self.scale = config.attention_scale
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            value: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            attention_mask: Optional mask tensor to prevent attention to certain positions
            causal_mask: Whether to apply causal (lower triangular) masking
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Attention output tensor and optionally attention weights
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Compute attention scores
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale by sqrt(head_dim)
        attention_scores = attention_scores * self.scale
        
        # Apply causal mask if requested
        if causal_mask:
            causal_mask_tensor = torch.tril(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool)
            )
            attention_scores = attention_scores.masked_fill(
                ~causal_mask_tensor, float('-inf')
            )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention scores shape
            if attention_mask.dim() == 2:  # (batch_size, seq_len)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            elif attention_mask.dim() == 3:  # (batch_size, seq_len, seq_len)
                attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, float('-inf')
            )
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        attention_output = torch.matmul(attention_weights, value)
        
        if return_attention_weights:
            return attention_output, attention_weights
        return attention_output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Implements parallel attention heads with learned linear projections
    for queries, keys, and values.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(
            config.hidden_size, 
            config.hidden_size, 
            bias=config.use_bias
        )
        self.key_projection = nn.Linear(
            config.hidden_size, 
            config.hidden_size, 
            bias=config.use_bias
        )
        self.value_projection = nn.Linear(
            config.hidden_size, 
            config.hidden_size, 
            bias=config.use_bias
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            config.hidden_size, 
            config.hidden_size, 
            bias=config.use_bias
        )
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(config)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using the specified initialization range."""
        for module in [self.query_projection, self.key_projection, 
                      self.value_projection, self.output_projection]:
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask tensor
            causal_mask: Whether to apply causal masking
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Attention output and optionally attention weights
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project to Q, K, V
        query = self.query_projection(hidden_states)  # (batch_size, seq_len, hidden_size)
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        if return_attention_weights:
            attention_output, attention_weights = self.attention(
                query, key, value, attention_mask, causal_mask, return_attention_weights=True
            )
        else:
            attention_output = self.attention(
                query, key, value, attention_mask, causal_mask, return_attention_weights=False
            )
        
        # Reshape back to original format
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, seq_len, hidden_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        # Apply output projection
        output = self.output_projection(attention_output)
        
        if return_attention_weights:
            return output, attention_weights
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Implements sinusoidal positional encoding as described in
    "Attention Is All You Need" paper.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        if config.position_embedding_type == "absolute":
            # Create learnable positional embeddings
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, 
                config.hidden_size
            )
            self._init_weights()
        elif config.position_embedding_type == "relative":
            # For relative positional encoding, we'll implement it later
            raise NotImplementedError("Relative positional encoding not yet implemented")
    
    def _init_weights(self):
        """Initialize positional embedding weights."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            input_ids: Input token embeddings of shape (batch_size, seq_len, hidden_size)
            position_ids: Optional position indices of shape (batch_size, seq_len)
            
        Returns:
            Input embeddings with positional encoding added
        """
        batch_size, seq_len = input_ids.shape[:2]
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings = self.position_embeddings(position_ids)
        
        return self.dropout(input_ids + position_embeddings)


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create attention mask from input token IDs.
    
    Args:
        input_ids: Input token IDs of shape (batch_size, seq_len)
        pad_token_id: ID of the padding token
        
    Returns:
        Attention mask of shape (batch_size, seq_len) where 1 indicates valid tokens
        and 0 indicates padding tokens
    """
    return (input_ids != pad_token_id).long()


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (lower triangular) mask for autoregressive generation.
    
    Args:
        seq_len: Sequence length
        device: Device to create the mask on
        
    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)) 