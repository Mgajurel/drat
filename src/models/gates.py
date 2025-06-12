"""
Differentiable Recomputation Gates for Memory-Efficient Transformers.

This module implements learnable gates that decide whether to store or 
recompute intermediate activations during forward and backward passes,
enabling memory-efficient training of large transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
import math
import logging

from .config import TransformerConfig

logger = logging.getLogger(__name__)


class RecomputationGate(nn.Module):
    """
    A single differentiable gate for storage vs. recomputation decisions.
    
    Uses sigmoid activation to produce probabilistic storage decisions.
    The gate output is in [0, 1] where:
    - 0 means "always recompute" (save memory)
    - 1 means "always store" (save computation)
    - Values in between represent probabilistic decisions
    """
    
    def __init__(
        self,
        hidden_size: int,
        gate_type: str = "global",
        init_bias: float = 0.0,
        temperature: float = 1.0,
        use_straight_through: bool = True
    ):
        """
        Initialize a recomputation gate.
        
        Args:
            hidden_size: Size of hidden representations
            gate_type: Type of gate ("global", "per_head", "per_token")
            init_bias: Initial bias for gate (affects initial storage probability)
            temperature: Temperature for sigmoid (lower = more binary decisions)
            use_straight_through: Whether to use straight-through estimator
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_type = gate_type
        self.init_bias = init_bias
        self.temperature = temperature
        self.use_straight_through = use_straight_through
        
        # Gate parameters based on type
        if gate_type == "global":
            # Single global gate for the entire layer
            self.gate_param = nn.Parameter(torch.zeros(1))
        elif gate_type == "per_head":
            # One gate per attention head (assumes hidden_size is divisible by num_heads)
            # We'll set this dynamically based on config
            self.gate_param = nn.Parameter(torch.zeros(1))  # Will be resized
        elif gate_type == "per_token":
            # One gate per token position (dynamic based on sequence length)
            self.gate_param = nn.Parameter(torch.zeros(1))  # Will be resized
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
        
        # Initialize gate parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gate parameters."""
        # Initialize with bias to control initial storage probability
        # init_bias = 0 -> 50% storage probability
        # init_bias > 0 -> higher storage probability
        # init_bias < 0 -> lower storage probability
        nn.init.constant_(self.gate_param, self.init_bias)
    
    def resize_gate_param(self, new_size: int):
        """Resize gate parameter for dynamic sizing."""
        if self.gate_type in ["per_head", "per_token"]:
            current_size = self.gate_param.size(0)
            if current_size != new_size:
                # Create new parameter with correct size
                new_param = nn.Parameter(torch.zeros(new_size))
                nn.init.constant_(new_param, self.init_bias)
                self.gate_param = new_param
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        num_heads: Optional[int] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gate values for storage decisions.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            num_heads: Number of attention heads (for per_head gates)
            training: Whether in training mode
            
        Returns:
            Tuple of (gate_values, binary_decisions)
            - gate_values: Continuous values in [0, 1]
            - binary_decisions: Binary decisions (0 or 1) for actual storage
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute gate values based on gate type
        if self.gate_type == "global":
            # Single gate value for entire batch/sequence
            gate_logits = self.gate_param.expand(batch_size, seq_len, 1)
        
        elif self.gate_type == "per_head":
            if num_heads is None:
                raise ValueError("num_heads must be provided for per_head gates")
            
            # Resize if needed
            self.resize_gate_param(num_heads)
            
            # One gate per attention head
            head_dim = hidden_size // num_heads
            gate_logits = self.gate_param.view(1, 1, num_heads, 1)
            gate_logits = gate_logits.expand(batch_size, seq_len, num_heads, head_dim)
            gate_logits = gate_logits.reshape(batch_size, seq_len, hidden_size)
        
        elif self.gate_type == "per_token":
            # Resize if needed
            self.resize_gate_param(seq_len)
            
            # One gate per token position
            gate_logits = self.gate_param.view(1, seq_len, 1)
            gate_logits = gate_logits.expand(batch_size, seq_len, hidden_size)
        
        # Apply temperature scaling and sigmoid
        gate_values = torch.sigmoid(gate_logits / self.temperature)
        
        # Generate binary decisions
        if training and self.use_straight_through:
            # Use straight-through estimator for binary decisions
            # Forward: binary, Backward: continuous gradient
            binary_decisions = (gate_values > 0.5).float()
            # Straight-through: copy gradients from continuous to binary
            binary_decisions = gate_values + (binary_decisions - gate_values).detach()
        else:
            # Use continuous values (for inference or when not using straight-through)
            binary_decisions = gate_values
        
        return gate_values, binary_decisions
    
    def get_storage_probability(self) -> float:
        """Get the current storage probability."""
        with torch.no_grad():
            gate_value = torch.sigmoid(self.gate_param / self.temperature)
            return gate_value.mean().item()
    
    def set_storage_probability(self, prob: float):
        """Set the storage probability by adjusting gate parameters."""
        if not 0 <= prob <= 1:
            raise ValueError("Probability must be between 0 and 1")
        
        # Compute required logit value
        logit = math.log(prob / (1 - prob + 1e-8)) * self.temperature
        
        with torch.no_grad():
            self.gate_param.fill_(logit)


class GatedTransformerLayer(nn.Module):
    """
    Transformer layer with differentiable recomputation gates.
    
    Extends the standard transformer layer to include gates that decide
    whether to store or recompute intermediate activations.
    """
    
    def __init__(self, config: TransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Import here to avoid circular imports
        from .attention import MultiHeadAttention
        from .feedforward import FeedForwardNetwork, LayerNorm
        
        # Standard transformer components
        self.attention = MultiHeadAttention(config)
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
        
        # Recomputation gates
        self.attention_gate = RecomputationGate(
            hidden_size=config.hidden_size,
            gate_type=getattr(config, 'gate_type', 'global'),
            init_bias=getattr(config, 'gate_init_bias', 0.0),
            temperature=getattr(config, 'gate_temperature', 1.0),
            use_straight_through=getattr(config, 'use_straight_through', True)
        )
        
        self.ff_gate = RecomputationGate(
            hidden_size=config.hidden_size,
            gate_type=getattr(config, 'gate_type', 'global'),
            init_bias=getattr(config, 'gate_init_bias', 0.0),
            temperature=getattr(config, 'gate_temperature', 1.0),
            use_straight_through=getattr(config, 'use_straight_through', True)
        )
        
        # Storage for intermediate activations
        self._stored_attention_output = None
        self._stored_ff_output = None
        self._attention_gate_decision = None
        self._ff_gate_decision = None
    
    def _init_weights(self):
        """Initialize weights for the gated transformer layer."""
        # Initialize standard components
        if hasattr(self.attention, '_init_weights'):
            self.attention._init_weights()
        
        if hasattr(self.feed_forward, '_init_weights'):
            self.feed_forward._init_weights()
        
        # Initialize layer norm weights
        nn.init.ones_(self.attention_layer_norm.weight)
        if self.attention_layer_norm.bias is not None:
            nn.init.zeros_(self.attention_layer_norm.bias)
        
        nn.init.ones_(self.ff_layer_norm.weight)
        if self.ff_layer_norm.bias is not None:
            nn.init.zeros_(self.ff_layer_norm.bias)
        
        # Gates initialize themselves
    
    def _gated_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """Apply attention with gated recomputation."""
        # Compute gate decision
        gate_values, gate_decision = self.attention_gate(
            hidden_states,
            num_heads=self.config.num_attention_heads,
            training=self.training
        )
        
        # Store gate decision for potential reuse
        self._attention_gate_decision = gate_decision
        
        # Always compute attention output (for now - recomputation logic can be added later)
        if return_attention_weights:
            attention_output, attention_weights = self.attention(
                hidden_states, attention_mask, causal_mask, return_attention_weights=True
            )
        else:
            attention_output = self.attention(
                hidden_states, attention_mask, causal_mask, return_attention_weights=False
            )
        
        # Apply gate to modulate attention output
        # This creates a differentiable path for gradients to flow to gate parameters
        if self.config.gate_type == "global":
            # For global gates, apply the same gate value to all positions
            gated_output = attention_output * gate_values
        else:
            # For per-head or per-token gates, gate_values already has the right shape
            gated_output = attention_output * gate_values
        
        if return_attention_weights:
            return gated_output, attention_weights
        return gated_output
    
    def _gated_feed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network with gated recomputation."""
        # Compute gate decision
        gate_values, gate_decision = self.ff_gate(
            hidden_states,
            training=self.training
        )
        
        # Store gate decision for potential reuse
        self._ff_gate_decision = gate_decision
        
        # Always compute feed-forward output (for now - recomputation logic can be added later)
        ff_output = self.feed_forward(hidden_states)
        
        # Apply gate to modulate feed-forward output
        # This creates a differentiable path for gradients to flow to gate parameters
        if self.config.gate_type == "global":
            # For global gates, apply the same gate value to all positions
            gated_output = ff_output * gate_values
        else:
            # For per-head or per-token gates, gate_values already has the right shape
            gated_output = ff_output * gate_values
        
        return gated_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        return_attention_weights: bool = False,
        return_gate_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Apply gated transformer layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            causal_mask: Whether to apply causal masking
            return_attention_weights: Whether to return attention weights
            return_gate_info: Whether to return gate information
            
        Returns:
            Output tensor or tuple with gate information
        """
        # Store input for residual connection
        residual = hidden_states
        
        # Pre-layer norm or post-layer norm
        if self.config.layer_norm_type == "pre":
            # Pre-layer norm: LayerNorm -> Attention -> Residual
            hidden_states = self.attention_layer_norm(hidden_states)
            
            if return_attention_weights:
                attention_output, attention_weights = self._gated_attention(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=True
                )
            else:
                attention_output = self._gated_attention(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=False
                )
            
            # Apply dropout and residual connection
            hidden_states = residual + self.dropout(attention_output)
            
            # Feed-forward with pre-layer norm
            residual = hidden_states
            hidden_states = self.ff_layer_norm(hidden_states)
            ff_output = self._gated_feed_forward(hidden_states)
            hidden_states = residual + self.dropout(ff_output)
            
        else:  # post-layer norm
            # Post-layer norm: Attention -> Residual -> LayerNorm
            if return_attention_weights:
                attention_output, attention_weights = self._gated_attention(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=True
                )
            else:
                attention_output = self._gated_attention(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=False
                )
            
            # Apply dropout, residual connection, and layer norm
            hidden_states = self.attention_layer_norm(residual + self.dropout(attention_output))
            
            # Feed-forward with post-layer norm
            residual = hidden_states
            ff_output = self._gated_feed_forward(hidden_states)
            hidden_states = self.ff_layer_norm(residual + self.dropout(ff_output))
        
        # Prepare return values
        if return_gate_info:
            gate_info = {
                'attention_gate_prob': self.attention_gate.get_storage_probability(),
                'ff_gate_prob': self.ff_gate.get_storage_probability(),
                'layer_idx': self.layer_idx
            }
            
            if return_attention_weights:
                return hidden_states, attention_weights, gate_info
            return hidden_states, gate_info
        
        if return_attention_weights:
            return hidden_states, attention_weights
        return hidden_states
    
    def clear_stored_activations(self):
        """Clear stored activations to free memory."""
        self._stored_attention_output = None
        self._stored_ff_output = None
        self._attention_gate_decision = None
        self._ff_gate_decision = None
    
    def get_gate_statistics(self) -> Dict[str, float]:
        """Get statistics about gate behavior."""
        return {
            'attention_gate_prob': self.attention_gate.get_storage_probability(),
            'ff_gate_prob': self.ff_gate.get_storage_probability(),
            'layer_idx': self.layer_idx
        }


class GateManager(nn.Module):
    """
    Manager for coordinating multiple recomputation gates across layers.
    
    Provides utilities for:
    - Setting gate probabilities across layers
    - Collecting gate statistics
    - Managing memory vs. computation trade-offs
    """
    
    def __init__(self, gated_layers: nn.ModuleList):
        super().__init__()
        self.gated_layers = gated_layers
    
    def set_all_gate_probabilities(self, attention_prob: float, ff_prob: float):
        """Set storage probabilities for all gates."""
        for layer in self.gated_layers:
            if hasattr(layer, 'attention_gate'):
                layer.attention_gate.set_storage_probability(attention_prob)
            if hasattr(layer, 'ff_gate'):
                layer.ff_gate.set_storage_probability(ff_prob)
    
    def get_all_gate_statistics(self) -> Dict[str, Any]:
        """Get statistics for all gates."""
        stats = {
            'layer_stats': [],
            'avg_attention_prob': 0.0,
            'avg_ff_prob': 0.0
        }
        
        attention_probs = []
        ff_probs = []
        
        for layer in self.gated_layers:
            if hasattr(layer, 'get_gate_statistics'):
                layer_stats = layer.get_gate_statistics()
                stats['layer_stats'].append(layer_stats)
                attention_probs.append(layer_stats['attention_gate_prob'])
                ff_probs.append(layer_stats['ff_gate_prob'])
        
        if attention_probs:
            stats['avg_attention_prob'] = sum(attention_probs) / len(attention_probs)
        if ff_probs:
            stats['avg_ff_prob'] = sum(ff_probs) / len(ff_probs)
        
        return stats
    
    def clear_all_stored_activations(self):
        """Clear stored activations in all layers."""
        for layer in self.gated_layers:
            if hasattr(layer, 'clear_stored_activations'):
                layer.clear_stored_activations()
    
    def set_memory_pressure_mode(self, pressure_level: str = "medium"):
        """
        Set gate probabilities based on memory pressure level.
        
        Args:
            pressure_level: "low", "medium", "high", or "extreme"
        """
        if pressure_level == "low":
            # Prefer storage (less recomputation)
            self.set_all_gate_probabilities(0.8, 0.8)
        elif pressure_level == "medium":
            # Balanced approach
            self.set_all_gate_probabilities(0.5, 0.5)
        elif pressure_level == "high":
            # Prefer recomputation (save memory)
            self.set_all_gate_probabilities(0.2, 0.2)
        elif pressure_level == "extreme":
            # Aggressive memory saving
            self.set_all_gate_probabilities(0.1, 0.1)
        else:
            raise ValueError(f"Unknown pressure level: {pressure_level}") 