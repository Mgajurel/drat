"""
Gated Transformer Model with Differentiable Recomputation Gates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging

from .config import TransformerConfig
from .transformer import TransformerEmbeddings
from .gates import GatedTransformerLayer, GateManager
from .feedforward import LayerNorm

logger = logging.getLogger(__name__)


class GatedTransformer(nn.Module):
    """
    Transformer Model with Differentiable Recomputation Gates.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings (same as baseline)
        self.embeddings = TransformerEmbeddings(config)
        
        # Gated transformer layers
        self.layers = nn.ModuleList([
            GatedTransformerLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            use_bias=config.use_bias
        )
        
        # Output projection (language modeling head)
        if config.tie_word_embeddings:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False
            )
        
        # Gate manager for coordinating all gates
        self.gate_manager = GateManager(self.layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        self.embeddings._init_weights()
        
        # Initialize gated transformer layers
        for layer in self.layers:
            if hasattr(layer, '_init_weights'):
                layer._init_weights()
        
        # Initialize final layer norm
        if hasattr(self.final_layer_norm, 'weight'):
            nn.init.ones_(self.final_layer_norm.weight)
        if hasattr(self.final_layer_norm, 'bias') and self.final_layer_norm.bias is not None:
            nn.init.zeros_(self.final_layer_norm.bias)
        
        # Initialize output projection
        if self.output_projection is not None:
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        causal_mask: bool = True,
        return_gate_info: bool = False,
        output_logits: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the gated transformer."""
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Apply embeddings
        hidden_states = self.embeddings(input_ids, position_ids)
        
        # Prepare return containers
        all_gate_info = [] if return_gate_info else None
        
        # Apply transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # Forward through layer
            if return_gate_info:
                hidden_states, gate_info = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    return_gate_info=True
                )
                all_gate_info.append(gate_info)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask
                )
        
        # Final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Prepare output dictionary
        outputs = {
            'last_hidden_state': hidden_states
        }
        
        # Compute logits if requested
        if output_logits:
            if self.config.tie_word_embeddings:
                logits = F.linear(hidden_states, self.embeddings.token_embeddings.weight)
            else:
                logits = self.output_projection(hidden_states)
            outputs['logits'] = logits
        
        # Add optional outputs
        if return_gate_info:
            outputs['gate_info'] = all_gate_info
            outputs['gate_statistics'] = self.gate_manager.get_all_gate_statistics()
        
        return outputs
    
    def set_memory_pressure(self, pressure_level: str = "medium"):
        """Set memory pressure mode for all gates."""
        self.gate_manager.set_memory_pressure_mode(pressure_level)
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gate statistics."""
        return self.gate_manager.get_all_gate_statistics()
    
    def clear_stored_activations(self):
        """Clear all stored activations to free memory."""
        self.gate_manager.clear_all_stored_activations()
    
    def set_gate_probabilities(self, attention_prob: float, ff_prob: float):
        """Set storage probabilities for all gates."""
        self.gate_manager.set_all_gate_probabilities(attention_prob, ff_prob) 