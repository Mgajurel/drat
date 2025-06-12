"""
Baseline Transformer Model Implementation.

This module provides the complete transformer model that integrates
all components: embeddings, attention, feed-forward networks, and
output layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import json
import logging

from .config import TransformerConfig
from .attention import PositionalEncoding, create_attention_mask
from .feedforward import TransformerLayer, LayerNorm

logger = logging.getLogger(__name__)


class TransformerEmbeddings(nn.Module):
    """
    Transformer embeddings including token and positional embeddings.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(config)
        
        # Layer normalization (optional, used in some variants)
        self.layer_norm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            use_bias=config.use_bias
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=self.config.initializer_range)
        # Set padding token embedding to zero
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                self.token_embeddings.weight[self.config.pad_token_id].fill_(0)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply embeddings to input tokens.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            position_ids: Optional position IDs of shape (batch_size, seq_len)
            
        Returns:
            Embedded representations of shape (batch_size, seq_len, hidden_size)
        """
        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Add positional encoding
        embeddings = self.positional_encoding(token_embeds, position_ids)
        
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BaselineTransformer(nn.Module):
    """
    Baseline Transformer Model.
    
    A standard transformer implementation suitable for language modeling
    and other sequence-to-sequence tasks.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = TransformerEmbeddings(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            use_bias=config.use_bias
        )
        
        # Output projection (language modeling head)
        if config.tie_word_embeddings:
            # Share weights between input embeddings and output projection
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False
            )
        
        # Initialize weights
        self._init_weights()
        
        # Calculate and log model size
        self._log_model_info()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        self.embeddings._init_weights()
        
        # Initialize transformer layers
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
    
    def _log_model_info(self):
        """Log model size and parameter information."""
        model_info = self.config.get_model_size_info()
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Baseline Transformer Model initialized:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Estimated memory: {model_info['memory_estimate_mb']:.1f} MB")
        logger.info(f"  - Hidden size: {self.config.hidden_size}")
        logger.info(f"  - Number of layers: {self.config.num_hidden_layers}")
        logger.info(f"  - Number of attention heads: {self.config.num_attention_heads}")
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input token embeddings."""
        return self.embeddings.token_embeddings
    
    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        """Set new input token embeddings."""
        self.embeddings.token_embeddings = new_embeddings
    
    def get_output_embeddings(self) -> Optional[nn.Linear]:
        """Get the output projection layer."""
        if self.config.tie_word_embeddings:
            return self.embeddings.token_embeddings
        return self.output_projection
    
    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Set new output projection layer."""
        if not self.config.tie_word_embeddings:
            self.output_projection = new_embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
        return_hidden_states: bool = False,
        output_logits: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the transformer.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            position_ids: Optional position IDs of shape (batch_size, seq_len)
            causal_mask: Whether to apply causal masking for autoregressive generation
            return_attention_weights: Whether to return attention weights from all layers
            return_hidden_states: Whether to return hidden states from all layers
            output_logits: Whether to compute output logits
            
        Returns:
            Dictionary containing:
            - last_hidden_state: Final hidden states
            - logits: Output logits (if output_logits=True)
            - hidden_states: All hidden states (if return_hidden_states=True)
            - attention_weights: All attention weights (if return_attention_weights=True)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = create_attention_mask(input_ids, self.config.pad_token_id)
        
        # Get embeddings
        hidden_states = self.embeddings(input_ids, position_ids)
        
        # Store hidden states and attention weights if requested
        all_hidden_states = [hidden_states] if return_hidden_states else None
        all_attention_weights = [] if return_attention_weights else None
        
        # Pass through transformer layers
        for layer in self.layers:
            if return_attention_weights:
                hidden_states, attention_weights = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    return_attention_weights=True
                )
                all_attention_weights.append(attention_weights)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    return_attention_weights=False
                )
            
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Apply final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Prepare output dictionary
        outputs = {
            "last_hidden_state": hidden_states
        }
        
        # Compute logits if requested
        if output_logits:
            if self.config.tie_word_embeddings:
                # Use tied embeddings for output projection
                logits = F.linear(hidden_states, self.embeddings.token_embeddings.weight)
            else:
                logits = self.output_projection(hidden_states)
            outputs["logits"] = logits
        
        # Add optional outputs
        if return_hidden_states:
            outputs["hidden_states"] = all_hidden_states
        if return_attention_weights:
            outputs["attention_weights"] = all_attention_weights
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate sequences using the transformer model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_length: Maximum length of generated sequences
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token sequences of shape (batch_size, max_length)
        """
        self.eval()
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        batch_size, current_length = input_ids.shape
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # Forward pass
                outputs = self.forward(
                    generated,
                    causal_mask=True,
                    output_logits=True
                )
                
                # Get next token logits
                next_token_logits = outputs["logits"][:, -1, :]  # (batch_size, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or select next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Update finished sequences
                if eos_token_id is not None:
                    finished = finished | (next_tokens == eos_token_id)
                
                # Add next tokens to generated sequence
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
                
                # Stop if all sequences are finished
                if finished.all():
                    break
        
        return generated
    
    def save_pretrained(self, save_directory: Union[str, Path]):
        """Save model and configuration to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.save_pretrained(save_directory)
        
        # Save model state dict
        model_file = save_directory / "pytorch_model.bin"
        torch.save(self.state_dict(), model_file)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> 'BaselineTransformer':
        """Load model from directory."""
        model_path = Path(model_path)
        
        # Load configuration
        config = TransformerConfig.from_pretrained(model_path)
        
        # Create model
        model = cls(config)
        
        # Load state dict
        model_file = model_path / "pytorch_model.bin"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_file}")
        
        return model
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get detailed parameter count information."""
        param_counts = {}
        
        # Embedding parameters
        param_counts['embeddings'] = sum(p.numel() for p in self.embeddings.parameters())
        
        # Transformer layer parameters
        param_counts['transformer_layers'] = sum(p.numel() for p in self.layers.parameters())
        
        # Output projection parameters
        if self.output_projection is not None:
            param_counts['output_projection'] = sum(p.numel() for p in self.output_projection.parameters())
        else:
            param_counts['output_projection'] = 0
        
        # Final layer norm parameters
        param_counts['final_layer_norm'] = sum(p.numel() for p in self.final_layer_norm.parameters())
        
        # Total parameters
        param_counts['total'] = sum(param_counts.values())
        
        return param_counts 