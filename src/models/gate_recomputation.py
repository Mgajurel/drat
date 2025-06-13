"""
Gate-Triggered Recomputation Logic for Memory-Efficient Transformers.

This module implements the integration between differentiable gates and 
recomputation hooks, enabling selective activation storage/recomputation
based on learned gate decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable, List, Tuple, Union
import logging
from contextlib import contextmanager

from .gates import RecomputationGate, GatedTransformerLayer
from .recomputation_hooks import (
    RecomputationHookManager, 
    RecomputationStrategy,
    ActivationInfo,
    CheckpointRecomputationHook
)

logger = logging.getLogger(__name__)


class GateTriggeredRecomputation:
    """
    Manages gate-triggered recomputation decisions and execution.
    
    This class bridges the gap between gate decisions (from RecomputationGate)
    and actual recomputation logic (from RecomputationHookManager).
    """
    
    def __init__(
        self,
        strategy: RecomputationStrategy = RecomputationStrategy.CUSTOM,
        storage_threshold: float = 0.5,
        memory_threshold_mb: float = 1000.0,
        enable_profiling: bool = False
    ):
        """
        Initialize gate-triggered recomputation.
        
        Args:
            strategy: Recomputation strategy to use
            storage_threshold: Gate value threshold for storage decisions
            memory_threshold_mb: Memory threshold for triggering recomputation
            enable_profiling: Whether to enable memory profiling
        """
        self.strategy = strategy
        self.storage_threshold = storage_threshold
        self.memory_threshold_mb = memory_threshold_mb
        self.enable_profiling = enable_profiling
        
        # Initialize hook manager
        self.hook_manager = RecomputationHookManager(
            strategy=strategy,
            memory_threshold_mb=memory_threshold_mb,
            enable_profiling=enable_profiling
        )
        
        # Initialize checkpoint hook for fallback
        self.checkpoint_hook = CheckpointRecomputationHook()
        
        # Storage for gate decisions and activations
        self.gate_decisions: Dict[str, torch.Tensor] = {}
        self.activation_cache: Dict[str, torch.Tensor] = {}
        
        # Statistics tracking
        self.stats = {
            'total_activations': 0,
            'stored_activations': 0,
            'recomputed_activations': 0,
            'memory_saved_mb': 0.0,
            'recomputation_time_ms': 0.0
        }
    
    def make_storage_decision(
        self,
        gate_values: torch.Tensor,
        gate_decisions: torch.Tensor,
        module_name: str,
        layer_idx: int,
        module_type: str
    ) -> bool:
        """
        Make storage decision based on gate values.
        
        Args:
            gate_values: Continuous gate values [0, 1]
            gate_decisions: Binary gate decisions
            module_name: Name of the module
            layer_idx: Layer index
            module_type: Type of module ("attention" or "feedforward")
            
        Returns:
            Boolean decision: True = store, False = recompute
        """
        # Store gate decision for later use
        self.gate_decisions[module_name] = gate_decisions
        
        # Use mean gate value for storage decision
        mean_gate_value = gate_values.mean().item()
        should_store = mean_gate_value > self.storage_threshold
        
        # Log decision
        logger.debug(
            f"Storage decision for {module_name} (layer {layer_idx}, {module_type}): "
            f"gate_value={mean_gate_value:.3f}, store={should_store}"
        )
        
        return should_store
    
    def store_activation(
        self,
        activation: torch.Tensor,
        module_name: str,
        layer_idx: int,
        module_type: str
    ):
        """
        Store activation in cache.
        
        Args:
            activation: Activation tensor to store
            module_name: Name of the module
            layer_idx: Layer index
            module_type: Type of module
        """
        # Store activation
        self.activation_cache[module_name] = activation.detach().clone()
        
        # Update statistics
        self.stats['total_activations'] += 1
        self.stats['stored_activations'] += 1
        
        # Track memory usage
        memory_mb = activation.numel() * activation.element_size() / (1024 * 1024)
        
        logger.debug(
            f"Stored activation for {module_name}: "
            f"shape={activation.shape}, memory={memory_mb:.2f}MB"
        )
    
    def prepare_recomputation(
        self,
        recompute_fn: Callable,
        module_name: str,
        layer_idx: int,
        module_type: str,
        original_activation: torch.Tensor
    ):
        """
        Prepare for recomputation by storing the recomputation function.
        
        Args:
            recompute_fn: Function to recompute the activation
            module_name: Name of the module
            layer_idx: Layer index
            module_type: Type of module
            original_activation: Original activation (for memory calculation)
        """
        # Create activation info for recomputation
        gate_decision = self.gate_decisions.get(module_name, torch.tensor(0.0))
        
        activation_info = ActivationInfo(
            tensor=None,  # Not stored
            recompute_fn=recompute_fn,
            gate_decision=gate_decision,
            layer_idx=layer_idx,
            module_type=module_type,
            is_stored=False,
            memory_saved_mb=original_activation.numel() * original_activation.element_size() / (1024 * 1024)
        )
        
        # Store in hook manager
        self.hook_manager.stored_activations[module_name] = activation_info
        
        # Update statistics
        self.stats['total_activations'] += 1
        self.stats['recomputed_activations'] += 1
        self.stats['memory_saved_mb'] += activation_info.memory_saved_mb
        
        logger.debug(
            f"Prepared recomputation for {module_name}: "
            f"memory_saved={activation_info.memory_saved_mb:.2f}MB"
        )
    
    def recompute_activation(
        self,
        module_name: str,
        fallback_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Recompute activation during backward pass.
        
        Args:
            module_name: Name of the module
            fallback_fn: Fallback recomputation function
            
        Returns:
            Recomputed activation tensor
        """
        import time
        start_time = time.time()
        
        # Check if we have stored activation
        if module_name in self.activation_cache:
            activation = self.activation_cache[module_name]
            logger.debug(f"Retrieved stored activation for {module_name}")
            return activation
        
        # Check if we have recomputation info
        if module_name in self.hook_manager.stored_activations:
            activation_info = self.hook_manager.stored_activations[module_name]
            
            if activation_info.recompute_fn is not None:
                # Use stored recomputation function
                activation = activation_info.recompute_fn()
                logger.debug(f"Recomputed activation for {module_name}")
            elif fallback_fn is not None:
                # Use fallback function
                activation = fallback_fn()
                logger.debug(f"Used fallback recomputation for {module_name}")
            else:
                raise RuntimeError(f"No recomputation method available for {module_name}")
        else:
            if fallback_fn is not None:
                activation = fallback_fn()
                logger.debug(f"Used fallback recomputation for {module_name} (no stored info)")
            else:
                raise RuntimeError(f"No activation or recomputation info for {module_name}")
        
        # Track recomputation time
        recomputation_time = (time.time() - start_time) * 1000  # Convert to ms
        self.stats['recomputation_time_ms'] += recomputation_time
        
        return activation
    
    def clear_cache(self):
        """Clear activation cache and reset statistics."""
        self.activation_cache.clear()
        self.gate_decisions.clear()
        self.hook_manager.clear_stored_activations()
        
        # Reset statistics
        self.stats = {
            'total_activations': 0,
            'stored_activations': 0,
            'recomputed_activations': 0,
            'memory_saved_mb': 0.0,
            'recomputation_time_ms': 0.0
        }
        
        logger.debug("Cleared recomputation cache and statistics")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recomputation statistics."""
        stats = self.stats.copy()
        
        # Add efficiency metrics
        if stats['total_activations'] > 0:
            stats['storage_rate'] = stats['stored_activations'] / stats['total_activations']
            stats['recomputation_rate'] = stats['recomputed_activations'] / stats['total_activations']
        else:
            stats['storage_rate'] = 0.0
            stats['recomputation_rate'] = 0.0
        
        # Add hook manager statistics (but preserve our own activation counts)
        hook_stats = self.hook_manager.get_memory_stats()
        # Only add memory-related stats, not activation counts which we track ourselves
        stats['memory_used_mb'] = hook_stats.get('memory_used_mb', 0.0)
        # Don't override memory_saved_mb as we track it ourselves
        
        if self.enable_profiling:
            profiling_stats = self.hook_manager.get_profiling_stats()
            stats.update(profiling_stats)
        
        return stats


class GatedRecomputationLayer(nn.Module):
    """
    Enhanced gated transformer layer with actual recomputation logic.
    
    This extends GatedTransformerLayer to implement actual storage/recomputation
    decisions based on gate values, rather than just modulating outputs.
    """
    
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        recomputation_strategy: RecomputationStrategy = RecomputationStrategy.CUSTOM,
        storage_threshold: float = 0.5
    ):
        """
        Initialize gated recomputation layer.
        
        Args:
            config: Transformer configuration
            layer_idx: Layer index
            recomputation_strategy: Strategy for recomputation
            storage_threshold: Threshold for storage decisions
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.storage_threshold = storage_threshold
        
        # Initialize base gated layer
        self.gated_layer = GatedTransformerLayer(config, layer_idx)
        
        # Initialize gate-triggered recomputation
        self.gate_recomputation = GateTriggeredRecomputation(
            strategy=recomputation_strategy,
            storage_threshold=storage_threshold,
            enable_profiling=True
        )
        
        # Storage for intermediate activations
        self._stored_attention_input = None
        self._stored_ff_input = None
        self._attention_recompute_fn = None
        self._ff_recompute_fn = None
    
    def _gated_attention_with_recomputation(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply attention with gate-triggered recomputation.
        """
        # Get gate decision
        gate_values, gate_decisions = self.gated_layer.attention_gate(
            hidden_states,
            num_heads=self.config.num_attention_heads,
            training=self.training
        )
        
        # Make storage decision
        module_name = f"attention_layer_{self.layer_idx}"
        should_store = self.gate_recomputation.make_storage_decision(
            gate_values, gate_decisions, module_name, self.layer_idx, "attention"
        )
        
        # Define recomputation function
        def recompute_attention():
            with torch.enable_grad():
                if return_attention_weights:
                    return self.gated_layer.attention(
                        hidden_states, attention_mask, causal_mask, return_attention_weights=True
                    )
                else:
                    return self.gated_layer.attention(
                        hidden_states, attention_mask, causal_mask, return_attention_weights=False
                    )
        
        if should_store:
            # Compute and store attention output
            if return_attention_weights:
                attention_output, attention_weights = recompute_attention()
                self.gate_recomputation.store_activation(
                    attention_output, module_name, self.layer_idx, "attention"
                )
                return attention_output, attention_weights
            else:
                attention_output = recompute_attention()
                self.gate_recomputation.store_activation(
                    attention_output, module_name, self.layer_idx, "attention"
                )
                return attention_output
        else:
            # Prepare for recomputation
            if return_attention_weights:
                # For now, compute once to get the shape, but prepare for recomputation
                attention_output, attention_weights = recompute_attention()
                self.gate_recomputation.prepare_recomputation(
                    recompute_attention, module_name, self.layer_idx, "attention", attention_output
                )
                return attention_output, attention_weights
            else:
                # For now, compute once to get the shape, but prepare for recomputation
                attention_output = recompute_attention()
                self.gate_recomputation.prepare_recomputation(
                    recompute_attention, module_name, self.layer_idx, "attention", attention_output
                )
                return attention_output
    
    def _gated_ff_with_recomputation(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply feed-forward with gate-triggered recomputation.
        """
        # Get gate decision
        gate_values, gate_decisions = self.gated_layer.ff_gate(
            hidden_states,
            training=self.training
        )
        
        # Make storage decision
        module_name = f"ff_layer_{self.layer_idx}"
        should_store = self.gate_recomputation.make_storage_decision(
            gate_values, gate_decisions, module_name, self.layer_idx, "feedforward"
        )
        
        # Define recomputation function
        def recompute_ff():
            with torch.enable_grad():
                return self.gated_layer.feed_forward(hidden_states)
        
        if should_store:
            # Compute and store feed-forward output
            ff_output = recompute_ff()
            self.gate_recomputation.store_activation(
                ff_output, module_name, self.layer_idx, "feedforward"
            )
            return ff_output
        else:
            # Prepare for recomputation
            # For now, compute once to get the shape, but prepare for recomputation
            ff_output = recompute_ff()
            self.gate_recomputation.prepare_recomputation(
                recompute_ff, module_name, self.layer_idx, "feedforward", ff_output
            )
            return ff_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        return_attention_weights: bool = False,
        return_gate_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with gate-triggered recomputation.
        """
        # Store input for residual connection
        residual = hidden_states
        
        # Pre-layer norm or post-layer norm
        if self.config.layer_norm_type == "pre":
            # Pre-layer norm: LayerNorm -> Attention -> Residual
            hidden_states = self.gated_layer.attention_layer_norm(hidden_states)
            
            if return_attention_weights:
                attention_output, attention_weights = self._gated_attention_with_recomputation(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=True
                )
            else:
                attention_output = self._gated_attention_with_recomputation(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=False
                )
            
            # Apply dropout and residual connection
            hidden_states = residual + self.gated_layer.dropout(attention_output)
            
            # Feed-forward with pre-layer norm
            residual = hidden_states
            hidden_states = self.gated_layer.ff_layer_norm(hidden_states)
            ff_output = self._gated_ff_with_recomputation(hidden_states)
            hidden_states = residual + self.gated_layer.dropout(ff_output)
            
        else:  # post-layer norm
            # Post-layer norm: Attention -> Residual -> LayerNorm
            if return_attention_weights:
                attention_output, attention_weights = self._gated_attention_with_recomputation(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=True
                )
            else:
                attention_output = self._gated_attention_with_recomputation(
                    hidden_states, attention_mask, causal_mask, return_attention_weights=False
                )
            
            # Apply dropout, residual connection, and layer norm
            hidden_states = self.gated_layer.attention_layer_norm(residual + self.gated_layer.dropout(attention_output))
            
            # Feed-forward with post-layer norm
            residual = hidden_states
            ff_output = self._gated_ff_with_recomputation(hidden_states)
            hidden_states = self.gated_layer.ff_layer_norm(residual + self.gated_layer.dropout(ff_output))
        
        # Prepare return values
        if return_gate_info:
            gate_info = self.gated_layer.get_gate_statistics()
            recomputation_stats = self.gate_recomputation.get_statistics()
            gate_info.update(recomputation_stats)
            
            if return_attention_weights:
                return hidden_states, attention_weights, gate_info
            return hidden_states, gate_info
        
        if return_attention_weights:
            return hidden_states, attention_weights
        return hidden_states
    
    def clear_cache(self):
        """Clear recomputation cache."""
        self.gate_recomputation.clear_cache()
        self.gated_layer.clear_stored_activations()
    
    def get_recomputation_statistics(self) -> Dict[str, Any]:
        """Get recomputation statistics."""
        return self.gate_recomputation.get_statistics()


@contextmanager
def gate_recomputation_context(
    layers: List[GatedRecomputationLayer],
    enable_recomputation: bool = True
):
    """
    Context manager for gate-triggered recomputation across multiple layers.
    
    Args:
        layers: List of gated recomputation layers
        enable_recomputation: Whether to enable recomputation
    """
    if enable_recomputation:
        try:
            yield layers
        finally:
            # Clear caches after use
            for layer in layers:
                layer.clear_cache()
    else:
        # Disable recomputation
        yield layers


def create_gated_recomputation_model(
    config,
    num_layers: int,
    recomputation_strategy: RecomputationStrategy = RecomputationStrategy.CUSTOM,
    storage_threshold: float = 0.5
) -> nn.ModuleList:
    """
    Create a list of gated recomputation layers.
    
    Args:
        config: Transformer configuration
        num_layers: Number of layers
        recomputation_strategy: Strategy for recomputation
        storage_threshold: Threshold for storage decisions
        
    Returns:
        ModuleList of gated recomputation layers
    """
    layers = nn.ModuleList([
        GatedRecomputationLayer(
            config=config,
            layer_idx=i,
            recomputation_strategy=recomputation_strategy,
            storage_threshold=storage_threshold
        )
        for i in range(num_layers)
    ])
    
    return layers 