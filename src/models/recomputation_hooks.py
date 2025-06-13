"""
Recomputation Hook Architecture for Memory-Efficient Transformers.

This module implements PyTorch hooks and custom mechanisms for intercepting
and managing activation recomputation based on gate decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from typing import Optional, Dict, Any, Callable, List, Tuple, Union
import weakref
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RecomputationStrategy(Enum):
    """Strategies for recomputation."""
    CHECKPOINT = "checkpoint"  # Use torch.utils.checkpoint
    CUSTOM = "custom"          # Use custom recomputation logic
    HYBRID = "hybrid"          # Mix of both strategies


@dataclass
class ActivationInfo:
    """Information about stored/discarded activations."""
    tensor: Optional[torch.Tensor]
    recompute_fn: Optional[Callable]
    gate_decision: torch.Tensor
    layer_idx: int
    module_type: str  # "attention" or "feedforward"
    is_stored: bool
    memory_saved_mb: float = 0.0


class RecomputationHookManager:
    """
    Manager for recomputation hooks across the transformer model.
    
    Coordinates hook registration, activation storage/recomputation,
    and memory management based on gate decisions.
    """
    
    def __init__(
        self,
        strategy: RecomputationStrategy = RecomputationStrategy.CUSTOM,
        memory_threshold_mb: float = 1000.0,
        enable_profiling: bool = False
    ):
        """
        Initialize the hook manager.
        
        Args:
            strategy: Recomputation strategy to use
            memory_threshold_mb: Memory threshold for triggering recomputation
            enable_profiling: Whether to enable memory profiling
        """
        self.strategy = strategy
        self.memory_threshold_mb = memory_threshold_mb
        self.enable_profiling = enable_profiling
        
        # Storage for activations and recomputation functions
        self.stored_activations: Dict[str, ActivationInfo] = {}
        self.recompute_functions: Dict[str, Callable] = {}
        
        # Hook handles for cleanup
        self.forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Memory tracking
        self.memory_saved_mb = 0.0
        self.memory_used_mb = 0.0
        
        # Profiling data
        self.profiling_data = {
            'recomputation_count': 0,
            'memory_savings': [],
            'recomputation_times': []
        }
    
    def register_module(
        self,
        module: nn.Module,
        module_name: str,
        layer_idx: int,
        module_type: str,
        gate_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ):
        """
        Register a module for recomputation hooks.
        
        Args:
            module: The PyTorch module to hook
            module_name: Unique name for the module
            layer_idx: Layer index in the transformer
            module_type: Type of module ("attention" or "feedforward")
            gate_fn: Function that returns (gate_values, gate_decisions)
        """
        # Create forward hook for activation capture
        def forward_hook(module, input, output):
            return self._forward_hook(
                module, input, output, module_name, layer_idx, module_type, gate_fn
            )
        
        # Create backward hook for recomputation
        def backward_hook(module, grad_input, grad_output):
            return self._backward_hook(
                module, grad_input, grad_output, module_name, layer_idx, module_type
            )
        
        # Register hooks
        forward_handle = module.register_forward_hook(forward_hook)
        backward_handle = module.register_full_backward_hook(backward_hook)
        
        self.forward_hooks.append(forward_handle)
        self.backward_hooks.append(backward_handle)
        
        logger.debug(f"Registered hooks for {module_name} (layer {layer_idx}, type {module_type})")
    
    def _forward_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
        module_name: str,
        layer_idx: int,
        module_type: str,
        gate_fn: Callable
    ) -> torch.Tensor:
        """
        Forward hook for capturing activations and making storage decisions.
        """
        # Get gate decision for this activation
        hidden_states = input[0] if isinstance(input, tuple) else input
        gate_values, gate_decisions = gate_fn(hidden_states)
        
        # Determine storage decision (simplified: use mean gate value)
        storage_prob = gate_values.mean().item()
        should_store = storage_prob > 0.5
        
        if should_store:
            # Store the activation
            activation_info = ActivationInfo(
                tensor=output.detach().clone(),
                recompute_fn=None,
                gate_decision=gate_decisions,
                layer_idx=layer_idx,
                module_type=module_type,
                is_stored=True,
                memory_saved_mb=0.0
            )
            self.stored_activations[module_name] = activation_info
            
            # Track memory usage
            memory_mb = output.numel() * output.element_size() / (1024 * 1024)
            self.memory_used_mb += memory_mb
            
        else:
            # Don't store, prepare for recomputation
            def recompute_fn():
                """Recomputation function for this activation."""
                with torch.enable_grad():
                    # Re-run the forward pass
                    recomputed_output = module(*input)
                    return recomputed_output
            
            activation_info = ActivationInfo(
                tensor=None,
                recompute_fn=recompute_fn,
                gate_decision=gate_decisions,
                layer_idx=layer_idx,
                module_type=module_type,
                is_stored=False,
                memory_saved_mb=output.numel() * output.element_size() / (1024 * 1024)
            )
            self.stored_activations[module_name] = activation_info
            
            # Track memory savings
            self.memory_saved_mb += activation_info.memory_saved_mb
        
        return output
    
    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
        module_name: str,
        layer_idx: int,
        module_type: str
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Backward hook for handling recomputation during backpropagation.
        """
        if module_name not in self.stored_activations:
            return grad_input
        
        activation_info = self.stored_activations[module_name]
        
        if not activation_info.is_stored and activation_info.recompute_fn is not None:
            # Need to recompute the activation
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            # Perform recomputation
            try:
                recomputed_activation = activation_info.recompute_fn()
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    recompute_time = start_time.elapsed_time(end_time)
                    self.profiling_data['recomputation_times'].append(recompute_time)
                
                self.profiling_data['recomputation_count'] += 1
                
                logger.debug(f"Recomputed activation for {module_name} (layer {layer_idx})")
                
            except Exception as e:
                logger.error(f"Recomputation failed for {module_name}: {e}")
                # Fall back to stored activation if available
                if activation_info.tensor is not None:
                    recomputed_activation = activation_info.tensor
                else:
                    raise e
        
        return grad_input
    
    def clear_stored_activations(self):
        """Clear all stored activations to free memory."""
        memory_freed = 0.0
        for activation_info in self.stored_activations.values():
            if activation_info.tensor is not None:
                memory_freed += activation_info.tensor.numel() * activation_info.tensor.element_size() / (1024 * 1024)
        
        self.stored_activations.clear()
        self.memory_used_mb = 0.0
        
        logger.debug(f"Cleared stored activations, freed {memory_freed:.2f} MB")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return {
            'memory_used_mb': self.memory_used_mb,
            'memory_saved_mb': self.memory_saved_mb,
            'total_activations': len(self.stored_activations),
            'stored_activations': sum(1 for info in self.stored_activations.values() if info.is_stored),
            'recompute_activations': sum(1 for info in self.stored_activations.values() if not info.is_stored)
        }
    
    def get_profiling_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        if not self.enable_profiling:
            return {}
        
        return {
            'recomputation_count': self.profiling_data['recomputation_count'],
            'avg_recomputation_time_ms': (
                sum(self.profiling_data['recomputation_times']) / 
                len(self.profiling_data['recomputation_times'])
                if self.profiling_data['recomputation_times'] else 0.0
            ),
            'total_memory_saved_mb': self.memory_saved_mb,
            'memory_efficiency': (
                self.memory_saved_mb / (self.memory_saved_mb + self.memory_used_mb)
                if (self.memory_saved_mb + self.memory_used_mb) > 0 else 0.0
            )
        }
    
    def cleanup(self):
        """Remove all hooks and clear memory."""
        # Remove forward hooks
        for handle in self.forward_hooks:
            handle.remove()
        self.forward_hooks.clear()
        
        # Remove backward hooks
        for handle in self.backward_hooks:
            handle.remove()
        self.backward_hooks.clear()
        
        # Clear stored activations
        self.clear_stored_activations()
        
        logger.debug("Cleaned up all recomputation hooks")


class CheckpointRecomputationHook:
    """
    Recomputation hook using torch.utils.checkpoint.
    
    This provides a simpler interface using PyTorch's built-in
    gradient checkpointing functionality.
    """
    
    def __init__(self, preserve_rng_state: bool = True):
        """
        Initialize checkpoint-based recomputation.
        
        Args:
            preserve_rng_state: Whether to preserve RNG state during recomputation
        """
        self.preserve_rng_state = preserve_rng_state
        self.checkpoint_segments: List[Callable] = []
    
    def checkpoint_function(
        self,
        function: Callable,
        *args,
        use_reentrant: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply gradient checkpointing to a function.
        
        Args:
            function: Function to checkpoint
            *args: Arguments to the function
            use_reentrant: Whether to use reentrant checkpointing
            **kwargs: Keyword arguments to the function
            
        Returns:
            Output of the checkpointed function
        """
        return checkpoint(
            function,
            *args,
            use_reentrant=use_reentrant,
            preserve_rng_state=self.preserve_rng_state,
            **kwargs
        )
    
    def checkpoint_sequential(
        self,
        functions: List[Callable],
        segments: int,
        input_tensor: torch.Tensor,
        use_reentrant: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply sequential checkpointing to a list of functions.
        
        Args:
            functions: List of functions to apply sequentially
            segments: Number of checkpoint segments
            input_tensor: Input tensor
            use_reentrant: Whether to use reentrant checkpointing (False recommended)
            **kwargs: Additional arguments
            
        Returns:
            Output tensor after applying all functions
        """
        return checkpoint_sequential(
            functions,
            segments,
            input_tensor,
            use_reentrant=use_reentrant,
            preserve_rng_state=self.preserve_rng_state,
            **kwargs
        )


@contextmanager
def recomputation_context(
    hook_manager: RecomputationHookManager,
    enable_recomputation: bool = True
):
    """
    Context manager for recomputation operations.
    
    Args:
        hook_manager: The hook manager to use
        enable_recomputation: Whether to enable recomputation in this context
    """
    if enable_recomputation:
        try:
            yield hook_manager
        finally:
            # Cleanup is handled by the hook manager
            pass
    else:
        # Disable recomputation in this context
        yield None


def create_recomputation_wrapper(
    module: nn.Module,
    gate_fn: Callable,
    strategy: RecomputationStrategy = RecomputationStrategy.CUSTOM
) -> Callable:
    """
    Create a recomputation wrapper for a module.
    
    Args:
        module: Module to wrap
        gate_fn: Gate function for storage decisions
        strategy: Recomputation strategy to use
        
    Returns:
        Wrapped function that handles recomputation
    """
    if strategy == RecomputationStrategy.CHECKPOINT:
        checkpoint_hook = CheckpointRecomputationHook()
        
        def wrapped_forward(*args, **kwargs):
            return checkpoint_hook.checkpoint_function(module, *args, **kwargs)
        
        return wrapped_forward
    
    elif strategy == RecomputationStrategy.CUSTOM:
        def wrapped_forward(*args, **kwargs):
            # Get gate decision
            input_tensor = args[0] if args else None
            if input_tensor is not None:
                gate_values, gate_decisions = gate_fn(input_tensor)
                storage_prob = gate_values.mean().item()
                
                if storage_prob > 0.5:
                    # Store activation normally
                    return module(*args, **kwargs)
                else:
                    # Use checkpointing for memory efficiency
                    checkpoint_hook = CheckpointRecomputationHook()
                    return checkpoint_hook.checkpoint_function(module, *args, **kwargs)
            else:
                return module(*args, **kwargs)
        
        return wrapped_forward
    
    else:  # HYBRID
        # Combine both strategies based on memory pressure
        def wrapped_forward(*args, **kwargs):
            # Simple heuristic: use checkpointing for larger tensors
            input_tensor = args[0] if args else None
            if input_tensor is not None and input_tensor.numel() > 10000:
                checkpoint_hook = CheckpointRecomputationHook()
                return checkpoint_hook.checkpoint_function(module, *args, **kwargs)
            else:
                return module(*args, **kwargs)
        
        return wrapped_forward 