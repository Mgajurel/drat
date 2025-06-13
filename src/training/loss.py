"""
Resource-Aware Loss Function for Memory-Efficient Transformers.

Implements loss function: L_total = L_task + λ Σ(g·C_mem + (1−g)·C_recomp)
where:
- L_task: Task-specific loss (e.g., cross-entropy)
- λ: Resource penalty weight
- g: Gate values (continuous, differentiable)
- C_mem: Memory cost for storing activations
- C_recomp: Recomputation cost for discarded activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostMetrics:
    """Container for cost metrics and statistics."""
    memory_cost: float
    recomputation_cost: float
    total_resource_cost: float
    gate_statistics: Dict[str, Any]
    layer_costs: List[Dict[str, float]]


class ResourceAwareLoss(nn.Module):
    """
    Resource-aware loss function that balances task accuracy with memory/computation costs.
    
    The loss function is defined as:
    L_total = L_task + λ * Σ_layers(g_att * C_mem_att + (1-g_att) * C_recomp_att + 
                                    g_ff * C_mem_ff + (1-g_ff) * C_recomp_ff)
    
    Where:
    - L_task: Task-specific loss (cross-entropy, etc.)
    - λ: Resource penalty weight (lambda_resource)
    - g_att, g_ff: Gate values for attention and feed-forward components
    - C_mem: Memory cost for storing activations
    - C_recomp: Recomputation cost for discarded activations
    """
    
    def __init__(
        self,
        task_loss_fn: Optional[nn.Module] = None,
        lambda_resource: float = 0.01,
        memory_cost_base: float = 1.0,
        recomputation_cost_base: float = 2.0,
        cost_model: str = "uniform",
        layer_weights: Optional[List[float]] = None,
        normalize_costs: bool = True,
        ignore_index: int = -100
    ):
        """
        Initialize resource-aware loss function.
        
        Args:
            task_loss_fn: Task-specific loss function (default: CrossEntropyLoss)
            lambda_resource: Weight for resource penalty term
            memory_cost_base: Base cost for storing activations in memory
            recomputation_cost_base: Base cost for recomputing activations
            cost_model: Cost calculation model ("uniform", "layer_weighted", "activation_size")
            layer_weights: Custom weights per layer (for layer_weighted model)
            normalize_costs: Whether to normalize costs by number of layers
            ignore_index: Index to ignore in task loss calculation
        """
        super().__init__()
        
        # Task loss function
        if task_loss_fn is None:
            self.task_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.task_loss_fn = task_loss_fn
        
        # Resource penalty parameters
        self.lambda_resource = lambda_resource
        self.memory_cost_base = memory_cost_base
        self.recomputation_cost_base = recomputation_cost_base
        self.cost_model = cost_model
        self.layer_weights = layer_weights
        self.normalize_costs = normalize_costs
        
        # Validate cost model
        valid_models = ["uniform", "layer_weighted", "activation_size"]
        if cost_model not in valid_models:
            raise ValueError(f"cost_model must be one of {valid_models}, got {cost_model}")
    
    def compute_task_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute task-specific loss."""
        if logits.dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
        
        return self.task_loss_fn(logits, targets)
    
    def compute_resource_costs(
        self,
        gate_statistics: Dict[str, Any],
        hidden_size: int,
        seq_len: int,
        batch_size: int = 1
    ) -> CostMetrics:
        """
        Compute memory and recomputation costs based on gate statistics.
        
        Args:
            gate_statistics: Gate statistics from model forward pass
            hidden_size: Hidden dimension size
            seq_len: Sequence length
            batch_size: Batch size
            
        Returns:
            CostMetrics object with detailed cost breakdown
        """
        layer_stats = gate_statistics.get('layer_stats', [])
        num_layers = len(layer_stats)
        
        if num_layers == 0:
            # No gate statistics available
            return CostMetrics(
                memory_cost=0.0,
                recomputation_cost=0.0,
                total_resource_cost=0.0,
                gate_statistics=gate_statistics,
                layer_costs=[]
            )
        
        total_memory_cost = 0.0
        total_recomputation_cost = 0.0
        layer_costs = []
        
        for layer_idx, layer_stat in enumerate(layer_stats):
            # Get gate probabilities (continuous values for differentiability)
            g_att = layer_stat.get('attention_gate_prob', 0.5)
            g_ff = layer_stat.get('ff_gate_prob', 0.5)
            
            # Compute base costs based on activation sizes
            if self.cost_model == "activation_size":
                # Cost proportional to activation tensor size
                attention_size = batch_size * seq_len * hidden_size
                ff_size = batch_size * seq_len * hidden_size  # Simplified
                
                att_mem_cost = g_att * attention_size * self.memory_cost_base
                att_recomp_cost = (1 - g_att) * attention_size * self.recomputation_cost_base
                
                ff_mem_cost = g_ff * ff_size * self.memory_cost_base
                ff_recomp_cost = (1 - g_ff) * ff_size * self.recomputation_cost_base
                
            elif self.cost_model == "layer_weighted":
                # Apply layer-specific weights
                if self.layer_weights and layer_idx < len(self.layer_weights):
                    weight = self.layer_weights[layer_idx]
                else:
                    # Default: deeper layers have higher cost
                    weight = 1.0 + (layer_idx / num_layers)
                
                att_mem_cost = g_att * weight * self.memory_cost_base
                att_recomp_cost = (1 - g_att) * weight * self.recomputation_cost_base
                
                ff_mem_cost = g_ff * weight * self.memory_cost_base
                ff_recomp_cost = (1 - g_ff) * weight * self.recomputation_cost_base
                
            else:  # uniform
                # Uniform cost across all layers
                att_mem_cost = g_att * self.memory_cost_base
                att_recomp_cost = (1 - g_att) * self.recomputation_cost_base
                
                ff_mem_cost = g_ff * self.memory_cost_base
                ff_recomp_cost = (1 - g_ff) * self.recomputation_cost_base
            
            # Layer total costs
            layer_memory_cost = att_mem_cost + ff_mem_cost
            layer_recomp_cost = att_recomp_cost + ff_recomp_cost
            layer_total_cost = layer_memory_cost + layer_recomp_cost
            
            # Store layer-specific costs
            layer_costs.append({
                'layer_idx': layer_idx,
                'attention_gate_prob': g_att,
                'ff_gate_prob': g_ff,
                'memory_cost': float(layer_memory_cost),
                'recomputation_cost': float(layer_recomp_cost),
                'total_cost': float(layer_total_cost)
            })
            
            # Accumulate total costs
            total_memory_cost += layer_memory_cost
            total_recomputation_cost += layer_recomp_cost
        
        # Normalize costs if requested
        if self.normalize_costs and num_layers > 0:
            total_memory_cost /= num_layers
            total_recomputation_cost /= num_layers
        
        total_resource_cost = total_memory_cost + total_recomputation_cost
        
        return CostMetrics(
            memory_cost=float(total_memory_cost),
            recomputation_cost=float(total_recomputation_cost),
            total_resource_cost=float(total_resource_cost),
            gate_statistics=gate_statistics,
            layer_costs=layer_costs
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gate_statistics: Optional[Dict[str, Any]] = None,
        hidden_size: Optional[int] = None,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute resource-aware loss.
        
        Args:
            logits: Model output logits
            targets: Target labels
            gate_statistics: Gate statistics from model forward pass
            hidden_size: Hidden dimension size (for cost calculation)
            return_metrics: Whether to return detailed metrics
            
        Returns:
            Loss tensor or tuple of (loss, metrics) if return_metrics=True
        """
        # Compute task loss
        task_loss = self.compute_task_loss(logits, targets)
        
        # Initialize resource loss
        resource_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        cost_metrics = None
        
        # Compute resource costs if gate statistics are provided
        if gate_statistics is not None and self.lambda_resource > 0:
            # Infer dimensions from logits if not provided
            if hidden_size is None:
                hidden_size = logits.size(-1)  # Assume vocab_size ≈ hidden_size for estimation
            
            batch_size, seq_len = logits.shape[:2] if logits.dim() == 3 else (logits.shape[0] // targets.numel(), targets.numel())
            
            # Compute cost metrics
            cost_metrics = self.compute_resource_costs(
                gate_statistics, hidden_size, seq_len, batch_size
            )
            
            # Convert to tensor for gradient computation
            resource_loss = torch.tensor(
                cost_metrics.total_resource_cost,
                device=logits.device,
                requires_grad=True
            )
        
        # Combine losses
        total_loss = task_loss + self.lambda_resource * resource_loss
        
        if return_metrics:
            metrics = {
                'task_loss': float(task_loss.item()),
                'resource_loss': float(resource_loss.item()),
                'total_loss': float(total_loss.item()),
                'lambda_resource': self.lambda_resource,
                'cost_metrics': cost_metrics
            }
            return total_loss, metrics
        
        return total_loss
    
    def set_lambda(self, lambda_resource: float):
        """Update the resource penalty weight."""
        self.lambda_resource = lambda_resource
    
    def get_config(self) -> Dict[str, Any]:
        """Get loss function configuration."""
        return {
            'lambda_resource': self.lambda_resource,
            'memory_cost_base': self.memory_cost_base,
            'recomputation_cost_base': self.recomputation_cost_base,
            'cost_model': self.cost_model,
            'layer_weights': self.layer_weights,
            'normalize_costs': self.normalize_costs
        } 