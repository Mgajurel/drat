"""
Training Script with Comprehensive Logging for Resource-Aware Transformers.

This module provides a complete training pipeline with memory monitoring,
cost tracking, and gate activation logging for transformer models with
recomputation gates.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np

# Logging integrations
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("Weights & Biases not available. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available. Install with: pip install tensorboard")

# Local imports
from ..models.config import TransformerConfig
from ..models.transformer import BaselineTransformer
from ..models.gated_transformer import GatedTransformer
from .loss import ResourceAwareLoss, CostMetrics
from .cost_tracker import CostTracker, BatchCostMetrics
from ..utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Model configuration
    model_config: Optional[TransformerConfig] = None
    use_gated_model: bool = True
    use_resource_aware_loss: bool = True
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Training schedule
    num_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    
    # Loss configuration
    lambda_resource: float = 0.01
    cost_model: str = "uniform"
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging and evaluation
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_top_k: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Logging backends
    use_wandb: bool = True
    use_tensorboard: bool = True
    wandb_project: str = "resource-aware-transformer"
    wandb_run_name: Optional[str] = None
    
    # Cost tracking
    track_memory: bool = True
    track_computation_time: bool = True
    track_gate_statistics: bool = True
    memory_log_interval: int = 100
    
    # Device configuration
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Debugging
    debug_mode: bool = False
    profile_steps: int = 0
    
    def __post_init__(self):
        """Validate and setup configuration."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate logging backends
        if self.use_wandb and not WANDB_AVAILABLE:
            logger.warning("W&B requested but not available. Disabling W&B logging.")
            self.use_wandb = False
        
        if self.use_tensorboard and not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard requested but not available. Disabling TensorBoard logging.")
            self.use_tensorboard = False


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int = 0
    epoch: int = 0
    
    # Loss metrics
    total_loss: float = 0.0
    task_loss: float = 0.0
    resource_loss: float = 0.0
    
    # Performance metrics
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    learning_rate: float = 0.0
    
    # Memory metrics
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Cost metrics
    memory_cost: float = 0.0
    recomputation_cost: float = 0.0
    total_resource_cost: float = 0.0
    
    # Gate statistics
    avg_attention_gate_prob: float = 0.0
    avg_ff_gate_prob: float = 0.0
    gate_entropy: float = 0.0
    
    # Training dynamics
    gradient_norm: float = 0.0
    gradient_scale: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss/total": self.total_loss,
            "loss/task": self.task_loss,
            "loss/resource": self.resource_loss,
            "performance/tokens_per_second": self.tokens_per_second,
            "performance/samples_per_second": self.samples_per_second,
            "performance/learning_rate": self.learning_rate,
            "memory/allocated_mb": self.memory_allocated_mb,
            "memory/reserved_mb": self.memory_reserved_mb,
            "memory/peak_mb": self.peak_memory_mb,
            "cost/memory": self.memory_cost,
            "cost/recomputation": self.recomputation_cost,
            "cost/total_resource": self.total_resource_cost,
            "gates/avg_attention_prob": self.avg_attention_gate_prob,
            "gates/avg_ff_prob": self.avg_ff_gate_prob,
            "gates/entropy": self.gate_entropy,
            "training/gradient_norm": self.gradient_norm,
            "training/gradient_scale": self.gradient_scale,
        }


class ResourceAwareTrainer:
    """Comprehensive trainer for resource-aware transformer models."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[nn.Module] = None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = model or self._create_model()
        self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = self._create_loss_function()
        
        # Initialize optimizer and scheduler
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler or self._create_scheduler()
        
        # Initialize data loaders
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Initialize cost tracking
        self.cost_tracker = CostTracker(device=self.device, track_flops=True)
        
        # Initialize logging
        self.logger_backends = self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.best_model_path = None
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Metrics tracking
        self.metrics_history = []
        self.step_times = []
        
        logger.info(f"Initialized ResourceAwareTrainer on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_model(self) -> nn.Module:
        """Create the transformer model."""
        if self.config.model_config is None:
            raise ValueError("model_config is required when model is not provided")
        
        if self.config.use_gated_model:
            model = GatedTransformer(self.config.model_config)
        else:
            model = BaselineTransformer(self.config.model_config)
        
        return model
    
    def _create_loss_function(self) -> ResourceAwareLoss:
        """Create the resource-aware loss function."""
        if self.config.use_resource_aware_loss:
            return ResourceAwareLoss(
                lambda_resource=self.config.lambda_resource,
                cost_model=self.config.cost_model,
                use_real_time_costs=self.config.track_memory
            )
        else:
            return ResourceAwareLoss(lambda_resource=0.0)  # Standard cross-entropy
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps or (self.config.num_epochs * 1000),
            eta_min=self.config.learning_rate * 0.1
        )
    
    def _setup_logging(self) -> Dict[str, Any]:
        """Setup logging backends."""
        backends = {}
        
        # Setup W&B
        if self.config.use_wandb:
            run_name = self.config.wandb_run_name or f"run_{int(time.time())}"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config=self.config.__dict__,
            )
            backends['wandb'] = wandb
            logger.info("Initialized Weights & Biases logging")
        
        # Setup TensorBoard
        if self.config.use_tensorboard:
            tb_log_dir = Path(self.config.output_dir) / "tensorboard"
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            backends['tensorboard'] = SummaryWriter(tb_log_dir)
            logger.info(f"Initialized TensorBoard logging at {tb_log_dir}")
        
        return backends
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        self.model.train()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train one epoch
            epoch_metrics = self._train_epoch()
            
            # Evaluate if needed
            if self.eval_dataloader and (epoch + 1) % self.config.eval_interval == 0:
                eval_metrics = self._evaluate()
                self._log_metrics(eval_metrics, prefix="eval")
                
                # Save best model
                if eval_metrics.total_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics.total_loss
                    self.best_model_path = self.save_checkpoint(f"best_model_epoch_{epoch}")
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            logger.info(f"Epoch {epoch + 1} metrics: {epoch_metrics.to_dict()}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}")
        
        # Cleanup
        self._cleanup_logging()
        logger.info("Training completed!")
    
    def _train_epoch(self) -> TrainingMetrics:
        """Train for one epoch."""
        if not self.train_dataloader:
            raise ValueError("train_dataloader is required for training")
        
        accumulated_metrics = TrainingMetrics()
        batch_count = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            step_start_time = time.time()
            
            # Start batch cost tracking
            if self.config.track_memory:
                self.cost_tracker.start_batch(self.global_step)
            
            # Train step
            metrics = self._train_step(batch, batch_idx)
            
            # End batch cost tracking
            if self.config.track_memory:
                batch_costs = self.cost_tracker.end_batch()
                self._update_metrics_with_costs(metrics, batch_costs)
            
            # Accumulate metrics
            self._accumulate_metrics(accumulated_metrics, metrics)
            batch_count += 1
            
            # Log step
            step_time = time.time() - step_start_time
            self.step_times.append(step_time)
            
            if self.global_step % self.config.log_interval == 0:
                self._log_metrics(metrics)
                logger.info(f"Step {self.global_step}: loss={metrics.total_loss:.4f}")
            
            self.global_step += 1
            
            # Check max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        # Average accumulated metrics
        self._average_metrics(accumulated_metrics, batch_count)
        return accumulated_metrics
    
    def _train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> TrainingMetrics:
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with automatic mixed precision
        with autocast(device_type=self.device.type, enabled=self.config.use_amp):
            # Model forward pass
            if self.config.use_gated_model:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    return_gate_info=True
                )
                logits = outputs['logits']
                gate_statistics = outputs.get('gate_statistics', {})
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask')
                )
                logits = outputs['logits']
                gate_statistics = {}
            
            # Compute loss
            targets = batch.get('labels', batch['input_ids'])
            loss_output = self.loss_fn(
                logits=logits,
                targets=targets,
                gate_statistics=gate_statistics,
                hidden_size=self.config.model_config.hidden_size,
                return_metrics=True,
                batch_idx=batch_idx
            )
            
            if isinstance(loss_output, tuple):
                loss, cost_metrics = loss_output
            else:
                loss, cost_metrics = loss_output, None
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        else:
            grad_norm = 0.0
        
        # Create metrics
        metrics = TrainingMetrics(
            step=self.global_step,
            epoch=self.current_epoch,
            total_loss=loss.item() * self.config.gradient_accumulation_steps,
            learning_rate=self.scheduler.get_last_lr()[0],
            gradient_norm=grad_norm,
            gradient_scale=self.scaler.get_scale() if self.scaler else 1.0
        )
        
        # Add cost metrics if available
        if cost_metrics:
            # Handle both dictionary and CostMetrics object formats
            if isinstance(cost_metrics, dict):
                # Extract CostMetrics from dictionary if present
                actual_cost_metrics = cost_metrics.get('cost_metrics')
                if actual_cost_metrics:
                    metrics.task_loss = cost_metrics.get('task_loss', 0.0)
                    metrics.resource_loss = cost_metrics.get('resource_loss', 0.0)
                    metrics.memory_cost = actual_cost_metrics.memory_cost
                    metrics.recomputation_cost = actual_cost_metrics.recomputation_cost
                    metrics.total_resource_cost = actual_cost_metrics.total_resource_cost
                else:
                    # Fallback to dictionary values
                    metrics.task_loss = cost_metrics.get('task_loss', 0.0)
                    metrics.resource_loss = cost_metrics.get('resource_loss', 0.0)
            else:
                # Handle CostMetrics object directly
                metrics.task_loss = cost_metrics.total_resource_cost - cost_metrics.memory_cost - cost_metrics.recomputation_cost
                metrics.resource_loss = cost_metrics.memory_cost + cost_metrics.recomputation_cost
                metrics.memory_cost = cost_metrics.memory_cost
                metrics.recomputation_cost = cost_metrics.recomputation_cost
                metrics.total_resource_cost = cost_metrics.total_resource_cost
        
        # Add gate statistics if available
        if gate_statistics:
            self._update_metrics_with_gate_stats(metrics, gate_statistics)
        
        # Add memory statistics
        if self.config.track_memory:
            self._update_metrics_with_memory_stats(metrics)
        
        return metrics
    
    def _evaluate(self) -> TrainingMetrics:
        """Evaluate the model."""
        if not self.eval_dataloader:
            raise ValueError("eval_dataloader is required for evaluation")
        
        self.model.eval()
        accumulated_metrics = TrainingMetrics()
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.config.use_gated_model:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        return_gate_info=True
                    )
                    logits = outputs['logits']
                    gate_statistics = outputs.get('gate_statistics', {})
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask')
                    )
                    logits = outputs['logits']
                    gate_statistics = {}
                
                # Compute loss
                targets = batch.get('labels', batch['input_ids'])
                loss_output = self.loss_fn(
                    logits=logits,
                    targets=targets,
                    gate_statistics=gate_statistics,
                    hidden_size=self.config.model_config.hidden_size,
                    return_metrics=True
                )
                
                if isinstance(loss_output, tuple):
                    loss, cost_metrics = loss_output
                else:
                    loss, cost_metrics = loss_output, None
                
                # Create metrics
                metrics = TrainingMetrics(
                    step=self.global_step,
                    epoch=self.current_epoch,
                    total_loss=loss.item()
                )
                
                # Add cost and gate metrics
                if cost_metrics:
                    # Handle both dictionary and CostMetrics object formats
                    if isinstance(cost_metrics, dict):
                        actual_cost_metrics = cost_metrics.get('cost_metrics')
                        if actual_cost_metrics:
                            metrics.memory_cost = actual_cost_metrics.memory_cost
                            metrics.recomputation_cost = actual_cost_metrics.recomputation_cost
                            metrics.total_resource_cost = actual_cost_metrics.total_resource_cost
                    else:
                        metrics.memory_cost = cost_metrics.memory_cost
                        metrics.recomputation_cost = cost_metrics.recomputation_cost
                        metrics.total_resource_cost = cost_metrics.total_resource_cost
                
                if gate_statistics:
                    self._update_metrics_with_gate_stats(metrics, gate_statistics)
                
                self._accumulate_metrics(accumulated_metrics, metrics)
                batch_count += 1
        
        self.model.train()
        self._average_metrics(accumulated_metrics, batch_count)
        return accumulated_metrics
    
    def _update_metrics_with_costs(self, metrics: TrainingMetrics, batch_costs: BatchCostMetrics):
        """Update metrics with batch cost information."""
        metrics.memory_allocated_mb = batch_costs.total_memory_mb
        metrics.peak_memory_mb = batch_costs.peak_memory_mb
        
        # Calculate tokens/samples per second
        if batch_costs.total_time_ms > 0:
            time_seconds = batch_costs.total_time_ms / 1000.0
            metrics.samples_per_second = self.config.batch_size / time_seconds
            # Approximate tokens per second (assuming average sequence length)
            avg_seq_len = getattr(self.config.model_config, 'max_sequence_length', 512) * 0.7
            metrics.tokens_per_second = (self.config.batch_size * avg_seq_len) / time_seconds
    
    def _update_metrics_with_gate_stats(self, metrics: TrainingMetrics, gate_stats: Dict[str, Any]):
        """Update metrics with gate activation statistics."""
        layer_stats = gate_stats.get('layer_stats', [])
        if layer_stats:
            att_probs = []
            ff_probs = []
            
            for stats in layer_stats:
                # Convert tensor gate probs to float for metrics (no gradients needed for metrics)
                att_prob = stats.get('attention_gate_prob', 0.5)
                ff_prob = stats.get('ff_gate_prob', 0.5)
                
                # Handle both tensor and scalar cases
                if isinstance(att_prob, torch.Tensor):
                    att_prob = att_prob.item()
                if isinstance(ff_prob, torch.Tensor):
                    ff_prob = ff_prob.item()
                    
                att_probs.append(att_prob)
                ff_probs.append(ff_prob)
            
            metrics.avg_attention_gate_prob = np.mean(att_probs)
            metrics.avg_ff_gate_prob = np.mean(ff_probs)
            
            # Calculate gate entropy (measure of gate decision diversity)
            all_probs = att_probs + ff_probs
            if all_probs:
                # Binary entropy: -p*log(p) - (1-p)*log(1-p)
                entropies = []
                for p in all_probs:
                    p = max(1e-8, min(1-1e-8, p))  # Avoid log(0)
                    entropy = -p * np.log(p) - (1-p) * np.log(1-p)
                    entropies.append(entropy)
                metrics.gate_entropy = np.mean(entropies)
    
    def _update_metrics_with_memory_stats(self, metrics: TrainingMetrics):
        """Update metrics with current memory statistics."""
        if torch.cuda.is_available():
            metrics.memory_allocated_mb = torch.cuda.memory_allocated(self.device) / 1024**2
            metrics.memory_reserved_mb = torch.cuda.memory_reserved(self.device) / 1024**2
    
    def _accumulate_metrics(self, accumulated: TrainingMetrics, current: TrainingMetrics):
        """Accumulate metrics for averaging."""
        for attr in ['total_loss', 'task_loss', 'resource_loss', 'memory_cost', 
                    'recomputation_cost', 'total_resource_cost', 'avg_attention_gate_prob',
                    'avg_ff_gate_prob', 'gate_entropy', 'gradient_norm', 'memory_allocated_mb',
                    'memory_reserved_mb', 'peak_memory_mb', 'tokens_per_second', 'samples_per_second']:
            current_val = getattr(current, attr, 0.0)
            accumulated_val = getattr(accumulated, attr, 0.0)
            setattr(accumulated, attr, accumulated_val + current_val)
    
    def _average_metrics(self, metrics: TrainingMetrics, count: int):
        """Average accumulated metrics."""
        if count == 0:
            return
        
        for attr in ['total_loss', 'task_loss', 'resource_loss', 'memory_cost', 
                    'recomputation_cost', 'total_resource_cost', 'avg_attention_gate_prob',
                    'avg_ff_gate_prob', 'gate_entropy', 'gradient_norm', 'memory_allocated_mb',
                    'memory_reserved_mb', 'peak_memory_mb', 'tokens_per_second', 'samples_per_second']:
            current_val = getattr(metrics, attr, 0.0)
            setattr(metrics, attr, current_val / count)
    
    def _log_metrics(self, metrics: TrainingMetrics, prefix: str = "train"):
        """Log metrics to all configured backends."""
        metrics_dict = metrics.to_dict()
        
        # Add prefix to all metrics
        if prefix != "train":
            metrics_dict = {f"{prefix}/{k}" if not k.startswith(prefix) else k: v 
                           for k, v in metrics_dict.items()}
        
        # Log to W&B
        if 'wandb' in self.logger_backends:
            self.logger_backends['wandb'].log(metrics_dict, step=self.global_step)
        
        # Log to TensorBoard
        if 'tensorboard' in self.logger_backends:
            tb_writer = self.logger_backends['tensorboard']
            for key, value in metrics_dict.items():
                tb_writer.add_scalar(key, value, self.global_step)
            tb_writer.flush()
        
        # Store metrics for analysis
        self.metrics_history.append((self.global_step, metrics_dict))
    
    def save_checkpoint(self, checkpoint_name: str) -> str:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.__dict__,
            'metrics_history': self.metrics_history[-100:],  # Save last 100 metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Cleanup old checkpoints (keep only top_k)
        self._cleanup_old_checkpoints(checkpoint_dir)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        if 'metrics_history' in checkpoint:
            self.metrics_history.extend(checkpoint['metrics_history'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resumed at step {self.global_step}, epoch {self.current_epoch}")
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) > self.config.save_top_k:
            # Sort by modification time
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old checkpoints
            for checkpoint in checkpoints[self.config.save_top_k:]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    def _cleanup_logging(self):
        """Cleanup logging backends."""
        if 'wandb' in self.logger_backends:
            wandb.finish()
        
        if 'tensorboard' in self.logger_backends:
            self.logger_backends['tensorboard'].close()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training session."""
        total_steps = self.global_step
        total_time = sum(self.step_times) if self.step_times else 0
        avg_step_time = total_time / len(self.step_times) if self.step_times else 0
        
        summary = {
            'total_steps': total_steps,
            'total_epochs': self.current_epoch + 1,
            'total_training_time': total_time,
            'average_step_time': avg_step_time,
            'best_eval_loss': self.best_eval_loss,
            'final_learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else None,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        if self.metrics_history:
            # Get final metrics
            final_metrics = self.metrics_history[-1][1]
            summary['final_metrics'] = final_metrics
        
        return summary 