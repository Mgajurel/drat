"""
Tests for the ResourceAwareTrainer and training infrastructure.

This module tests the training script functionality including:
- TrainingConfig validation
- ResourceAwareTrainer initialization
- Training loop execution
- Metrics logging and tracking
- Checkpoint saving and loading
- Memory and cost tracking integration
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import time

from src.training.trainer import (
    ResourceAwareTrainer, TrainingConfig, TrainingMetrics
)
from src.training.loss import ResourceAwareLoss
from src.training.cost_tracker import CostTracker
from src.models.config import get_small_config
from src.models.gated_transformer import GatedTransformer
from src.models.transformer import BaselineTransformer


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.use_gated_model is True
        assert config.use_resource_aware_loss is True
        assert config.learning_rate == 3e-4
        assert config.weight_decay == 0.01
        assert config.num_epochs == 10
        assert config.batch_size == 8
        assert config.lambda_resource == 0.01
        assert config.cost_model == "uniform"
        assert config.use_amp is True
        assert config.track_memory is True
        # W&B and TensorBoard may be disabled if not available
        # Just check they are boolean values
        assert isinstance(config.use_wandb, bool)
        assert isinstance(config.use_tensorboard, bool)
    
    def test_config_post_init(self):
        """Test configuration post-initialization validation."""
        config = TrainingConfig(device="auto")
        
        # Should set device to cuda or cpu
        assert config.device in ["cuda", "cpu"]
        
        # Should create output directory
        assert Path(config.output_dir).exists()
    
    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        config = TrainingConfig(
            learning_rate=1e-3,
            num_epochs=5,
            batch_size=16,
            lambda_resource=0.1,
            cost_model="layer_weighted",
            output_dir="./test_outputs"
        )
        
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 5
        assert config.batch_size == 16
        assert config.lambda_resource == 0.1
        assert config.cost_model == "layer_weighted"
        assert config.output_dir == "./test_outputs"


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = TrainingMetrics()
        
        assert metrics.step == 0
        assert metrics.epoch == 0
        assert metrics.total_loss == 0.0
        assert metrics.task_loss == 0.0
        assert metrics.resource_loss == 0.0
        assert metrics.memory_allocated_mb == 0.0
        assert metrics.gradient_norm == 0.0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = TrainingMetrics(
            step=10,
            epoch=1,
            total_loss=2.5,
            task_loss=2.0,
            resource_loss=0.5,
            learning_rate=3e-4
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["step"] == 10
        assert metrics_dict["epoch"] == 1
        assert metrics_dict["loss/total"] == 2.5
        assert metrics_dict["loss/task"] == 2.0
        assert metrics_dict["loss/resource"] == 0.5
        assert metrics_dict["performance/learning_rate"] == 3e-4
        
        # Check all expected keys are present
        expected_keys = [
            "step", "epoch", "loss/total", "loss/task", "loss/resource",
            "performance/tokens_per_second", "performance/samples_per_second",
            "performance/learning_rate", "memory/allocated_mb", "memory/reserved_mb",
            "memory/peak_mb", "cost/memory", "cost/recomputation", "cost/total_resource",
            "gates/avg_attention_prob", "gates/avg_ff_prob", "gates/entropy",
            "training/gradient_norm", "training/gradient_scale"
        ]
        
        for key in expected_keys:
            assert key in metrics_dict


class TestResourceAwareTrainer:
    """Test ResourceAwareTrainer functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        config = get_small_config()
        config.vocab_size = 1000
        config.max_sequence_length = 64
        config.max_position_embeddings = 64
        config.num_hidden_layers = 2  # Smaller for faster tests
        config.num_attention_heads = 4
        config.hidden_size = 256
        config.intermediate_size = 512
        return config
    
    @pytest.fixture
    def training_config(self, temp_dir, model_config):
        """Create test training configuration."""
        return TrainingConfig(
            model_config=model_config,
            num_epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            log_interval=1,
            eval_interval=2,
            save_interval=5,
            output_dir=str(temp_dir),
            use_wandb=False,  # Disable external logging for tests
            use_tensorboard=False,
            track_memory=True,
            device="cpu"  # Use CPU for tests
        )
    
    @pytest.fixture
    def mock_dataloader(self, model_config):
        """Create mock data loader."""
        batch_size = 2
        seq_len = model_config.max_sequence_length
        vocab_size = model_config.vocab_size
        
        # Create sample batch
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
        
        # Mock dataloader that returns the same batch
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([batch, batch]))
        mock_loader.__len__ = Mock(return_value=2)
        
        return mock_loader
    
    def test_trainer_initialization(self, training_config):
        """Test trainer initialization."""
        trainer = ResourceAwareTrainer(config=training_config)
        
        assert trainer.config == training_config
        assert trainer.device == torch.device("cpu")
        assert isinstance(trainer.model, GatedTransformer)  # Default is gated model
        assert isinstance(trainer.loss_fn, ResourceAwareLoss)
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert isinstance(trainer.cost_tracker, CostTracker)
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert trainer.metrics_history == []
    
    def test_trainer_with_standard_model(self, training_config):
        """Test trainer with standard transformer model."""
        training_config.use_gated_model = False
        trainer = ResourceAwareTrainer(config=training_config)
        
        assert isinstance(trainer.model, BaselineTransformer)
    
    def test_trainer_with_custom_model(self, training_config, model_config):
        """Test trainer with provided model."""
        model = GatedTransformer(model_config)
        trainer = ResourceAwareTrainer(
            config=training_config,
            model=model
        )
        
        assert trainer.model is model
    
    def test_create_loss_function(self, training_config):
        """Test loss function creation."""
        trainer = ResourceAwareTrainer(config=training_config)
        
        loss_fn = trainer._create_loss_function()
        assert isinstance(loss_fn, ResourceAwareLoss)
        assert loss_fn.lambda_resource == training_config.lambda_resource
        assert loss_fn.cost_model == training_config.cost_model
    
    def test_create_optimizer(self, training_config):
        """Test optimizer creation."""
        trainer = ResourceAwareTrainer(config=training_config)
        
        optimizer = trainer._create_optimizer()
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]['lr'] == training_config.learning_rate
        assert optimizer.param_groups[0]['weight_decay'] == training_config.weight_decay
    
    def test_create_scheduler(self, training_config):
        """Test scheduler creation."""
        trainer = ResourceAwareTrainer(config=training_config)
        
        scheduler = trainer._create_scheduler()
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_setup_logging_disabled(self, training_config):
        """Test logging setup when backends are disabled."""
        training_config.use_wandb = False
        training_config.use_tensorboard = False
        
        trainer = ResourceAwareTrainer(config=training_config)
        backends = trainer._setup_logging()
        
        assert 'wandb' not in backends
        assert 'tensorboard' not in backends
    
    def test_train_step(self, training_config, mock_dataloader):
        """Test single training step."""
        trainer = ResourceAwareTrainer(
            config=training_config,
            train_dataloader=mock_dataloader
        )
        
        # Get a batch from the mock dataloader
        batch = next(iter(mock_dataloader))
        
        # Execute training step
        metrics = trainer._train_step(batch, 0)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.step == 0
        assert metrics.epoch == 0
        assert metrics.total_loss > 0  # Should have some loss
        assert metrics.learning_rate > 0
    
    def test_update_metrics_with_gate_stats(self, training_config):
        """Test updating metrics with gate statistics."""
        trainer = ResourceAwareTrainer(config=training_config)
        metrics = TrainingMetrics()
        
        gate_stats = {
            'layer_stats': [
                {'attention_gate_prob': 0.8, 'ff_gate_prob': 0.6},
                {'attention_gate_prob': 0.7, 'ff_gate_prob': 0.9}
            ]
        }
        
        trainer._update_metrics_with_gate_stats(metrics, gate_stats)
        
        assert metrics.avg_attention_gate_prob == 0.75  # (0.8 + 0.7) / 2
        assert metrics.avg_ff_gate_prob == 0.75  # (0.6 + 0.9) / 2
        assert metrics.gate_entropy > 0  # Should have some entropy
    
    def test_accumulate_and_average_metrics(self, training_config):
        """Test metrics accumulation and averaging."""
        trainer = ResourceAwareTrainer(config=training_config)
        
        accumulated = TrainingMetrics()
        current = TrainingMetrics(total_loss=2.0, task_loss=1.5, resource_loss=0.5)
        
        trainer._accumulate_metrics(accumulated, current)
        
        assert accumulated.total_loss == 2.0
        assert accumulated.task_loss == 1.5
        assert accumulated.resource_loss == 0.5
        
        # Test averaging
        trainer._average_metrics(accumulated, 2)
        
        assert accumulated.total_loss == 1.0
        assert accumulated.task_loss == 0.75
        assert accumulated.resource_loss == 0.25
    
    def test_save_and_load_checkpoint(self, training_config, temp_dir):
        """Test checkpoint saving and loading."""
        trainer = ResourceAwareTrainer(config=training_config)
        
        # Modify some trainer state
        trainer.global_step = 100
        trainer.current_epoch = 5
        trainer.best_eval_loss = 1.5
        
        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint("test_checkpoint")
        
        assert Path(checkpoint_path).exists()
        
        # Create new trainer and load checkpoint
        new_trainer = ResourceAwareTrainer(config=training_config)
        new_trainer.load_checkpoint(checkpoint_path)
        
        assert new_trainer.global_step == 100
        assert new_trainer.current_epoch == 5
        assert new_trainer.best_eval_loss == 1.5
    
    def test_get_training_summary(self, training_config):
        """Test training summary generation."""
        trainer = ResourceAwareTrainer(config=training_config)
        
        # Add some mock data
        trainer.global_step = 50
        trainer.current_epoch = 2
        trainer.step_times = [0.1, 0.2, 0.15]
        trainer.best_eval_loss = 2.3
        trainer.metrics_history = [(10, {'loss/total': 3.0}), (20, {'loss/total': 2.5})]
        
        summary = trainer.get_training_summary()
        
        assert summary['total_steps'] == 50
        assert summary['total_epochs'] == 3  # current_epoch + 1
        assert summary['total_training_time'] == 0.45  # sum of step_times
        assert summary['average_step_time'] == 0.15  # average of step_times
        assert summary['best_eval_loss'] == 2.3
        assert 'model_parameters' in summary
        assert 'trainable_parameters' in summary
        assert 'final_metrics' in summary
    
    def test_log_metrics(self, training_config):
        """Test metrics logging."""
        trainer = ResourceAwareTrainer(config=training_config)
        trainer.global_step = 10  # Set the trainer's global step
        
        metrics = TrainingMetrics(
            step=10,
            total_loss=2.0,
            learning_rate=1e-3
        )
        
        # Test without logging backends (should not raise errors)
        trainer._log_metrics(metrics)
        
        # Check metrics are stored in history
        assert len(trainer.metrics_history) == 1
        assert trainer.metrics_history[0][0] == 10  # step (uses trainer.global_step)
        assert 'loss/total' in trainer.metrics_history[0][1]  # metrics dict
    
    def test_cleanup_old_checkpoints(self, training_config, temp_dir):
        """Test cleanup of old checkpoints."""
        training_config.save_top_k = 2
        trainer = ResourceAwareTrainer(config=training_config)
        
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create multiple checkpoint files
        for i in range(5):
            checkpoint_file = checkpoint_dir / f"checkpoint_epoch_{i}.pt"
            checkpoint_file.write_text("dummy checkpoint")
            time.sleep(0.01)  # Ensure different modification times
        
        # Run cleanup
        trainer._cleanup_old_checkpoints(checkpoint_dir)
        
        # Should only keep the most recent save_top_k files
        remaining_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(remaining_files) == training_config.save_top_k


class TestTrainingIntegration:
    """Integration tests for complete training workflow."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def minimal_setup(self, temp_dir):
        """Create minimal training setup for integration tests."""
        # Model config
        model_config = get_small_config()
        model_config.vocab_size = 100
        model_config.max_sequence_length = 16
        model_config.max_position_embeddings = 16
        model_config.num_hidden_layers = 1
        model_config.num_attention_heads = 2
        model_config.hidden_size = 64
        model_config.intermediate_size = 128
        # Trigger recomputation of derived fields
        model_config.__post_init__()
        
        # Training config
        training_config = TrainingConfig(
            model_config=model_config,
            num_epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            log_interval=1,
            output_dir=str(temp_dir),
            use_wandb=False,
            use_tensorboard=False,
            device="cpu"
        )
        
        # Create simple dataset
        input_ids = torch.randint(0, model_config.vocab_size, (2, model_config.max_sequence_length))
        labels = torch.randint(0, model_config.vocab_size, (2, model_config.max_sequence_length))
        
        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }
        
        # Mock dataloader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([batch]))
        mock_loader.__len__ = Mock(return_value=1)
        
        return training_config, mock_loader
    
    def test_short_training_run(self, minimal_setup):
        """Test a short training run end-to-end."""
        training_config, train_dataloader = minimal_setup
        
        trainer = ResourceAwareTrainer(
            config=training_config,
            train_dataloader=train_dataloader
        )
        
        # Record initial state
        initial_step = trainer.global_step
        initial_loss_params = trainer.loss_fn.lambda_resource
        
        # Store initial parameter values to check if they change
        initial_params = {name: param.clone().detach() for name, param in trainer.model.named_parameters()}
        
        # Run training (should complete without errors)
        trainer.train()
        
        # Check training progressed
        assert trainer.global_step > initial_step
        assert trainer.current_epoch >= 0
        
        # Check metrics were collected
        assert len(trainer.metrics_history) > 0
        
        # Check model parameters were updated (at least some should have changed)
        params_changed = False
        for name, param in trainer.model.named_parameters():
            if not torch.equal(initial_params[name], param):
                params_changed = True
                break
        assert params_changed, "Model parameters should have been updated during training"
    
    def test_training_with_evaluation(self, minimal_setup):
        """Test training with evaluation dataset."""
        training_config, dataloader = minimal_setup
        training_config.eval_interval = 1  # Evaluate every step
        
        trainer = ResourceAwareTrainer(
            config=training_config,
            train_dataloader=dataloader,
            eval_dataloader=dataloader  # Use same data for eval
        )
        
        # Run training
        trainer.train()
        
        # Should have run evaluation
        assert trainer.best_eval_loss < float('inf')
    
    def test_training_with_checkpointing(self, minimal_setup, temp_dir):
        """Test training with checkpoint saving."""
        training_config, dataloader = minimal_setup
        training_config.save_interval = 1  # Save every step
        
        trainer = ResourceAwareTrainer(
            config=training_config,
            train_dataloader=dataloader
        )
        
        # Run training
        trainer.train()
        
        # Check checkpoint was saved
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0
    
    def test_resume_from_checkpoint(self, minimal_setup):
        """Test resuming training from checkpoint."""
        training_config, dataloader = minimal_setup
        
        # First training session
        trainer1 = ResourceAwareTrainer(
            config=training_config,
            train_dataloader=dataloader
        )
        trainer1.global_step = 10
        trainer1.current_epoch = 1
        checkpoint_path = trainer1.save_checkpoint("resume_test")
        
        # Second training session (resume)
        training_config.resume_from_checkpoint = checkpoint_path
        trainer2 = ResourceAwareTrainer(
            config=training_config,
            train_dataloader=dataloader
        )
        
        # Load checkpoint manually for testing (normally done in train())
        trainer2.load_checkpoint(checkpoint_path)
        
        # Should have resumed state
        assert trainer2.global_step == 10
        assert trainer2.current_epoch == 1


if __name__ == "__main__":
    pytest.main([__file__]) 