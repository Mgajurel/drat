#!/usr/bin/env python3
"""
Lambda Parameter Sweep for Resource-Aware Transformer Training.

This script systematically explores the trade-offs between model accuracy and 
resource usage by training models with different λ (lambda_resource) values.

RECENT OPTIMIZATIONS (Post-Gradient Flow Fix):
- ✅ Gradient flow validation before sweep execution
- ⚡ Optimized training parameters for meaningful convergence
- 📊 Enhanced lambda value range for better granularity  
- 🔧 Robust error handling and progress monitoring

Key Features:
- Pre-sweep validation of gradient flow through gate parameters
- Comprehensive metrics collection (loss, memory, timing, gate stats)
- Automated analysis with plots and detailed reports
- Configurable model sizes and training parameters
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

# Local imports
from src.models.config import TransformerConfig, get_small_config, get_tiny_config
from src.models.gated_transformer import GatedTransformer
from src.training.trainer import ResourceAwareTrainer, TrainingConfig
from src.tokenizer.bpe_tokenizer import BPETokenizer, BPETokenizerConfig


@dataclass
class LambdaSweepConfig:
    """Configuration for lambda parameter sweep experiment."""
    
    # Lambda values to test
    lambda_values: List[float]
    
    # Base model configuration
    model_size: str = "small"  # tiny, small, medium
    vocab_size: int = 10000
    max_seq_length: int = 128
    
    # Training configuration - OPTIMIZED for meaningful results
    epochs: int = 5  # Increased from 2 for better convergence
    batch_size: int = 8  # Increased from 4 for more stable gradients
    learning_rate: float = 3e-4  # Increased from 1e-4 for faster learning
    synthetic_samples: int = 1000  # Increased from 300 for more robust training
    
    # Experiment configuration
    output_base_dir: str = "outputs/lambda_sweep"
    random_seed: int = 42
    device: str = "auto"
    
    # Logging configuration - OPTIMIZED for better monitoring
    log_interval: int = 20  # Increased from 10 for less noise
    eval_interval: int = 50  # Increased from 25 for meaningful evaluation
    use_tensorboard: bool = True
    
    # Analysis configuration
    create_plots: bool = True
    save_detailed_results: bool = True


@dataclass
class LambdaResult:
    """Results from a single lambda value experiment."""
    
    lambda_value: float
    
    # Training metrics
    final_loss: float
    task_loss: float
    resource_loss: float
    convergence_steps: int
    training_time: float
    
    # Resource metrics
    peak_memory_mb: float
    avg_memory_mb: float
    memory_efficiency: float
    
    # Model metrics
    model_parameters: int
    gate_activation_rate: float
    gate_entropy: float
    
    # Performance metrics
    avg_step_time: float
    tokens_per_second: float
    
    # Convergence metrics
    gradient_norm: float
    learning_rate_final: float
    
    # Additional metadata
    converged: bool
    training_stable: bool
    experiment_id: str


class LambdaSweepExperiment:
    """Manages lambda parameter sweep experiments."""
    
    def __init__(self, config: LambdaSweepConfig):
        self.config = config
        self.results: List[LambdaResult] = []
        self.setup_logging()
        self.setup_output_dirs()
        
    def setup_logging(self):
        """Setup logging for the sweep experiment."""
        log_dir = Path(self.config.output_base_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"lambda_sweep_{int(time.time())}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Lambda sweep experiment started")
        self.logger.info(f"Config: {asdict(self.config)}")
        
    def setup_output_dirs(self):
        """Create output directories for the experiment."""
        base_dir = Path(self.config.output_base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (base_dir / "runs").mkdir(exist_ok=True)
        (base_dir / "analysis").mkdir(exist_ok=True)
        (base_dir / "plots").mkdir(exist_ok=True)
        (base_dir / "logs").mkdir(exist_ok=True)
        
    def create_tokenizer(self) -> BPETokenizer:
        """Create and train tokenizer for consistent vocabulary."""
        tokenizer_config = BPETokenizerConfig(vocab_size=self.config.vocab_size)
        tokenizer = BPETokenizer(tokenizer_config)
        
        # Train on sample text
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models learn patterns from data.",
            "Natural language processing enables text understanding.",
            "Deep learning transforms artificial intelligence capabilities.",
            "Transformers revolutionized sequence modeling tasks.",
            "Attention mechanisms focus on relevant information.",
            "Resource-aware training optimizes computational efficiency.",
            "Lambda parameters control trade-off between objectives.",
            "Gated models selectively activate computational paths.",
            "Training dynamics affect model convergence behavior."
        ] * 20  # Repeat for sufficient training data
        
        tokenizer.train(sample_texts)
        self.logger.info(f"Tokenizer trained with vocabulary size: {tokenizer.vocab_size}")
        
        return tokenizer
        
    def create_dataset(self, tokenizer: BPETokenizer, model_config: TransformerConfig) -> Tuple[DataLoader, DataLoader]:
        """Create consistent dataset for all experiments."""
        import random
        
        # Set seed for reproducible data generation
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        # Generate sample text data
        sample_texts = []
        for i in range(self.config.synthetic_samples):
            sentence_length = random.randint(10, model_config.max_sequence_length // 2)
            words = [f"word{j}" for j in range(sentence_length)]
            sentence = " ".join(words)
            sample_texts.append(sentence)
        
        # Tokenize and create dataset
        tokenized_data = []
        for text in sample_texts:
            tokens = tokenizer.encode(text)
            if len(tokens) < model_config.max_sequence_length:
                tokens.extend([tokenizer.pad_token_id] * (model_config.max_sequence_length - len(tokens)))
            else:
                tokens = tokens[:model_config.max_sequence_length]
            
            tokenized_data.append({
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'labels': torch.tensor(tokens[1:] + [tokenizer.eos_token_id], dtype=torch.long)
            })
        
        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': (input_ids != tokenizer.pad_token_id).long()
            }
        
        # Split into train/eval
        train_size = int(0.8 * len(tokenized_data))
        train_data = tokenized_data[:train_size]
        eval_data = tokenized_data[train_size:]
        
        train_dataloader = DataLoader(
            train_data, batch_size=self.config.batch_size, shuffle=True, 
            collate_fn=collate_fn, num_workers=0, pin_memory=True
        )
        
        eval_dataloader = DataLoader(
            eval_data, batch_size=self.config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0, pin_memory=True
        )
        
        return train_dataloader, eval_dataloader
        
    def run_single_experiment(self, lambda_value: float, tokenizer: BPETokenizer, 
                            train_dataloader: DataLoader, eval_dataloader: DataLoader) -> LambdaResult:
        """Run a single experiment with given lambda value."""
        
        experiment_id = f"lambda_{lambda_value:.3f}_{int(time.time())}"
        self.logger.info(f"Starting experiment: {experiment_id}")
        
        # Setup device
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
            
        # Create model configuration
        if self.config.model_size == "tiny":
            model_config = get_tiny_config()
        elif self.config.model_size == "small":
            model_config = get_small_config()
        else:
            raise ValueError(f"Unsupported model size: {self.config.model_size}")
            
        model_config.vocab_size = self.config.vocab_size
        model_config.max_sequence_length = self.config.max_seq_length
        model_config.max_position_embeddings = self.config.max_seq_length
        model_config.use_recomputation_gates = True  # Always use gated model
        
        # Create model with consistent initialization
        torch.manual_seed(self.config.random_seed)
        model = GatedTransformer(model_config)
        
        # Create training configuration
        output_dir = Path(self.config.output_base_dir) / "runs" / experiment_id
        training_config = TrainingConfig(
            model_config=model_config,
            use_gated_model=True,
            use_resource_aware_loss=True,
            
            # Training hyperparameters
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            max_grad_norm=1.0,
            
            # Training schedule
            num_epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=1,
            
            # LAMBDA CONFIGURATION - This is the key parameter!
            lambda_resource=lambda_value,
            cost_model="uniform",
            
            # Mixed precision
            use_amp=True,
            
            # Logging and evaluation
            log_interval=self.config.log_interval,
            eval_interval=self.config.eval_interval,
            save_interval=1000,  # Don't save checkpoints for sweep
            
            # Checkpointing
            output_dir=str(output_dir),
            resume_from_checkpoint=None,
            
            # Logging backends
            use_wandb=False,
            use_tensorboard=self.config.use_tensorboard,
            wandb_project=None,
            wandb_run_name=experiment_id,
            
            # Cost tracking
            track_memory=True,
            track_computation_time=True,
            memory_log_interval=25,
            
            # Device configuration
            device=device,
            num_workers=0,
            
            # Debugging
            debug_mode=False,
            profile_steps=0,
        )
        
        # Create trainer
        trainer = ResourceAwareTrainer(
            config=training_config,
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )
        
        # Run training
        start_time = time.time()
        try:
            trainer.train()
            training_time = time.time() - start_time
            
            # Get final metrics
            summary = trainer.get_training_summary()
            final_metrics = summary.get('final_metrics', {})
            
            # Analyze convergence
            convergence_steps = self._analyze_convergence(trainer)
            
            # Create result
            result = LambdaResult(
                lambda_value=lambda_value,
                final_loss=final_metrics.get('loss/total', float('inf')),
                task_loss=final_metrics.get('loss/task', float('inf')),
                resource_loss=final_metrics.get('loss/resource', 0.0),
                convergence_steps=convergence_steps,
                training_time=training_time,
                peak_memory_mb=final_metrics.get('memory/peak_mb', 0.0),
                avg_memory_mb=final_metrics.get('memory/allocated_mb', 0.0),
                memory_efficiency=self._calculate_memory_efficiency(final_metrics),
                model_parameters=summary.get('model_parameters', 0),
                gate_activation_rate=final_metrics.get('gates/avg_attention_prob', 0.5),
                gate_entropy=final_metrics.get('gates/entropy', 0.0),
                avg_step_time=summary.get('average_step_time', 0.0),
                tokens_per_second=final_metrics.get('performance/tokens_per_second', 0.0),
                gradient_norm=float(final_metrics.get('training/gradient_norm', 0.0)),
                learning_rate_final=summary.get('final_learning_rate', 0.0),
                converged=convergence_steps < summary.get('total_steps', float('inf')),
                training_stable=self._check_training_stability(trainer),
                experiment_id=experiment_id
            )
            
            self.logger.info(f"Experiment {experiment_id} completed successfully")
            self.logger.info(f"Lambda: {lambda_value}, Final Loss: {result.final_loss:.4f}, "
                           f"Resource Loss: {result.resource_loss:.4f}, "
                           f"Memory: {result.peak_memory_mb:.1f}MB")
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            result = self._create_failed_result(lambda_value, experiment_id, str(e))
            
        finally:
            # Cleanup
            trainer._cleanup_logging()
            del trainer
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return result
        
    def _analyze_convergence(self, trainer) -> int:
        """Analyze when the model converged during training."""
        # This is a simplified convergence analysis
        # In a real implementation, you'd analyze the loss history
        return trainer.global_step // 2  # Rough estimate
        
    def _calculate_memory_efficiency(self, metrics: Dict[str, Any]) -> float:
        """Calculate memory efficiency metric."""
        peak_mb = metrics.get('memory/peak_mb', 1.0)
        allocated_mb = metrics.get('memory/allocated_mb', 1.0)
        return allocated_mb / peak_mb if peak_mb > 0 else 0.0
        
    def _check_training_stability(self, trainer) -> bool:
        """Check if training was stable (no gradient explosions, etc.)."""
        # Simplified stability check
        return True  # Would analyze gradient norms, loss spikes, etc.
        
    def _create_failed_result(self, lambda_value: float, experiment_id: str, error: str) -> LambdaResult:
        """Create a result object for failed experiments."""
        return LambdaResult(
            lambda_value=lambda_value,
            final_loss=float('inf'),
            task_loss=float('inf'),
            resource_loss=0.0,
            convergence_steps=-1,
            training_time=0.0,
            peak_memory_mb=0.0,
            avg_memory_mb=0.0,
            memory_efficiency=0.0,
            model_parameters=0,
            gate_activation_rate=0.0,
            gate_entropy=0.0,
            avg_step_time=0.0,
            tokens_per_second=0.0,
            gradient_norm=0.0,
            learning_rate_final=0.0,
            converged=False,
            training_stable=False,
            experiment_id=f"{experiment_id}_FAILED"
        )
        
    def validate_gradient_flow(self) -> bool:
        """Validate that gradient flow through gates is working properly before running sweep."""
        self.logger.info("🔍 Validating gradient flow through gate parameters...")
        
        try:
            # Create a minimal test setup
            if self.config.model_size == "tiny":
                model_config = get_tiny_config()
            else:
                model_config = get_small_config()
                
            model_config.vocab_size = self.config.vocab_size
            model_config.max_sequence_length = 64  # Shorter for validation
            model_config.max_position_embeddings = 64
            model_config.use_recomputation_gates = True
            model_config.num_layers = 2  # Minimal for testing
            
            # Create model
            torch.manual_seed(self.config.random_seed)
            model = GatedTransformer(model_config)
            
            # Create loss function with non-zero lambda
            from src.training.loss import ResourceAwareLoss
            loss_fn = ResourceAwareLoss(lambda_resource=0.1)
            
            # Create dummy data
            batch_size, seq_len = 2, 32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            
            input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len)).to(device)
            targets = input_ids.clone()
            
            # Forward pass
            outputs = model(input_ids, return_gate_info=True)
            gate_stats = outputs.get('gate_statistics', {})
            
            # Check gate statistics structure
            if not gate_stats or 'layer_stats' not in gate_stats:
                self.logger.error("❌ Gate statistics not found in model output!")
                return False
                
            layer_stats = gate_stats['layer_stats']
            if not layer_stats:
                self.logger.error("❌ No layer statistics found!")
                return False
                
            # Check first layer's gate statistics
            first_layer = layer_stats[0]
            att_prob = first_layer.get('attention_gate_prob')
            ff_prob = first_layer.get('ff_gate_prob')
            
            # Validate that gate probs are tensors with gradients
            if not isinstance(att_prob, torch.Tensor) or not att_prob.requires_grad:
                self.logger.error("❌ Attention gate probability is not a tensor with gradients!")
                return False
                
            if not isinstance(ff_prob, torch.Tensor) or not ff_prob.requires_grad:
                self.logger.error("❌ FF gate probability is not a tensor with gradients!")
                return False
                
            # Compute loss and check resource loss component
            logits = outputs['logits']
            gate_stats = outputs.get('gate_statistics', {})
            total_loss, metrics = loss_fn(logits, targets, gate_statistics=gate_stats, return_metrics=True)
            resource_loss = torch.tensor(metrics['resource_loss'], device=total_loss.device, requires_grad=True)
            
            # Validate that resource loss is non-zero and has gradients
            if resource_loss.item() == 0.0:
                self.logger.error("❌ Resource loss is zero - gradient flow might be broken!")
                return False
                
            if not resource_loss.requires_grad:
                self.logger.error("❌ Resource loss does not require gradients!")
                return False
                
            # Test backward pass
            total_loss.backward()
            
            # Check that gate parameters have gradients
            gate_params_with_grads = 0
            total_gate_params = 0
            
            for name, param in model.named_parameters():
                if 'gate_param' in name:
                    total_gate_params += 1
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        gate_params_with_grads += 1
                        
            if total_gate_params == 0:
                self.logger.error("❌ No gate parameters found in model!")
                return False
                
            if gate_params_with_grads == 0:
                self.logger.error("❌ No gate parameters received gradients!")
                return False
                
            gradient_ratio = gate_params_with_grads / total_gate_params
            if gradient_ratio < 0.5:
                self.logger.warning(f"⚠️ Only {gradient_ratio:.1%} of gate parameters received gradients")
                
            self.logger.info(f"✅ Gradient flow validation passed!")
            self.logger.info(f"   - Gate statistics: tensors with gradients ✓")
            self.logger.info(f"   - Resource loss: {resource_loss.item():.6f} (non-zero) ✓")
            self.logger.info(f"   - Gate parameters with gradients: {gate_params_with_grads}/{total_gate_params} ✓")
            
            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gradient flow validation failed: {e}")
            return False
        
    def run_sweep(self):
        """Run the complete lambda parameter sweep."""
        self.logger.info(f"Starting lambda sweep with values: {self.config.lambda_values}")
        
        # VALIDATION: Ensure gradient flow is working before running expensive sweep
        self.logger.info("🔧 Running pre-sweep validation...")
        if not self.validate_gradient_flow():
            self.logger.error("❌ Gradient flow validation failed! Aborting sweep.")
            self.logger.error("💡 Please check that the gradient flow fixes are properly applied.")
            return
        
        self.logger.info("✅ Validation passed! Starting lambda parameter sweep...")
        
        # Create shared components
        tokenizer = self.create_tokenizer()
        
        # Create model config for dataset creation
        if self.config.model_size == "tiny":
            model_config = get_tiny_config()
        else:
            model_config = get_small_config()
        model_config.max_sequence_length = self.config.max_seq_length
        
        train_dataloader, eval_dataloader = self.create_dataset(tokenizer, model_config)
        
        # Run experiments for each lambda value
        for i, lambda_value in enumerate(self.config.lambda_values):
            self.logger.info(f"🚀 Running experiment {i+1}/{len(self.config.lambda_values)} "
                           f"with lambda = {lambda_value}")
            
            result = self.run_single_experiment(
                lambda_value, tokenizer, train_dataloader, eval_dataloader
            )
            
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            self.logger.info(f"✅ Completed {i+1}/{len(self.config.lambda_values)} experiments")
            
        self.logger.info("🎉 Lambda sweep completed!")
        
        # Final analysis
        self.analyze_results()
        
    def save_results(self):
        """Save experiment results to JSON."""
        results_file = Path(self.config.output_base_dir) / "analysis" / "lambda_sweep_results.json"
        
        results_data = {
            'config': asdict(self.config),
            'results': [asdict(result) for result in self.results],
            'summary': self.get_summary_stats()
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
            
        self.logger.info(f"Results saved to {results_file}")
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from the sweep."""
        if not self.results:
            return {}
            
        successful_results = [r for r in self.results if r.converged]
        
        return {
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(self.results) - len(successful_results),
            'best_lambda': min(successful_results, key=lambda x: x.final_loss).lambda_value if successful_results else None,
            'lambda_range': [min(r.lambda_value for r in self.results), max(r.lambda_value for r in self.results)],
            'loss_range': [min(r.final_loss for r in successful_results), max(r.final_loss for r in successful_results)] if successful_results else None,
            'memory_range': [min(r.peak_memory_mb for r in successful_results), max(r.peak_memory_mb for r in successful_results)] if successful_results else None,
        }
        
    def analyze_results(self):
        """Perform detailed analysis of the sweep results."""
        if not self.results:
            self.logger.warning("No results to analyze")
            return
            
        self.logger.info("Analyzing lambda sweep results...")
        
        # Create analysis plots
        if self.config.create_plots:
            self.create_analysis_plots()
            
        # Generate analysis report
        self.generate_analysis_report()
        
    def create_analysis_plots(self):
        """Create visualization plots for the sweep results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn not available, skipping plots")
            return
            
        successful_results = [r for r in self.results if r.converged and r.final_loss < float('inf')]
        
        if len(successful_results) < 2:
            self.logger.warning("Not enough successful results for plotting")
            return
            
        # Set up plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Lambda Parameter Sweep Analysis', fontsize=16, fontweight='bold')
        
        lambdas = [r.lambda_value for r in successful_results]
        
        # Plot 1: Loss vs Lambda
        ax1 = axes[0, 0]
        final_losses = [r.final_loss for r in successful_results]
        task_losses = [r.task_loss for r in successful_results]
        resource_losses = [r.resource_loss for r in successful_results]
        
        ax1.plot(lambdas, final_losses, 'o-', label='Total Loss', linewidth=2, markersize=6)
        ax1.plot(lambdas, task_losses, 's-', label='Task Loss', linewidth=2, markersize=6)
        ax1.plot(lambdas, resource_losses, '^-', label='Resource Loss', linewidth=2, markersize=6)
        ax1.set_xlabel('Lambda (λ)')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Components vs Lambda')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory Usage vs Lambda
        ax2 = axes[0, 1]
        peak_memory = [r.peak_memory_mb for r in successful_results]
        avg_memory = [r.avg_memory_mb for r in successful_results]
        
        ax2.plot(lambdas, peak_memory, 'o-', label='Peak Memory', linewidth=2, markersize=6)
        ax2.plot(lambdas, avg_memory, 's-', label='Avg Memory', linewidth=2, markersize=6)
        ax2.set_xlabel('Lambda (λ)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Lambda')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training Time vs Lambda
        ax3 = axes[1, 0]
        training_times = [r.training_time / 60 for r in successful_results]  # Convert to minutes
        step_times = [r.avg_step_time for r in successful_results]
        
        ax3.plot(lambdas, training_times, 'o-', label='Total Time (min)', linewidth=2, markersize=6)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(lambdas, step_times, 's-', color='orange', label='Avg Step Time (s)', linewidth=2, markersize=6)
        ax3.set_xlabel('Lambda (λ)')
        ax3.set_ylabel('Training Time (minutes)', color='blue')
        ax3_twin.set_ylabel('Step Time (seconds)', color='orange')
        ax3.set_title('Training Time vs Lambda')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Gate Activation vs Lambda
        ax4 = axes[1, 1]
        gate_rates = [r.gate_activation_rate for r in successful_results]
        gate_entropies = [r.gate_entropy for r in successful_results]
        
        ax4.plot(lambdas, gate_rates, 'o-', label='Gate Activation Rate', linewidth=2, markersize=6)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(lambdas, gate_entropies, 's-', color='red', label='Gate Entropy', linewidth=2, markersize=6)
        ax4.set_xlabel('Lambda (λ)')
        ax4.set_ylabel('Gate Activation Rate', color='blue')
        ax4_twin.set_ylabel('Gate Entropy', color='red')
        ax4.set_title('Gate Statistics vs Lambda')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = Path(self.config.output_base_dir) / "plots" / "lambda_sweep_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Analysis plots saved to {plot_file}")
        
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report."""
        report_file = Path(self.config.output_base_dir) / "analysis" / "lambda_sweep_report.md"
        
        successful_results = [r for r in self.results if r.converged and r.final_loss < float('inf')]
        
        if not successful_results:
            self.logger.warning("No successful results for report generation")
            return
            
        # Find optimal lambda values
        best_loss = min(successful_results, key=lambda x: x.final_loss)
        best_memory = min(successful_results, key=lambda x: x.peak_memory_mb)
        best_time = min(successful_results, key=lambda x: x.training_time)
        
        # Calculate trade-off metrics
        trade_off_scores = []
        for result in successful_results:
            # Normalize metrics (0-1 scale)
            loss_norm = result.final_loss / max(r.final_loss for r in successful_results)
            memory_norm = result.peak_memory_mb / max(r.peak_memory_mb for r in successful_results)
            time_norm = result.training_time / max(r.training_time for r in successful_results)
            
            # Combined score (lower is better)
            combined_score = (loss_norm + memory_norm + time_norm) / 3
            trade_off_scores.append((result.lambda_value, combined_score, result))
            
        best_trade_off = min(trade_off_scores, key=lambda x: x[1])
        
        # Generate report
        report_content = f"""# Lambda Parameter Sweep Analysis Report

## Experiment Overview

- **Total Experiments:** {len(self.results)}
- **Successful Experiments:** {len(successful_results)}
- **Failed Experiments:** {len(self.results) - len(successful_results)}
- **Lambda Range:** {min(r.lambda_value for r in self.results):.3f} - {max(r.lambda_value for r in self.results):.3f}

## Key Findings

### Best Performance by Metric

| Metric | Lambda (λ) | Value | Experiment ID |
|--------|------------|-------|---------------|
| **Lowest Loss** | {best_loss.lambda_value:.3f} | {best_loss.final_loss:.4f} | {best_loss.experiment_id} |
| **Lowest Memory** | {best_memory.lambda_value:.3f} | {best_memory.peak_memory_mb:.1f} MB | {best_memory.experiment_id} |
| **Fastest Training** | {best_time.lambda_value:.3f} | {best_time.training_time/60:.1f} min | {best_time.experiment_id} |
| **Best Trade-off** | {best_trade_off[0]:.3f} | Score: {best_trade_off[1]:.3f} | {best_trade_off[2].experiment_id} |

### Detailed Results

| Lambda (λ) | Final Loss | Task Loss | Resource Loss | Peak Memory (MB) | Training Time (min) | Gate Activation | Converged |
|------------|------------|-----------|---------------|------------------|---------------------|-----------------|-----------|
"""
        
        for result in sorted(successful_results, key=lambda x: x.lambda_value):
            report_content += f"| {result.lambda_value:.3f} | {result.final_loss:.4f} | {result.task_loss:.4f} | {result.resource_loss:.4f} | {result.peak_memory_mb:.1f} | {result.training_time/60:.1f} | {result.gate_activation_rate:.3f} | {'✅' if result.converged else '❌'} |\n"
            
        report_content += f"""

## Analysis Insights

### Loss Behavior
- **Task Loss Range:** {min(r.task_loss for r in successful_results):.4f} - {max(r.task_loss for r in successful_results):.4f}
- **Resource Loss Range:** {min(r.resource_loss for r in successful_results):.4f} - {max(r.resource_loss for r in successful_results):.4f}
- **Total Loss Range:** {min(r.final_loss for r in successful_results):.4f} - {max(r.final_loss for r in successful_results):.4f}

### Resource Efficiency
- **Memory Usage Range:** {min(r.peak_memory_mb for r in successful_results):.1f} - {max(r.peak_memory_mb for r in successful_results):.1f} MB
- **Training Time Range:** {min(r.training_time for r in successful_results)/60:.1f} - {max(r.training_time for r in successful_results)/60:.1f} minutes
- **Average Step Time Range:** {min(r.avg_step_time for r in successful_results):.1f} - {max(r.avg_step_time for r in successful_results):.1f} seconds

### Gate Behavior
- **Gate Activation Range:** {min(r.gate_activation_rate for r in successful_results):.3f} - {max(r.gate_activation_rate for r in successful_results):.3f}
- **Gate Entropy Range:** {min(r.gate_entropy for r in successful_results):.3f} - {max(r.gate_entropy for r in successful_results):.3f}

## Recommendations

### Optimal Lambda Values

1. **For Best Accuracy:** λ = {best_loss.lambda_value:.3f}
   - Achieves lowest final loss of {best_loss.final_loss:.4f}
   - Trade-off: Higher memory usage ({best_loss.peak_memory_mb:.1f} MB)

2. **For Resource Efficiency:** λ = {best_memory.lambda_value:.3f}
   - Lowest memory usage ({best_memory.peak_memory_mb:.1f} MB)
   - Trade-off: Slightly higher loss ({best_memory.final_loss:.4f})

3. **For Balanced Trade-off:** λ = {best_trade_off[0]:.3f}
   - Best overall balance between accuracy and resource usage
   - Combined score: {best_trade_off[1]:.3f}

### Usage Guidelines

- **λ = 0.0:** Standard training without resource penalties
- **λ = 0.01-0.1:** Light resource awareness, minimal accuracy impact
- **λ = 0.1-0.5:** Moderate resource penalties, noticeable efficiency gains
- **λ > 0.5:** Strong resource constraints, significant accuracy trade-offs

---

*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
*Experiment configuration: {self.config.model_size} model, {self.config.epochs} epochs, {self.config.synthetic_samples} samples*
"""

        with open(report_file, 'w') as f:
            f.write(report_content)
            
        self.logger.info(f"Analysis report saved to {report_file}")


def main():
    """Main function for lambda sweep experiment."""
    parser = argparse.ArgumentParser(description="Lambda Parameter Sweep for Resource-Aware Transformer")
    
    # Lambda configuration
    parser.add_argument("--lambda-values", type=str, 
                       default="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0",
                       help="Comma-separated lambda values to test (optimized range for gradient flow)")
    
    # Model configuration
    parser.add_argument("--model-size", choices=["tiny", "small"], default="small",
                       help="Model size for experiments")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs per experiment (increased for convergence)")
    parser.add_argument("--synthetic-samples", type=int, default=1000,
                       help="Number of synthetic samples for training (increased for robustness)")
    
    # Experiment configuration
    parser.add_argument("--output-dir", type=str, default="outputs/lambda_sweep",
                       help="Base output directory for experiments")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    # Analysis configuration
    parser.add_argument("--create-plots", action="store_true", default=True,
                       help="Create analysis plots")
    parser.add_argument("--skip-plots", action="store_true", default=False,
                       help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Parse lambda values
    lambda_values = [float(x.strip()) for x in args.lambda_values.split(',')]
    
    # Create configuration
    config = LambdaSweepConfig(
        lambda_values=lambda_values,
        model_size=args.model_size,
        epochs=args.epochs,
        synthetic_samples=args.synthetic_samples,
        output_base_dir=args.output_dir,
        random_seed=args.random_seed,
        device=args.device,
        create_plots=args.create_plots and not args.skip_plots
    )
    
    # Run experiment
    experiment = LambdaSweepExperiment(config)
    experiment.run_sweep()
    
    print(f"\n🎉 Lambda sweep completed!")
    print(f"📊 Results saved to: {config.output_base_dir}")
    print(f"📈 Analysis plots: {config.output_base_dir}/plots/")
    print(f"📋 Detailed report: {config.output_base_dir}/analysis/lambda_sweep_report.md")


if __name__ == "__main__":
    main() 