#!/usr/bin/env python3
"""
Example Training Script for Resource-Aware Transformer.

This script demonstrates how to use the ResourceAwareTrainer with comprehensive
logging, cost tracking, and gate activation monitoring.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

# Local imports
from src.models.config import TransformerConfig, get_small_config
from src.models.gated_transformer import GatedTransformer
from src.training.trainer import ResourceAwareTrainer, TrainingConfig
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessing import TextPreprocessor
from src.tokenizer.bpe_tokenizer import BPETokenizer, BPETokenizerConfig


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def create_sample_dataset(tokenizer, config: TransformerConfig, num_samples: int = 1000) -> DataLoader:
    """Create a simple synthetic dataset for demonstration."""
    import random
    
    # Generate sample text data
    sample_texts = []
    for i in range(num_samples):
        # Create random sentences with different lengths
        sentence_length = random.randint(10, config.max_sequence_length // 2)
        words = [f"word{j}" for j in range(sentence_length)]
        sentence = " ".join(words)
        sample_texts.append(sentence)
    
    # Tokenize and create dataset
    tokenized_data = []
    for text in sample_texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < config.max_sequence_length:
            # Pad to max length
            tokens.extend([tokenizer.pad_token_id] * (config.max_sequence_length - len(tokens)))
        else:
            tokens = tokens[:config.max_sequence_length]
        
        tokenized_data.append({
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(tokens[1:] + [tokenizer.eos_token_id], dtype=torch.long)
        })
    
    # Create DataLoader
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != tokenizer.pad_token_id).long()
        }
    
    return DataLoader(
        tokenized_data,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Keep simple for example
        pin_memory=True
    )


def load_real_dataset(data_path: str, tokenizer, config: TransformerConfig) -> tuple[DataLoader, DataLoader]:
    """Load a real dataset for training."""
    try:
        from src.data.dataset_loader import DatasetConfig
        
        # Create dataset configuration
        dataset_config = DatasetConfig(
            data_path=data_path,
            batch_size=8,
            max_length=config.max_sequence_length,
            num_workers=2,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0
        )
        
        # Create dataset loader
        dataset_loader = DatasetLoader(dataset_config)
        
        # Load datasets (this will automatically split if single file)
        dataset_loader.load_datasets(data=data_path)
        
        # Get data loaders
        train_dataloader = dataset_loader.get_dataloader('train')
        eval_dataloader = dataset_loader.get_dataloader('val')
        
        if train_dataloader is None or eval_dataloader is None:
            raise ValueError("Failed to create data loaders")
        
        return train_dataloader, eval_dataloader
        
    except Exception as e:
        logging.warning(f"Failed to load real dataset from {data_path}: {e}")
        logging.info("Falling back to synthetic dataset")
        return None, None


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Resource-Aware Transformer")
    
    # Model configuration
    parser.add_argument("--model-size", choices=["tiny", "small", "medium"], default="small",
                       help="Model size configuration")
    parser.add_argument("--use-gated-model", action="store_true", default=True,
                       help="Use gated transformer model")
    parser.add_argument("--vocab-size", type=int, default=10000,
                       help="Vocabulary size")
    parser.add_argument("--max-seq-length", type=int, default=128,
                       help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    
    # Loss configuration
    parser.add_argument("--lambda-resource", type=float, default=0.01,
                       help="Resource penalty weight")
    parser.add_argument("--cost-model", choices=["uniform", "layer_weighted", "activation_size"], 
                       default="uniform", help="Cost model for resource penalty")
    
    # Data configuration
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to training data")
    parser.add_argument("--synthetic-samples", type=int, default=1000,
                       help="Number of synthetic samples if no real data")
    
    # Training configuration
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Steps between logging")
    parser.add_argument("--eval-interval", type=int, default=100,
                       help="Steps between evaluation")
    parser.add_argument("--save-interval", type=int, default=500,
                       help="Steps between checkpoint saves")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    # Logging configuration
    parser.add_argument("--use-wandb", action="store_true", default=False,
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="resource-aware-transformer",
                       help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                       help="W&B run name")
    parser.add_argument("--use-tensorboard", action="store_true", default=True,
                       help="Use TensorBoard logging")
    
    # Tracking configuration
    parser.add_argument("--track-memory", action="store_true", default=True,
                       help="Track memory usage")
    parser.add_argument("--track-computation-time", action="store_true", default=True,
                       help="Track computation time")
    parser.add_argument("--memory-log-interval", type=int, default=50,
                       help="Steps between memory logging")
    
    # System configuration
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--num-workers", type=int, default=2,
                       help="Number of data loader workers")
    parser.add_argument("--use-amp", action="store_true", default=True,
                       help="Use automatic mixed precision")
    
    # Debugging
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--profile-steps", type=int, default=0,
                       help="Number of steps to profile (0 = disabled)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(args.output_dir) / "training.log"
    setup_logging(args.log_level, str(log_file))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Resource-Aware Transformer Training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create model configuration
    if args.model_size == "tiny":
        from src.models.config import get_tiny_config
        model_config = get_tiny_config()
    elif args.model_size == "small":
        model_config = get_small_config()
    elif args.model_size == "medium":
        from src.models.config import get_medium_config
        model_config = get_medium_config()
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    # Update model config with arguments
    model_config.vocab_size = args.vocab_size
    model_config.max_sequence_length = args.max_seq_length
    model_config.max_position_embeddings = args.max_seq_length
    model_config.use_recomputation_gates = args.use_gated_model
    
    logger.info(f"Model configuration: {model_config}")
    
    # Create tokenizer
    tokenizer_config = BPETokenizerConfig(vocab_size=args.vocab_size)
    tokenizer = BPETokenizer(tokenizer_config)
    
    # Train tokenizer on sample text if not using real data
    if not args.data_path:
        logger.info("Training tokenizer on sample text...")
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world, this is a sample text for training.",
            "Machine learning and artificial intelligence are fascinating fields.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models can learn complex patterns from data.",
            "Transformers have revolutionized the field of natural language processing.",
            "Attention mechanisms allow models to focus on relevant parts of input.",
            "Training neural networks requires large amounts of data and computation.",
            "Tokenization is the process of converting text into numerical tokens.",
            "Language models can generate coherent and contextually relevant text."
        ] * 10  # Repeat to have enough training data
        tokenizer.train(sample_texts)
        logger.info(f"Tokenizer trained with vocabulary size: {tokenizer.vocab_size}")
    
    # Load or create dataset
    train_dataloader = eval_dataloader = None
    
    if args.data_path:
        logger.info(f"Loading dataset from {args.data_path}")
        train_dataloader, eval_dataloader = load_real_dataset(
            args.data_path, tokenizer, model_config
        )
    
    if train_dataloader is None:
        logger.info(f"Creating synthetic dataset with {args.synthetic_samples} samples")
        train_dataloader = create_sample_dataset(
            tokenizer, model_config, args.synthetic_samples
        )
        # Create smaller eval dataset
        eval_dataloader = create_sample_dataset(
            tokenizer, model_config, args.synthetic_samples // 5
        )
    
    # Create model
    if args.use_gated_model:
        model = GatedTransformer(model_config)
        logger.info("Created GatedTransformer")
    else:
        from src.models.transformer import BaselineTransformer
        model = BaselineTransformer(model_config)
        logger.info("Created standard TransformerModel")
    
    # Create training configuration
    training_config = TrainingConfig(
        model_config=model_config,
        use_gated_model=args.use_gated_model,
        use_resource_aware_loss=True,
        
        # Training hyperparameters
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        
        # Training schedule
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Loss configuration
        lambda_resource=args.lambda_resource,
        cost_model=args.cost_model,
        
        # Mixed precision
        use_amp=args.use_amp,
        
        # Logging and evaluation
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        
        # Checkpointing
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from,
        
        # Logging backends
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        
        # Cost tracking
        track_memory=args.track_memory,
        track_computation_time=args.track_computation_time,
        memory_log_interval=args.memory_log_interval,
        
        # Device configuration
        device=device,
        num_workers=args.num_workers,
        
        # Debugging
        debug_mode=args.debug,
        profile_steps=args.profile_steps,
    )
    
    logger.info(f"Training configuration: {training_config}")
    
    # Create trainer
    trainer = ResourceAwareTrainer(
        config=training_config,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
    
    logger.info("Created ResourceAwareTrainer")
    
    try:
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Get training summary
        summary = trainer.get_training_summary()
        logger.info(f"Training completed! Summary: {summary}")
        
        # Save final summary
        summary_path = Path(args.output_dir) / "training_summary.json"
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training summary saved to {summary_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save emergency checkpoint
        emergency_path = trainer.save_checkpoint("emergency_checkpoint")
        logger.info(f"Emergency checkpoint saved to {emergency_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        trainer._cleanup_logging()
        logger.info("Training session ended")


if __name__ == "__main__":
    main() 