"""
Baseline Transformer Model Example.

This example demonstrates how to use the baseline transformer model
for various tasks including:
- Model initialization and configuration
- Forward pass and inference
- Text generation
- Model serialization
- Integration with tokenizer
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our components
from src.models import (
    TransformerConfig,
    BaselineTransformer,
    get_tiny_config,
    get_small_config,
    create_attention_mask
)
from src.tokenizer import BPETokenizer, BPETokenizerConfig


def example_1_basic_model_usage():
    """Example 1: Basic model initialization and forward pass."""
    print("\n" + "="*60)
    print("Example 1: Basic Model Usage")
    print("="*60)
    
    # Create a small configuration for demonstration
    config = get_tiny_config()
    print(f"Model configuration:")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Number of layers: {config.num_hidden_layers}")
    print(f"  - Number of attention heads: {config.num_attention_heads}")
    
    # Initialize the model
    model = BaselineTransformer(config)
    
    # Get model size information
    param_counts = model.get_parameter_count()
    print(f"\nModel parameters:")
    for component, count in param_counts.items():
        print(f"  - {component}: {count:,}")
    
    # Create sample input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Sample input: {input_ids[0].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"\nOutput shapes:")
    print(f"  - Hidden states: {outputs['last_hidden_state'].shape}")
    print(f"  - Logits: {outputs['logits'].shape}")
    
    # Get predictions
    predictions = torch.argmax(outputs['logits'], dim=-1)
    print(f"  - Predictions: {predictions[0].tolist()}")


def example_2_attention_analysis():
    """Example 2: Analyzing attention patterns."""
    print("\n" + "="*60)
    print("Example 2: Attention Analysis")
    print("="*60)
    
    config = get_tiny_config()
    model = BaselineTransformer(config)
    
    # Create input with some structure
    batch_size, seq_len = 1, 8
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # Sequential tokens
    
    # Forward pass with attention weights
    with torch.no_grad():
        outputs = model(
            input_ids,
            return_attention_weights=True,
            return_hidden_states=True
        )
    
    attention_weights = outputs['attention_weights']
    print(f"Number of layers with attention: {len(attention_weights)}")
    print(f"Attention shape per layer: {attention_weights[0].shape}")
    
    # Analyze first layer, first head attention
    first_layer_attention = attention_weights[0][0, 0]  # [seq_len, seq_len]
    print(f"\nFirst layer, first head attention pattern:")
    print(f"Shape: {first_layer_attention.shape}")
    
    # Show attention matrix (rounded for readability)
    attention_matrix = first_layer_attention.numpy()
    print("Attention matrix (rows attend to columns):")
    for i in range(seq_len):
        row = " ".join([f"{val:.3f}" for val in attention_matrix[i]])
        print(f"  Token {i+1}: {row}")


def example_3_text_generation():
    """Example 3: Text generation with different strategies."""
    print("\n" + "="*60)
    print("Example 3: Text Generation")
    print("="*60)
    
    config = get_tiny_config()
    model = BaselineTransformer(config)
    
    # Create a prompt
    prompt = torch.tensor([[1, 2, 3]])  # Simple 3-token prompt
    print(f"Prompt tokens: {prompt[0].tolist()}")
    
    # Greedy generation
    print("\nGreedy generation:")
    with torch.no_grad():
        generated_greedy = model.generate(
            prompt,
            max_length=10,
            do_sample=False,
            temperature=1.0
        )
    print(f"Generated: {generated_greedy[0].tolist()}")
    
    # Sampling with temperature
    print("\nSampling with temperature=0.8:")
    with torch.no_grad():
        generated_sample = model.generate(
            prompt,
            max_length=10,
            do_sample=True,
            temperature=0.8
        )
    print(f"Generated: {generated_sample[0].tolist()}")
    
    # Top-k sampling
    print("\nTop-k sampling (k=5):")
    with torch.no_grad():
        generated_topk = model.generate(
            prompt,
            max_length=10,
            do_sample=True,
            top_k=5,
            temperature=1.0
        )
    print(f"Generated: {generated_topk[0].tolist()}")


def example_4_model_serialization():
    """Example 4: Model saving and loading."""
    print("\n" + "="*60)
    print("Example 4: Model Serialization")
    print("="*60)
    
    config = get_tiny_config()
    original_model = BaselineTransformer(config)
    
    # Create test input
    test_input = torch.randint(1, config.vocab_size, (1, 5))
    
    # Get original output
    with torch.no_grad():
        original_output = original_model(test_input)
    
    # Save model
    save_path = Path("./temp_model")
    original_model.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")
    
    # Load model
    loaded_model = BaselineTransformer.from_pretrained(save_path)
    print("Model loaded successfully")
    
    # Test that outputs match
    with torch.no_grad():
        loaded_output = loaded_model(test_input)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(original_output['logits'] - loaded_output['logits']))
    print(f"Maximum difference in logits: {max_diff.item():.2e}")
    
    # Clean up
    import shutil
    if save_path.exists():
        shutil.rmtree(save_path)
        print("Temporary files cleaned up")


def example_5_integration_with_tokenizer():
    """Example 5: Integration with BPE tokenizer."""
    print("\n" + "="*60)
    print("Example 5: Integration with Tokenizer")
    print("="*60)
    
    # Create tokenizer configuration
    tokenizer_config = BPETokenizerConfig(
        vocab_size=1000,
        min_frequency=1,
        special_tokens=['<PAD>', '<UNK>', '<BOS>', '<EOS>'],
        pad_token='<PAD>',
        unk_token='<UNK>',
        bos_token='<BOS>',
        eos_token='<EOS>'
    )
    
    # Initialize tokenizer
    tokenizer = BPETokenizer(tokenizer_config)
    
    # Train on sample corpus
    sample_corpus = [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Transformers are powerful models.",
        "Natural language processing rocks!"
    ]
    
    print("Training tokenizer on sample corpus...")
    tokenizer.train(sample_corpus)
    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")
    
    # Create model configuration matching tokenizer
    # Use safe defaults if special tokens are not available
    pad_token_id = tokenizer.special_tokens.get('<PAD>', 0)
    bos_token_id = tokenizer.special_tokens.get('<BOS>', 1)
    eos_token_id = tokenizer.special_tokens.get('<EOS>', 2)
    unk_token_id = tokenizer.special_tokens.get('<UNK>', 3)
    
    model_config = TransformerConfig(
        vocab_size=len(tokenizer.vocab),
        max_sequence_length=64,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        unk_token_id=unk_token_id
    )
    
    # Initialize model
    model = BaselineTransformer(model_config)
    print(f"Model initialized with vocab size: {model_config.vocab_size}")
    
    # Tokenize sample text
    sample_text = "Hello world, this is a test."
    tokens = tokenizer.encode(sample_text)
    print(f"\nOriginal text: '{sample_text}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{tokenizer.decode(tokens)}'")
    
    # Convert to tensor and add batch dimension
    input_ids = torch.tensor([tokens])
    print(f"Input tensor shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    
    # Generate continuation
    print("\nGenerating text continuation...")
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=min(20, model_config.max_sequence_length),
            do_sample=True,
            temperature=0.8,
            eos_token_id=model_config.eos_token_id
        )
    
    # Decode generated text
    generated_tokens = generated[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generated text: '{generated_text}'")


def example_6_training_simulation():
    """Example 6: Simulating a training step."""
    print("\n" + "="*60)
    print("Example 6: Training Simulation")
    print("="*60)
    
    config = get_tiny_config()
    model = BaselineTransformer(config)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    
    # Create sample batch
    batch_size, seq_len = 4, 12
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    
    # Create targets (shifted input for language modeling)
    target_ids = torch.cat([input_ids[:, 1:], torch.zeros(batch_size, 1, dtype=torch.long)], dim=1)
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(input_ids)
    logits = outputs['logits']
    
    # Compute loss
    loss = loss_fn(logits.view(-1, config.vocab_size), target_ids.view(-1))
    print(f"\nLoss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"Gradient norm: {total_grad_norm:.4f}")
    
    # Optimizer step
    optimizer.step()
    print("Optimizer step completed")
    
    # Evaluation mode
    model.eval()
    with torch.no_grad():
        eval_outputs = model(input_ids)
        eval_loss = loss_fn(eval_outputs['logits'].view(-1, config.vocab_size), target_ids.view(-1))
    print(f"Evaluation loss: {eval_loss.item():.4f}")


def example_7_model_comparison():
    """Example 7: Comparing different model sizes."""
    print("\n" + "="*60)
    print("Example 7: Model Size Comparison")
    print("="*60)
    
    configs = {
        "Tiny": get_tiny_config(),
        "Small": get_small_config()
    }
    
    for name, config in configs.items():
        print(f"\n{name} Model:")
        print(f"  - Vocabulary: {config.vocab_size:,}")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Layers: {config.num_hidden_layers}")
        print(f"  - Attention heads: {config.num_attention_heads}")
        
        # Get size info
        size_info = config.get_model_size_info()
        print(f"  - Parameters: {size_info['total_parameters']:,}")
        print(f"  - Memory estimate: {size_info['memory_estimate_mb']:.1f} MB")
        
        # Create model and measure actual parameters
        model = BaselineTransformer(config)
        actual_params = sum(p.numel() for p in model.parameters())
        print(f"  - Actual parameters: {actual_params:,}")
        
        # Test inference time
        input_ids = torch.randint(1, config.vocab_size, (1, 10))
        
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"  - Average inference time: {avg_time*1000:.2f} ms")


def main():
    """Run all examples."""
    print("Baseline Transformer Model Examples")
    print("="*60)
    
    try:
        example_1_basic_model_usage()
        example_2_attention_analysis()
        example_3_text_generation()
        example_4_model_serialization()
        example_5_integration_with_tokenizer()
        example_6_training_simulation()
        example_7_model_comparison()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 