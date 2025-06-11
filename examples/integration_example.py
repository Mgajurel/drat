"""
Complete Text Processing Pipeline Integration Example

This example demonstrates how to use the integrated text processing pipeline
that combines preprocessing, tokenization, and dataset loading into a unified
workflow suitable for machine learning applications.
"""

import torch
import logging
from pathlib import Path
from typing import List, Dict, Any

# Import the integrated components
from src.integration.pipeline import (
    TextProcessingPipeline, 
    PipelineConfig,
    create_default_pipeline,
    create_production_pipeline
)
from src.integration.validation import validate_pipeline, run_consistency_tests
from src.integration.benchmarks import run_quick_benchmark, run_production_benchmark
from src.data.preprocessing import PreprocessingConfig
from src.tokenizer.bpe_tokenizer import BPETokenizerConfig
from src.data.dataset_loader import DatasetConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_sample_data() -> Dict[str, List[str]]:
    """Prepare sample text data for demonstration."""
    
    # Sample texts representing different domains and lengths
    train_texts = [
        "Machine learning is revolutionizing how we process and understand data.",
        "Natural language processing enables computers to comprehend human communication.",
        "Deep learning models require large datasets and computational resources.",
        "Text preprocessing is crucial for achieving optimal model performance.",
        "Tokenization breaks down text into meaningful units for analysis.",
        "PyTorch provides excellent tools for building neural network architectures.",
        "Data augmentation techniques can improve model generalization capabilities.",
        "Transfer learning allows leveraging pre-trained models for new tasks.",
        "Attention mechanisms have transformed the field of natural language understanding.",
        "Transformers have become the backbone of modern NLP applications.",
        "BERT and GPT models have set new standards for language understanding.",
        "Fine-tuning pre-trained models is more efficient than training from scratch.",
        "Batch processing significantly improves computational efficiency in training.",
        "Validation datasets help prevent overfitting during model development.",
        "Cross-validation provides robust estimates of model performance.",
        "Hyperparameter tuning is essential for optimizing model performance.",
        "Regularization techniques help prevent overfitting in neural networks.",
        "Dropout is a simple yet effective regularization method.",
        "Learning rate scheduling can improve training stability and convergence.",
        "Early stopping prevents overfitting by monitoring validation metrics."
    ]
    
    val_texts = [
        "Evaluation metrics provide insights into model strengths and weaknesses.",
        "Precision and recall are fundamental metrics for classification tasks.",
        "F1-score balances precision and recall in a single metric.",
        "Confusion matrices visualize classification performance across classes.",
        "ROC curves help evaluate binary classification model performance.",
        "Cross-entropy loss is commonly used for multi-class classification.",
        "Mean squared error is appropriate for regression tasks.",
        "Gradient descent optimizes model parameters during training."
    ]
    
    test_texts = [
        "Feature engineering remains important even with deep learning approaches.",
        "Model interpretability is crucial for understanding decision processes.",
        "Ensemble methods combine multiple models for improved performance.",
        "Data quality significantly impacts model performance and reliability."
    ]
    
    return {
        'train': train_texts,
        'val': val_texts,
        'test': test_texts
    }


def example_1_basic_pipeline():
    """Example 1: Basic pipeline usage with default settings."""
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("="*60)
    
    # Create default pipeline
    pipeline = create_default_pipeline()
    print("✓ Created default pipeline")
    
    # Prepare sample data
    data = prepare_sample_data()
    all_texts = data['train'] + data['val'] + data['test']
    
    # Train tokenizer
    print("Training tokenizer...")
    pipeline.train_tokenizer(all_texts, vocab_size=2000)
    print(f"✓ Trained tokenizer with vocab size: {len(pipeline.tokenizer.vocab)}")
    
    # Process texts
    print("Processing texts...")
    sequences = pipeline.process_texts(all_texts[:10])
    print(f"✓ Processed {len(sequences)} texts into sequences")
    print(f"  Average sequence length: {sum(len(seq) for seq in sequences) / len(sequences):.1f}")
    
    # Create datasets
    print("Creating datasets...")
    pipeline.create_datasets(texts=all_texts)
    stats = pipeline.dataset_loader.get_stats()
    print("✓ Created datasets:")
    for split, split_stats in stats.items():
        if split_stats['num_sequences'] > 0:
            print(f"  {split}: {split_stats['num_sequences']} sequences")
    
    # Get performance statistics
    perf_stats = pipeline.get_performance_stats()
    print(f"✓ Performance: {perf_stats['texts_per_second']:.1f} texts/sec, "
          f"{perf_stats['tokens_per_second']:.1f} tokens/sec")


def example_2_custom_configuration():
    """Example 2: Custom pipeline configuration for specific requirements."""
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Pipeline Configuration")
    print("="*60)
    
    # Create custom configuration
    preprocessing_config = PreprocessingConfig(
        lowercase=True,
        remove_extra_whitespace=True,
        remove_special_chars=False,  # Keep punctuation
        min_length=10,
        max_length=500,
        filter_language='en'
    )
    
    tokenization_config = BPETokenizerConfig(
        vocab_size=5000,
        min_frequency=2,
        special_tokens=['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>', '<SEP>'],
        pad_token='<PAD>',
        unk_token='<UNK>',
        bos_token='<BOS>',
        eos_token='<EOS>',
        mask_token='<MASK>'
    )
    
    dataset_config = DatasetConfig(
        batch_size=16,
        max_length=128,
        padding_strategy='dynamic',
        train_split=0.8,
        val_split=0.15,
        test_split=0.05,
        shuffle_train=True,
        drop_last=True
    )
    
    config = PipelineConfig(
        preprocessing=preprocessing_config,
        tokenization=tokenization_config,
        dataset=dataset_config,
        cache_processed_data=True,
        cache_dir="cache",
        validate_consistency=True,
        log_performance=True
    )
    
    # Create pipeline with custom config
    pipeline = TextProcessingPipeline(config)
    print("✓ Created pipeline with custom configuration")
    
    # Prepare data
    data = prepare_sample_data()
    
    # Train tokenizer with larger vocabulary
    print("Training tokenizer with custom settings...")
    pipeline.train_tokenizer(data['train'], vocab_size=5000)
    print(f"✓ Trained tokenizer with vocab size: {len(pipeline.tokenizer.vocab)}")
    
    # Process data with separate splits
    print("Creating datasets with separate splits...")
    pipeline.create_datasets(
        train_data=data['train'],
        val_data=data['val'],
        test_data=data['test']
    )
    
    # Demonstrate dataloader usage
    print("Demonstrating dataloader functionality...")
    train_dataloader = pipeline.get_dataloader('train')
    val_dataloader = pipeline.get_dataloader('val')
    
    # Process a few batches
    print("Processing training batches:")
    for i, batch in enumerate(train_dataloader):
        batch_size, seq_len = batch['input_ids'].shape
        print(f"  Batch {i+1}: {batch_size} sequences, max length {seq_len}")
        if i >= 2:  # Show first 3 batches
            break
    
    print("Processing validation batches:")
    for i, batch in enumerate(val_dataloader):
        batch_size, seq_len = batch['input_ids'].shape
        print(f"  Batch {i+1}: {batch_size} sequences, max length {seq_len}")
        if i >= 1:  # Show first 2 batches
            break


def example_3_production_pipeline():
    """Example 3: Production-ready pipeline with optimization."""
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Production Pipeline")
    print("="*60)
    
    # Create production pipeline
    pipeline = create_production_pipeline(
        vocab_size=10000,
        batch_size=32,
        max_length=256
    )
    print("✓ Created production-optimized pipeline")
    
    # Prepare larger dataset (simulated)
    data = prepare_sample_data()
    # Extend training data for production demo
    extended_train = data['train'] * 3  # Simulate larger dataset
    
    # Train tokenizer
    print("Training production tokenizer...")
    pipeline.train_tokenizer(extended_train, vocab_size=10000)
    print(f"✓ Trained production tokenizer with vocab size: {len(pipeline.tokenizer.vocab)}")
    
    # Process with caching
    print("Processing texts with caching enabled...")
    sequences = pipeline.process_texts(
        extended_train, 
        return_tensors=False, 
        cache_key="production_data"
    )
    print(f"✓ Processed {len(sequences)} sequences with caching")
    
    # Create optimized datasets
    print("Creating optimized datasets...")
    pipeline.create_datasets(sequences=sequences)
    
    # Demonstrate efficient batching
    dataloader = pipeline.get_dataloader('train')
    print("Demonstrating efficient batching:")
    
    batch_sizes = []
    sequence_lengths = []
    
    for i, batch in enumerate(dataloader):
        batch_size, seq_len = batch['input_ids'].shape
        batch_sizes.append(batch_size)
        sequence_lengths.append(seq_len)
        
        if i >= 4:  # Analyze first 5 batches
            break
    
    print(f"  Average batch size: {sum(batch_sizes) / len(batch_sizes):.1f}")
    print(f"  Sequence length range: {min(sequence_lengths)}-{max(sequence_lengths)}")
    
    # Performance statistics
    perf_stats = pipeline.get_performance_stats()
    print("Production performance metrics:")
    print(f"  Texts per second: {perf_stats['texts_per_second']:.1f}")
    print(f"  Tokens per second: {perf_stats['tokens_per_second']:.1f}")
    print(f"  Average sequence length: {perf_stats.get('avg_sequence_length', 0):.1f}")


def example_4_validation_and_testing():
    """Example 4: Pipeline validation and consistency testing."""
    
    print("\n" + "="*60)
    print("EXAMPLE 4: Pipeline Validation and Testing")
    print("="*60)
    
    # Create pipeline for validation
    pipeline = create_default_pipeline()
    data = prepare_sample_data()
    all_texts = data['train'] + data['val']
    
    # Train tokenizer
    pipeline.train_tokenizer(all_texts, vocab_size=3000)
    print("✓ Prepared pipeline for validation")
    
    # Create datasets for validation
    pipeline.create_datasets(texts=all_texts)
    
    # Run comprehensive validation
    print("Running comprehensive validation...")
    validation_results = validate_pipeline(pipeline)
    
    print("Validation Results:")
    print(f"  Overall status: {'PASSED' if validation_results.passed else 'FAILED'}")
    print(f"  Tests passed: {validation_results.passed_tests}/{validation_results.total_tests}")
    
    if validation_results.errors:
        print("  Errors:")
        for error in validation_results.errors:
            print(f"    - {error}")
    
    if validation_results.warnings:
        print("  Warnings:")
        for warning in validation_results.warnings:
            print(f"    - {warning}")
    
    # Run consistency tests
    print("\nRunning consistency tests...")
    consistency_results = run_consistency_tests(pipeline, data)
    
    print("Consistency Results:")
    for test_name, passed in consistency_results.items():
        if test_name != 'errors':
            status = "PASSED" if passed else "FAILED"
            print(f"  {test_name}: {status}")
    
    if consistency_results.get('errors'):
        print("  Consistency errors:")
        for error in consistency_results['errors']:
            print(f"    - {error}")


def example_5_performance_benchmarking():
    """Example 5: Performance benchmarking and optimization."""
    
    print("\n" + "="*60)
    print("EXAMPLE 5: Performance Benchmarking")
    print("="*60)
    
    # Create pipeline for benchmarking
    pipeline = create_default_pipeline()
    data = prepare_sample_data()
    
    # Prepare larger dataset for meaningful benchmarks
    benchmark_texts = (data['train'] + data['val']) * 2  # Double the data
    
    # Train tokenizer
    pipeline.train_tokenizer(benchmark_texts, vocab_size=4000)
    print("✓ Prepared pipeline for benchmarking")
    
    # Run quick benchmark
    print("Running quick benchmark...")
    quick_metrics = run_quick_benchmark(pipeline, benchmark_texts)
    
    print("Quick Benchmark Results:")
    print(f"  Total time: {quick_metrics.total_time:.3f}s")
    print(f"  Texts per second: {quick_metrics.texts_per_second:.1f}")
    print(f"  Tokens per second: {quick_metrics.tokens_per_second:.1f}")
    print(f"  Peak memory: {quick_metrics.peak_memory_mb:.1f} MB")
    print(f"  Memory efficiency: {quick_metrics.memory_efficiency:.1f} tokens/MB")
    
    # Run detailed benchmarking
    print("\nRunning detailed component benchmarks...")
    runner = BenchmarkRunner(warmup_iterations=2, measurement_iterations=5)
    
    # Benchmark preprocessing
    prep_results = runner.benchmark_preprocessing(pipeline, benchmark_texts[:20])
    print("Preprocessing performance:")
    for batch_size, results in prep_results.items():
        timing = results['timing']
        print(f"  {batch_size}: {timing['texts_per_second']:.1f} texts/sec")
    
    # Benchmark tokenization
    tok_results = runner.benchmark_tokenization(pipeline, benchmark_texts[:20])
    print("Tokenization performance:")
    for batch_size, results in tok_results.items():
        timing = results['timing']
        print(f"  {batch_size}: {timing['tokens_per_second']:.1f} tokens/sec")


def example_6_ml_training_simulation():
    """Example 6: Simulating a machine learning training workflow."""
    
    print("\n" + "="*60)
    print("EXAMPLE 6: ML Training Workflow Simulation")
    print("="*60)
    
    # Step 1: Setup pipeline for ML training
    config = PipelineConfig()
    config.dataset.batch_size = 8
    config.dataset.max_length = 64
    config.dataset.shuffle_train = True
    
    pipeline = TextProcessingPipeline(config)
    data = prepare_sample_data()
    
    # Step 2: Train tokenizer
    print("Step 1: Training tokenizer...")
    pipeline.train_tokenizer(data['train'], vocab_size=2000)
    vocab_size = len(pipeline.tokenizer.vocab)
    print(f"✓ Tokenizer ready with vocab size: {vocab_size}")
    
    # Step 3: Create datasets
    print("Step 2: Creating datasets...")
    pipeline.create_datasets(
        train_data=data['train'],
        val_data=data['val'],
        test_data=data['test']
    )
    
    stats = pipeline.dataset_loader.get_stats()
    print("✓ Datasets created:")
    for split, split_stats in stats.items():
        if split_stats['num_sequences'] > 0:
            print(f"  {split}: {split_stats['num_sequences']} sequences")
    
    # Step 4: Simulate training loop
    print("Step 3: Simulating training loop...")
    
    train_dataloader = pipeline.get_dataloader('train')
    val_dataloader = pipeline.get_dataloader('val')
    
    # Simulate training epochs
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}:")
        
        # Training phase
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Simulate forward pass (just compute a dummy loss)
            batch_size, seq_len = input_ids.shape
            dummy_loss = torch.randn(1).abs().item()  # Random positive loss
            train_loss += dummy_loss
            train_batches += 1
            
            if batch_idx == 0:
                print(f"  Training batch shape: {input_ids.shape}")
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        print(f"  Training - Batches: {train_batches}, Avg Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        val_loss = 0.0
        val_batches = 0
        
        for batch in val_dataloader:
            input_ids = batch['input_ids']
            
            # Simulate validation forward pass
            dummy_loss = torch.randn(1).abs().item()
            val_loss += dummy_loss
            val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        print(f"  Validation - Batches: {val_batches}, Avg Loss: {avg_val_loss:.4f}")
    
    # Step 5: Final evaluation on test set
    print("\nStep 4: Final evaluation...")
    test_dataloader = pipeline.get_dataloader('test')
    
    test_batches = 0
    total_sequences = 0
    
    for batch in test_dataloader:
        input_ids = batch['input_ids']
        total_sequences += input_ids.shape[0]
        test_batches += 1
    
    print(f"✓ Test evaluation complete: {test_batches} batches, {total_sequences} sequences")
    
    # Performance summary
    final_stats = pipeline.get_performance_stats()
    print("\nFinal Performance Summary:")
    print(f"  Total texts processed: {final_stats['texts_processed']}")
    print(f"  Total tokens generated: {final_stats['tokens_generated']}")
    print(f"  Processing speed: {final_stats.get('texts_per_second', 0):.1f} texts/sec")


def example_7_configuration_management():
    """Example 7: Configuration management and persistence."""
    
    print("\n" + "="*60)
    print("EXAMPLE 7: Configuration Management")
    print("="*60)
    
    # Create custom configuration
    config = PipelineConfig(
        preprocessing=PreprocessingConfig(
            lowercase=True,
            remove_extra_whitespace=True,
            min_length=5,
            max_length=200
        ),
        tokenization=BPETokenizerConfig(
            vocab_size=3000,
            min_frequency=2
        ),
        dataset=DatasetConfig(
            batch_size=16,
            max_length=100
        ),
        cache_processed_data=True,
        validate_consistency=True
    )
    
    # Save configuration
    config_path = "example_pipeline_config.json"
    pipeline = TextProcessingPipeline(config)
    pipeline.save_config(config_path)
    print(f"✓ Saved configuration to {config_path}")
    
    # Load configuration
    loaded_pipeline = TextProcessingPipeline.from_config_file(config_path)
    print("✓ Loaded configuration from file")
    
    # Verify configuration
    print("Configuration verification:")
    print(f"  Batch size: {loaded_pipeline.config.dataset.batch_size}")
    print(f"  Vocab size: {loaded_pipeline.config.tokenization.vocab_size}")
    print(f"  Max length: {loaded_pipeline.config.dataset.max_length}")
    print(f"  Caching enabled: {loaded_pipeline.config.cache_processed_data}")
    
    # Clean up
    Path(config_path).unlink()
    print("✓ Cleaned up configuration file")


def main():
    """Run all integration examples."""
    
    print("TEXT PROCESSING PIPELINE INTEGRATION EXAMPLES")
    print("=" * 80)
    print("This example demonstrates the complete integration of text preprocessing,")
    print("tokenization, and dataset loading into a unified pipeline suitable for")
    print("machine learning applications.")
    
    try:
        example_1_basic_pipeline()
        example_2_custom_configuration()
        example_3_production_pipeline()
        example_4_validation_and_testing()
        example_5_performance_benchmarking()
        example_6_ml_training_simulation()
        example_7_configuration_management()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey takeaways:")
        print("1. The integrated pipeline simplifies text processing workflows")
        print("2. Custom configurations allow optimization for specific use cases")
        print("3. Validation and benchmarking ensure reliability and performance")
        print("4. The pipeline is ready for production ML training workflows")
        print("5. Configuration management enables reproducible experiments")
        
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 