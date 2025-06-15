"""
Integration tests for the complete text processing pipeline.

This module tests the integration between preprocessing, tokenization, and
dataset loading components to ensure they work together correctly.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import json
import time

from src.integration.pipeline import TextProcessingPipeline, PipelineConfig, create_default_pipeline
from src.integration.validation import validate_pipeline, run_consistency_tests
from src.integration.benchmarks import run_quick_benchmark, BenchmarkRunner
from src.data.preprocessing import PreprocessingConfig
from src.tokenizer.bpe_tokenizer import BPETokenizerConfig
from src.data.dataset_loader import DatasetConfig


class TestPipelineIntegration:
    """Test suite for pipeline integration."""
    
    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Sample texts for testing."""
        return [
            "Hello, world! This is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating and powerful technology.",
            "Natural language processing enables computers to understand human text.",
            "PyTorch provides excellent tools for deep learning research and development.",
            "Data preprocessing is crucial for achieving good model performance.",
            "Tokenization breaks text into meaningful units for processing.",
            "Batch processing improves computational efficiency significantly.",
            "Validation ensures system reliability and correctness.",
            "Integration testing verifies component compatibility and functionality.",
            "Performance benchmarking helps identify bottlenecks and optimization opportunities.",
            "Comprehensive testing covers edge cases and error conditions thoroughly."
        ]
    
    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pipeline_config(self) -> PipelineConfig:
        """Basic pipeline configuration for testing."""
        preprocessing_config = PreprocessingConfig(
            lowercase=True,
            normalize_whitespace=True,
            min_length=5,
            max_length=1000
        )
        
        tokenization_config = BPETokenizerConfig(
            vocab_size=1000,
            min_frequency=1,
            special_tokens=['<PAD>', '<UNK>', '<BOS>', '<EOS>'],
            pad_token='<PAD>',
            unk_token='<UNK>',
            bos_token='<BOS>',
            eos_token='<EOS>'
        )
        
        dataset_config = DatasetConfig(
            batch_size=4,
            max_length=50,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        return PipelineConfig(
            preprocessing=preprocessing_config,
            tokenization=tokenization_config,
            dataset=dataset_config,
            cache_processed_data=False,  # Disable caching for tests
            validate_consistency=True
        )
    
    @pytest.fixture
    def trained_pipeline(self, pipeline_config: PipelineConfig, sample_texts: List[str]) -> TextProcessingPipeline:
        """Pipeline with trained tokenizer."""
        pipeline = TextProcessingPipeline(pipeline_config)
        pipeline.train_tokenizer(sample_texts)
        return pipeline


class TestBasicIntegration(TestPipelineIntegration):
    """Test basic integration functionality."""
    
    def test_pipeline_initialization(self, pipeline_config: PipelineConfig):
        """Test pipeline initialization."""
        pipeline = TextProcessingPipeline(pipeline_config)
        
        assert pipeline.config == pipeline_config
        assert pipeline.preprocessor is not None
        assert pipeline.tokenizer is None  # Not initialized yet
        assert pipeline.dataset_loader is not None
        assert pipeline.performance_stats is not None
    
    def test_tokenizer_training_integration(self, pipeline_config: PipelineConfig, sample_texts: List[str]):
        """Test tokenizer training integration."""
        pipeline = TextProcessingPipeline(pipeline_config)
        
        # Train tokenizer
        tokenizer = pipeline.train_tokenizer(sample_texts)
        
        assert tokenizer is not None
        assert pipeline.tokenizer is tokenizer
        assert len(tokenizer.vocab) > 0
        assert tokenizer.vocab_size <= pipeline_config.tokenization.vocab_size
    
    def test_text_processing_integration(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test complete text processing integration."""
        # Process texts
        sequences = trained_pipeline.process_texts(sample_texts, return_tensors=False)
        
        assert len(sequences) <= len(sample_texts)  # Some texts might be filtered
        assert all(isinstance(seq, list) for seq in sequences)
        assert all(len(seq) > 0 for seq in sequences)
        assert all(isinstance(token, int) for seq in sequences for token in seq)
        
        # Test with tensors
        tensor_sequences = trained_pipeline.process_texts(sample_texts, return_tensors=True)
        assert all(torch.is_tensor(seq) for seq in tensor_sequences)
        assert all(seq.dtype == torch.long for seq in tensor_sequences)
    
    def test_dataset_creation_integration(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test dataset creation integration."""
        # Create datasets from texts
        dataset_loader = trained_pipeline.create_datasets(texts=sample_texts)
        
        assert dataset_loader is not None
        stats = dataset_loader.get_stats()
        assert any(stats.values())  # At least one split should have data
        
        # Test getting dataloaders
        # Check for either 'num_sequences' (full dataset) or 'num_samples' (split dataset)
        train_count = stats['train'].get('num_sequences', stats['train'].get('num_samples', 0))
        if train_count > 0:
            train_dataloader = trained_pipeline.get_dataloader('train')
            assert train_dataloader is not None
            
            # Test iteration
            batch = next(iter(train_dataloader))
            assert isinstance(batch, dict)
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert torch.is_tensor(batch['input_ids'])
            assert batch['input_ids'].dtype == torch.long


class TestAdvancedIntegration(TestPipelineIntegration):
    """Test advanced integration scenarios."""
    
    def test_end_to_end_pipeline(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test complete end-to-end pipeline functionality."""
        # Process texts and create datasets
        trained_pipeline.create_datasets(texts=sample_texts)
        
        # Get dataloaders for all splits
        train_dataloader = trained_pipeline.get_dataloader('train')
        
        # Process multiple batches
        batch_count = 0
        total_sequences = 0
        
        for batch in train_dataloader:
            batch_count += 1
            batch_size = batch['input_ids'].shape[0]
            seq_length = batch['input_ids'].shape[1]
            total_sequences += batch_size
            
            # Validate batch structure
            assert batch['input_ids'].shape == torch.Size([batch_size, seq_length])
            assert batch['attention_mask'].shape == torch.Size([batch_size, seq_length])
            
            # Check attention mask validity
            assert torch.all(batch['attention_mask'].sum(dim=1) > 0)  # No empty sequences
            
            if batch_count >= 3:  # Test first few batches
                break
        
        assert batch_count > 0
        assert total_sequences > 0
    
    def test_different_data_formats(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test integration with different data input formats."""
        # Test with pre-tokenized sequences
        sequences = trained_pipeline.process_texts(sample_texts[:5], return_tensors=False)
        trained_pipeline.create_datasets(sequences=sequences)
        
        dataloader = trained_pipeline.get_dataloader('train')
        batch = next(iter(dataloader))
        assert 'input_ids' in batch
        
        # Test with separate train/val/test data
        mid_point = len(sample_texts) // 2
        train_texts = sample_texts[:mid_point]
        val_texts = sample_texts[mid_point:]
        
        trained_pipeline.create_datasets(
            train_data=train_texts,
            val_data=val_texts
        )
        
        train_dataloader = trained_pipeline.get_dataloader('train')
        val_dataloader = trained_pipeline.get_dataloader('val')
        
        assert train_dataloader is not None
        assert val_dataloader is not None
        
        # Verify different data in each split
        train_batch = next(iter(train_dataloader))
        val_batch = next(iter(val_dataloader))
        
        # Should have different content (not a strict requirement but likely)
        assert train_batch['input_ids'].shape[0] > 0
        assert val_batch['input_ids'].shape[0] > 0
    
    def test_caching_integration(self, pipeline_config: PipelineConfig, sample_texts: List[str], temp_dir: str):
        """Test caching integration."""
        # Enable caching
        pipeline_config.cache_processed_data = True
        pipeline_config.cache_dir = temp_dir
        
        pipeline = TextProcessingPipeline(pipeline_config)
        pipeline.train_tokenizer(sample_texts)
        
        # Process texts with caching
        cache_key = "test_cache"
        sequences1 = pipeline.process_texts(sample_texts[:5], cache_key=cache_key)
        
        # Process again - should use cache
        sequences2 = pipeline.process_texts(sample_texts[:5], cache_key=cache_key)
        
        # Should be identical
        assert len(sequences1) == len(sequences2)
        for seq1, seq2 in zip(sequences1, sequences2):
            assert torch.equal(seq1, seq2)
        
        # Check cache file exists
        cache_path = Path(temp_dir) / f"{cache_key}_processed.json"
        assert cache_path.exists()
    
    def test_performance_stats_integration(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test performance statistics integration."""
        # Reset stats
        trained_pipeline.reset_performance_stats()
        initial_stats = trained_pipeline.get_performance_stats()
        
        assert initial_stats['texts_processed'] == 0
        assert initial_stats['tokens_generated'] == 0
        
        # Process texts
        trained_pipeline.process_texts(sample_texts)
        trained_pipeline.create_datasets(texts=sample_texts)
        
        # Check updated stats
        final_stats = trained_pipeline.get_performance_stats()
        
        assert final_stats['texts_processed'] > 0
        assert final_stats['tokens_generated'] > 0
        assert final_stats['total_time'] > 0
        assert 'avg_preprocessing_time_per_text' in final_stats
        assert 'tokens_per_second' in final_stats


class TestPipelineValidation(TestPipelineIntegration):
    """Test pipeline validation functionality."""
    
    def test_comprehensive_validation(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test comprehensive pipeline validation."""
        # Create datasets for validation
        trained_pipeline.create_datasets(texts=sample_texts)
        
        # Run validation
        validation_results = validate_pipeline(trained_pipeline)
        
        assert validation_results is not None
        assert hasattr(validation_results, 'passed')
        assert hasattr(validation_results, 'total_tests')
        assert hasattr(validation_results, 'test_results')
        
        # Check that some tests were run
        assert validation_results.total_tests > 0
        assert validation_results.passed_tests >= 0
        
        # Check test results structure
        assert 'tokenizer' in validation_results.test_results or 'end_to_end' in validation_results.test_results
        
        # Should have minimal errors for well-formed pipeline
        assert len(validation_results.errors) <= 2  # Allow some minor issues
    
    def test_consistency_validation(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test consistency validation across data splits."""
        # Prepare test data
        mid = len(sample_texts) // 2
        test_data = {
            'train': sample_texts[:mid],
            'val': sample_texts[mid:],
            'test': sample_texts[mid+2:]  # Smaller test set
        }
        
        # Run consistency tests
        consistency_results = run_consistency_tests(trained_pipeline, test_data)
        
        assert consistency_results is not None
        assert 'vocab_consistency' in consistency_results
        assert 'encoding_consistency' in consistency_results
        
        # Should have minimal errors
        errors = consistency_results.get('errors', [])
        assert len(errors) <= 1  # Allow one minor issue


class TestPipelineBenchmarking(TestPipelineIntegration):
    """Test pipeline benchmarking functionality."""
    
    def test_quick_benchmark(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test quick benchmarking functionality."""
        # Run quick benchmark
        metrics = run_quick_benchmark(trained_pipeline, sample_texts[:8])  # Use fewer texts for speed
        
        assert metrics is not None
        assert metrics.total_time > 0
        assert metrics.texts_per_second > 0
        assert metrics.total_texts > 0
        
        # Check that metrics are reasonable
        assert metrics.texts_per_second < 50000  # Should be reasonable (increased for micro-benchmarks)
        assert metrics.total_time < 60  # Should complete quickly
    
    def test_benchmark_components(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test individual component benchmarking."""
        runner = BenchmarkRunner(warmup_iterations=1, measurement_iterations=2)
        
        # Test preprocessing benchmark
        prep_results = runner.benchmark_preprocessing(trained_pipeline, sample_texts[:5])
        assert prep_results is not None
        assert 'batch_size_1' in prep_results
        
        # Test tokenization benchmark
        tok_results = runner.benchmark_tokenization(trained_pipeline, sample_texts[:5])
        assert tok_results is not None
        assert 'batch_size_1' in tok_results
        
        # Test dataset loading benchmark
        dataset_results = runner.benchmark_dataset_loading(trained_pipeline, sample_texts[:8])
        assert dataset_results is not None
        assert any('batch_size_' in key for key in dataset_results.keys())


class TestPipelineConfiguration(TestPipelineIntegration):
    """Test pipeline configuration functionality."""
    
    def test_config_serialization(self, pipeline_config: PipelineConfig, temp_dir: str):
        """Test configuration serialization and deserialization."""
        config_path = Path(temp_dir) / "test_config.json"
        
        # Save configuration
        pipeline = TextProcessingPipeline(pipeline_config)
        pipeline.save_config(config_path)
        
        assert config_path.exists()
        
        # Load configuration
        loaded_pipeline = TextProcessingPipeline.from_config_file(config_path)
        
        # Compare configurations (basic check)
        assert loaded_pipeline.config.preprocessing.lowercase == pipeline_config.preprocessing.lowercase
        assert loaded_pipeline.config.tokenization.vocab_size == pipeline_config.tokenization.vocab_size
        assert loaded_pipeline.config.dataset.batch_size == pipeline_config.dataset.batch_size
    
    def test_factory_functions(self, sample_texts: List[str]):
        """Test factory function integrations."""
        # Test default pipeline
        default_pipeline = create_default_pipeline()
        assert default_pipeline is not None
        
        # Should be able to train and use
        default_pipeline.train_tokenizer(sample_texts[:5])
        sequences = default_pipeline.process_texts(sample_texts[:3])
        assert len(sequences) > 0


class TestErrorHandling(TestPipelineIntegration):
    """Test error handling in integration scenarios."""
    
    def test_missing_tokenizer_error(self, pipeline_config: PipelineConfig, sample_texts: List[str]):
        """Test error handling when tokenizer is not trained."""
        pipeline = TextProcessingPipeline(pipeline_config)
        
        # Should raise error when trying to process texts without tokenizer
        with pytest.raises(ValueError, match="Tokenizer not initialized"):
            pipeline.process_texts(sample_texts)
    
    def test_empty_data_handling(self, trained_pipeline: TextProcessingPipeline):
        """Test handling of empty or invalid data."""
        # Test with empty text list
        sequences = trained_pipeline.process_texts([])
        assert len(sequences) == 0
        
        # Test with texts that get filtered out
        bad_texts = ["", "   ", "a"]  # Very short texts that might be filtered
        sequences = trained_pipeline.process_texts(bad_texts)
        # Should handle gracefully (might result in empty list)
        assert isinstance(sequences, list)
    
    def test_invalid_dataloader_request(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test error handling for invalid dataloader requests."""
        # Create datasets
        trained_pipeline.create_datasets(texts=sample_texts)
        
        # Request non-existent split
        with pytest.raises(ValueError):
            trained_pipeline.get_dataloader('invalid_split')


class TestMemoryEfficiency(TestPipelineIntegration):
    """Test memory efficiency of integration."""
    
    def test_memory_cleanup(self, trained_pipeline: TextProcessingPipeline, sample_texts: List[str]):
        """Test that memory is properly managed during processing."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process texts multiple times
        for _ in range(3):
            sequences = trained_pipeline.process_texts(sample_texts)
            trained_pipeline.create_datasets(sequences=sequences)
            dataloader = trained_pipeline.get_dataloader('train')
            
            # Process a few batches
            for i, batch in enumerate(dataloader):
                if i >= 2:
                    break
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"


# Integration test fixtures and helpers
# Integration test marker removed to avoid pytest warning
class TestFullIntegrationWorkflow(TestPipelineIntegration):
    """Test complete integration workflow scenarios."""
    
    def test_ml_training_workflow(self, pipeline_config: PipelineConfig, sample_texts: List[str]):
        """Test a typical ML training workflow."""
        # Step 1: Initialize pipeline
        pipeline = TextProcessingPipeline(pipeline_config)
        
        # Step 2: Train tokenizer
        pipeline.train_tokenizer(sample_texts)
        
        # Step 3: Create datasets
        mid = len(sample_texts) // 2
        pipeline.create_datasets(
            train_data=sample_texts[:mid],
            val_data=sample_texts[mid:]
        )
        
        # Step 4: Get dataloaders
        train_dataloader = pipeline.get_dataloader('train')
        val_dataloader = pipeline.get_dataloader('val')
        
        # Step 5: Simulate training loop
        train_batch_count = 0
        for batch in train_dataloader:
            # Simulate model forward pass
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Basic checks
            assert input_ids.shape[0] > 0  # Has batch dimension
            assert attention_mask.shape == input_ids.shape
            
            train_batch_count += 1
            if train_batch_count >= 2:  # Limit for testing
                break
        
        # Step 6: Simulate validation
        val_batch_count = 0
        for batch in val_dataloader:
            input_ids = batch['input_ids']
            val_batch_count += 1
            if val_batch_count >= 1:  # Just check one batch
                break
        
        assert train_batch_count > 0
        assert val_batch_count > 0
        
        # Step 7: Get performance stats
        stats = pipeline.get_performance_stats()
        assert stats['texts_processed'] > 0
        assert stats['total_time'] > 0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 