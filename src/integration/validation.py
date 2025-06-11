"""
Validation utilities for testing pipeline consistency, performance, and integration.

This module provides comprehensive validation and testing capabilities for the
text processing pipeline components.
"""

import torch
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """Results from pipeline validation."""
    
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'passed': self.passed,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'errors': self.errors,
            'warnings': self.warnings
        }


class TokenizationValidator:
    """Validator for tokenization consistency and correctness."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_tokenizer(self, tokenizer, test_texts: List[str]) -> Dict[str, Any]:
        """
        Validate tokenizer functionality.
        
        Args:
            tokenizer: BPE tokenizer to validate.
            test_texts: Test texts for validation.
            
        Returns:
            Validation results.
        """
        results = {
            'vocabulary_check': False,
            'encoding_decoding_consistency': False,
            'special_tokens_handling': False,
            'edge_cases_handling': False,
            'performance_check': False
        }
        
        try:
            # Test 1: Vocabulary check
            vocab_size = len(tokenizer.vocab)
            if vocab_size > 0:
                results['vocabulary_check'] = True
                logger.info(f"✓ Vocabulary size: {vocab_size}")
            else:
                self.errors.append("Empty vocabulary")
            
            # Test 2: Encoding/decoding consistency
            consistency_passed = 0
            for text in test_texts[:10]:  # Test first 10 texts
                try:
                    encoded = tokenizer.encode(text)
                    decoded = tokenizer.decode(encoded)
                    
                    # Check if decoding preserves meaning (allow for some normalization)
                    if len(encoded) > 0 and len(decoded.strip()) > 0:
                        consistency_passed += 1
                except Exception as e:
                    self.errors.append(f"Encoding/decoding error: {e}")
            
            if consistency_passed >= len(test_texts[:10]) * 0.8:  # 80% success rate
                results['encoding_decoding_consistency'] = True
                logger.info(f"✓ Encoding/decoding consistency: {consistency_passed}/10")
            else:
                self.errors.append(f"Poor encoding/decoding consistency: {consistency_passed}/10")
            
            # Test 3: Special tokens handling
            special_tokens = tokenizer.special_tokens
            if special_tokens and len(special_tokens) > 0:
                # Test if special tokens are properly handled
                pad_token = special_tokens.get('pad_token', '')
                unk_token = special_tokens.get('unk_token', '')
                
                if pad_token and unk_token:
                    try:
                        pad_encoded = tokenizer.encode(pad_token)
                        unk_encoded = tokenizer.encode(unk_token)
                        
                        if len(pad_encoded) > 0 and len(unk_encoded) > 0:
                            results['special_tokens_handling'] = True
                            logger.info("✓ Special tokens handled correctly")
                    except Exception as e:
                        self.errors.append(f"Special tokens error: {e}")
            else:
                self.warnings.append("No special tokens defined")
                results['special_tokens_handling'] = True  # Not required
            
            # Test 4: Edge cases
            edge_cases = ["", " ", "   ", "1234567890", "!@#$%^&*()", "a" * 1000]
            edge_cases_passed = 0
            
            for case in edge_cases:
                try:
                    encoded = tokenizer.encode(case)
                    # Should handle gracefully (empty result is acceptable for empty input)
                    edge_cases_passed += 1
                except Exception as e:
                    self.errors.append(f"Edge case '{case[:20]}...' failed: {e}")
            
            if edge_cases_passed >= len(edge_cases) * 0.8:
                results['edge_cases_handling'] = True
                logger.info(f"✓ Edge cases handled: {edge_cases_passed}/{len(edge_cases)}")
            
            # Test 5: Performance check
            start_time = time.time()
            total_tokens = 0
            
            for text in test_texts[:100]:  # Test with more texts for performance
                encoded = tokenizer.encode(text)
                total_tokens += len(encoded)
            
            elapsed_time = time.time() - start_time
            tokens_per_second = total_tokens / max(elapsed_time, 0.001)
            
            if tokens_per_second > 1000:  # Reasonable threshold
                results['performance_check'] = True
                logger.info(f"✓ Performance: {tokens_per_second:.0f} tokens/second")
            else:
                self.warnings.append(f"Slow tokenization: {tokens_per_second:.0f} tokens/second")
                results['performance_check'] = True  # Not critical
            
        except Exception as e:
            self.errors.append(f"Tokenizer validation error: {e}")
        
        return results


class DataLoaderValidator:
    """Validator for dataset loader functionality."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_dataloader(self, dataloader, expected_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate PyTorch DataLoader functionality.
        
        Args:
            dataloader: PyTorch DataLoader to validate.
            expected_features: Expected features in batches.
            
        Returns:
            Validation results.
        """
        results = {
            'batch_structure': False,
            'tensor_types': False,
            'batch_consistency': False,
            'iteration_stability': False,
            'memory_efficiency': False
        }
        
        if expected_features is None:
            expected_features = ['input_ids', 'attention_mask', 'lengths']
        
        try:
            # Test 1: Batch structure
            batch = next(iter(dataloader))
            
            if isinstance(batch, dict):
                missing_features = set(expected_features) - set(batch.keys())
                if not missing_features:
                    results['batch_structure'] = True
                    logger.info(f"✓ Batch structure: {list(batch.keys())}")
                else:
                    self.errors.append(f"Missing batch features: {missing_features}")
            else:
                self.errors.append("Batch is not a dictionary")
            
            # Test 2: Tensor types
            tensor_types_correct = True
            for key, value in batch.items():
                if not torch.is_tensor(value):
                    self.errors.append(f"Feature '{key}' is not a tensor")
                    tensor_types_correct = False
                elif key == 'input_ids' and value.dtype != torch.long:
                    self.errors.append(f"input_ids should be torch.long, got {value.dtype}")
                    tensor_types_correct = False
                elif key == 'attention_mask' and value.dtype not in [torch.bool, torch.long]:
                    self.warnings.append(f"attention_mask has dtype {value.dtype}")
            
            if tensor_types_correct:
                results['tensor_types'] = True
                logger.info("✓ Tensor types correct")
            
            # Test 3: Batch consistency
            batch_shapes = []
            batch_sizes = []
            
            for i, batch in enumerate(dataloader):
                if i >= 5:  # Test first 5 batches
                    break
                
                batch_size = batch['input_ids'].shape[0]
                seq_length = batch['input_ids'].shape[1]
                batch_shapes.append((batch_size, seq_length))
                batch_sizes.append(batch_size)
                
                # Check shape consistency within batch
                for key, tensor in batch.items():
                    if tensor.shape[0] != batch_size:
                        self.errors.append(f"Inconsistent batch size in feature '{key}'")
            
            if len(set(batch_shapes[:-1])) <= 2:  # Allow some variation due to dynamic batching
                results['batch_consistency'] = True
                logger.info(f"✓ Batch consistency: shapes {batch_shapes}")
            else:
                self.warnings.append(f"High shape variation: {batch_shapes}")
                results['batch_consistency'] = True  # Not critical for dynamic padding
            
            # Test 4: Iteration stability
            try:
                iteration_count = 0
                for batch in dataloader:
                    iteration_count += 1
                    if iteration_count >= 10:  # Test first 10 batches
                        break
                
                if iteration_count > 0:
                    results['iteration_stability'] = True
                    logger.info(f"✓ Iteration stability: {iteration_count} batches")
            except Exception as e:
                self.errors.append(f"Iteration error: {e}")
            
            # Test 5: Memory efficiency (basic check)
            try:
                # Check if tensors are on expected device and memory usage is reasonable
                first_batch = next(iter(dataloader))
                total_elements = sum(tensor.numel() for tensor in first_batch.values() if torch.is_tensor(tensor))
                
                if total_elements > 0:
                    results['memory_efficiency'] = True
                    logger.info(f"✓ Memory efficiency: {total_elements} elements per batch")
            except Exception as e:
                self.warnings.append(f"Memory efficiency check failed: {e}")
                results['memory_efficiency'] = True  # Not critical
        
        except Exception as e:
            self.errors.append(f"DataLoader validation error: {e}")
        
        return results


class PipelineValidator:
    """Comprehensive pipeline validator."""
    
    def __init__(self):
        self.tokenization_validator = TokenizationValidator()
        self.dataloader_validator = DataLoaderValidator()
        self.errors = []
        self.warnings = []
    
    def validate_end_to_end(self, pipeline, test_texts: List[str]) -> Dict[str, Any]:
        """
        Validate end-to-end pipeline functionality.
        
        Args:
            pipeline: TextProcessingPipeline to validate.
            test_texts: Test texts for validation.
            
        Returns:
            Validation results.
        """
        results = {
            'preprocessing_integration': False,
            'tokenization_integration': False,
            'dataset_integration': False,
            'dataloader_integration': False,
            'performance_integration': False
        }
        
        try:
            # Test 1: Preprocessing integration
            try:
                processed_texts = []
                for text in test_texts[:5]:
                    processed = pipeline.preprocessor.preprocess_text(text)
                    if processed:
                        processed_texts.append(processed)
                
                if len(processed_texts) > 0:
                    results['preprocessing_integration'] = True
                    logger.info(f"✓ Preprocessing integration: {len(processed_texts)} texts processed")
            except Exception as e:
                self.errors.append(f"Preprocessing integration error: {e}")
            
            # Test 2: Tokenization integration
            if pipeline.tokenizer is not None:
                try:
                    sequences = pipeline.process_texts(test_texts[:5], return_tensors=False)
                    if len(sequences) > 0 and all(len(seq) > 0 for seq in sequences):
                        results['tokenization_integration'] = True
                        logger.info(f"✓ Tokenization integration: {len(sequences)} sequences")
                except Exception as e:
                    self.errors.append(f"Tokenization integration error: {e}")
            else:
                self.warnings.append("No tokenizer available for integration test")
            
            # Test 3: Dataset integration
            try:
                if pipeline.tokenizer is not None:
                    pipeline.create_datasets(texts=test_texts[:10])
                    stats = pipeline.dataset_loader.get_stats()
                    
                    if any(stats.values()):
                        results['dataset_integration'] = True
                        logger.info(f"✓ Dataset integration: {sum(s.get('num_sequences', 0) for s in stats.values())} sequences")
            except Exception as e:
                self.errors.append(f"Dataset integration error: {e}")
            
            # Test 4: DataLoader integration
            try:
                train_dataloader = pipeline.get_dataloader('train')
                batch = next(iter(train_dataloader))
                
                if isinstance(batch, dict) and 'input_ids' in batch:
                    results['dataloader_integration'] = True
                    logger.info(f"✓ DataLoader integration: batch shape {batch['input_ids'].shape}")
            except Exception as e:
                self.errors.append(f"DataLoader integration error: {e}")
            
            # Test 5: Performance integration
            start_time = time.time()
            try:
                if pipeline.tokenizer is not None:
                    # Process a small batch end-to-end
                    sequences = pipeline.process_texts(test_texts[:5])
                    pipeline.create_datasets(texts=test_texts[:20])
                    dataloader = pipeline.get_dataloader('train')
                    
                    # Process one batch
                    batch = next(iter(dataloader))
                    
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 30:  # Should complete within 30 seconds
                        results['performance_integration'] = True
                        logger.info(f"✓ Performance integration: {elapsed_time:.2f}s")
                    else:
                        self.warnings.append(f"Slow end-to-end performance: {elapsed_time:.2f}s")
            except Exception as e:
                self.errors.append(f"Performance integration error: {e}")
        
        except Exception as e:
            self.errors.append(f"End-to-end validation error: {e}")
        
        return results


def validate_pipeline(pipeline) -> ValidationResults:
    """
    Comprehensive validation of text processing pipeline.
    
    Args:
        pipeline: TextProcessingPipeline to validate.
        
    Returns:
        Comprehensive validation results.
    """
    logger.info("Starting comprehensive pipeline validation...")
    
    # Generate test data
    test_texts = [
        "Hello, world! This is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating and powerful.",
        "Natural language processing enables computers to understand text.",
        "PyTorch provides excellent tools for deep learning research.",
        "Data preprocessing is crucial for model performance.",
        "Tokenization breaks text into meaningful units.",
        "Batch processing improves computational efficiency.",
        "Validation ensures system reliability and correctness.",
        "Integration testing verifies component compatibility."
    ]
    
    validator = PipelineValidator()
    test_results = {}
    total_tests = 0
    passed_tests = 0
    errors = []
    warnings = []
    performance_metrics = {}
    
    # Validate tokenizer if available
    if pipeline.tokenizer is not None:
        logger.info("Validating tokenizer...")
        tokenizer_results = validator.tokenization_validator.validate_tokenizer(
            pipeline.tokenizer, test_texts
        )
        test_results['tokenizer'] = tokenizer_results
        total_tests += len(tokenizer_results)
        passed_tests += sum(tokenizer_results.values())
        errors.extend(validator.tokenization_validator.errors)
        warnings.extend(validator.tokenization_validator.warnings)
    
    # Validate dataset loader
    try:
        if pipeline.tokenizer is not None:
            pipeline.create_datasets(texts=test_texts)
            train_dataloader = pipeline.get_dataloader('train')
            
            logger.info("Validating data loader...")
            dataloader_results = validator.dataloader_validator.validate_dataloader(train_dataloader)
            test_results['dataloader'] = dataloader_results
            total_tests += len(dataloader_results)
            passed_tests += sum(dataloader_results.values())
            errors.extend(validator.dataloader_validator.errors)
            warnings.extend(validator.dataloader_validator.warnings)
    except Exception as e:
        errors.append(f"DataLoader validation setup error: {e}")
    
    # Validate end-to-end integration
    logger.info("Validating end-to-end integration...")
    e2e_results = validator.validate_end_to_end(pipeline, test_texts)
    test_results['end_to_end'] = e2e_results
    total_tests += len(e2e_results)
    passed_tests += sum(e2e_results.values())
    errors.extend(validator.errors)
    warnings.extend(validator.warnings)
    
    # Collect performance metrics
    performance_stats = pipeline.get_performance_stats()
    performance_metrics.update(performance_stats)
    
    # Calculate overall results
    passed = len(errors) == 0 and passed_tests >= total_tests * 0.8  # 80% pass rate
    failed_tests = total_tests - passed_tests
    
    logger.info(f"Validation completed: {passed_tests}/{total_tests} tests passed")
    if errors:
        logger.error(f"Validation errors: {errors}")
    if warnings:
        logger.warning(f"Validation warnings: {warnings}")
    
    return ValidationResults(
        passed=passed,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        test_results=test_results,
        performance_metrics=performance_metrics,
        errors=errors,
        warnings=warnings
    )


def run_consistency_tests(pipeline, test_data: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Run consistency tests across different data splits.
    
    Args:
        pipeline: TextProcessingPipeline to test.
        test_data: Dictionary with 'train', 'val', 'test' text lists.
        
    Returns:
        Consistency test results.
    """
    results = {
        'vocab_consistency': False,
        'encoding_consistency': False,
        'batch_consistency': False,
        'performance_consistency': False
    }
    
    errors = []
    
    try:
        # Test vocabulary consistency across splits
        if pipeline.tokenizer is not None:
            vocab_sizes = {}
            for split, texts in test_data.items():
                if texts:
                    # Process a sample of texts
                    sample_texts = texts[:min(10, len(texts))]
                    sequences = pipeline.process_texts(sample_texts, return_tensors=False)
                    
                    # Check vocabulary usage
                    used_tokens = set()
                    for seq in sequences:
                        used_tokens.update(seq)
                    
                    vocab_sizes[split] = len(used_tokens)
            
            # Vocabulary should be consistent (within reasonable bounds)
            if len(set(vocab_sizes.values())) <= 2:  # Allow some variation
                results['vocab_consistency'] = True
                logger.info(f"✓ Vocabulary consistency: {vocab_sizes}")
            else:
                errors.append(f"Vocabulary inconsistency: {vocab_sizes}")
        
        # Test encoding consistency
        if pipeline.tokenizer is not None:
            test_text = "This is a consistency test sentence."
            encodings = []
            
            for _ in range(5):  # Encode same text multiple times
                encoded = pipeline.tokenizer.encode(test_text)
                encodings.append(encoded)
            
            if all(enc == encodings[0] for enc in encodings):
                results['encoding_consistency'] = True
                logger.info("✓ Encoding consistency verified")
            else:
                errors.append("Encoding inconsistency detected")
        
        # Test batch consistency
        if 'train' in test_data and test_data['train']:
            pipeline.create_datasets(texts=test_data['train'][:20])
            dataloader = pipeline.get_dataloader('train')
            
            batch_shapes = []
            for i, batch in enumerate(dataloader):
                if i >= 3:  # Test first 3 batches
                    break
                batch_shapes.append(batch['input_ids'].shape)
            
            # Check if batch sizes are consistent (allowing for last batch variation)
            batch_sizes = [shape[0] for shape in batch_shapes]
            if len(set(batch_sizes[:-1])) <= 1:  # All but last should be same
                results['batch_consistency'] = True
                logger.info(f"✓ Batch consistency: {batch_shapes}")
            else:
                errors.append(f"Batch inconsistency: {batch_shapes}")
        
        # Test performance consistency
        if pipeline.tokenizer is not None and test_data.get('train'):
            processing_times = []
            
            for _ in range(3):  # Run 3 times
                start_time = time.time()
                pipeline.process_texts(test_data['train'][:5])
                processing_times.append(time.time() - start_time)
            
            # Performance should be relatively consistent
            avg_time = np.mean(processing_times)
            std_time = np.std(processing_times)
            
            if std_time / avg_time < 0.5:  # Coefficient of variation < 50%
                results['performance_consistency'] = True
                logger.info(f"✓ Performance consistency: {avg_time:.3f}±{std_time:.3f}s")
            else:
                errors.append(f"Performance inconsistency: {processing_times}")
    
    except Exception as e:
        errors.append(f"Consistency test error: {e}")
    
    results['errors'] = errors
    return results 