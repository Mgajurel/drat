"""
Performance benchmarking utilities for text processing pipeline.

This module provides comprehensive benchmarking and performance analysis
capabilities for evaluating pipeline efficiency and scalability.
"""

import torch
import time
import gc
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import defaultdict

# Optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for pipeline benchmarking."""
    
    # Timing metrics
    total_time: float = 0.0
    preprocessing_time: float = 0.0
    tokenization_time: float = 0.0
    dataset_loading_time: float = 0.0
    batch_creation_time: float = 0.0
    
    # Throughput metrics
    texts_per_second: float = 0.0
    tokens_per_second: float = 0.0
    batches_per_second: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_efficiency: float = 0.0  # tokens per MB
    
    # Data metrics
    total_texts: int = 0
    total_tokens: int = 0
    total_batches: int = 0
    avg_text_length: float = 0.0
    avg_sequence_length: float = 0.0
    
    # Quality metrics
    preprocessing_retention_rate: float = 0.0  # Texts retained after preprocessing
    tokenization_coverage: float = 0.0  # Vocabulary coverage
    
    # Scalability metrics
    linear_scaling_coefficient: float = 0.0
    memory_scaling_coefficient: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timing': {
                'total_time': self.total_time,
                'preprocessing_time': self.preprocessing_time,
                'tokenization_time': self.tokenization_time,
                'dataset_loading_time': self.dataset_loading_time,
                'batch_creation_time': self.batch_creation_time
            },
            'throughput': {
                'texts_per_second': self.texts_per_second,
                'tokens_per_second': self.tokens_per_second,
                'batches_per_second': self.batches_per_second
            },
            'memory': {
                'peak_memory_mb': self.peak_memory_mb,
                'avg_memory_mb': self.avg_memory_mb,
                'memory_efficiency': self.memory_efficiency
            },
            'data': {
                'total_texts': self.total_texts,
                'total_tokens': self.total_tokens,
                'total_batches': self.total_batches,
                'avg_text_length': self.avg_text_length,
                'avg_sequence_length': self.avg_sequence_length
            },
            'quality': {
                'preprocessing_retention_rate': self.preprocessing_retention_rate,
                'tokenization_coverage': self.tokenization_coverage
            },
            'scalability': {
                'linear_scaling_coefficient': self.linear_scaling_coefficient,
                'memory_scaling_coefficient': self.memory_scaling_coefficient
            }
        }


class MemoryMonitor:
    """Monitor memory usage during benchmarking."""
    
    def __init__(self):
        self.measurements = []
        if HAS_PSUTIL:
            self.process = psutil.Process()
            self.baseline_memory = self.get_memory_usage()
        else:
            self.process = None
            self.baseline_memory = 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL and self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.measurements = [self.get_memory_usage()]
    
    def record_measurement(self):
        """Record current memory usage."""
        self.measurements.append(self.get_memory_usage())
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage above baseline."""
        if not self.measurements:
            return 0.0
        return max(self.measurements) - self.baseline_memory
    
    def get_average_memory(self) -> float:
        """Get average memory usage above baseline."""
        if not self.measurements:
            return 0.0
        return np.mean(self.measurements) - self.baseline_memory
    
    def get_memory_trend(self) -> List[float]:
        """Get memory usage trend (relative to baseline)."""
        return [m - self.baseline_memory for m in self.measurements]


class BenchmarkRunner:
    """Comprehensive benchmark runner for text processing pipeline."""
    
    def __init__(self, warmup_iterations: int = 3, measurement_iterations: int = 10):
        """
        Initialize benchmark runner.
        
        Args:
            warmup_iterations: Number of warmup iterations.
            measurement_iterations: Number of measurement iterations.
        """
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.memory_monitor = MemoryMonitor()
        self.results_history = []
    
    def benchmark_preprocessing(
        self,
        pipeline,
        test_texts: List[str],
        batch_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark preprocessing performance.
        
        Args:
            pipeline: TextProcessingPipeline to benchmark.
            test_texts: Test texts for benchmarking.
            batch_sizes: Different batch sizes to test.
            
        Returns:
            Preprocessing benchmark results.
        """
        if batch_sizes is None:
            batch_sizes = [1, 10, 50, 100]
        
        logger.info("Benchmarking preprocessing performance...")
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(test_texts):
                continue
            
            batch_texts = test_texts[:batch_size]
            timing_results = []
            memory_results = []
            quality_results = []
            
            # Warmup
            for _ in range(self.warmup_iterations):
                for text in batch_texts:
                    pipeline.preprocessor.preprocess_text(text, validate=False)
            
            # Measurements
            for iteration in range(self.measurement_iterations):
                self.memory_monitor.start_monitoring()
                
                start_time = time.time()
                processed_texts = []
                
                for text in batch_texts:
                    processed = pipeline.preprocessor.preprocess_text(text, validate=True)
                    if processed:
                        processed_texts.append(processed)
                    self.memory_monitor.record_measurement()
                
                elapsed_time = time.time() - start_time
                timing_results.append(elapsed_time)
                memory_results.append(self.memory_monitor.get_peak_memory())
                
                # Quality metrics
                retention_rate = len(processed_texts) / len(batch_texts)
                avg_length = np.mean([len(text) for text in processed_texts]) if processed_texts else 0
                quality_results.append({
                    'retention_rate': retention_rate,
                    'avg_length': avg_length
                })
            
            # Calculate statistics
            results[f'batch_size_{batch_size}'] = {
                'timing': {
                    'mean_time': np.mean(timing_results),
                    'std_time': np.std(timing_results),
                    'min_time': np.min(timing_results),
                    'max_time': np.max(timing_results),
                    'texts_per_second': batch_size / np.mean(timing_results)
                },
                'memory': {
                    'mean_memory': np.mean(memory_results),
                    'peak_memory': np.max(memory_results),
                    'memory_per_text': np.mean(memory_results) / batch_size
                },
                'quality': {
                    'avg_retention_rate': np.mean([q['retention_rate'] for q in quality_results]),
                    'avg_text_length': np.mean([q['avg_length'] for q in quality_results])
                }
            }
        
        return results
    
    def benchmark_tokenization(
        self,
        pipeline,
        test_texts: List[str],
        batch_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark tokenization performance.
        
        Args:
            pipeline: TextProcessingPipeline to benchmark.
            test_texts: Test texts for benchmarking.
            batch_sizes: Different batch sizes to test.
            
        Returns:
            Tokenization benchmark results.
        """
        if pipeline.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        if batch_sizes is None:
            batch_sizes = [1, 10, 50, 100]
        
        logger.info("Benchmarking tokenization performance...")
        results = {}
        
        # Preprocess texts for tokenization
        preprocessed_texts = []
        for text in test_texts:
            processed = pipeline.preprocessor.preprocess_text(text, validate=False)
            if processed:
                preprocessed_texts.append(processed)
        
        for batch_size in batch_sizes:
            if batch_size > len(preprocessed_texts):
                continue
            
            batch_texts = preprocessed_texts[:batch_size]
            timing_results = []
            memory_results = []
            token_counts = []
            
            # Warmup
            for _ in range(self.warmup_iterations):
                for text in batch_texts:
                    pipeline.tokenizer.encode(text)
            
            # Measurements
            for iteration in range(self.measurement_iterations):
                self.memory_monitor.start_monitoring()
                
                start_time = time.time()
                total_tokens = 0
                
                for text in batch_texts:
                    tokens = pipeline.tokenizer.encode(text)
                    total_tokens += len(tokens)
                    self.memory_monitor.record_measurement()
                
                elapsed_time = time.time() - start_time
                timing_results.append(elapsed_time)
                memory_results.append(self.memory_monitor.get_peak_memory())
                token_counts.append(total_tokens)
            
            # Calculate statistics
            avg_tokens = np.mean(token_counts)
            avg_time = np.mean(timing_results)
            
            results[f'batch_size_{batch_size}'] = {
                'timing': {
                    'mean_time': avg_time,
                    'std_time': np.std(timing_results),
                    'texts_per_second': batch_size / avg_time,
                    'tokens_per_second': avg_tokens / avg_time
                },
                'memory': {
                    'mean_memory': np.mean(memory_results),
                    'peak_memory': np.max(memory_results),
                    'tokens_per_mb': avg_tokens / max(np.mean(memory_results), 0.1)
                },
                'tokenization': {
                    'avg_tokens_per_text': avg_tokens / batch_size,
                    'total_tokens': avg_tokens,
                    'vocab_coverage': min(len(set(sum([pipeline.tokenizer.encode(text) for text in batch_texts], []))), len(pipeline.tokenizer.vocab))
                }
            }
        
        return results
    
    def benchmark_dataset_loading(
        self,
        pipeline,
        test_texts: List[str],
        batch_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark dataset loading and DataLoader performance.
        
        Args:
            pipeline: TextProcessingPipeline to benchmark.
            test_texts: Test texts for benchmarking.
            batch_sizes: Different batch sizes to test.
            
        Returns:
            Dataset loading benchmark results.
        """
        if pipeline.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        if batch_sizes is None:
            batch_sizes = [8, 16, 32, 64]
        
        logger.info("Benchmarking dataset loading performance...")
        results = {}
        
        for batch_size in batch_sizes:
            # Update pipeline config for this batch size
            original_batch_size = pipeline.config.dataset.batch_size
            pipeline.config.dataset.batch_size = batch_size
            
            timing_results = []
            memory_results = []
            batch_stats = []
            
            # Warmup
            for _ in range(self.warmup_iterations):
                try:
                    pipeline.create_datasets(texts=test_texts[:50])
                    dataloader = pipeline.get_dataloader('train')
                    next(iter(dataloader))
                except Exception:
                    pass
            
            # Measurements
            for iteration in range(self.measurement_iterations):
                gc.collect()  # Clean memory
                self.memory_monitor.start_monitoring()
                
                start_time = time.time()
                
                # Create datasets
                pipeline.create_datasets(texts=test_texts)
                dataloader = pipeline.get_dataloader('train')
                
                # Process batches
                batch_count = 0
                total_sequences = 0
                
                for batch in dataloader:
                    batch_count += 1
                    total_sequences += batch['input_ids'].shape[0]
                    self.memory_monitor.record_measurement()
                    
                    if batch_count >= 10:  # Limit for benchmarking
                        break
                
                elapsed_time = time.time() - start_time
                timing_results.append(elapsed_time)
                memory_results.append(self.memory_monitor.get_peak_memory())
                batch_stats.append({
                    'batch_count': batch_count,
                    'total_sequences': total_sequences,
                    'avg_batch_size': total_sequences / batch_count if batch_count > 0 else 0
                })
            
            # Calculate statistics
            avg_time = np.mean(timing_results)
            avg_batches = np.mean([s['batch_count'] for s in batch_stats])
            
            results[f'batch_size_{batch_size}'] = {
                'timing': {
                    'mean_time': avg_time,
                    'std_time': np.std(timing_results),
                    'batches_per_second': avg_batches / avg_time,
                    'sequences_per_second': np.mean([s['total_sequences'] for s in batch_stats]) / avg_time
                },
                'memory': {
                    'mean_memory': np.mean(memory_results),
                    'peak_memory': np.max(memory_results),
                    'memory_per_batch': np.mean(memory_results) / avg_batches
                },
                'batching': {
                    'avg_batch_count': avg_batches,
                    'avg_batch_size': np.mean([s['avg_batch_size'] for s in batch_stats]),
                    'batch_size_consistency': np.std([s['avg_batch_size'] for s in batch_stats])
                }
            }
            
            # Restore original batch size
            pipeline.config.dataset.batch_size = original_batch_size
        
        return results
    
    def benchmark_end_to_end(
        self,
        pipeline,
        test_texts: List[str],
        iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Benchmark complete end-to-end pipeline performance.
        
        Args:
            pipeline: TextProcessingPipeline to benchmark.
            test_texts: Test texts for benchmarking.
            iterations: Number of iterations (uses default if None).
            
        Returns:
            End-to-end benchmark results.
        """
        if pipeline.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        if iterations is None:
            iterations = max(3, self.measurement_iterations // 2)
        
        logger.info("Benchmarking end-to-end pipeline performance...")
        
        timing_breakdown = {
            'preprocessing': [],
            'tokenization': [],
            'dataset_creation': [],
            'dataloader_iteration': [],
            'total': []
        }
        memory_results = []
        data_stats = []
        
        # Warmup
        for _ in range(self.warmup_iterations):
            try:
                pipeline.process_texts(test_texts[:10])
                pipeline.create_datasets(texts=test_texts[:20])
                dataloader = pipeline.get_dataloader('train')
                next(iter(dataloader))
            except Exception:
                pass
        
        # Measurements
        for iteration in range(iterations):
            gc.collect()
            self.memory_monitor.start_monitoring()
            
            total_start = time.time()
            
            # Step 1: Text processing (preprocessing + tokenization)
            processing_start = time.time()
            sequences = pipeline.process_texts(test_texts, return_tensors=False)
            processing_time = time.time() - processing_start
            
            # Step 2: Dataset creation
            dataset_start = time.time()
            pipeline.create_datasets(sequences=sequences)
            dataset_time = time.time() - dataset_start
            
            # Step 3: DataLoader iteration
            dataloader_start = time.time()
            dataloader = pipeline.get_dataloader('train')
            batch_count = 0
            
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 5:  # Process first 5 batches
                    break
            
            dataloader_time = time.time() - dataloader_start
            total_time = time.time() - total_start
            
            # Record results
            timing_breakdown['preprocessing'].append(processing_time * 0.4)  # Estimated split
            timing_breakdown['tokenization'].append(processing_time * 0.6)  # Estimated split
            timing_breakdown['dataset_creation'].append(dataset_time)
            timing_breakdown['dataloader_iteration'].append(dataloader_time)
            timing_breakdown['total'].append(total_time)
            
            memory_results.append(self.memory_monitor.get_peak_memory())
            
            # Data statistics
            total_tokens = sum(len(seq) for seq in sequences)
            data_stats.append({
                'total_texts': len(test_texts),
                'total_sequences': len(sequences),
                'total_tokens': total_tokens,
                'avg_sequence_length': total_tokens / len(sequences) if sequences else 0,
                'retention_rate': len(sequences) / len(test_texts)
            })
        
        # Calculate comprehensive statistics
        results = {}
        
        for component, times in timing_breakdown.items():
            results[f'{component}_timing'] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        
        # Overall performance metrics
        avg_total_time = np.mean(timing_breakdown['total'])
        avg_data = {key: np.mean([d[key] for d in data_stats]) for key in data_stats[0].keys()}
        
        results['performance'] = {
            'texts_per_second': avg_data['total_texts'] / avg_total_time,
            'tokens_per_second': avg_data['total_tokens'] / avg_total_time,
            'sequences_per_second': avg_data['total_sequences'] / avg_total_time,
            'memory_efficiency': avg_data['total_tokens'] / max(np.mean(memory_results), 0.1)
        }
        
        results['memory'] = {
            'mean_memory': np.mean(memory_results),
            'peak_memory': np.max(memory_results),
            'memory_std': np.std(memory_results)
        }
        
        results['data_quality'] = {
            'avg_retention_rate': avg_data['retention_rate'],
            'avg_sequence_length': avg_data['avg_sequence_length'],
            'total_tokens': avg_data['total_tokens']
        }
        
        return results
    
    def run_comprehensive_benchmark(
        self,
        pipeline,
        test_texts: List[str],
        save_results: bool = True,
        results_path: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        Run comprehensive benchmark suite.
        
        Args:
            pipeline: TextProcessingPipeline to benchmark.
            test_texts: Test texts for benchmarking.
            save_results: Whether to save results to file.
            results_path: Path to save results.
            
        Returns:
            Comprehensive performance metrics.
        """
        logger.info("Starting comprehensive benchmark suite...")
        
        benchmark_results = {
            'preprocessing': {},
            'tokenization': {},
            'dataset_loading': {},
            'end_to_end': {},
            'metadata': {
                'test_text_count': len(test_texts),
                'warmup_iterations': self.warmup_iterations,
                'measurement_iterations': self.measurement_iterations,
                'timestamp': time.time()
            }
        }
        
        # Run individual benchmarks
        try:
            benchmark_results['preprocessing'] = self.benchmark_preprocessing(pipeline, test_texts)
        except Exception as e:
            logger.error(f"Preprocessing benchmark failed: {e}")
        
        try:
            benchmark_results['tokenization'] = self.benchmark_tokenization(pipeline, test_texts)
        except Exception as e:
            logger.error(f"Tokenization benchmark failed: {e}")
        
        try:
            benchmark_results['dataset_loading'] = self.benchmark_dataset_loading(pipeline, test_texts)
        except Exception as e:
            logger.error(f"Dataset loading benchmark failed: {e}")
        
        try:
            benchmark_results['end_to_end'] = self.benchmark_end_to_end(pipeline, test_texts)
        except Exception as e:
            logger.error(f"End-to-end benchmark failed: {e}")
        
        # Extract comprehensive metrics
        metrics = self._extract_metrics(benchmark_results)
        
        # Save results if requested
        if save_results:
            if results_path is None:
                results_path = f"benchmark_results_{int(time.time())}.json"
            
            with open(results_path, 'w') as f:
                json.dump({
                    'metrics': metrics.to_dict(),
                    'detailed_results': benchmark_results
                }, f, indent=2)
            
            logger.info(f"Benchmark results saved to {results_path}")
        
        self.results_history.append(metrics)
        return metrics
    
    def _extract_metrics(self, benchmark_results: Dict[str, Any]) -> PerformanceMetrics:
        """Extract PerformanceMetrics from benchmark results."""
        metrics = PerformanceMetrics()
        
        try:
            # Extract timing metrics from end-to-end results
            e2e = benchmark_results.get('end_to_end', {})
            if e2e:
                metrics.total_time = e2e.get('total_timing', {}).get('mean_time', 0.0)
                metrics.preprocessing_time = e2e.get('preprocessing_timing', {}).get('mean_time', 0.0)
                metrics.tokenization_time = e2e.get('tokenization_timing', {}).get('mean_time', 0.0)
                metrics.dataset_loading_time = e2e.get('dataset_creation_timing', {}).get('mean_time', 0.0)
                metrics.batch_creation_time = e2e.get('dataloader_iteration_timing', {}).get('mean_time', 0.0)
                
                # Throughput metrics
                perf = e2e.get('performance', {})
                metrics.texts_per_second = perf.get('texts_per_second', 0.0)
                metrics.tokens_per_second = perf.get('tokens_per_second', 0.0)
                
                # Memory metrics
                memory = e2e.get('memory', {})
                metrics.peak_memory_mb = memory.get('peak_memory', 0.0)
                metrics.avg_memory_mb = memory.get('mean_memory', 0.0)
                metrics.memory_efficiency = perf.get('memory_efficiency', 0.0)
                
                # Data metrics
                data_quality = e2e.get('data_quality', {})
                metrics.total_tokens = data_quality.get('total_tokens', 0)
                metrics.avg_sequence_length = data_quality.get('avg_sequence_length', 0.0)
                metrics.preprocessing_retention_rate = data_quality.get('avg_retention_rate', 0.0)
            
            # Extract batch metrics from dataset loading
            dataset_results = benchmark_results.get('dataset_loading', {})
            if dataset_results:
                # Use results from largest batch size
                largest_batch = max(dataset_results.keys(), key=lambda x: int(x.split('_')[-1]))
                batch_data = dataset_results[largest_batch]
                
                metrics.batches_per_second = batch_data.get('timing', {}).get('batches_per_second', 0.0)
                
                batching = batch_data.get('batching', {})
                metrics.total_batches = batching.get('avg_batch_count', 0)
            
            # Calculate metadata metrics
            metadata = benchmark_results.get('metadata', {})
            metrics.total_texts = metadata.get('test_text_count', 0)
            
            if metrics.total_texts > 0 and metrics.total_time > 0:
                metrics.avg_text_length = metrics.total_tokens / metrics.total_texts
        
        except Exception as e:
            logger.warning(f"Error extracting metrics: {e}")
        
        return metrics
    
    def generate_performance_report(
        self,
        metrics: PerformanceMetrics,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable performance report.
        
        Args:
            metrics: Performance metrics to report.
            save_path: Optional path to save report.
            
        Returns:
            Formatted report string.
        """
        report_lines = [
            "=" * 80,
            "TEXT PROCESSING PIPELINE PERFORMANCE REPORT",
            "=" * 80,
            "",
            "TIMING METRICS:",
            f"  Total Time:               {metrics.total_time:.3f}s",
            f"  Preprocessing Time:       {metrics.preprocessing_time:.3f}s ({metrics.preprocessing_time/max(metrics.total_time, 0.001)*100:.1f}%)",
            f"  Tokenization Time:        {metrics.tokenization_time:.3f}s ({metrics.tokenization_time/max(metrics.total_time, 0.001)*100:.1f}%)",
            f"  Dataset Loading Time:     {metrics.dataset_loading_time:.3f}s ({metrics.dataset_loading_time/max(metrics.total_time, 0.001)*100:.1f}%)",
            f"  Batch Creation Time:      {metrics.batch_creation_time:.3f}s ({metrics.batch_creation_time/max(metrics.total_time, 0.001)*100:.1f}%)",
            "",
            "THROUGHPUT METRICS:",
            f"  Texts per Second:         {metrics.texts_per_second:.1f}",
            f"  Tokens per Second:        {metrics.tokens_per_second:.1f}",
            f"  Batches per Second:       {metrics.batches_per_second:.1f}",
            "",
            "MEMORY METRICS:",
            f"  Peak Memory Usage:        {metrics.peak_memory_mb:.1f} MB",
            f"  Average Memory Usage:     {metrics.avg_memory_mb:.1f} MB",
            f"  Memory Efficiency:        {metrics.memory_efficiency:.1f} tokens/MB",
            "",
            "DATA METRICS:",
            f"  Total Texts Processed:    {metrics.total_texts:,}",
            f"  Total Tokens Generated:   {metrics.total_tokens:,}",
            f"  Total Batches Created:    {metrics.total_batches}",
            f"  Average Text Length:      {metrics.avg_text_length:.1f} chars",
            f"  Average Sequence Length:  {metrics.avg_sequence_length:.1f} tokens",
            "",
            "QUALITY METRICS:",
            f"  Preprocessing Retention:  {metrics.preprocessing_retention_rate:.1%}",
            f"  Tokenization Coverage:    {metrics.tokenization_coverage:.1%}",
            "",
            "=" * 80
        ]
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Performance report saved to {save_path}")
        
        return report


def run_quick_benchmark(pipeline, test_texts: List[str]) -> PerformanceMetrics:
    """Run a quick benchmark with minimal iterations."""
    runner = BenchmarkRunner(warmup_iterations=1, measurement_iterations=3)
    return runner.run_comprehensive_benchmark(pipeline, test_texts, save_results=False)


def run_production_benchmark(pipeline, test_texts: List[str], results_dir: str = "benchmark_results") -> PerformanceMetrics:
    """Run a comprehensive production-ready benchmark."""
    Path(results_dir).mkdir(exist_ok=True)
    
    runner = BenchmarkRunner(warmup_iterations=5, measurement_iterations=20)
    metrics = runner.run_comprehensive_benchmark(
        pipeline, 
        test_texts, 
        save_results=True,
        results_path=f"{results_dir}/comprehensive_benchmark_{int(time.time())}.json"
    )
    
    # Generate report
    report = runner.generate_performance_report(
        metrics,
        save_path=f"{results_dir}/performance_report_{int(time.time())}.txt"
    )
    
    print(report)
    return metrics 