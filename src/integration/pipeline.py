"""
Unified text processing pipeline integrating preprocessing, tokenization, and dataset loading.

This module provides a high-level interface for the complete text processing pipeline
from raw text to batched tensors ready for model training.
"""

import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import logging
import time
import json

from ..data.preprocessing import TextPreprocessor, PreprocessingConfig
from ..tokenizer.bpe_tokenizer import BPETokenizer, BPETokenizerConfig
from ..data.dataset_loader import DatasetLoader, DatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the complete text processing pipeline."""
    
    # Preprocessing configuration
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    
    # Tokenization configuration
    tokenization: BPETokenizerConfig = field(default_factory=BPETokenizerConfig)
    
    # Dataset loading configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Pipeline-specific settings
    cache_processed_data: bool = True
    cache_dir: str = "cache"
    validate_consistency: bool = True
    log_performance: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'preprocessing': self.preprocessing.to_dict(),
            'tokenization': self.tokenization.to_dict(),
            'dataset': self.dataset.to_dict(),
            'cache_processed_data': self.cache_processed_data,
            'cache_dir': self.cache_dir,
            'validate_consistency': self.validate_consistency,
            'log_performance': self.log_performance
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        preprocessing_config = PreprocessingConfig.from_dict(
            config_dict.get('preprocessing', {})
        )
        tokenization_config = BPETokenizerConfig.from_dict(
            config_dict.get('tokenization', {})
        )
        dataset_config = DatasetConfig.from_dict(
            config_dict.get('dataset', {})
        )
        
        return cls(
            preprocessing=preprocessing_config,
            tokenization=tokenization_config,
            dataset=dataset_config,
            cache_processed_data=config_dict.get('cache_processed_data', True),
            cache_dir=config_dict.get('cache_dir', 'cache'),
            validate_consistency=config_dict.get('validate_consistency', True),
            log_performance=config_dict.get('log_performance', True)
        )


class TextProcessingPipeline:
    """
    Unified pipeline for text preprocessing, tokenization, and dataset loading.
    
    This class integrates all components into a cohesive pipeline that can process
    raw text data and produce batched tensors ready for model training.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the text processing pipeline.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = TextPreprocessor(config.preprocessing)
        self.tokenizer = None  # Initialized when needed
        self.dataset_loader = DatasetLoader(config.dataset)
        
        # Performance tracking
        self.performance_stats = {
            'preprocessing_time': 0.0,
            'tokenization_time': 0.0,
            'dataset_loading_time': 0.0,
            'total_time': 0.0,
            'texts_processed': 0,
            'tokens_generated': 0,
            'sequences_created': 0
        }
        
        logger.info(f"Initialized TextProcessingPipeline with config: {config}")
    
    def train_tokenizer(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> BPETokenizer:
        """
        Train the BPE tokenizer on provided texts.
        
        Args:
            texts: Training texts for tokenizer.
            vocab_size: Vocabulary size (overrides config if provided).
            save_path: Path to save trained tokenizer.
            
        Returns:
            Trained BPE tokenizer.
        """
        start_time = time.time()
        
        # Preprocess texts for tokenizer training
        logger.info(f"Preprocessing {len(texts)} texts for tokenizer training...")
        preprocessed_texts = []
        
        for text in texts:
            processed = self.preprocessor.preprocess_text(text, validate=False)
            if processed:  # Only include non-empty texts
                preprocessed_texts.append(processed)
        
        logger.info(f"Preprocessed {len(preprocessed_texts)}/{len(texts)} texts")
        
        # Update vocab size if provided
        tokenizer_config = self.config.tokenization
        if vocab_size is not None:
            tokenizer_config.vocab_size = vocab_size
        
        # Train tokenizer
        self.tokenizer = BPETokenizer(tokenizer_config)
        logger.info("Training BPE tokenizer...")
        self.tokenizer.train(preprocessed_texts)
        
        # Save tokenizer if path provided
        if save_path:
            self.tokenizer.save(save_path)
            logger.info(f"Saved trained tokenizer to {save_path}")
        
        # Update performance stats
        training_time = time.time() - start_time
        self.performance_stats['tokenization_time'] += training_time
        self.performance_stats['texts_processed'] += len(texts)
        
        logger.info(f"Tokenizer training completed in {training_time:.2f}s")
        return self.tokenizer
    
    def load_tokenizer(self, tokenizer_path: Union[str, Path]) -> BPETokenizer:
        """
        Load a pre-trained tokenizer.
        
        Args:
            tokenizer_path: Path to saved tokenizer.
            
        Returns:
            Loaded BPE tokenizer.
        """
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
        return self.tokenizer
    
    def process_texts(
        self,
        texts: List[str],
        return_tensors: bool = True,
        cache_key: Optional[str] = None
    ) -> List[Union[List[int], torch.Tensor]]:
        """
        Process texts through the complete pipeline.
        
        Args:
            texts: Raw input texts.
            return_tensors: Whether to return torch tensors.
            cache_key: Optional cache key for processed results.
            
        Returns:
            List of tokenized sequences (as lists or tensors).
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call train_tokenizer() or load_tokenizer() first.")
        
        start_time = time.time()
        
        # Check cache
        cache_path = None
        if cache_key and self.config.cache_processed_data:
            cache_path = self.cache_dir / f"{cache_key}_processed.json"
            if cache_path.exists():
                logger.info(f"Loading cached processed data from {cache_path}")
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                sequences = cached_data['sequences']
                if return_tensors:
                    sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
                return sequences
        
        # Process texts
        logger.info(f"Processing {len(texts)} texts through pipeline...")
        
        # Step 1: Preprocessing
        preprocessing_start = time.time()
        preprocessed_texts = []
        preprocessing_stats = []
        
        for text in texts:
            processed = self.preprocessor.preprocess_text(text, validate=True)
            if processed:
                preprocessed_texts.append(processed)
                preprocessing_stats.append(self.preprocessor.last_stats)
        
        preprocessing_time = time.time() - preprocessing_start
        logger.info(f"Preprocessing completed: {len(preprocessed_texts)}/{len(texts)} texts in {preprocessing_time:.2f}s")
        
        # Step 2: Tokenization
        tokenization_start = time.time()
        sequences = []
        total_tokens = 0
        
        for text in preprocessed_texts:
            tokens = self.tokenizer.encode(text)
            sequences.append(tokens)
            total_tokens += len(tokens)
        
        tokenization_time = time.time() - tokenization_start
        logger.info(f"Tokenization completed: {total_tokens} tokens in {tokenization_time:.2f}s")
        
        # Cache results if requested
        if cache_path and self.config.cache_processed_data:
            cache_data = {
                'sequences': sequences,
                'preprocessing_stats': preprocessing_stats,
                'total_tokens': total_tokens,
                'processing_time': time.time() - start_time
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            logger.info(f"Cached processed data to {cache_path}")
        
        # Convert to tensors if requested
        if return_tensors:
            sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        
        # Update performance stats
        total_time = time.time() - start_time
        self.performance_stats['preprocessing_time'] += preprocessing_time
        self.performance_stats['tokenization_time'] += tokenization_time
        self.performance_stats['total_time'] += total_time
        self.performance_stats['texts_processed'] += len(texts)
        self.performance_stats['tokens_generated'] += total_tokens
        self.performance_stats['sequences_created'] += len(sequences)
        
        logger.info(f"Pipeline processing completed in {total_time:.2f}s")
        return sequences
    
    def create_datasets(
        self,
        sequences: Optional[List[List[int]]] = None,
        texts: Optional[List[str]] = None,
        train_data: Optional[Union[List[str], List[List[int]], str, Path]] = None,
        val_data: Optional[Union[List[str], List[List[int]], str, Path]] = None,
        test_data: Optional[Union[List[str], List[List[int]], str, Path]] = None,
        cache_key: Optional[str] = None
    ) -> DatasetLoader:
        """
        Create datasets from sequences or texts.
        
        Args:
            sequences: Pre-tokenized sequences.
            texts: Raw texts to process.
            train_data: Training data (texts, sequences, or file path).
            val_data: Validation data.
            test_data: Test data.
            cache_key: Cache key for processing.
            
        Returns:
            Configured DatasetLoader with loaded datasets.
        """
        start_time = time.time()
        
        if sequences is not None:
            # Use provided sequences directly
            self.dataset_loader.load_datasets(data=sequences)
            
        elif texts is not None:
            # Process texts to sequences
            sequences = self.process_texts(texts, return_tensors=False, cache_key=cache_key)
            self.dataset_loader.load_datasets(data=sequences)
            
        elif any([train_data, val_data, test_data]):
            # Process separate datasets
            processed_train = None
            processed_val = None
            processed_test = None
            
            if train_data is not None:
                if isinstance(train_data, (str, Path)):
                    processed_train = train_data  # File path
                elif isinstance(train_data[0], str):  # List of texts
                    processed_train = self.process_texts(
                        train_data, return_tensors=False, 
                        cache_key=f"{cache_key}_train" if cache_key else None
                    )
                else:  # List of sequences
                    processed_train = train_data
            
            if val_data is not None:
                if isinstance(val_data, (str, Path)):
                    processed_val = val_data  # File path
                elif isinstance(val_data[0], str):  # List of texts
                    processed_val = self.process_texts(
                        val_data, return_tensors=False,
                        cache_key=f"{cache_key}_val" if cache_key else None
                    )
                else:  # List of sequences
                    processed_val = val_data
            
            if test_data is not None:
                if isinstance(test_data, (str, Path)):
                    processed_test = test_data  # File path
                elif isinstance(test_data[0], str):  # List of texts
                    processed_test = self.process_texts(
                        test_data, return_tensors=False,
                        cache_key=f"{cache_key}_test" if cache_key else None
                    )
                else:  # List of sequences
                    processed_test = test_data
            
            self.dataset_loader.load_datasets(
                train_data=processed_train,
                val_data=processed_val,
                test_data=processed_test
            )
        
        else:
            raise ValueError("Must provide either sequences, texts, or train/val/test data")
        
        # Update performance stats
        dataset_time = time.time() - start_time
        self.performance_stats['dataset_loading_time'] += dataset_time
        
        logger.info(f"Dataset creation completed in {dataset_time:.2f}s")
        return self.dataset_loader
    
    def get_dataloader(self, split: str = 'train') -> torch.utils.data.DataLoader:
        """
        Get PyTorch DataLoader for specified split.
        
        Args:
            split: Dataset split ('train', 'val', 'test').
            
        Returns:
            PyTorch DataLoader.
        """
        dataloader = self.dataset_loader.get_dataloader(split)
        if dataloader is None:
            raise ValueError(f"No dataloader available for split '{split}'")
        return dataloader
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """
        Validate the complete pipeline functionality.
        
        Returns:
            Validation results.
        """
        if self.config.validate_consistency:
            from .validation import validate_pipeline
            return validate_pipeline(self)
        else:
            return {'validation': 'skipped'}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        # Calculate derived metrics
        if stats['texts_processed'] > 0:
            stats['avg_preprocessing_time_per_text'] = stats['preprocessing_time'] / stats['texts_processed']
            stats['avg_tokenization_time_per_text'] = stats['tokenization_time'] / stats['texts_processed']
            stats['avg_total_time_per_text'] = stats['total_time'] / stats['texts_processed']
        
        if stats['tokens_generated'] > 0:
            stats['tokens_per_second'] = stats['tokens_generated'] / max(stats['tokenization_time'], 0.001)
        
        if stats['sequences_created'] > 0:
            stats['avg_sequence_length'] = stats['tokens_generated'] / stats['sequences_created']
        
        return stats
    
    def save_config(self, path: Union[str, Path]):
        """Save pipeline configuration."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Saved pipeline configuration to {path}")
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'TextProcessingPipeline':
        """Load pipeline from configuration file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = PipelineConfig.from_dict(config_dict)
        return cls(config)
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        for key in self.performance_stats:
            self.performance_stats[key] = 0.0


def create_default_pipeline() -> TextProcessingPipeline:
    """Create a pipeline with default configuration."""
    config = PipelineConfig()
    return TextProcessingPipeline(config)


def create_production_pipeline(
    vocab_size: int = 50000,
    batch_size: int = 32,
    max_length: int = 512
) -> TextProcessingPipeline:
    """Create a pipeline optimized for production use."""
    from ..data.preprocessing import create_default_preprocessor
    from ..data.dataset_loader import create_default_loader
    
    # Production-ready preprocessing
    preprocessing_config = create_default_preprocessor().config
    
    # Production tokenizer config
    tokenizer_config = BPETokenizerConfig(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>'],
        pad_token='<PAD>',
        unk_token='<UNK>',
        bos_token='<BOS>',
        eos_token='<EOS>',
        mask_token='<MASK>'
    )
    
    # Production dataset config
    dataset_config = create_default_loader().config
    dataset_config.batch_size = batch_size
    dataset_config.max_length = max_length
    
    config = PipelineConfig(
        preprocessing=preprocessing_config,
        tokenization=tokenizer_config,
        dataset=dataset_config,
        cache_processed_data=True,
        validate_consistency=True,
        log_performance=True
    )
    
    return TextProcessingPipeline(config) 