"""
Dataset Loader for handling tokenized sequences with batching and configuration.

This module provides dataset loading functionality with support for:
- Tokenized sequence management
- Batching with dynamic padding
- Train/validation/test splits
- Memory-efficient loading
- Configurable parameters
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import numpy as np
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    
    # Data paths
    data_path: str = "data/processed"
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Batching configuration
    batch_size: int = 32
    max_length: int = 512
    pad_token_id: int = 0
    dynamic_padding: bool = True
    drop_last: bool = False
    
    # Data splitting (if using single file)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Loading configuration
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Sequence handling
    truncate_sequences: bool = True
    min_length: int = 1
    pack_sequences: bool = False  # Pack multiple short sequences into one batch
    
    # Memory optimization
    cache_dataset: bool = True
    max_cache_size: int = 10000  # Maximum number of samples to cache
    use_memory_mapping: bool = False
    
    # Data augmentation
    random_crop: bool = False
    crop_ratio: float = 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data_path': self.data_path,
            'train_file': self.train_file,
            'val_file': self.val_file,
            'test_file': self.test_file,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'pad_token_id': self.pad_token_id,
            'dynamic_padding': self.dynamic_padding,
            'drop_last': self.drop_last,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'prefetch_factor': self.prefetch_factor,
            'truncate_sequences': self.truncate_sequences,
            'min_length': self.min_length,
            'pack_sequences': self.pack_sequences,
            'cache_dataset': self.cache_dataset,
            'max_cache_size': self.max_cache_size,
            'use_memory_mapping': self.use_memory_mapping,
            'random_crop': self.random_crop,
            'crop_ratio': self.crop_ratio,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DatasetConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def validate(self):
        """Validate configuration parameters."""
        if not (0 < self.train_ratio <= 1):
            raise ValueError("train_ratio must be between 0 and 1")
        if not (0 <= self.val_ratio <= 1):
            raise ValueError("val_ratio must be between 0 and 1")
        if not (0 <= self.test_ratio <= 1):
            raise ValueError("test_ratio must be between 0 and 1")
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.min_length < 0:
            raise ValueError("min_length must be non-negative")
        if not (0 < self.crop_ratio <= 1):
            raise ValueError("crop_ratio must be between 0 and 1")


class TokenizedDataset(Dataset):
    """
    Dataset for handling tokenized sequences.
    
    Supports various input formats:
    - List of token sequences
    - File paths to tokenized data
    - Memory-mapped files for large datasets
    """
    
    def __init__(
        self,
        data: Union[List[List[int]], str, Path],
        config: DatasetConfig,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data: Tokenized sequences or path to data file.
            config: Dataset configuration.
            transform: Optional transform function to apply to samples.
        """
        self.config = config
        self.transform = transform
        self.cache = {} if config.cache_dataset else None
        
        # Load or set data
        if isinstance(data, (str, Path)):
            self.data_path = Path(data)
            self.sequences = self._load_from_file(self.data_path)
        else:
            self.data_path = None
            self.sequences = data
        
        # Filter sequences by length
        self.sequences = self._filter_sequences(self.sequences)
        
        # Create index mapping for memory-mapped files
        self.index_map = list(range(len(self.sequences)))
        
        logger.info(f"Loaded dataset with {len(self.sequences)} sequences")
        
        # Calculate statistics
        self.stats = self._calculate_stats()
    
    def _load_from_file(self, file_path: Path) -> List[List[int]]:
        """Load tokenized sequences from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'sequences' in data:
                    return data['sequences']
                elif isinstance(data, list):
                    return data
                else:
                    raise ValueError("Invalid JSON format")
        
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        elif file_path.suffix == '.txt':
            sequences = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Assume space-separated token IDs
                        try:
                            sequence = [int(x) for x in line.split()]
                            sequences.append(sequence)
                        except ValueError:
                            logger.warning(f"Skipping invalid line: {line}")
            return sequences
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _filter_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        """Filter sequences by length criteria."""
        filtered = []
        for seq in sequences:
            if self.config.min_length <= len(seq) <= self.config.max_length:
                filtered.append(seq)
            elif self.config.truncate_sequences and len(seq) > self.config.max_length:
                # Truncate long sequences
                filtered.append(seq[:self.config.max_length])
        
        logger.info(f"Filtered {len(sequences)} -> {len(filtered)} sequences")
        return filtered
    
    def _calculate_stats(self) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        if not self.sequences:
            return {}
        
        lengths = [len(seq) for seq in self.sequences]
        return {
            'num_sequences': len(self.sequences),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'total_tokens': sum(lengths),
            'vocab_size': len(set(token for seq in self.sequences for token in seq))
        }
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        if self.cache and idx in self.cache:
            return self.cache[idx]
        
        sequence = self.sequences[idx]
        
        # Apply random cropping if enabled
        if self.config.random_crop and len(sequence) > self.config.min_length:
            crop_length = max(
                self.config.min_length,
                int(len(sequence) * self.config.crop_ratio)
            )
            start_idx = np.random.randint(0, len(sequence) - crop_length + 1)
            sequence = sequence[start_idx:start_idx + crop_length]
        
        # Create sample
        sample = {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'length': len(sequence),
            'original_idx': idx
        }
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        # Cache if enabled
        if self.cache and len(self.cache) < self.config.max_cache_size:
            self.cache[idx] = sample
        
        return sample
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.stats.copy()
    
    def save_cache(self, cache_path: Union[str, Path]):
        """Save cached data to disk."""
        if self.cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.cache, f)
    
    def load_cache(self, cache_path: Union[str, Path]):
        """Load cached data from disk."""
        if Path(cache_path).exists():
            with open(cache_path, 'rb') as f:
                self.cache = pickle.load(f)


class DataCollator:
    """
    Collator for batching tokenized sequences with padding.
    """
    
    def __init__(self, config: DatasetConfig):
        """Initialize collator with configuration."""
        self.config = config
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of samples from dataset.
            
        Returns:
            Batched and padded tensors.
        """
        # Extract input_ids and lengths
        input_ids = [sample['input_ids'] for sample in batch]
        lengths = [sample['length'] for sample in batch]
        original_indices = [sample['original_idx'] for sample in batch]
        
        # Determine padding length
        if self.config.dynamic_padding:
            max_len = max(lengths)
            # Optionally round up to nearest multiple for efficiency
            max_len = min(max_len, self.config.max_length)
        else:
            max_len = self.config.max_length
        
        # Pad sequences
        batch_size = len(batch)
        padded_input_ids = torch.full(
            (batch_size, max_len),
            self.config.pad_token_id,
            dtype=torch.long
        )
        
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        for i, (seq, length) in enumerate(zip(input_ids, lengths)):
            actual_len = min(length, max_len)
            padded_input_ids[i, :actual_len] = seq[:actual_len]
            attention_mask[i, :actual_len] = True
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'lengths': torch.tensor(lengths, dtype=torch.long),
            'original_indices': torch.tensor(original_indices, dtype=torch.long)
        }


class DatasetLoader:
    """
    Main dataset loader with configuration and splitting capabilities.
    """
    
    def __init__(self, config: DatasetConfig):
        """Initialize dataset loader."""
        self.config = config
        self.config.validate()
        
        self.datasets = {}
        self.dataloaders = {}
        self.collator = DataCollator(config)
        
        logger.info(f"Initialized DatasetLoader with config: {config}")
    
    def load_datasets(
        self,
        data: Optional[Union[str, Path, List[List[int]]]] = None,
        train_data: Optional[Union[str, Path, List[List[int]]]] = None,
        val_data: Optional[Union[str, Path, List[List[int]]]] = None,
        test_data: Optional[Union[str, Path, List[List[int]]]] = None
    ):
        """
        Load datasets from various sources.
        
        Args:
            data: Single dataset to split into train/val/test.
            train_data: Training dataset.
            val_data: Validation dataset.
            test_data: Test dataset.
        """
        if data is not None:
            # Load single dataset and split
            full_dataset = TokenizedDataset(data, self.config)
            self._split_dataset(full_dataset)
        
        else:
            # Load separate datasets
            if train_data is not None:
                self.datasets['train'] = TokenizedDataset(train_data, self.config)
            
            if val_data is not None:
                self.datasets['val'] = TokenizedDataset(val_data, self.config)
            
            if test_data is not None:
                self.datasets['test'] = TokenizedDataset(test_data, self.config)
        
        # Create data loaders
        self._create_dataloaders()
        
        # Log dataset information
        self._log_dataset_info()
    
    def _split_dataset(self, dataset: TokenizedDataset):
        """Split dataset into train/val/test."""
        total_size = len(dataset)
        
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)
        test_size = total_size - train_size - val_size
        
        # Ensure sizes are positive
        if test_size < 0:
            val_size += test_size
            test_size = 0
        
        if val_size < 0:
            train_size += val_size
            val_size = 0
        
        splits = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        if train_size > 0:
            self.datasets['train'] = splits[0]
        if val_size > 0:
            self.datasets['val'] = splits[1]
        if test_size > 0:
            self.datasets['test'] = splits[2]
    
    def _create_dataloaders(self):
        """Create PyTorch DataLoaders for each dataset."""
        for split, dataset in self.datasets.items():
            shuffle = self.config.shuffle if split == 'train' else False
            
            self.dataloaders[split] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last,
                collate_fn=self.collator,
                prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None
            )
    
    def _log_dataset_info(self):
        """Log information about loaded datasets."""
        for split, dataset in self.datasets.items():
            size = len(dataset)
            if hasattr(dataset, 'get_stats'):
                stats = dataset.get_stats()
                logger.info(f"{split.upper()} dataset: {size} samples, "
                           f"avg_length={stats.get('mean_length', 0):.1f}, "
                           f"vocab_size={stats.get('vocab_size', 0)}")
            else:
                logger.info(f"{split.upper()} dataset: {size} samples")
    
    def get_dataloader(self, split: str) -> Optional[DataLoader]:
        """Get DataLoader for specified split."""
        return self.dataloaders.get(split)
    
    def get_dataset(self, split: str) -> Optional[Dataset]:
        """Get Dataset for specified split."""
        return self.datasets.get(split)
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all datasets."""
        stats = {}
        for split, dataset in self.datasets.items():
            if hasattr(dataset, 'get_stats'):
                stats[split] = dataset.get_stats()
            else:
                stats[split] = {'num_samples': len(dataset)}
        return stats
    
    def save_config(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'DatasetLoader':
        """Create DatasetLoader from configuration file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = DatasetConfig.from_dict(config_dict)
        return cls(config)


def create_default_loader() -> DatasetLoader:
    """Create a dataset loader with default configuration."""
    config = DatasetConfig(
        batch_size=32,
        max_length=512,
        dynamic_padding=True,
        shuffle=True,
        num_workers=2,
        cache_dataset=True
    )
    return DatasetLoader(config)


def create_large_dataset_loader() -> DatasetLoader:
    """Create a dataset loader optimized for large datasets."""
    config = DatasetConfig(
        batch_size=64,
        max_length=1024,
        dynamic_padding=True,
        shuffle=True,
        num_workers=4,
        cache_dataset=False,  # Disable caching for large datasets
        use_memory_mapping=True,
        pin_memory=True,
        prefetch_factor=4
    )
    return DatasetLoader(config)


def create_small_dataset_loader() -> DatasetLoader:
    """Create a dataset loader optimized for small datasets."""
    config = DatasetConfig(
        batch_size=16,
        max_length=256,
        dynamic_padding=True,
        shuffle=True,
        num_workers=0,  # Single process for small datasets
        cache_dataset=True,
        max_cache_size=50000
    )
    return DatasetLoader(config) 