"""
Tests for the dataset loader functionality.
"""

import pytest
import torch
import tempfile
import json
import pickle
from pathlib import Path
from src.data.dataset_loader import (
    DatasetConfig,
    TokenizedDataset,
    DataCollator,
    DatasetLoader,
    create_default_loader,
    create_large_dataset_loader,
    create_small_dataset_loader
)


class TestDatasetConfig:
    """Test dataset configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DatasetConfig()
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.pad_token_id == 0
        assert config.dynamic_padding is True
        assert config.shuffle is True
        assert config.train_ratio == 0.8
        assert config.val_ratio == 0.1
        assert config.test_ratio == 0.1
    
    def test_config_validation_valid(self):
        """Test valid configuration passes validation."""
        config = DatasetConfig(
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            batch_size=16,
            max_length=256
        )
        config.validate()  # Should not raise
    
    def test_config_validation_invalid_ratios(self):
        """Test invalid ratio configurations."""
        # Ratios don't sum to 1
        config = DatasetConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
        with pytest.raises(ValueError, match="must equal 1.0"):
            config.validate()
        
        # Negative ratio
        config = DatasetConfig(train_ratio=-0.1, val_ratio=0.6, test_ratio=0.5)
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            config.validate()
    
    def test_config_validation_invalid_params(self):
        """Test invalid parameter configurations."""
        # Invalid batch size
        config = DatasetConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()
        
        # Invalid max length
        config = DatasetConfig(max_length=-1)
        with pytest.raises(ValueError, match="max_length must be positive"):
            config.validate()
        
        # Invalid min length
        config = DatasetConfig(min_length=-1)
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            config.validate()
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = DatasetConfig(batch_size=64, max_length=1024)
        config_dict = config.to_dict()
        assert config_dict['batch_size'] == 64
        assert config_dict['max_length'] == 1024
        assert 'train_ratio' in config_dict
    
    def test_config_from_dict(self):
        """Test configuration deserialization."""
        config_dict = {
            'batch_size': 128,
            'max_length': 2048,
            'shuffle': False,
            'cache_dataset': False
        }
        config = DatasetConfig.from_dict(config_dict)
        assert config.batch_size == 128
        assert config.max_length == 2048
        assert config.shuffle is False
        assert config.cache_dataset is False


class TestTokenizedDataset:
    """Test tokenized dataset functionality."""
    
    def test_dataset_from_list(self):
        """Test creating dataset from list of sequences."""
        sequences = [
            [1, 2, 3, 4, 5],
            [6, 7, 8],
            [9, 10, 11, 12],
            [13, 14]
        ]
        config = DatasetConfig(min_length=2, max_length=10)
        dataset = TokenizedDataset(sequences, config)
        
        assert len(dataset) == 4
        assert dataset.sequences == sequences
    
    def test_dataset_length_filtering(self):
        """Test sequence length filtering."""
        sequences = [
            [1],  # Too short
            [2, 3, 4],  # Valid
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Too long, will be truncated
            [16, 17, 18, 19]  # Valid
        ]
        config = DatasetConfig(min_length=2, max_length=6, truncate_sequences=True)
        dataset = TokenizedDataset(sequences, config)
        
        assert len(dataset) == 3  # One filtered out (too short)
        assert len(dataset.sequences[1]) <= 6  # Long sequence truncated
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        sequences = [[1, 2, 3], [4, 5, 6, 7]]
        config = DatasetConfig()
        dataset = TokenizedDataset(sequences, config)
        
        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'length' in sample
        assert 'original_idx' in sample
        assert torch.equal(sample['input_ids'], torch.tensor([1, 2, 3]))
        assert sample['length'] == 3
        assert sample['original_idx'] == 0
    
    def test_dataset_random_crop(self):
        """Test random cropping functionality."""
        sequences = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        config = DatasetConfig(random_crop=True, crop_ratio=0.5, min_length=2)
        dataset = TokenizedDataset(sequences, config)
        
        # Get multiple samples to test randomness
        samples = [dataset[0] for _ in range(10)]
        lengths = [sample['length'] for sample in samples]
        
        # Should have some variation in lengths due to random cropping
        assert min(lengths) >= 2  # At least min_length
        assert max(lengths) <= 10  # At most original length
    
    def test_dataset_caching(self):
        """Test dataset caching functionality."""
        sequences = [[1, 2, 3], [4, 5, 6]]
        config = DatasetConfig(cache_dataset=True, max_cache_size=10)
        dataset = TokenizedDataset(sequences, config)
        
        # Access same item multiple times
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        # Should be the same object (cached)
        assert sample1 is sample2
        assert len(dataset.cache) > 0
    
    def test_dataset_stats(self):
        """Test dataset statistics calculation."""
        sequences = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9, 10]
        ]
        config = DatasetConfig()
        dataset = TokenizedDataset(sequences, config)
        
        stats = dataset.get_stats()
        assert stats['num_sequences'] == 3
        assert stats['min_length'] == 2
        assert stats['max_length'] == 5
        assert stats['total_tokens'] == 10
        assert stats['vocab_size'] == 10  # All unique tokens
    
    def test_dataset_file_loading_json(self):
        """Test loading dataset from JSON file."""
        sequences = [[1, 2, 3], [4, 5, 6]]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sequences, f)
            temp_path = f.name
        
        try:
            config = DatasetConfig()
            dataset = TokenizedDataset(temp_path, config)
            assert len(dataset) == 2
            assert dataset.sequences == sequences
        finally:
            Path(temp_path).unlink()
    
    def test_dataset_file_loading_txt(self):
        """Test loading dataset from text file."""
        content = "1 2 3\n4 5 6 7\n8 9\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            config = DatasetConfig()
            dataset = TokenizedDataset(temp_path, config)
            assert len(dataset) == 3
            assert dataset.sequences[0] == [1, 2, 3]
            assert dataset.sequences[1] == [4, 5, 6, 7]
            assert dataset.sequences[2] == [8, 9]
        finally:
            Path(temp_path).unlink()


class TestDataCollator:
    """Test data collator functionality."""
    
    def test_collator_basic_padding(self):
        """Test basic padding functionality."""
        config = DatasetConfig(pad_token_id=0, max_length=5, dynamic_padding=False)
        collator = DataCollator(config)
        
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'length': 3,
                'original_idx': 0
            },
            {
                'input_ids': torch.tensor([4, 5]),
                'length': 2,
                'original_idx': 1
            }
        ]
        
        collated = collator(batch)
        
        assert collated['input_ids'].shape == (2, 5)
        assert torch.equal(collated['input_ids'][0], torch.tensor([1, 2, 3, 0, 0]))
        assert torch.equal(collated['input_ids'][1], torch.tensor([4, 5, 0, 0, 0]))
        assert torch.equal(collated['attention_mask'][0], torch.tensor([True, True, True, False, False]))
        assert torch.equal(collated['attention_mask'][1], torch.tensor([True, True, False, False, False]))
    
    def test_collator_dynamic_padding(self):
        """Test dynamic padding functionality."""
        config = DatasetConfig(pad_token_id=0, max_length=10, dynamic_padding=True)
        collator = DataCollator(config)
        
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'length': 3,
                'original_idx': 0
            },
            {
                'input_ids': torch.tensor([4, 5]),
                'length': 2,
                'original_idx': 1
            }
        ]
        
        collated = collator(batch)
        
        # Should pad to max length in batch (3), not config max_length (10)
        assert collated['input_ids'].shape == (2, 3)
        assert torch.equal(collated['input_ids'][0], torch.tensor([1, 2, 3]))
        assert torch.equal(collated['input_ids'][1], torch.tensor([4, 5, 0]))


class TestDatasetLoader:
    """Test dataset loader functionality."""
    
    def test_loader_initialization(self):
        """Test dataset loader initialization."""
        config = DatasetConfig(batch_size=16)
        loader = DatasetLoader(config)
        
        assert loader.config.batch_size == 16
        assert isinstance(loader.collator, DataCollator)
        assert len(loader.datasets) == 0
        assert len(loader.dataloaders) == 0
    
    def test_loader_single_dataset_split(self):
        """Test loading and splitting a single dataset."""
        sequences = [[i, i+1, i+2] for i in range(100)]
        config = DatasetConfig(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            batch_size=8
        )
        loader = DatasetLoader(config)
        loader.load_datasets(data=sequences)
        
        assert 'train' in loader.datasets
        assert 'val' in loader.datasets
        assert 'test' in loader.datasets
        
        # Check approximate split sizes
        total = len(sequences)
        assert len(loader.datasets['train']) == int(total * 0.8)
        assert len(loader.datasets['val']) == int(total * 0.1)
    
    def test_loader_separate_datasets(self):
        """Test loading separate train/val/test datasets."""
        train_data = [[1, 2, 3], [4, 5, 6]]
        val_data = [[7, 8, 9]]
        test_data = [[10, 11, 12]]
        
        config = DatasetConfig(batch_size=2)
        loader = DatasetLoader(config)
        loader.load_datasets(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )
        
        assert len(loader.datasets['train']) == 2
        assert len(loader.datasets['val']) == 1
        assert len(loader.datasets['test']) == 1
    
    def test_loader_dataloader_creation(self):
        """Test DataLoader creation."""
        sequences = [[i, i+1] for i in range(20)]
        config = DatasetConfig(batch_size=4, shuffle=True)
        loader = DatasetLoader(config)
        loader.load_datasets(data=sequences)
        
        train_dataloader = loader.get_dataloader('train')
        assert train_dataloader is not None
        assert train_dataloader.batch_size == 4
        
        # Test iteration
        batch = next(iter(train_dataloader))
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert batch['input_ids'].shape[0] <= 4  # Batch size
    
    def test_loader_statistics(self):
        """Test getting dataset statistics."""
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        config = DatasetConfig()
        loader = DatasetLoader(config)
        loader.load_datasets(data=sequences)
        
        stats = loader.get_stats()
        assert 'train' in stats
        assert 'val' in stats
        assert 'test' in stats
        
        # Check that stats contain expected fields
        train_stats = stats['train']
        assert 'num_sequences' in train_stats
        assert 'mean_length' in train_stats
    
    def test_loader_config_save_load(self):
        """Test saving and loading configuration."""
        config = DatasetConfig(batch_size=64, max_length=1024)
        loader = DatasetLoader(config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            loader.save_config(config_path)
            
            # Load config and create new loader
            new_loader = DatasetLoader.from_config_file(config_path)
            
            assert new_loader.config.batch_size == 64
            assert new_loader.config.max_length == 1024
            
        finally:
            Path(config_path).unlink()


class TestDatasetLoaderFactories:
    """Test dataset loader factory functions."""
    
    def test_default_loader(self):
        """Test default loader creation."""
        loader = create_default_loader()
        assert loader.config.batch_size == 32
        assert loader.config.max_length == 512
        assert loader.config.dynamic_padding is True
        assert loader.config.cache_dataset is True
    
    def test_large_dataset_loader(self):
        """Test large dataset loader creation."""
        loader = create_large_dataset_loader()
        assert loader.config.batch_size == 64
        assert loader.config.max_length == 1024
        assert loader.config.num_workers == 4
        assert loader.config.cache_dataset is False  # Disabled for large datasets
    
    def test_small_dataset_loader(self):
        """Test small dataset loader creation."""
        loader = create_small_dataset_loader()
        assert loader.config.batch_size == 16
        assert loader.config.max_length == 256
        assert loader.config.num_workers == 0  # Single process
        assert loader.config.cache_dataset is True


class TestIntegration:
    """Integration tests for the complete dataset loading pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to batches."""
        # Create sample data
        sequences = []
        for i in range(50):
            length = 5 + (i % 10)  # Variable lengths
            seq = list(range(i * 10, i * 10 + length))
            sequences.append(seq)
        
        # Create loader and load data
        config = DatasetConfig(
            batch_size=8,
            max_length=20,
            dynamic_padding=True,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0
        )
        loader = DatasetLoader(config)
        loader.load_datasets(data=sequences)
        
        # Test training dataloader
        train_dataloader = loader.get_dataloader('train')
        batch_count = 0
        total_samples = 0
        
        for batch in train_dataloader:
            batch_count += 1
            batch_size = batch['input_ids'].shape[0]
            total_samples += batch_size
            
            # Check batch structure
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'lengths' in batch
            assert batch['input_ids'].dtype == torch.long
            assert batch['attention_mask'].dtype == torch.bool
            
            # Check padding
            for i in range(batch_size):
                actual_length = batch['lengths'][i].item()
                attention_sum = batch['attention_mask'][i].sum().item()
                assert actual_length == attention_sum
        
        assert batch_count > 0
        assert total_samples <= len(loader.datasets['train'])
    
    def test_memory_efficiency(self):
        """Test memory efficiency features."""
        # Create larger dataset to test memory features
        sequences = [[i + j for j in range(10)] for i in range(1000)]
        
        config = DatasetConfig(
            batch_size=32,
            cache_dataset=True,
            max_cache_size=100,  # Small cache
            num_workers=0  # Single process for testing
        )
        
        loader = DatasetLoader(config)
        loader.load_datasets(data=sequences)
        
        # Access some data to populate cache
        dataset = loader.get_dataset('train')
        for i in range(50):
            _ = dataset[i]
        
        # Check cache size is respected
        if hasattr(dataset, 'cache') and dataset.cache:
            assert len(dataset.cache) <= config.max_cache_size


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        config = DatasetConfig()
        dataset = TokenizedDataset([], config)
        
        assert len(dataset) == 0
        stats = dataset.get_stats()
        assert stats == {}
    
    def test_invalid_file_format(self):
        """Test handling of invalid file formats."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("invalid content")
            temp_path = f.name
        
        try:
            config = DatasetConfig()
            with pytest.raises(ValueError, match="Unsupported file format"):
                TokenizedDataset(temp_path, config)
        finally:
            Path(temp_path).unlink()
    
    def test_missing_file(self):
        """Test handling of missing files."""
        config = DatasetConfig()
        with pytest.raises(FileNotFoundError):
            TokenizedDataset("nonexistent_file.json", config)
    
    def test_invalid_json_content(self):
        """Test handling of invalid JSON content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"invalid": "format"}, f)
            temp_path = f.name
        
        try:
            config = DatasetConfig()
            with pytest.raises(ValueError, match="Invalid JSON format"):
                TokenizedDataset(temp_path, config)
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__]) 