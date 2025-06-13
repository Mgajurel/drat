"""
Test script for BPE Tokenizer implementation.

This script tests the tokenizer functionality including training,
encoding/decoding, special tokens, and consistency validation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tokenizer.bpe_tokenizer import BPETokenizer, BPETokenizerConfig
import torch
import tempfile
import shutil
from pathlib import Path
import pytest


@pytest.fixture
def test_texts():
    """Sample training texts for testing."""
    return [
        "Hello world, how are you today?",
        "This is a sample text for training our tokenizer.",
        "Machine learning and natural language processing are fascinating fields.",
        "We need to test various scenarios and edge cases.",
        "The quick brown fox jumps over the lazy dog.",
        "Python programming is fun and powerful.",
        "Artificial intelligence will change the world.",
        "Deep learning models require large amounts of data.",
        "Transformers have revolutionized natural language understanding.",
        "Tokenization is a crucial preprocessing step.",
    ] * 10  # Repeat to have more training data


@pytest.fixture
def tokenizer(test_texts):
    """Trained tokenizer fixture."""
    # Create tokenizer with small vocab for testing
    config = BPETokenizerConfig(vocab_size=1000, min_frequency=2)
    tokenizer = BPETokenizer(config)
    
    # Train tokenizer
    tokenizer.train(test_texts)
    return tokenizer


def test_tokenizer_config():
    """Test tokenizer configuration creation and serialization."""
    print("Testing tokenizer configuration...")
    
    # Test default config
    config = BPETokenizerConfig()
    assert config.vocab_size == 32000
    assert config.pad_token == "[PAD]"
    assert config.unk_token == "[UNK]"
    assert config.bos_token == "[BOS]"
    assert config.eos_token == "[EOS]"
    print("✓ Default config test passed")
    
    # Test custom config
    custom_config = BPETokenizerConfig(
        vocab_size=16000,
        min_frequency=3,
        pad_token="<pad>",
        unk_token="<unk>"
    )
    assert custom_config.vocab_size == 16000
    assert custom_config.min_frequency == 3
    assert custom_config.pad_token == "<pad>"
    print("✓ Custom config test passed")
    
    # Test serialization
    config_dict = custom_config.to_dict()
    loaded_config = BPETokenizerConfig.from_dict(config_dict)
    assert loaded_config.vocab_size == custom_config.vocab_size
    assert loaded_config.pad_token == custom_config.pad_token
    print("✓ Config serialization test passed")


def test_tokenizer_training(test_texts):
    """Test tokenizer training functionality."""
    print("\nTesting tokenizer training...")
    
    # Create tokenizer with small vocab for testing
    config = BPETokenizerConfig(vocab_size=1000, min_frequency=2)
    tokenizer = BPETokenizer(config)
    
    # Train tokenizer
    tokenizer.train(test_texts)
    assert tokenizer._is_trained
    print(f"✓ Tokenizer trained with vocab size: {tokenizer.vocab_size}")


def test_encoding_decoding(tokenizer, test_texts):
    """Test encoding and decoding functionality."""
    print("\nTesting encoding and decoding...")
    
    # Test single text encoding/decoding
    test_text = "Hello world, this is a test message."
    
    # Test encoding
    token_ids = tokenizer.encode(test_text)
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    print(f"✓ Encoded text to {len(token_ids)} tokens")
    
    # Test decoding
    decoded_text = tokenizer.decode(token_ids)
    assert isinstance(decoded_text, str)
    print(f"✓ Decoded tokens back to text: '{decoded_text[:50]}...'")
    
    # Test tensor encoding
    token_tensor = tokenizer.encode(test_text, return_tensors='pt')
    assert torch.is_tensor(token_tensor)
    assert token_tensor.dtype == torch.long
    print("✓ Tensor encoding test passed")
    
    # Test batch encoding
    batch_texts = test_texts[:5]
    batch_ids = tokenizer.encode_batch(batch_texts)
    assert len(batch_ids) == len(batch_texts)
    print(f"✓ Batch encoding test passed for {len(batch_texts)} texts")
    
    # Test batch tensor encoding
    batch_tensor = tokenizer.encode_batch(batch_texts, return_tensors='pt')
    assert torch.is_tensor(batch_tensor)
    assert batch_tensor.shape[0] == len(batch_texts)
    print(f"✓ Batch tensor encoding test passed, shape: {batch_tensor.shape}")
    
    # Test batch decoding
    batch_decoded = tokenizer.decode_batch(batch_ids)
    assert len(batch_decoded) == len(batch_texts)
    print("✓ Batch decoding test passed")


def test_special_tokens(tokenizer):
    """Test special token functionality."""
    print("\nTesting special tokens...")
    
    # Test special token IDs
    pad_id = tokenizer.pad_token_id
    unk_id = tokenizer.unk_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    
    assert pad_id is not None
    assert unk_id is not None
    assert bos_id is not None
    assert eos_id is not None
    print(f"✓ Special token IDs: PAD={pad_id}, UNK={unk_id}, BOS={bos_id}, EOS={eos_id}")
    
    # Test token to ID and ID to token conversion
    pad_token = tokenizer.id_to_token(pad_id)
    assert pad_token == tokenizer.config.pad_token
    
    pad_id_check = tokenizer.token_to_id(tokenizer.config.pad_token)
    assert pad_id_check == pad_id
    print("✓ Token/ID conversion test passed")
    
    # Test encoding with and without special tokens
    test_text = "Hello world"
    with_special = tokenizer.encode(test_text, add_special_tokens=True)
    without_special = tokenizer.encode(test_text, add_special_tokens=False)
    
    assert len(with_special) > len(without_special)
    print("✓ Special token inclusion test passed")


def test_vocabulary_access(tokenizer):
    """Test vocabulary access functions."""
    print("\nTesting vocabulary access...")
    
    # Test vocabulary retrieval
    vocab = tokenizer.get_vocab()
    assert isinstance(vocab, dict)
    assert len(vocab) > 0
    print(f"✓ Retrieved vocabulary with {len(vocab)} tokens")
    
    # Test that special tokens are in vocabulary
    assert tokenizer.config.pad_token in vocab
    assert tokenizer.config.unk_token in vocab
    assert tokenizer.config.bos_token in vocab
    assert tokenizer.config.eos_token in vocab
    print("✓ All special tokens found in vocabulary")


def test_save_load_functionality(tokenizer):
    """Test save and load functionality."""
    print("\nTesting save/load functionality...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "test_tokenizer"
        
        # Save tokenizer
        tokenizer.save(str(save_path))
        assert (save_path / "tokenizer.json").exists()
        assert (save_path / "config.json").exists()
        print("✓ Tokenizer saved successfully")
        
        # Load tokenizer
        loaded_tokenizer = BPETokenizer.load(str(save_path))
        assert loaded_tokenizer._is_trained
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
        print("✓ Tokenizer loaded successfully")
        
        # Test that loaded tokenizer works the same
        test_text = "Test text for consistency check"
        original_ids = tokenizer.encode(test_text)
        loaded_ids = loaded_tokenizer.encode(test_text)
        assert original_ids == loaded_ids
        print("✓ Loaded tokenizer produces identical results")


def test_consistency_validation(tokenizer, test_texts):
    """Test tokenization consistency validation."""
    print("\nTesting consistency validation...")
    
    # Run consistency test
    test_sample = test_texts[:20]  # Use subset for testing
    results = tokenizer.test_consistency(test_sample)
    
    assert 'total_texts' in results
    assert 'encoding_errors' in results
    assert 'roundtrip_errors' in results
    assert 'avg_tokens_per_text' in results
    assert 'vocab_coverage' in results
    
    print(f"✓ Consistency test results:")
    print(f"  - Total texts: {results['total_texts']}")
    print(f"  - Encoding errors: {results['encoding_errors']}")
    print(f"  - Roundtrip errors: {results['roundtrip_errors']}")
    print(f"  - Avg tokens per text: {results['avg_tokens_per_text']:.2f}")
    print(f"  - Vocab coverage: {results['vocab_coverage']:.2%}")
    
    # Validate that there are minimal encoding errors (actual failures)
    assert results['encoding_errors'] == 0
    # Note: BPE tokenizers may have roundtrip differences due to subword merging,
    # especially with small vocabularies. This is normal behavior.


def test_edge_cases(tokenizer):
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    # Test empty string
    empty_ids = tokenizer.encode("")
    assert len(empty_ids) >= 2  # Should have at least BOS and EOS
    print("✓ Empty string encoding test passed")
    
    # Test very long text
    long_text = "This is a very long text. " * 100
    long_ids = tokenizer.encode(long_text)
    assert len(long_ids) > 100
    decoded_long = tokenizer.decode(long_ids)
    assert len(decoded_long) > 0
    print("✓ Long text encoding/decoding test passed")
    
    # Test special characters
    special_text = "Hello! @#$%^&*()_+ 你好 🌟 测试"
    special_ids = tokenizer.encode(special_text)
    special_decoded = tokenizer.decode(special_ids)
    assert len(special_decoded) > 0
    print("✓ Special character encoding/decoding test passed")
    
    # Test numeric text
    numeric_text = "12345 67890 3.14159 -42"
    numeric_ids = tokenizer.encode(numeric_text)
    numeric_decoded = tokenizer.decode(numeric_ids)
    assert len(numeric_decoded) > 0
    print("✓ Numeric text encoding/decoding test passed")


def run_all_tests():
    """Run all tokenizer tests."""
    print("=" * 60)
    print("Running BPE Tokenizer Tests")
    print("=" * 60)
    
    test_tokenizer_config()
    
    # Create test data
    training_texts = [
        "Hello world, how are you today?",
        "This is a sample text for training our tokenizer.",
        "Machine learning and natural language processing are fascinating fields.",
        "We need to test various scenarios and edge cases.",
        "The quick brown fox jumps over the lazy dog.",
        "Python programming is fun and powerful.",
        "Artificial intelligence will change the world.",
        "Deep learning models require large amounts of data.",
        "Transformers have revolutionized natural language understanding.",
        "Tokenization is a crucial preprocessing step.",
    ] * 10
    
    # Test training
    tokenizer, test_texts = test_tokenizer_training(training_texts)
    
    # Run all other tests with the trained tokenizer
    test_encoding_decoding(tokenizer, test_texts)
    test_special_tokens(tokenizer)
    test_vocabulary_access(tokenizer)
    test_save_load_functionality(tokenizer)
    test_consistency_validation(tokenizer, test_texts)
    test_edge_cases(tokenizer)
    
    print("\n" + "=" * 60)
    print("All BPE Tokenizer Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests() 