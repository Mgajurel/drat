"""
Example usage of the BPE Tokenizer for the Differentiable Recomputation Gates project.

This script demonstrates how to train and use the BPE tokenizer with sample text data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tokenizer.bpe_tokenizer import BPETokenizer, BPETokenizerConfig
import torch


def create_sample_data():
    """Create sample text data for tokenizer training."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is revolutionizing artificial intelligence.",
        "Natural language processing enables computers to understand human text.",
        "Transformers have become the dominant architecture for many NLP tasks.",
        "Attention mechanisms allow models to focus on relevant parts of input.",
        "Deep learning requires large datasets and computational resources.",
        "Python is a popular programming language for data science.",
        "Neural networks can learn complex patterns from data.",
        "Gradient descent is used to optimize model parameters.",
        "Backpropagation computes gradients for neural network training.",
        "Tokenization splits text into smaller units called tokens.",
        "Byte Pair Encoding is an effective subword tokenization method.",
        "Special tokens like [PAD], [UNK], [BOS], and [EOS] serve specific purposes.",
        "Vocabulary size affects model performance and computational requirements.",
        "Preprocessing is crucial for training robust language models.",
        "Evaluation metrics help assess model quality and performance.",
        "Research in AI continues to advance rapidly with new discoveries.",
        "Applications of NLP include translation, summarization, and chatbots.",
        "The field of artificial intelligence has a rich history and bright future.",
        "Understanding data is essential for building effective machine learning systems."
    ] * 20  # Repeat to create more training data


def main():
    """Main example demonstrating BPE tokenizer usage."""
    print("BPE Tokenizer Example")
    print("=" * 50)
    
    # 1. Create training data
    print("\n1. Creating sample training data...")
    training_texts = create_sample_data()
    print(f"Created {len(training_texts)} training texts")
    
    # 2. Configure and create tokenizer
    print("\n2. Configuring BPE tokenizer...")
    config = BPETokenizerConfig(
        vocab_size=5000,       # Reasonable size for example
        min_frequency=3,       # Require tokens to appear at least 3 times
        unk_token="[UNK]",     # Unknown token
        pad_token="[PAD]",     # Padding token
        bos_token="[BOS]",     # Beginning of sequence
        eos_token="[EOS]",     # End of sequence
        mask_token="[MASK]"    # Mask token for masked language modeling
    )
    
    tokenizer = BPETokenizer(config)
    print(f"Tokenizer configured with vocab_size={config.vocab_size}")
    
    # 3. Train the tokenizer
    print("\n3. Training tokenizer...")
    tokenizer.train(training_texts, save_path="models/bpe_tokenizer")
    print(f"Training complete! Actual vocab size: {tokenizer.vocab_size}")
    
    # 4. Test the tokenizer
    print("\n4. Testing tokenizer functionality...")
    
    # Test encoding
    test_text = "This is a sample text for testing our BPE tokenizer implementation."
    print(f"\nOriginal text: '{test_text}'")
    
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    
    # Test decoding
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: '{decoded_text}'")
    
    # 5. Show special tokens
    print("\n5. Special token information:")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"UNK token ID: {tokenizer.unk_token_id}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"MASK token ID: {tokenizer.mask_token_id}")
    
    # 6. Test batch processing
    print("\n6. Testing batch processing...")
    batch_texts = [
        "Hello world!",
        "Machine learning is amazing.",
        "Natural language processing rocks!"
    ]
    
    # Encode batch
    batch_ids = tokenizer.encode_batch(batch_texts)
    print(f"Batch encoded to {len(batch_ids)} sequences")
    
    # Encode as tensor
    batch_tensor = tokenizer.encode_batch(batch_texts, return_tensors='pt')
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    # Decode batch
    batch_decoded = tokenizer.decode_batch(batch_ids)
    print("Batch decoded texts:")
    for i, text in enumerate(batch_decoded):
        print(f"  {i+1}: '{text}'")
    
    # 7. Show vocabulary statistics
    print("\n7. Vocabulary statistics:")
    vocab = tokenizer.get_vocab()
    print(f"Total vocabulary size: {len(vocab)}")
    
    # Show some example tokens
    print("Sample vocabulary tokens:")
    sample_tokens = list(vocab.keys())[:10]
    for token in sample_tokens:
        token_id = vocab[token]
        print(f"  '{token}' -> {token_id}")
    
    # 8. Test consistency
    print("\n8. Testing tokenization consistency...")
    test_texts = training_texts[:50]  # Use subset for testing
    consistency_results = tokenizer.test_consistency(test_texts)
    
    print("Consistency test results:")
    print(f"  - Total texts tested: {consistency_results['total_texts']}")
    print(f"  - Encoding errors: {consistency_results['encoding_errors']}")
    print(f"  - Roundtrip errors: {consistency_results['roundtrip_errors']}")
    print(f"  - Average tokens per text: {consistency_results['avg_tokens_per_text']:.2f}")
    print(f"  - Vocabulary coverage: {consistency_results['vocab_coverage']:.2%}")
    
    # 9. Demonstrate loading saved tokenizer
    print("\n9. Testing save/load functionality...")
    print("Loading tokenizer from saved files...")
    loaded_tokenizer = BPETokenizer.load("models/bpe_tokenizer")
    
    # Verify loaded tokenizer works identically
    test_sentence = "Testing loaded tokenizer consistency."
    original_ids = tokenizer.encode(test_sentence)
    loaded_ids = loaded_tokenizer.encode(test_sentence)
    
    if original_ids == loaded_ids:
        print("✓ Loaded tokenizer produces identical results!")
    else:
        print("✗ Loaded tokenizer results differ from original")
    
    print("\n" + "=" * 50)
    print("BPE Tokenizer example completed successfully!")
    print("The tokenizer is ready for use in transformer model training.")
    print("=" * 50)


if __name__ == "__main__":
    main() 