"""
Example usage of the Text Preprocessing Pipeline.

This script demonstrates various preprocessing configurations and use cases
for preparing text data for tokenization and model training.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.preprocessing import (
    TextPreprocessor,
    PreprocessingConfig,
    create_default_preprocessor,
    create_strict_preprocessor,
    create_minimal_preprocessor
)


def basic_preprocessing_example():
    """Demonstrate basic text preprocessing."""
    print("=== Basic Text Preprocessing Example ===")
    
    # Create a default preprocessor
    preprocessor = create_default_preprocessor()
    
    # Sample texts with various issues
    sample_texts = [
        "This is a <b>clean</b> text with https://example.com URL.",
        "Contact us at info@example.com for help!",
        "Call 123-456-7890 today!",
        "Short",  # Too short, will be filtered
        "THIS IS ALL CAPS TEXT THAT NEEDS NORMALIZATION",
        "Text   with    multiple     spaces\t\tand\ttabs.",
        "Text with émojis 🌟 and àccënts!",
        "",  # Empty, will be filtered
        "!@#$%^&*()",  # Only special chars, will be filtered
    ]
    
    print(f"Processing {len(sample_texts)} sample texts...")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. Original: {repr(text)}")
        processed = preprocessor.preprocess_text(text)
        if processed:
            print(f"   Processed: {repr(processed)}")
        else:
            print(f"   FILTERED OUT")
    
    # Show statistics
    stats = preprocessor.get_statistics()
    print(f"\nProcessing Statistics:")
    print(f"- Total processed: {stats['total_processed']}")
    print(f"- Total filtered: {stats['total_filtered']}")
    print(f"- Filter rate: {stats['filter_rate']:.2%}")
    print(f"- Characters removed: {stats['chars_removed']}")


def custom_configuration_example():
    """Demonstrate custom preprocessing configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create a custom configuration for social media text
    social_media_config = PreprocessingConfig(
        remove_html_tags=True,
        remove_urls=True,
        remove_emails=True,
        remove_phone_numbers=True,
        lowercase=True,
        normalize_unicode=True,
        normalize_whitespace=True,
        min_length=5,  # Shorter minimum for social media
        max_length=280,  # Twitter-like limit
        min_alpha_ratio=0.3,  # Lower threshold for emoji-heavy content
        max_digit_ratio=0.7,  # Allow more numbers
        custom_patterns=[
            r'@\w+',  # Remove @mentions
            r'#\w+',  # Remove hashtags
            r'RT\s+',  # Remove retweet indicators
        ],
        preserve_sentence_boundaries=False,  # Less formal text
    )
    
    preprocessor = TextPreprocessor(social_media_config)
    
    social_media_texts = [
        "RT @user: Check out this amazing product! #amazing #product https://example.com",
        "@friend1 @friend2 LOL 😂😂😂 this is hilarious!!!",
        "Just got my order #12345! So excited 🎉",
        "Contact support@company.com if you need help.",
        "😍😍😍",  # Only emojis, will be filtered due to low alpha ratio
    ]
    
    print("Processing social media texts with custom config...")
    
    for i, text in enumerate(social_media_texts, 1):
        print(f"\n{i}. Original: {repr(text)}")
        processed = preprocessor.preprocess_text(text)
        if processed:
            print(f"   Processed: {repr(processed)}")
        else:
            print(f"   FILTERED OUT")


def quality_analysis_example():
    """Demonstrate text quality analysis."""
    print("\n=== Text Quality Analysis Example ===")
    
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "This is a high-quality text with proper grammar and good content.",
        "12345 67890 numbers only text 98765",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # Repeated characters
        "Hello world! This has good balance of text and punctuation.",
        "short",  # Too short
        "Very long text " * 100,  # Too long
    ]
    
    print("Analyzing text quality...")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. Text: {repr(text[:50] + '...' if len(text) > 50 else text)}")
        
        is_valid, metrics = preprocessor.is_valid_text(text)
        
        print(f"   Length: {metrics.length}")
        print(f"   Alpha ratio: {metrics.alpha_ratio:.3f}")
        print(f"   Digit ratio: {metrics.digit_ratio:.3f}")
        print(f"   Repeated char ratio: {metrics.repeated_char_ratio:.3f}")
        print(f"   Unique words: {metrics.unique_words}")
        print(f"   Average word length: {metrics.avg_word_length:.2f}")
        print(f"   Sentence count: {metrics.sentence_count}")
        print(f"   Valid: {'✓' if is_valid else '✗'}")


def batch_processing_example():
    """Demonstrate batch text processing."""
    print("\n=== Batch Processing Example ===")
    
    preprocessor = create_default_preprocessor()
    
    # Simulate a batch of documents
    documents = [
        "This is the first document with good quality content.",
        "Second document contains <html>tags</html> and https://urls.com",
        "Document 3 has email@example.com and phone 123-456-7890",
        "Short doc",  # Will be filtered
        "Fourth document with proper length and good content quality.",
        "",  # Empty, will be filtered
        "Final document in the batch with sufficient content length.",
    ]
    
    print(f"Processing batch of {len(documents)} documents...")
    
    # Process all documents at once
    processed_docs = preprocessor.preprocess_batch(documents, validate=True)
    
    print(f"Successfully processed {len(processed_docs)} documents:")
    for i, doc in enumerate(processed_docs, 1):
        print(f"{i}. {doc[:60]}{'...' if len(doc) > 60 else ''}")
    
    # Show processing statistics
    stats = preprocessor.get_statistics()
    print(f"\nBatch Processing Statistics:")
    print(f"- Input documents: {len(documents)}")
    print(f"- Output documents: {len(processed_docs)}")
    print(f"- Filter rate: {stats['filter_rate']:.2%}")


def file_processing_example():
    """Demonstrate file-based text processing."""
    print("\n=== File Processing Example ===")
    
    # Create a sample input file
    input_file = Path("sample_input.txt")
    output_file = Path("sample_output.txt")
    
    # Sample content with various text quality issues
    sample_content = """This is a good quality line with proper content.
Short line.
Another good line that meets all quality criteria and length requirements.

This line has <html>tags</html> and https://example.com URLs that need cleaning.
Contact info@example.com for more details or call 555-123-4567.
THIS LINE IS ALL CAPS AND NEEDS NORMALIZATION FOR PROPER PROCESSING.

Low quality line with 12345 67890 too many numbers 98765 43210.
Normal line with good text quality and appropriate content length.
"""
    
    try:
        # Write sample content to input file
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        print(f"Created sample input file: {input_file}")
        print(f"Input file contains {len(sample_content.splitlines())} lines")
        
        # Create preprocessor and process file
        preprocessor = create_default_preprocessor()
        stats = preprocessor.preprocess_file(input_file, output_file)
        
        print(f"\nFile processing completed!")
        print(f"Output file: {output_file}")
        print(f"Processing statistics:")
        print(f"- Lines processed: {stats['total_processed']}")
        print(f"- Lines filtered: {stats['total_filtered']}")
        print(f"- Filter rate: {stats['filter_rate']:.2%}")
        print(f"- Characters removed: {stats['chars_removed']}")
        
        # Show sample of processed content
        print(f"\nSample processed content:")
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:5]  # Show first 5 lines
            for i, line in enumerate(lines, 1):
                print(f"{i}. {line.strip()}")
    
    finally:
        # Clean up temporary files
        for file_path in [input_file, output_file]:
            if file_path.exists():
                file_path.unlink()
                print(f"Cleaned up: {file_path}")


def configuration_save_load_example():
    """Demonstrate saving and loading preprocessing configurations."""
    print("\n=== Configuration Save/Load Example ===")
    
    # Create a custom configuration
    custom_config = PreprocessingConfig(
        remove_html_tags=True,
        remove_urls=False,  # Keep URLs for this use case
        lowercase=False,     # Preserve case
        min_length=20,
        max_length=1000,
        min_alpha_ratio=0.6,
        custom_patterns=[r'\d{3}-\d{3}-\d{4}'],  # Remove phone numbers
        preserve_paragraph_boundaries=True,
    )
    
    preprocessor = TextPreprocessor(custom_config)
    config_file = Path("custom_preprocessing_config.json")
    
    try:
        # Save configuration
        preprocessor.save_config(config_file)
        print(f"Saved configuration to: {config_file}")
        
        # Create new preprocessor and load configuration
        new_preprocessor = TextPreprocessor()
        new_preprocessor.load_config(config_file)
        
        print("Loaded configuration successfully!")
        
        # Test that configurations match
        sample_text = "Test text with URL: https://example.com and phone: 123-456-7890"
        
        original_result = preprocessor.preprocess_text(sample_text, validate=False)
        loaded_result = new_preprocessor.preprocess_text(sample_text, validate=False)
        
        print(f"Original config result: {repr(original_result)}")
        print(f"Loaded config result:   {repr(loaded_result)}")
        print(f"Results match: {'✓' if original_result == loaded_result else '✗'}")
    
    finally:
        # Clean up
        if config_file.exists():
            config_file.unlink()
            print(f"Cleaned up: {config_file}")


def preprocessor_comparison_example():
    """Compare different preprocessor configurations."""
    print("\n=== Preprocessor Comparison Example ===")
    
    # Create different preprocessors
    preprocessors = {
        "Default": create_default_preprocessor(),
        "Strict": create_strict_preprocessor(),
        "Minimal": create_minimal_preprocessor(),
    }
    
    test_text = "This is a <b>TEST</b> text with MIXED case, numbers 123, and symbols @#$! Visit https://example.com for more info."
    
    print(f"Original text: {repr(test_text)}")
    print()
    
    for name, preprocessor in preprocessors.items():
        print(f"{name} Preprocessor:")
        result = preprocessor.preprocess_text(test_text, validate=False)
        print(f"  Result: {repr(result)}")
        
        # Show key configuration differences
        config = preprocessor.config
        print(f"  Config: lowercase={config.lowercase}, "
              f"remove_special_chars={config.remove_special_chars}, "
              f"min_length={config.min_length}")
        print()


def main():
    """Run all preprocessing examples."""
    print("Text Preprocessing Pipeline Examples")
    print("=" * 50)
    
    examples = [
        basic_preprocessing_example,
        custom_configuration_example,
        quality_analysis_example,
        batch_processing_example,
        file_processing_example,
        configuration_save_load_example,
        preprocessor_comparison_example,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
        print()


if __name__ == "__main__":
    main() 