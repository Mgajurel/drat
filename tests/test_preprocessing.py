"""
Tests for the text preprocessing pipeline.
"""

import pytest
import tempfile
from pathlib import Path
from src.data.preprocessing import (
    TextPreprocessor,
    PreprocessingConfig,
    TextQualityMetrics,
    create_default_preprocessor,
    create_strict_preprocessor,
    create_minimal_preprocessor
)


class TestPreprocessingConfig:
    """Test preprocessing configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PreprocessingConfig()
        assert config.remove_html_tags is True
        assert config.remove_urls is True
        assert config.lowercase is True
        assert config.min_length == 10
        assert config.max_length == 10000
        assert config.min_alpha_ratio == 0.5
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = PreprocessingConfig(min_length=20, max_length=5000)
        config_dict = config.to_dict()
        assert config_dict['min_length'] == 20
        assert config_dict['max_length'] == 5000
        assert 'remove_html_tags' in config_dict
    
    def test_config_from_dict(self):
        """Test configuration deserialization."""
        config_dict = {
            'min_length': 30,
            'max_length': 8000,
            'lowercase': False,
            'remove_urls': False,
            'custom_patterns': ['test_pattern']
        }
        config = PreprocessingConfig.from_dict(config_dict)
        assert config.min_length == 30
        assert config.max_length == 8000
        assert config.lowercase is False
        assert config.remove_urls is False
        assert config.custom_patterns == ['test_pattern']


class TestTextQualityMetrics:
    """Test text quality metrics."""
    
    def test_quality_metrics_creation(self):
        """Test quality metrics object creation."""
        metrics = TextQualityMetrics(
            length=100,
            alpha_ratio=0.8,
            digit_ratio=0.1,
            space_ratio=0.15,
            repeated_char_ratio=0.05,
            unique_chars=50,
            unique_words=20,
            avg_word_length=4.5,
            sentence_count=3,
            encoding_errors=0
        )
        assert metrics.length == 100
        assert metrics.alpha_ratio == 0.8
        assert metrics.unique_words == 20
    
    def test_is_valid_with_default_config(self):
        """Test text validity with default configuration."""
        config = PreprocessingConfig()
        
        # Valid text
        valid_metrics = TextQualityMetrics(
            length=50, alpha_ratio=0.7, digit_ratio=0.2, space_ratio=0.1,
            repeated_char_ratio=0.1, unique_chars=30, unique_words=10,
            avg_word_length=4.0, sentence_count=2, encoding_errors=0
        )
        assert valid_metrics.is_valid(config) is True
        
        # Invalid: too short
        short_metrics = TextQualityMetrics(
            length=5, alpha_ratio=0.8, digit_ratio=0.1, space_ratio=0.1,
            repeated_char_ratio=0.1, unique_chars=5, unique_words=1,
            avg_word_length=4.0, sentence_count=1, encoding_errors=0
        )
        assert short_metrics.is_valid(config) is False
        
        # Invalid: low alpha ratio
        low_alpha_metrics = TextQualityMetrics(
            length=50, alpha_ratio=0.3, digit_ratio=0.1, space_ratio=0.6,
            repeated_char_ratio=0.1, unique_chars=30, unique_words=10,
            avg_word_length=4.0, sentence_count=2, encoding_errors=0
        )
        assert low_alpha_metrics.is_valid(config) is False


class TestTextPreprocessor:
    """Test text preprocessing functionality."""
    
    def test_init_with_default_config(self):
        """Test preprocessor initialization with default config."""
        preprocessor = TextPreprocessor()
        assert preprocessor.config.remove_html_tags is True
        assert preprocessor.config.lowercase is True
        assert 'total_processed' in preprocessor.stats
    
    def test_init_with_custom_config(self):
        """Test preprocessor initialization with custom config."""
        config = PreprocessingConfig(lowercase=False, min_length=5)
        preprocessor = TextPreprocessor(config)
        assert preprocessor.config.lowercase is False
        assert preprocessor.config.min_length == 5
    
    def test_html_tag_removal(self):
        """Test HTML tag removal."""
        preprocessor = TextPreprocessor()
        text = "This is <b>bold</b> and <i>italic</i> text."
        cleaned = preprocessor.clean_text(text)
        assert "<b>" not in cleaned
        assert "<i>" not in cleaned
        assert "bold" in cleaned
        assert "italic" in cleaned
    
    def test_url_removal(self):
        """Test URL removal."""
        preprocessor = TextPreprocessor()
        text = "Visit https://example.com for more info."
        cleaned = preprocessor.clean_text(text)
        assert "https://example.com" not in cleaned
        assert "Visit" in cleaned
        assert "for more info" in cleaned
    
    def test_email_removal(self):
        """Test email address removal."""
        preprocessor = TextPreprocessor()
        text = "Contact us at info@example.com for help."
        cleaned = preprocessor.clean_text(text)
        assert "info@example.com" not in cleaned
        assert "Contact us at" in cleaned
        assert "for help" in cleaned
    
    def test_phone_removal(self):
        """Test phone number removal."""
        preprocessor = TextPreprocessor()
        text = "Call us at 1234567890 today."
        cleaned = preprocessor.clean_text(text)
        assert "1234567890" not in cleaned
        assert "Call us at" in cleaned
        assert "today" in cleaned
    
    def test_custom_pattern_removal(self):
        """Test custom pattern removal."""
        config = PreprocessingConfig(custom_patterns=[r'\d{4}-\d{4}'])
        preprocessor = TextPreprocessor(config)
        text = "Reference number: 1234-5678 for tracking."
        cleaned = preprocessor.clean_text(text)
        assert "1234-5678" not in cleaned
        assert "Reference number:" in cleaned
    
    def test_allowed_chars_filtering(self):
        """Test allowed characters filtering."""
        config = PreprocessingConfig(allowed_chars="abcdefghijklmnopqrstuvwxyz ")
        preprocessor = TextPreprocessor(config)
        text = "Hello123! World@#$"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "Hello World"
    
    def test_special_chars_removal(self):
        """Test special characters removal."""
        config = PreprocessingConfig(remove_special_chars=True)
        preprocessor = TextPreprocessor(config)
        text = "Hello @#$% World!"
        cleaned = preprocessor.clean_text(text)
        assert "@#$%" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned
    
    def test_lowercasing(self):
        """Test text lowercasing."""
        preprocessor = TextPreprocessor()
        text = "Hello WORLD Test"
        normalized = preprocessor.normalize_text(text)
        assert normalized == "hello world test"
    
    def test_no_lowercasing(self):
        """Test text without lowercasing."""
        config = PreprocessingConfig(lowercase=False)
        preprocessor = TextPreprocessor(config)
        text = "Hello WORLD Test"
        normalized = preprocessor.normalize_text(text)
        assert normalized == "Hello WORLD Test"
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        preprocessor = TextPreprocessor()
        text = "café naïve résumé"  # Text with accented characters
        normalized = preprocessor.normalize_text(text)
        # Should still contain accented characters after NFKC normalization
        assert "café" in normalized
    
    def test_accent_removal(self):
        """Test accent removal."""
        config = PreprocessingConfig(remove_accents=True)
        preprocessor = TextPreprocessor(config)
        text = "café naïve résumé"
        normalized = preprocessor.normalize_text(text)
        assert normalized == "cafe naive resume"
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        preprocessor = TextPreprocessor()
        text = "Hello    world\t\ttest\n\n\nend"
        normalized = preprocessor.normalize_text(text)
        # Should normalize multiple spaces/tabs but preserve paragraph breaks
        assert "    " not in normalized
        assert "\t\t" not in normalized
        assert normalized.count('\n') <= 2  # Should reduce multiple newlines
    
    def test_paragraph_boundary_preservation(self):
        """Test paragraph boundary preservation."""
        config = PreprocessingConfig(preserve_paragraph_boundaries=True)
        preprocessor = TextPreprocessor(config)
        text = "Paragraph 1.\n\n\nParagraph 2.\n\n\n\nParagraph 3."
        normalized = preprocessor.normalize_text(text)
        # Should preserve paragraph boundaries with double newlines
        assert "\n\n" in normalized
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        preprocessor = TextPreprocessor()
        text = "Hello world! This is a test sentence with 123 numbers."
        metrics = preprocessor.calculate_quality_metrics(text)
        
        assert metrics.length == len(text)
        assert 0 < metrics.alpha_ratio < 1
        assert 0 < metrics.digit_ratio < 1
        assert metrics.unique_chars > 0
        assert metrics.unique_words > 0
        assert metrics.avg_word_length > 0
        assert metrics.sentence_count >= 1
    
    def test_empty_text_metrics(self):
        """Test quality metrics for empty text."""
        preprocessor = TextPreprocessor()
        metrics = preprocessor.calculate_quality_metrics("")
        
        assert metrics.length == 0
        assert metrics.alpha_ratio == 0
        assert metrics.digit_ratio == 0
        assert metrics.unique_chars == 0
        assert metrics.unique_words == 0
    
    def test_text_validation(self):
        """Test text validation."""
        preprocessor = TextPreprocessor()
        
        # Valid text
        valid_text = "This is a good quality text with proper length and content."
        is_valid, metrics = preprocessor.is_valid_text(valid_text)
        assert is_valid is True
        assert metrics.length > 10
        
        # Invalid: too short
        short_text = "Short"
        is_valid, metrics = preprocessor.is_valid_text(short_text)
        assert is_valid is False
    
    def test_preprocess_text_valid(self):
        """Test full text preprocessing with valid text."""
        preprocessor = TextPreprocessor()
        text = "This is a <b>test</b> text with https://example.com URL."
        processed = preprocessor.preprocess_text(text)
        
        assert processed is not None
        assert "<b>" not in processed
        assert "https://example.com" not in processed
        assert "test" in processed
        assert processed.islower()
    
    def test_preprocess_text_invalid(self):
        """Test full text preprocessing with invalid text."""
        preprocessor = TextPreprocessor()
        # Text that's too short after cleaning
        text = "Hi!"
        processed = preprocessor.preprocess_text(text)
        
        assert processed is None
        assert preprocessor.stats['total_filtered'] > 0
    
    def test_preprocess_text_without_validation(self):
        """Test text preprocessing without validation."""
        preprocessor = TextPreprocessor()
        text = "Hi!"  # Would normally be filtered out
        processed = preprocessor.preprocess_text(text, validate=False)
        
        assert processed is not None
        assert processed == "hi!"
    
    def test_preprocess_batch(self):
        """Test batch text preprocessing."""
        preprocessor = TextPreprocessor()
        texts = [
            "This is a good text with proper length.",
            "Short",  # Will be filtered out
            "Another good text that meets the criteria.",
            "",  # Will be filtered out
        ]
        
        processed = preprocessor.preprocess_batch(texts)
        assert len(processed) == 2  # Only 2 valid texts
        assert all("good" in text for text in processed)
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        preprocessor = TextPreprocessor()
        texts = [
            "This is a good text with proper length.",
            "Short",  # Will be filtered out
            "Another good text that meets all criteria.",
        ]
        
        preprocessor.preprocess_batch(texts)
        stats = preprocessor.get_statistics()
        
        assert stats['total_processed'] == 3
        assert stats['total_filtered'] == 1
        assert 0 < stats['filter_rate'] < 1
    
    def test_statistics_reset(self):
        """Test statistics reset."""
        preprocessor = TextPreprocessor()
        preprocessor.preprocess_text("Some test text")
        
        assert preprocessor.stats['total_processed'] > 0
        
        preprocessor.reset_statistics()
        assert preprocessor.stats['total_processed'] == 0
        assert preprocessor.stats['total_filtered'] == 0
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        config = PreprocessingConfig(min_length=25, lowercase=False)
        preprocessor = TextPreprocessor(config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            preprocessor.save_config(config_path)
            
            # Create new preprocessor and load config
            new_preprocessor = TextPreprocessor()
            new_preprocessor.load_config(config_path)
            
            assert new_preprocessor.config.min_length == 25
            assert new_preprocessor.config.lowercase is False
            
        finally:
            Path(config_path).unlink()
    
    def test_file_preprocessing(self):
        """Test file preprocessing."""
        preprocessor = TextPreprocessor()
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            input_path = f.name
            f.write("This is a good line with proper content.\n")
            f.write("Short\n")  # Will be filtered out
            f.write("Another good line that meets all the quality criteria.\n")
            f.write("\n")  # Empty line, will be filtered out
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name
        
        try:
            # Process file
            stats = preprocessor.preprocess_file(input_path, output_path)
            
            # Check statistics
            assert stats['total_processed'] == 4
            assert stats['total_filtered'] == 2
            assert stats['filter_rate'] == 0.5
            
            # Check output file
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2  # Only valid lines
                assert all("good" in line for line in lines)
                
        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()


class TestPreprocessorFactories:
    """Test preprocessor factory functions."""
    
    def test_default_preprocessor(self):
        """Test default preprocessor creation."""
        preprocessor = create_default_preprocessor()
        assert preprocessor.config.remove_html_tags is True
        assert preprocessor.config.remove_urls is True
        assert preprocessor.config.lowercase is True
        assert preprocessor.config.min_length == 10
        assert preprocessor.config.min_alpha_ratio == 0.5
    
    def test_strict_preprocessor(self):
        """Test strict preprocessor creation."""
        preprocessor = create_strict_preprocessor()
        assert preprocessor.config.remove_special_chars is True
        assert preprocessor.config.remove_accents is True
        assert preprocessor.config.min_length == 50
        assert preprocessor.config.min_alpha_ratio == 0.7
        assert preprocessor.config.max_digit_ratio == 0.2
    
    def test_minimal_preprocessor(self):
        """Test minimal preprocessor creation."""
        preprocessor = create_minimal_preprocessor()
        assert preprocessor.config.remove_urls is False
        assert preprocessor.config.remove_emails is False
        assert preprocessor.config.lowercase is False
        assert preprocessor.config.min_length == 1
        assert preprocessor.config.min_alpha_ratio == 0.1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_none_input(self):
        """Test handling of None input."""
        preprocessor = TextPreprocessor()
        # This should handle None gracefully
        result = preprocessor.preprocess_text("")
        assert result is None
    
    def test_unicode_edge_cases(self):
        """Test Unicode edge cases."""
        preprocessor = TextPreprocessor()
        # Test with various Unicode characters
        text = "Hello 世界 🌍 café naïve"
        processed = preprocessor.preprocess_text(text, validate=False)
        assert processed is not None
        assert len(processed) > 0
    
    def test_very_long_text(self):
        """Test with very long text."""
        config = PreprocessingConfig(max_length=100)
        preprocessor = TextPreprocessor(config)
        text = "A" * 1000 + " very long text with lots of content"
        processed = preprocessor.preprocess_text(text)
        assert processed is None  # Should be filtered out for being too long
    
    def test_only_special_characters(self):
        """Test text with only special characters."""
        preprocessor = TextPreprocessor()
        text = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/"
        processed = preprocessor.preprocess_text(text)
        # Should be filtered out due to low alpha ratio
        assert processed is None
    
    def test_repeated_characters(self):
        """Test text with many repeated characters."""
        config = PreprocessingConfig(max_repeated_char_ratio=0.1)
        preprocessor = TextPreprocessor(config)
        text = "Thissssss issssss annnnnn exampleeeeee"
        processed = preprocessor.preprocess_text(text)
        # Should be filtered out due to high repeated character ratio
        assert processed is None


if __name__ == "__main__":
    pytest.main([__file__]) 