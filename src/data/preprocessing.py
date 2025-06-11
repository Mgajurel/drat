"""
Text Preprocessing Pipeline for corpus data preparation.

This module provides comprehensive text preprocessing capabilities including:
- Text cleaning and normalization
- Character filtering and encoding handling
- Quality validation and metrics
- Configurable preprocessing steps
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import Counter
import json

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing pipeline."""
    
    # Text cleaning options
    remove_html_tags: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_special_chars: bool = False
    allowed_chars: Optional[str] = None  # If set, only keep these characters
    
    # Normalization options
    lowercase: bool = True
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    remove_accents: bool = False
    
    # Length filtering
    min_length: int = 10
    max_length: int = 10000
    
    # Language filtering
    remove_non_printable: bool = True
    keep_ascii_only: bool = False
    
    # Quality filtering
    min_alpha_ratio: float = 0.5  # Minimum ratio of alphabetic characters
    max_digit_ratio: float = 0.5  # Maximum ratio of digit characters
    max_repeated_char_ratio: float = 0.3  # Maximum ratio of repeated characters
    
    # Custom regex patterns to remove
    custom_patterns: List[str] = field(default_factory=list)
    
    # Preserve options
    preserve_sentence_boundaries: bool = True
    preserve_paragraph_boundaries: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'remove_html_tags': self.remove_html_tags,
            'remove_urls': self.remove_urls,
            'remove_emails': self.remove_emails,
            'remove_phone_numbers': self.remove_phone_numbers,
            'remove_special_chars': self.remove_special_chars,
            'allowed_chars': self.allowed_chars,
            'lowercase': self.lowercase,
            'normalize_unicode': self.normalize_unicode,
            'normalize_whitespace': self.normalize_whitespace,
            'remove_accents': self.remove_accents,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'remove_non_printable': self.remove_non_printable,
            'keep_ascii_only': self.keep_ascii_only,
            'min_alpha_ratio': self.min_alpha_ratio,
            'max_digit_ratio': self.max_digit_ratio,
            'max_repeated_char_ratio': self.max_repeated_char_ratio,
            'custom_patterns': self.custom_patterns,
            'preserve_sentence_boundaries': self.preserve_sentence_boundaries,
            'preserve_paragraph_boundaries': self.preserve_paragraph_boundaries,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PreprocessingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class TextQualityMetrics:
    """Metrics for evaluating text quality."""
    
    length: int
    alpha_ratio: float
    digit_ratio: float
    space_ratio: float
    repeated_char_ratio: float
    unique_chars: int
    unique_words: int
    avg_word_length: float
    sentence_count: int
    encoding_errors: int
    
    def is_valid(self, config: PreprocessingConfig) -> bool:
        """Check if text meets quality criteria."""
        return (
            config.min_length <= self.length <= config.max_length and
            self.alpha_ratio >= config.min_alpha_ratio and
            self.digit_ratio <= config.max_digit_ratio and
            self.repeated_char_ratio <= config.max_repeated_char_ratio
        )


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline.
    
    Provides configurable text cleaning, normalization, and quality filtering
    for preparing raw corpus data for tokenization.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration. Uses default if None.
        """
        self.config = config or PreprocessingConfig()
        self._compile_patterns()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'total_filtered': 0,
            'chars_removed': 0,
            'quality_filtered': 0,
            'length_filtered': 0,
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient processing."""
        # HTML tags
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # URLs
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email addresses
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Phone numbers (basic pattern)
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[0-9]{7,15}')
        
        # Special characters (keeping basic punctuation)
        self.special_chars_pattern = re.compile(r'[^\w\s\.\,\!\?\;\:\-\'\"]')
        
        # Whitespace normalization
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Non-printable characters
        self.non_printable_pattern = re.compile(r'[^\x20-\x7E\n\t]')
        
        # Repeated characters (3+ in a row)
        self.repeated_chars_pattern = re.compile(r'(.)\1{2,}')
        
        # Custom patterns
        self.custom_patterns = [re.compile(pattern) for pattern in self.config.custom_patterns]
    
    def clean_text(self, text: str) -> str:
        """
        Clean text according to configuration.
        
        Args:
            text: Input text to clean.
            
        Returns:
            Cleaned text.
        """
        original_length = len(text)
        
        # Remove HTML tags
        if self.config.remove_html_tags:
            text = self.html_pattern.sub(' ', text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        if self.config.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        # Remove phone numbers
        if self.config.remove_phone_numbers:
            text = self.phone_pattern.sub(' ', text)
        
        # Apply custom patterns
        for pattern in self.custom_patterns:
            text = pattern.sub(' ', text)
        
        # Handle character filtering
        if self.config.allowed_chars:
            # Keep only allowed characters
            allowed_set = set(self.config.allowed_chars)
            text = ''.join(c for c in text if c in allowed_set)
        elif self.config.remove_special_chars:
            # Remove special characters
            text = self.special_chars_pattern.sub(' ', text)
        
        # Remove non-printable characters
        if self.config.remove_non_printable:
            if self.config.keep_ascii_only:
                text = ''.join(c for c in text if ord(c) < 128)
            else:
                text = self.non_printable_pattern.sub('', text)
        
        # Track characters removed
        self.stats['chars_removed'] += original_length - len(text)
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text according to configuration.
        
        Args:
            text: Input text to normalize.
            
        Returns:
            Normalized text.
        """
        # Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove accents
        if self.config.remove_accents:
            text = ''.join(
                c for c in unicodedata.normalize('NFD', text)
                if unicodedata.category(c) != 'Mn'
            )
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            # Preserve paragraph boundaries if configured
            if self.config.preserve_paragraph_boundaries:
                # Replace multiple newlines with double newline
                text = re.sub(r'\n\s*\n', '\n\n', text)
                # Normalize other whitespace
                text = re.sub(r'[ \t]+', ' ', text)
                text = re.sub(r'\n[ \t]*', '\n', text)
            else:
                # Normalize all whitespace to single spaces
                text = self.whitespace_pattern.sub(' ', text)
        
        # Handle repeated characters
        if hasattr(self.config, 'normalize_repeated_chars') and self.config.normalize_repeated_chars:
            text = self.repeated_chars_pattern.sub(r'\1\1', text)
        
        return text.strip()
    
    def calculate_quality_metrics(self, text: str) -> TextQualityMetrics:
        """
        Calculate quality metrics for text.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Quality metrics object.
        """
        if not text:
            return TextQualityMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        length = len(text)
        
        # Character type ratios
        alpha_chars = sum(1 for c in text if c.isalpha())
        digit_chars = sum(1 for c in text if c.isdigit())
        space_chars = sum(1 for c in text if c.isspace())
        
        alpha_ratio = alpha_chars / length if length > 0 else 0
        digit_ratio = digit_chars / length if length > 0 else 0
        space_ratio = space_chars / length if length > 0 else 0
        
        # Repeated character ratio
        repeated_matches = self.repeated_chars_pattern.findall(text)
        repeated_chars = sum(len(match) + 2 for match in repeated_matches)  # +2 for the original char
        repeated_char_ratio = repeated_chars / length if length > 0 else 0
        
        # Unique characters and words
        unique_chars = len(set(text))
        words = text.split()
        unique_words = len(set(words))
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Sentence count (rough estimate)
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        # Encoding errors (try to detect)
        encoding_errors = text.count('�') + text.count('\ufffd')
        
        return TextQualityMetrics(
            length=length,
            alpha_ratio=alpha_ratio,
            digit_ratio=digit_ratio,
            space_ratio=space_ratio,
            repeated_char_ratio=repeated_char_ratio,
            unique_chars=unique_chars,
            unique_words=unique_words,
            avg_word_length=avg_word_length,
            sentence_count=sentence_count,
            encoding_errors=encoding_errors
        )
    
    def is_valid_text(self, text: str) -> Tuple[bool, TextQualityMetrics]:
        """
        Check if text meets quality criteria.
        
        Args:
            text: Text to validate.
            
        Returns:
            Tuple of (is_valid, quality_metrics).
        """
        metrics = self.calculate_quality_metrics(text)
        is_valid = metrics.is_valid(self.config)
        return is_valid, metrics
    
    def preprocess_text(self, text: str, validate: bool = True) -> Optional[str]:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Input text to preprocess.
            validate: Whether to apply quality validation.
            
        Returns:
            Preprocessed text or None if filtered out.
        """
        self.stats['total_processed'] += 1
        
        if not text or not text.strip():
            self.stats['total_filtered'] += 1
            return None
        
        # Clean text
        text = self.clean_text(text)
        
        # Normalize text
        text = self.normalize_text(text)
        
        # Validate if requested
        if validate:
            is_valid, metrics = self.is_valid_text(text)
            if not is_valid:
                if metrics.length < self.config.min_length or metrics.length > self.config.max_length:
                    self.stats['length_filtered'] += 1
                else:
                    self.stats['quality_filtered'] += 1
                self.stats['total_filtered'] += 1
                return None
        
        return text
    
    def preprocess_batch(self, texts: List[str], validate: bool = True) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts.
            validate: Whether to apply quality validation.
            
        Returns:
            List of preprocessed texts (filtered texts are excluded).
        """
        results = []
        for text in texts:
            processed = self.preprocess_text(text, validate=validate)
            if processed is not None:
                results.append(processed)
        
        return results
    
    def preprocess_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        encoding: str = 'utf-8',
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Preprocess a text file.
        
        Args:
            input_path: Path to input file.
            output_path: Path to output file.
            encoding: File encoding.
            validate: Whether to apply quality validation.
            
        Returns:
            Processing statistics.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Reset stats
        self.stats = {
            'total_processed': 0,
            'total_filtered': 0,
            'chars_removed': 0,
            'quality_filtered': 0,
            'length_filtered': 0,
        }
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'r', encoding=encoding) as infile, \
             open(output_path, 'w', encoding=encoding) as outfile:
            
            for line in infile:
                processed = self.preprocess_text(line.strip(), validate=validate)
                if processed is not None:
                    outfile.write(processed + '\n')
        
        # Add final statistics
        self.stats['input_file'] = str(input_path)
        self.stats['output_file'] = str(output_path)
        self.stats['filter_rate'] = (
            self.stats['total_filtered'] / self.stats['total_processed']
            if self.stats['total_processed'] > 0 else 0
        )
        
        return self.stats.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.stats.copy()
        if stats['total_processed'] > 0:
            stats['filter_rate'] = stats['total_filtered'] / stats['total_processed']
        else:
            stats['filter_rate'] = 0
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            'total_processed': 0,
            'total_filtered': 0,
            'chars_removed': 0,
            'quality_filtered': 0,
            'length_filtered': 0,
        }
    
    def save_config(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def load_config(self, path: Union[str, Path]):
        """Load configuration from file."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        self.config = PreprocessingConfig.from_dict(config_dict)
        self._compile_patterns()


def create_default_preprocessor() -> TextPreprocessor:
    """Create a preprocessor with sensible defaults for general text."""
    config = PreprocessingConfig(
        remove_html_tags=True,
        remove_urls=True,
        remove_emails=True,
        remove_phone_numbers=True,
        lowercase=True,
        normalize_unicode=True,
        normalize_whitespace=True,
        min_length=10,
        max_length=10000,
        min_alpha_ratio=0.5,
        max_digit_ratio=0.5,
        preserve_sentence_boundaries=True,
        preserve_paragraph_boundaries=True
    )
    return TextPreprocessor(config)


def create_strict_preprocessor() -> TextPreprocessor:
    """Create a preprocessor with strict filtering for high-quality text."""
    config = PreprocessingConfig(
        remove_html_tags=True,
        remove_urls=True,
        remove_emails=True,
        remove_phone_numbers=True,
        remove_special_chars=True,
        lowercase=True,
        normalize_unicode=True,
        normalize_whitespace=True,
        remove_accents=True,
        min_length=50,
        max_length=5000,
        min_alpha_ratio=0.7,
        max_digit_ratio=0.2,
        max_repeated_char_ratio=0.1,
        preserve_sentence_boundaries=True,
        preserve_paragraph_boundaries=True
    )
    return TextPreprocessor(config)


def create_minimal_preprocessor() -> TextPreprocessor:
    """Create a preprocessor with minimal filtering."""
    config = PreprocessingConfig(
        remove_html_tags=True,
        remove_urls=False,
        remove_emails=False,
        remove_phone_numbers=False,
        lowercase=False,
        normalize_unicode=True,
        normalize_whitespace=True,
        min_length=1,
        max_length=50000,
        min_alpha_ratio=0.1,
        max_digit_ratio=0.9,
        preserve_sentence_boundaries=True,
        preserve_paragraph_boundaries=True
    )
    return TextPreprocessor(config) 