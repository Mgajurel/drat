"""
BPE Tokenizer Implementation for Differentiable Recomputation Gates Project

This module implements a Byte Pair Encoding (BPE) tokenizer with configurable
vocabulary size and special tokens for the transformer model training.
"""

from typing import List, Dict, Optional, Union, Tuple
import json
import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder
import torch


class BPETokenizerConfig:
    """Configuration class for BPE tokenizer parameters."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]", 
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        mask_token: str = "[MASK]",
        lowercase: bool = False,
        dropout: Optional[float] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.lowercase = lowercase
        self.dropout = dropout
        
        # Default special tokens
        if special_tokens is None:
            self.special_tokens = [
                self.pad_token,
                self.unk_token, 
                self.bos_token,
                self.eos_token,
                self.mask_token
            ]
        else:
            self.special_tokens = special_tokens
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'unk_token': self.unk_token,
            'pad_token': self.pad_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'mask_token': self.mask_token,
            'lowercase': self.lowercase,
            'dropout': self.dropout
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BPETokenizerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class BPETokenizer:
    """
    BPE Tokenizer with configurable vocabulary and special tokens.
    
    This tokenizer uses the HuggingFace tokenizers library to implement
    Byte Pair Encoding with proper handling of special tokens and 
    consistent encoding/decoding functionality.
    """
    
    def __init__(self, config: Optional[BPETokenizerConfig] = None):
        self.config = config or BPETokenizerConfig()
        self.tokenizer = None
        self._is_trained = False
        
        # Initialize tokenizer structure
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize the tokenizer with BPE model and configuration."""
        # Create BPE model
        self.tokenizer = Tokenizer(BPE(unk_token=self.config.unk_token))
        
        # Set pre-tokenizer (split on whitespace)
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Set decoder
        self.tokenizer.decoder = BPEDecoder()
        
        # Configure post-processing to add special tokens
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.config.bos_token} $A {self.config.eos_token}",
            special_tokens=[
                (self.config.bos_token, self.bos_token_id),
                (self.config.eos_token, self.eos_token_id),
            ],
        )
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.tokenizer and self._is_trained:
            return self.tokenizer.get_vocab_size()
        return self.config.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        if self.tokenizer and self._is_trained:
            return self.tokenizer.token_to_id(self.config.pad_token)
        return 0  # Default PAD token ID
    
    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        if self.tokenizer and self._is_trained:
            return self.tokenizer.token_to_id(self.config.unk_token)
        return 1  # Default UNK token ID
    
    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        if self.tokenizer and self._is_trained:
            return self.tokenizer.token_to_id(self.config.bos_token)
        return 2  # Default BOS token ID
    
    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        if self.tokenizer and self._is_trained:
            return self.tokenizer.token_to_id(self.config.eos_token)
        return 3  # Default EOS token ID
    
    @property
    def mask_token_id(self) -> int:
        """Get MASK token ID.""" 
        if self.tokenizer and self._is_trained:
            return self.tokenizer.token_to_id(self.config.mask_token)
        return 4  # Default MASK token ID
    
    def train(self, texts: List[str], save_path: Optional[str] = None) -> None:
        """
        Train the BPE tokenizer on provided texts.
        
        Args:
            texts: List of text strings to train on
            save_path: Optional path to save the trained tokenizer
        """
        # Create trainer
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=True
        )
        
        # Train the tokenizer
        print(f"Training BPE tokenizer with vocab_size={self.config.vocab_size}")
        self.tokenizer.train_from_iterator(texts, trainer)
        self._is_trained = True
        
        # Save if path provided
        if save_path:
            self.save(save_path)
        
        print(f"Training complete. Vocabulary size: {self.vocab_size}")
    
    def train_from_files(self, file_paths: List[str], save_path: Optional[str] = None) -> None:
        """
        Train the BPE tokenizer from text files.
        
        Args:
            file_paths: List of file paths containing training text
            save_path: Optional path to save the trained tokenizer
        """
        # Create trainer
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=True
        )
        
        # Train the tokenizer
        print(f"Training BPE tokenizer from {len(file_paths)} files")
        self.tokenizer.train(file_paths, trainer)
        self._is_trained = True
        
        # Save if path provided
        if save_path:
            self.save(save_path)
        
        print(f"Training complete. Vocabulary size: {self.vocab_size}")
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            return_tensors: Return format ('pt' for PyTorch tensor)
            
        Returns:
            List of token IDs or PyTorch tensor
        """
        if not self._is_trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        # Encode text
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        token_ids = encoding.ids
        
        # Return as tensor if requested
        if return_tensors == 'pt':
            return torch.tensor(token_ids, dtype=torch.long)
        
        return token_ids
    
    def encode_batch(
        self, 
        texts: List[str], 
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None
    ) -> Union[List[List[int]], torch.Tensor]:
        """
        Encode batch of texts to token IDs.
        
        Args:
            texts: List of input texts to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            return_tensors: Return format ('pt' for PyTorch tensor)
            
        Returns:
            List of token ID lists or PyTorch tensor
        """
        if not self._is_trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        # Encode texts
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        token_ids = [encoding.ids for encoding in encodings]
        
        # Return as tensor if requested (requires padding)
        if return_tensors == 'pt':
            # Pad sequences to same length
            max_length = max(len(ids) for ids in token_ids)
            padded_ids = []
            for ids in token_ids:
                padded = ids + [self.pad_token_id] * (max_length - len(ids))
                padded_ids.append(padded)
            return torch.tensor(padded_ids, dtype=torch.long)
        
        return token_ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        if not self._is_trained:
            raise ValueError("Tokenizer must be trained before decoding")
        
        # Convert tensor to list if needed
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        
        # Decode tokens
        text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return text
    
    def decode_batch(
        self, 
        batch_token_ids: Union[List[List[int]], torch.Tensor], 
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode batch of token IDs back to texts.
        
        Args:
            batch_token_ids: Batch of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            List of decoded text strings
        """
        if not self._is_trained:
            raise ValueError("Tokenizer must be trained before decoding")
        
        # Convert tensor to list if needed
        if torch.is_tensor(batch_token_ids):
            batch_token_ids = batch_token_ids.tolist()
        
        # Decode tokens
        texts = self.tokenizer.decode_batch(batch_token_ids, skip_special_tokens=skip_special_tokens)
        return texts
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping."""
        if not self._is_trained:
            raise ValueError("Tokenizer must be trained before accessing vocabulary")
        return self.tokenizer.get_vocab()
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token to ID."""
        if not self._is_trained:
            return None
        return self.tokenizer.token_to_id(token)
    
    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert ID to token."""
        if not self._is_trained:
            return None
        return self.tokenizer.id_to_token(token_id)
    
    def save(self, save_path: str) -> None:
        """Save the trained tokenizer and config."""
        if not self._is_trained:
            raise ValueError("Cannot save untrained tokenizer")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        tokenizer_path = save_path / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_path))
        
        # Save config
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"Tokenizer saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'BPETokenizer':
        """Load a trained tokenizer and config."""
        load_path = Path(load_path)
        
        # Load config
        config_path = load_path / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = BPETokenizerConfig.from_dict(config_dict)
        
        # Create tokenizer instance
        tokenizer_instance = cls(config)
        
        # Load trained tokenizer
        tokenizer_path = load_path / "tokenizer.json"
        tokenizer_instance.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer_instance._is_trained = True
        
        print(f"Tokenizer loaded from {load_path}")
        return tokenizer_instance
    
    def test_consistency(self, test_texts: List[str]) -> Dict[str, any]:
        """
        Test tokenization consistency and return metrics.
        
        Args:
            test_texts: List of texts to test
            
        Returns:
            Dictionary with test results and metrics
        """
        if not self._is_trained:
            raise ValueError("Tokenizer must be trained before testing")
        
        results = {
            'total_texts': len(test_texts),
            'encoding_errors': 0,
            'decoding_errors': 0,
            'roundtrip_errors': 0,
            'avg_tokens_per_text': 0,
            'vocab_coverage': 0
        }
        
        total_tokens = 0
        unique_tokens = set()
        
        for text in test_texts:
            try:
                # Test encoding
                token_ids = self.encode(text)
                total_tokens += len(token_ids)
                
                # Track unique tokens
                for token_id in token_ids:
                    token = self.id_to_token(token_id)
                    if token:
                        unique_tokens.add(token)
                
                # Test decoding
                decoded_text = self.decode(token_ids)
                
                # Test roundtrip consistency (allowing for whitespace normalization)
                if text.strip() != decoded_text.strip():
                    results['roundtrip_errors'] += 1
                    
            except Exception as e:
                results['encoding_errors'] += 1
                print(f"Error processing text: {str(e)[:100]}...")
        
        # Calculate metrics
        if len(test_texts) > 0:
            results['avg_tokens_per_text'] = total_tokens / len(test_texts)
            results['vocab_coverage'] = len(unique_tokens) / self.vocab_size
        
        return results 