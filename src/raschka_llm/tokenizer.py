"""Tokenizer classes and utilities for text processing."""
from pathlib import Path
from typing import Dict, List
from collections import Counter
import re


def _tokenize_text(text: str) -> List[str]:
    """Internal helper to tokenize text using regex splitting.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed


class SimpleTokenizer:
    """Simple whitespace-based tokenizer for learning.
    
    Builds a vocabulary from text and provides encode/decode methods.
    """
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
        
    def encode(self, text):
        """Convert text into token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        # Step 1: Tokenize the text into words and punctuation
        preprocessed = _tokenize_text(text)

        # Step 2: Replace unknown tokens with the special <|unk|> token
        # This prevents KeyError for words not in the vocabulary
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]

        # Step 3: Convert each token string to its corresponding ID using the vocab passed in constructor
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        """Convert token IDs back into text.

        Args:
            ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        # Step 1: Convert each ID back to its token string
        # and join them with spaces
        text = " ".join([self.int_to_str[i] for i in ids])

        # Step 2: Remove spaces before punctuation for natural formatting
        # e.g., "Hello , world !" becomes "Hello, world!"
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


def build_vocab_from_text(text: str) -> Dict[str, int]:
    """Build vocabulary from text using the same tokenization as SimpleTokenizer.

    Args:
        text: Input text to build vocabulary from

    Returns:
        Dictionary mapping tokens to IDs
    """
    # Tokenize using the same logic as SimpleTokenizer.encode
    preprocessed = _tokenize_text(text)

    # Create vocabulary from unique sorted tokens
    all_tokens = sorted(set(preprocessed))

    # Add special tokens
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    # Create vocab dictionary
    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    return vocab