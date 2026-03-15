"""raschka_llm package - small utilities extracted while learning.
"""

__version__ = "0.0.0"

# Chapter 2: Tokenization and data loading
from .dataloader import TextDataset
from .tokenizer import SimpleTokenizer, build_vocab_from_text

# Chapter 3: Self-attention mechanisms
from .self_attention import (
    SelfAttention_v1,
    SelfAttention_v2,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
)

# Chapter 4: GPT model components
from .gpt_model import (
    GPT_CONFIG_124M,
    LayerNorm,
    GELU,
    FeedForward,
    TransformerBlock,
    DummyGPTModel,
)

__all__ = [
    # Chapter 2
    "TextDataset",
    "SimpleTokenizer",
    "build_vocab_from_text",
    # Chapter 3
    "SelfAttention_v1",
    "SelfAttention_v2",
    "CausalAttention",
    "MultiHeadAttentionWrapper",
    "MultiHeadAttention",
    # Chapter 4
    "GPT_CONFIG_124M",
    "LayerNorm",
    "GELU",
    "FeedForward",
    "TransformerBlock",
    "DummyGPTModel",
]
