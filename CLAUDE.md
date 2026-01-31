# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a learning repository following Sebastian Raschka's "Build a Large Language Model from Scratch". The goal is to build runnable, well-tested components and create a production-style LLM library progressively through chapters.

**Key principle**: Code is written directly in src/, validated with tests, and documented in chapter summaries.

## Development Commands

### Package Installation
```bash
# Install package in editable mode (recommended for development)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_tokenizer.py

# Run a specific test function
pytest tests/test_tokenizer.py::test_simple_tokenize

# Run with verbose output
pytest -v

# Run with output showing print statements
pytest -s
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Architecture & Code Organization

### Module Structure
The codebase builds components progressively by chapter:

- **Chapter 2**: Tokenization and data loading
  - `tokenizer.py`: SimpleTokenizer (educational), build_vocab_from_text()
  - `bpe_tokenizer.py`: BPE tokenization with tiktoken (production)
  - `dataloader.py`: GPTDatasetV1, create_dataloader_v1() for sliding window batching

- **Chapter 3**: Self-attention mechanisms
  - `self_attention.py`: Contains multiple attention implementations
    - `SelfAttention_v1`: Basic with nn.Parameter (educational)
    - `SelfAttention_v2`: Production with nn.Linear
    - `CausalAttention`: Masked attention for autoregressive models
    - `MultiHeadAttentionWrapper`: Simple multi-head (educational)
    - `MultiHeadAttention`: Efficient multi-head (production)

- **Chapter 4**: Complete GPT model (in progress)
  - `gpt_model.py`: GPT architecture components
  - `text_generator.py`: Text generation utilities

### Version Naming Convention
Classes often have `_v1`, `_v2` suffixes:
- `v1`: Educational implementation showing the concept clearly
- `v2` or no suffix: Production-ready implementation with best practices
- Both versions are kept to show progression and aid learning

### Public API (\_\_init\_\_.py)
The `src/raschka_llm/__init__.py` file defines the public API. When adding new modules:
1. Import the new classes/functions
2. Add them to `__all__` list
3. This enables clean imports: `from raschka_llm import GPTModel`

### Data Pipeline
```
Text File → Tokenizer → Token IDs → DataLoader (sliding window) →
Embeddings (token + position) → Attention Layers → Model
```

Key insight: The dataloader creates (input, target) pairs where target is input shifted by 1 position, enabling next-token prediction training.

### Attention Architecture Patterns
All attention modules follow this pattern:
1. Project input to Q, K, V using nn.Linear
2. Reshape/transpose for multi-head processing (if applicable)
3. Compute attention scores: queries @ keys.T
4. Apply causal mask (if autoregressive)
5. Scale by sqrt(d_out), apply softmax
6. Apply dropout (during training)
7. Compute context: attention_weights @ values
8. Merge heads and project output (if multi-head)

The `transpose(1, 2)` operation in MultiHeadAttention reorganizes from `[batch, tokens, heads, features]` to `[batch, heads, tokens, features]` to enable parallel processing of all heads.

## Working with Chapters

### Adding Code from Raschka's Source
When working through a chapter (referencing code from LLMs-from-scratch repo):

1. **Write directly in src/**: Implement components in the appropriate module
2. **Write tests**: Create corresponding test file to validate
3. **Update \_\_init\_\_.py**: Export new components
4. **Document**: Update or create CHAPTER_*_SUMMARY.md after understanding the concepts

### Chapter Summary Files
Each completed chapter has a `CHAPTER_N_SUMMARY.md` at the project root containing:
- Conceptual explanations of what was learned
- Code examples with detailed comments
- Visual diagrams and step-by-step breakdowns
- References to implementation files with line numbers (e.g., `self_attention.py:508`)
- Project structure snapshot

These summaries are teaching documents, not API references. They explain the "why" behind implementations.

### Interactive Concept Demos
The `notes/concepts/` directory contains standalone demos for complex topics:
- Each concept has a `.md` explanation and `.py` demo script
- Current examples: `transpose_demo.py`, `transpose_explanation.md`
- Pattern: Create these when a concept needs deeper exploration than inline comments provide

## Testing Patterns

### Test File Organization
- Test files mirror src/ structure: `test_tokenizer.py` tests `tokenizer.py`
- Path handling: Tests add parent/src to sys.path to import modules
- File fixtures: Tests expect `data/sample.txt` and may download `data/the-verdict.txt`

### Test Naming
- `test_<component>`: Test a specific component or function
- `test_<component>_<scenario>`: Test a specific scenario
- Example: `test_tokenizer_encode_decode`, `test_build_vocab_from_file`

## Important Implementation Details

### Embeddings
Token embeddings (`nn.Embedding`) are lookup tables that convert token IDs to dense vectors. Position embeddings add location information. Both are learned during training and combined via element-wise addition.

### Causal Masking
Autoregressive models use causal masking to prevent tokens from attending to future positions:
- `torch.triu(..., diagonal=1)` creates upper triangular mask
- Positions to mask are filled with `-torch.inf`
- Softmax converts -inf to 0 probability
- Mask is registered as a buffer (not a parameter) via `register_buffer()`

### Dropout in Attention
Dropout is applied to attention weights (not inputs/outputs) for regularization. It's only active during training (automatically disabled in eval mode).

### Tensor Shapes Reference
Common shapes throughout the codebase:
- Embeddings: `[batch, seq_len, d_model]`
- Attention scores: `[batch, (heads), seq_len, seq_len]`
- Multi-head before transpose: `[batch, seq_len, num_heads, head_dim]`
- Multi-head after transpose: `[batch, num_heads, seq_len, head_dim]`

## Data Files
- `data/sample.txt`: Small test file (committed)
- `data/the-verdict.txt`: Training data (downloaded by tests if missing)
- Large data files (*.pt, *.bin) are gitignored

## Pre-commit Hooks
The repository uses pre-commit hooks for code quality:
- black: Code formatting
- isort: Import sorting
- flake8: Linting

Hooks run automatically on commit. To bypass (not recommended): `git commit --no-verify`
