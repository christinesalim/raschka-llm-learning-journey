# Chapter 2: Working with Text Data - Summary

This document summarizes the key concepts and implementations from Chapter 2 of "Build a Large Language Model from Scratch" by Sebastian Raschka.

## Overview

Chapter 2 covers the foundational data processing pipeline for training language models:
1. **Tokenization**: Converting text into numerical representations
2. **Data Loading**: Creating training batches with sliding windows
3. **Embeddings**: Transforming token IDs into dense vector representations
4. **Positional Encoding**: Adding position information to embeddings

---

## 1. Tokenization

### Simple Tokenizer ([tokenizer.py](src/raschka_llm/tokenizer.py))

**Purpose**: Convert text into token IDs that a model can process.

**Key Components**:

```python
# 1. Build vocabulary from text
vocab = build_vocab_from_text(text)
# Creates: {"word": 0, "hello": 1, ..., "<|endoftext|>": n, "<|unk|>": n+1}

# 2. Create tokenizer with vocabulary
tokenizer = SimpleTokenizer(vocab)

# 3. Encode text to IDs
ids = tokenizer.encode("Hello, world!")  # [0, 1, 2, 3]

# 4. Decode IDs back to text
text = tokenizer.decode(ids)  # "Hello, world!"
```

**How it works**:
- Uses regex to split text: `re.split(r'([,.:;?_!"()\']|--|\s)', text)`
- Unknown tokens replaced with `<|unk|>` special token
- Special token `<|endoftext|>` marks text boundaries

### BPE Tokenizer ([bpe_tokenizer.py](src/raschka_llm/bpe_tokenizer.py))

**Purpose**: Production-grade tokenization using OpenAI's tiktoken library.

**Key Difference from Simple Tokenizer**:
- **Simple**: Splits on whitespace/punctuation (educational)
- **BPE**: Breaks words into subword units (handles unknown words better)

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
text = tokenizer.decode(ids)
```

**Why BPE is better**:
- Handles out-of-vocabulary words by breaking them into smaller units
- Example: "someunknownPlace" → ["some", "unknown", "Place"] (subwords)

---

## 2. Data Loading

### GPTDatasetV1 ([dataloader.py](src/raschka_llm/dataloader.py))

**Purpose**: Create training samples using a sliding window approach.

**Key Concept**: Input-Target Pairs
- **Input**: A sequence of tokens
- **Target**: The same sequence shifted forward by 1 position

```python
# Example with max_length=4, stride=2
token_ids = [10, 20, 30, 40, 50, 60, 70, 80]

Sample 0 (i=0):
  Input:  [10, 20, 30, 40]
  Target: [20, 30, 40, 50]  # Shifted by 1

Sample 1 (i=2):  # stride moves 2 positions
  Input:  [30, 40, 50, 60]
  Target: [40, 50, 60, 70]

Sample 2 (i=4):
  Input:  [50, 60, 70, 80]
  Target: [60, 70, 80, 90]
```

**Parameters**:
- `max_length`: Sequence length (e.g., 4 tokens)
- `stride`: How many positions to move the window (e.g., 2 = overlap of 2 tokens)
- `batch_size`: Number of samples grouped together

### PyTorch Dataset

**Three Required Methods**:

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # Pre-compute all samples

    def __len__(self):
        # Return number of samples
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return one sample
        return self.input_ids[idx], self.target_ids[idx]
```

### DataLoader

**Purpose**: Batch samples and handle shuffling.

```python
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,      # 8 samples per batch
    max_length=4,      # 4 tokens per sequence
    stride=4,          # No overlap
    shuffle=False      # Sequential order
)

# Get a batch
inputs, targets = next(iter(dataloader))
# inputs.shape:  [8, 4]  # 8 samples, 4 tokens each
# targets.shape: [8, 4]  # Same shape, shifted by 1
```

---

## 3. Token Embeddings

**Purpose**: Convert token IDs to dense vector representations.

**Why?**
- Token IDs are discrete integers (0, 1, 2, ...)
- Neural networks need continuous vectors to learn patterns

```python
vocab_size = 50257    # GPT-2 vocabulary size
output_dim = 256      # Embedding dimension

# Create embedding layer
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Convert token IDs to embeddings
inputs = torch.tensor([[373, 530, 286, 262]])  # Shape: [1, 4]
token_embeddings = token_embedding_layer(inputs)
# token_embeddings.shape: [1, 4, 256]
# 1 sample, 4 tokens, 256-dim vector for each token
```

**How it works**:
- `torch.nn.Embedding` is a lookup table
- Each token ID maps to a learnable 256-dimensional vector
- These vectors are learned during training

**Example**:
```
Token ID 373 → [0.123, -0.456, 0.789, ..., 0.234]  # 256 numbers
Token ID 530 → [-0.345, 0.678, -0.123, ..., 0.567] # 256 numbers
Token ID 286 → [0.456, 0.234, -0.789, ..., -0.123] # 256 numbers
Token ID 262 → [-0.678, -0.234, 0.567, ..., 0.890] # 256 numbers
```

---

## 4. Positional Embeddings

**Purpose**: Add position information to token embeddings.

**Why?**
- Transformers process all tokens in parallel
- Without position info, these would look identical:
  - "The cat chased the mouse"
  - "The mouse chased the cat"

```python
context_length = 4    # Sequence length
output_dim = 256      # Same as token embeddings

# Create positional embedding layer
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# Get embeddings for positions [0, 1, 2, 3]
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# pos_embeddings.shape: [4, 256]
```

**How it works**:
- Creates learnable vectors for each position (0, 1, 2, 3)
- Position 0 gets one vector, position 1 gets another, etc.

### Combining Token + Position Embeddings

```python
# Token embeddings: What each token means
token_embeddings.shape  # [batch_size, seq_len, embed_dim]
                       # [8, 4, 256]

# Position embeddings: Where each token is located
pos_embeddings.shape   # [seq_len, embed_dim]
                      # [4, 256]

# Combined: Element-wise addition (broadcasting handles batch dimension)
input_embeddings = token_embeddings + pos_embeddings
# input_embeddings.shape: [8, 4, 256]
# Now the model knows both WHAT each token is AND WHERE it appears
```

**Visual Example**:
```
Sequence: ["The", "cat", "sat"]

Token Embeddings:
  "The" → [0.1, 0.2, 0.3, ...]  # What "The" means
  "cat" → [0.4, 0.5, 0.6, ...]  # What "cat" means
  "sat" → [0.7, 0.8, 0.9, ...]  # What "sat" means

Position Embeddings:
  pos_0 → [0.01, 0.02, 0.03, ...]  # Position 0 encoding
  pos_1 → [0.04, 0.05, 0.06, ...]  # Position 1 encoding
  pos_2 → [0.07, 0.08, 0.09, ...]  # Position 2 encoding

Final Input:
  "The" at pos_0 → [0.11, 0.22, 0.33, ...]  # Token + Position
  "cat" at pos_1 → [0.44, 0.55, 0.66, ...]
  "sat" at pos_2 → [0.77, 0.88, 0.99, ...]
```

---

## Complete Pipeline

Here's how all the pieces fit together:

```python
# 1. Load and tokenize text
with open("data/the-verdict.txt", "r") as f:
    raw_text = f.read()

# 2. Create dataloader with sliding window
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)

# 3. Get a batch of token IDs
inputs, targets = next(iter(dataloader))
# inputs: [8, 4] - 8 samples, 4 tokens each

# 4. Convert to token embeddings
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
# token_embeddings: [8, 4, 256]

# 5. Add positional embeddings
context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# pos_embeddings: [4, 256]

# 6. Combine them
input_embeddings = token_embeddings + pos_embeddings
# input_embeddings: [8, 4, 256]
# Ready for the attention mechanism!
```

---

## Key Concepts Summary

### 1. Next Token Prediction
- **Input**: `[10, 20, 30, 40]`
- **Target**: `[20, 30, 40, 50]`
- Model learns to predict the next token at each position

### 2. Teacher Forcing (Training)
- Model always sees ground truth tokens as input
- Even if it would have predicted wrong at earlier positions
- Makes training more stable

### 3. Autoregressive Generation (Inference)
- Model uses its own predictions as input for next step
- Start: `[10]` → predict `20`
- Then: `[10, 20]` → predict `30`
- Then: `[10, 20, 30]` → predict `40`
- Continue until done

### 4. Embeddings are Learned
- Both token and position embeddings start random
- Trained via backpropagation to capture meaningful patterns
- After training, similar tokens have similar embeddings

---

## Project Structure

```
raschka-build-llm-from-scratch-learning-journey/
├── src/raschka_llm/
│   ├── tokenizer.py          # SimpleTokenizer, build_vocab_from_text
│   ├── bpe_tokenizer.py      # BPE tokenization with tiktoken
│   ├── dataloader.py         # GPTDatasetV1, create_dataloader_v1
│   └── __init__.py           # Package exports
├── tests/
│   └── test_tokenizer.py     # Tests for tokenization
├── data/
│   ├── sample.txt            # Sample text file
│   └── the-verdict.txt       # Training data
└── requirements.txt          # tiktoken==0.7.0, torch, etc.
```

---

## Key Takeaways

1. **Tokenization** converts text → numbers
2. **Sliding window** creates overlapping training samples
3. **Token embeddings** give meaning to each token
4. **Position embeddings** encode where tokens appear
5. **Combined embeddings** = what + where information

These embeddings are now ready to be processed by the **attention mechanism** in Chapter 3!

---

## Next Steps: Chapter 3 - Attention Mechanism

The attention mechanism will:
- Process these input embeddings
- Allow tokens to "attend to" other tokens in the sequence
- Learn which tokens are important for predicting the next token
- Form the core building block of transformer models

The embeddings we created are the **input** to the attention layers.
