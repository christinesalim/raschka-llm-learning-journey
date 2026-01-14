# Chapter 3: Self-Attention Mechanism - Summary

This document summarizes the key concepts and implementations from Chapter 3 of "Build a Large Language Model from Scratch" by Sebastian Raschka.

## Overview

Chapter 3 introduces the **self-attention mechanism** - the core building block that makes transformer models like GPT work. Self-attention allows each token in a sequence to "attend to" (look at) all other tokens and determine which ones are most relevant for understanding it.

**Connection to Chapter 2**: In Chapter 2, we created token embeddings that capture _what_ each token means and _where_ it appears. In Chapter 3, these embeddings become the input to self-attention, which lets tokens communicate and build context-aware representations.

---

## 1. The Self-Attention Concept

### What Problem Does It Solve?

Embeddings from Chapter 2 are **static** - the word "bank" has the same embedding whether it means "river bank" or "financial bank". Self-attention makes embeddings **dynamic** by letting context influence meaning.

### How It Works (Intuition)

For each token, self-attention answers: **"Which other tokens should I pay attention to?"**

**Example**: "The cat sat on the mat"

For the word **"sat"**:

1. **Query**: "I'm a verb, what should I pay attention to?"
2. **Keys from other words**:
   - "cat" key: "I'm the subject" → High attention (0.6)
   - "mat" key: "I'm an object" → Medium attention (0.3)
   - "the" key: "I'm just an article" → Low attention (0.1)
3. **Values**: Actual content from each word
4. **Output**: Weighted combination emphasizing "cat" and "mat"

The output for "sat" now contains information from the entire context, making it context-aware.

---

## 2. Self-Attention Architecture

### Implementation: SelfAttention_v1 ([self_attention.py](src/raschka_llm/self_attention.py))

**Key Components**: Three learnable weight matrices

```python
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        # Three learnable transformation matrices
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
```

**Parameters**:

- `d_in`: Input embedding dimension (e.g., 256 from Chapter 2)
- `d_out`: Output dimension for queries, keys, and values (e.g., 64)

**Why Three Matrices?**

- **W_query**: Transforms input into "what I'm looking for"
- **W_key**: Transforms input into "what I offer"
- **W_value**: Transforms input into "actual content to return"

### nn.Parameter Explained

```python
self.W_query = nn.Parameter(torch.rand(d_in, d_out))
```

**What this does**:

1. `torch.rand(d_in, d_out)` - Creates random matrix (e.g., 256×64)
2. `nn.Parameter(...)` - Tells PyTorch "this is learnable"
   - Tracks gradients during backpropagation
   - Gets updated by the optimizer during training
   - Automatically registered in `model.parameters()`

**Without `nn.Parameter`**: PyTorch would treat it as a regular tensor and wouldn't update it during training.

---

## 3. The Forward Pass

### Step-by-Step Execution

```python
def forward(self, x):
    # x shape: [batch, seq_len, d_in]
    # Example: [8, 4, 256]

    # Step 1: Create Q, K, V matrices
    keys    = x @ self.W_key      # [8, 4, 64]
    queries = x @ self.W_query    # [8, 4, 64]
    values  = x @ self.W_value    # [8, 4, 64]

    # Step 2: Calculate attention scores
    attn_scores = queries @ keys.T  # [8, 4, 4]

    # Step 3: Scale and normalize
    attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5,  # Scale by sqrt(d_out)
        dim=-1
    )

    # Step 4: Compute context vectors
    context_vec = attn_weights @ values  # [8, 4, 64]
    return context_vec
```

### Detailed Breakdown

#### Step 1: Create Q, K, V Matrices

```python
keys    = x @ self.W_key     # [batch, seq_len, d_in] @ [d_in, d_out]
                             # = [batch, seq_len, d_out]
```

**Matrix multiplication example**:

```
x:                W_key:           keys:
[8, 4, 256]    @  [256, 64]    =   [8, 4, 64]
↑   ↑   ↑         ↑     ↑          ↑  ↑  ↑
│   │   └─ d_in   │     └─ d_out   │  │  └─ Each token's key vector
│   └─ seq_len    └─ d_in          │  └─ 4 tokens
└─ batch                           └─ 8 samples
```

**What it means**:

- Each of the 4 tokens gets transformed into a 64-dimensional key vector
- This transformation is learned during training
- After training, W_key knows how to extract "what I offer" from each token

#### Step 2: Calculate Attention Scores

```python
attn_scores = queries @ keys.T  # [8, 4, 64] @ [8, 64, 4] = [8, 4, 4]
```

**Visual example** (for 4 tokens):

```
        Token 0  Token 1  Token 2  Token 3
Token 0   0.8      0.3      0.1      0.2     ← How much token 0 attends to each token
Token 1   0.2      0.9      0.4      0.1     ← How much token 1 attends to each token
Token 2   0.1      0.3      0.8      0.5     ← How much token 2 attends to each token
Token 3   0.3      0.1      0.2      0.9     ← How much token 3 attends to each token
```

**What it computes**: Compatibility between what token i wants (query) and what token j offers (key).

- Higher score = more relevant
- Diagonal tends to be high (tokens attend to themselves)

#### Step 3: Scale and Normalize

```python
attn_weights = torch.softmax(
    attn_scores / keys.shape[-1]**0.5,  # Divide by sqrt(64) = 8
    dim=-1
)
```

This step has **two critical operations**: scaling and softmax normalization.

##### Part A: Why Scale by √d_out?

**The Problem**: Dot products grow with dimensionality

When computing `queries @ keys.T`, you're doing dot products:
```python
query = [0.1, 0.2, 0.3, ..., 0.5]  # 64 numbers
key   = [0.4, 0.1, 0.6, ..., 0.2]  # 64 numbers

score = (0.1 × 0.4) + (0.2 × 0.1) + ... + (0.5 × 0.2)  # 64 terms summed
```

With 64 terms, sums can get very large (e.g., -10 to +15).

**Without Scaling Example**:
```python
# Scores before scaling
attn_scores = [[ 8.2, -3.1,  5.7, -2.5],
               [-3.1, 10.9, -1.2,  7.3],
               [ 5.7, -1.2,  9.5,  3.9],
               [-2.5,  7.3,  3.9, 11.2]]

# Apply softmax to large scores
softmax(attn_scores[0]) = [0.9999, 0.0000, 0.0001, 0.0000]
```

**Problem**: Almost all weight goes to the highest score!
- Token 0 gets 99.99% attention
- Other tokens get ~0%
- Gradients vanish → can't learn

**With Scaling**:
```python
# Divide by sqrt(64) = 8
scaled_scores = attn_scores / 8
              = [[ 1.03, -0.39,  0.71, -0.31],
                 [-0.39,  1.36, -0.15,  0.92],
                 [ 0.71, -0.15,  1.18,  0.49],
                 [-0.31,  0.92,  0.49,  1.40]]

# Apply softmax to scaled scores
softmax(scaled_scores[0]) = [0.39, 0.09, 0.28, 0.10]
```

**Better!** Attention is distributed across multiple tokens:
- Token 0: 39% attention
- Token 2: 28% attention
- Token 3: 10% attention
- Token 1: 9% attention

**Key Insight**: Scaling by √d_out keeps scores in a reasonable range, allowing the model to attend to multiple relevant tokens instead of fixating on one.

##### Part B: Softmax Explained

**What Softmax Does**: Converts any numbers into probabilities that sum to 1.0

**Formula**: `softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)`

**Step-by-Step Example**:
```python
scores = [1.0, 0.5, 0.2, 0.8]

# Step 1: Exponentiate each score
exp_scores = [e^1.0, e^0.5, e^0.2, e^0.8]
           = [2.718, 1.649, 1.221, 2.226]

# Step 2: Sum all exponentials
sum_exp = 2.718 + 1.649 + 1.221 + 2.226 = 7.814

# Step 3: Divide each by the sum
weights = [2.718/7.814, 1.649/7.814, 1.221/7.814, 2.226/7.814]
        = [0.348,      0.211,      0.156,      0.285]

# Verify: 0.348 + 0.211 + 0.156 + 0.285 = 1.000 ✓
```

**Why Exponentiate?**

1. **Always positive**: `e^x > 0` even when x is negative
   - Negative scores become small positive probabilities

2. **Emphasizes differences**: Higher scores get proportionally more weight
   - Score difference of 1.0 → ~2.7x more weight after softmax

3. **Smooth distribution**: Unlike "max" (winner-takes-all), softmax gives some weight to all tokens

**Complete Example with Real Numbers**:
```python
import torch

# Small example for clarity
d_out = 4
queries = torch.tensor([[1.0, 0.5, 0.2, 0.8]])  # 1 token
keys = torch.tensor([
    [0.5, 1.0, 0.3, 0.2],  # Token 0
    [0.8, 0.2, 0.9, 0.1],  # Token 1
    [0.3, 0.7, 0.4, 0.6],  # Token 2
    [0.9, 0.1, 0.5, 0.8]   # Token 3
])

# Compute attention scores
attn_scores = queries @ keys.T
# Result: [[1.37, 1.43, 1.08, 1.59]]

# Scale by sqrt(d_out) = sqrt(4) = 2
scaled_scores = attn_scores / (d_out ** 0.5)
# Result: [[0.685, 0.715, 0.540, 0.795]]

# Apply softmax
attn_weights = torch.softmax(scaled_scores, dim=-1)
# Result: [[0.229, 0.238, 0.199, 0.334]]

# These sum to 1.0 and represent attention distribution
```

**Interpretation**: The query token will:
- Pay **33.4%** attention to Token 3 (highest compatibility)
- Pay **23.8%** attention to Token 1
- Pay **22.9%** attention to Token 0
- Pay **19.9%** attention to Token 2 (lowest compatibility)

##### Why This Matters for Training

**Without Scaling** (Bad):
```
Attention weights: [0.99, 0.00, 0.01, 0.00]
                      ↓
Gradient for Token 1: ~0.00  (vanishes!)
```
Model can't learn to use Token 1.

**With Scaling** (Good):
```
Attention weights: [0.35, 0.25, 0.20, 0.20]
                      ↓
Gradient for Token 1: ~0.25  (healthy!)
```
Model can learn to adjust attention to all relevant tokens.

##### Summary of Step 3

**Scaling by √d_out**:
- Prevents extreme attention scores
- Maintains gradient flow during training
- Formula: `scores / sqrt(d_out)`
- Example: With d_out=64, divide by 8

**Softmax**:
- Converts scores → probabilities (0 to 1, sum to 1)
- Ensures all weights are positive
- Preserves relative importance (higher score → higher weight)
- Formula: `exp(x_i) / sum(exp(x_j))`

**Together**: They enable stable, distributed attention that trains effectively.

**After softmax**:

```
        Token 0  Token 1  Token 2  Token 3
Token 0   0.5      0.2      0.1      0.2     (sums to 1.0)
Token 1   0.1      0.6      0.2      0.1     (sums to 1.0)
Token 2   0.1      0.2      0.4      0.3     (sums to 1.0)
Token 3   0.2      0.1      0.1      0.6     (sums to 1.0)
```

#### Step 4: Compute Context Vectors

```python
context_vec = attn_weights @ values  # [8, 4, 4] @ [8, 4, 64] = [8, 4, 64]
```

**What this does**: Each token's output is a **weighted sum** of all value vectors.

**Example for token 1**:

```
output[1] = 0.1 * value[0] + 0.6 * value[1] + 0.2 * value[2] + 0.1 * value[3]
            ↑                ↑                ↑                ↑
         Low weight      High weight      Med weight       Low weight
      (token 0)         (itself)        (token 2)        (token 3)
```

The output embedding for token 1 now contains information weighted by attention scores.

---

## 4. Input Dimensions Explained

### Where do `batch` and `seq_len` come from?

The input `x` comes from Chapter 2's embedding pipeline:

```python
# From dataloader.py (Chapter 2)
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,      # ← This becomes "batch"
    max_length=4,      # ← This becomes "seq_len"
    stride=4,
    shuffle=False
)

# Get token IDs
inputs, targets = next(iter(dataloader))
# inputs.shape: [8, 4]
#                ↑  ↑
#             batch seq_len

# Convert to embeddings
token_embeddings = token_embedding_layer(inputs)
# token_embeddings.shape: [8, 4, 256]
#                          ↑  ↑   ↑
#                       batch seq d_in

# Add positional embeddings
input_embeddings = token_embeddings + pos_embeddings
# input_embeddings.shape: [8, 4, 256]

# THIS goes into self-attention
output = attention(input_embeddings)
```

### Dimension Reference

| Dimension | Name                 | Example Value | Comes From                                   |
| --------- | -------------------- | ------------- | -------------------------------------------- |
| `batch`   | Batch size           | 8             | `batch_size=8` in dataloader                 |
| `seq_len` | Sequence length      | 4             | `max_length=4` in dataloader                 |
| `d_in`    | Input embedding dim  | 256           | `output_dim=256` in token embedding          |
| `d_out`   | Attention output dim | 64            | Parameter to `SelfAttention_v1(d_in, d_out)` |

---

## 5. Complete Example: End-to-End

### From Raw Text to Attention Output

```python
import torch
import torch.nn as nn
from raschka_llm.dataloader import create_dataloader_v1

# Step 1: Load and prepare data (from Chapter 2)
with open("data/the-verdict.txt", "r") as f:
    raw_text = f.read()

# Step 2: Create dataloader
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)

# Step 3: Get a batch
inputs, targets = next(iter(dataloader))
# inputs.shape: [8, 4] - token IDs

# Step 4: Create embeddings
vocab_size = 50257
embed_dim = 256

token_embedding_layer = nn.Embedding(vocab_size, embed_dim)
pos_embedding_layer = nn.Embedding(4, embed_dim)

token_embeddings = token_embedding_layer(inputs)
pos_embeddings = pos_embedding_layer(torch.arange(4))
input_embeddings = token_embeddings + pos_embeddings
# input_embeddings.shape: [8, 4, 256]

# Step 5: Apply self-attention
attention = SelfAttention_v1(d_in=256, d_out=64)
context_vectors = attention(input_embeddings)
# context_vectors.shape: [8, 4, 64]

print(f"Input shape:  {input_embeddings.shape}")  # [8, 4, 256]
print(f"Output shape: {context_vectors.shape}")   # [8, 4, 64]
```

### What Happened?

1. **Dataloader**: Created 8 sequences of 4 tokens each
2. **Embeddings**: Converted token IDs to 256-dim vectors with position info
3. **Self-Attention**:
   - Each token "looked at" all other tokens
   - Computed attention weights (who to focus on)
   - Created context-aware representations
4. **Output**: 64-dim vectors that now contain contextual information

---

## 6. Key Concepts Summary

### Self-Attention vs Regular Attention

| Aspect             | Self-Attention          | Regular Attention                |
| ------------------ | ----------------------- | -------------------------------- |
| **What attends**   | Tokens in same sequence | Decoder attends to encoder       |
| **Use case**       | GPT (decoder-only)      | Translation (encoder-decoder)    |
| **Q, K, V source** | All from same input     | Q from decoder, K/V from encoder |

### Matrix Dimensions Flow

```
Input Embeddings
[batch, seq_len, d_in]
      [8, 4, 256]
           ↓
    ┌──────┴──────┐
    ↓      ↓      ↓
  W_query W_key W_value
  [256,64][256,64][256,64]
    ↓      ↓      ↓
  queries keys  values
  [8,4,64][8,4,64][8,4,64]
    │      │      │
    └──────┴──────┘
           ↓
    Attention Scores
      [8, 4, 4]
           ↓
    Attention Weights
      [8, 4, 4]
           ↓
    Context Vectors
      [8, 4, 64]
```

### Why Scaling by √d_out?

```python
attn_scores / keys.shape[-1]**0.5
```

**Problem without scaling**: As `d_out` increases, dot products get larger

- Large scores → extreme probabilities after softmax (0.99, 0.01, 0, 0)
- Extreme probabilities → vanishing gradients

**Solution**: Divide by √d_out to keep scores in reasonable range

- If d_out=64, divide by √64=8
- Keeps gradients healthy during training

### Learnable Parameters

The only learnable parameters are:

1. `W_query` - [d_in, d_out] - e.g., [256, 64]
2. `W_key` - [d_in, d_out] - e.g., [256, 64]
3. `W_value` - [d_in, d_out] - e.g., [256, 64]

**Total parameters**: 3 × (256 × 64) = 49,152 parameters

These get updated during training to learn:

- What patterns to look for (queries)
- What information to offer (keys)
- What content to return (values)

---

## 7. What's Next?

This is **vanilla self-attention**. GPT uses more advanced versions:

### Coming in Later Sections

1. **Causal/Masked Attention**: Prevent tokens from seeing future tokens

   - Required for autoregressive generation
   - "The cat sat" shouldn't see "on the mat" when predicting next word

2. **Multi-Head Attention**: Run multiple attention mechanisms in parallel

   - Different heads learn different patterns
   - Head 1: subject-verb relationships
   - Head 2: adjective-noun relationships
   - Head 3: long-range dependencies

3. **Attention in Transformer Blocks**: Combine with:
   - Layer normalization
   - Feed-forward networks
   - Residual connections

---

## 8. Visual Summary

### Self-Attention in One Picture

```
Input Sequence: "The cat sat on the mat"

Token Embeddings (from Ch 2):
The:  [0.1, 0.2, ..., 0.9]  256 dims
cat:  [0.4, 0.5, ..., 0.8]  256 dims
sat:  [0.7, 0.8, ..., 0.2]  256 dims
...

       ↓ W_query, W_key, W_value

Queries (what I want):
The:  [0.05, 0.12, ..., 0.44]  64 dims
cat:  [0.82, 0.21, ..., 0.19]  64 dims
sat:  [0.33, 0.67, ..., 0.11]  64 dims

       ↓ queries @ keys.T

Attention Scores (compatibility):
       The   cat   sat   on   the   mat
The:   0.8   0.3   0.1   0.2  0.4   0.1
cat:   0.2   0.9   0.4   0.1  0.3   0.2
sat:   0.1   0.6   0.8   0.5  0.2   0.3

       ↓ softmax

Attention Weights (probabilities):
       The   cat   sat   on   the   mat
The:   0.3   0.2   0.1   0.1  0.2   0.1  (sum=1)
cat:   0.1   0.4   0.2   0.1  0.1   0.1  (sum=1)
sat:   0.1   0.3   0.3   0.2  0.1   0.0  (sum=1)

       ↓ attn_weights @ values

Context Vectors (output):
The:  [0.15, 0.22, ..., 0.54]  64 dims (context-aware!)
cat:  [0.45, 0.31, ..., 0.29]  64 dims (context-aware!)
sat:  [0.23, 0.57, ..., 0.21]  64 dims (context-aware!)
```

**Key Insight**: Output embeddings now contain information from the entire sequence, weighted by relevance!

---

## Project Structure

```
raschka-build-llm-from-scratch-learning-journey/
├── src/raschka_llm/
│   ├── tokenizer.py          # SimpleTokenizer (Ch 2)
│   ├── bpe_tokenizer.py      # BPE tokenization (Ch 2)
│   ├── dataloader.py         # GPTDatasetV1 (Ch 2)
│   ├── self_attention.py     # SelfAttention_v1 (Ch 3) ← NEW
│   └── __init__.py
├── CHAPTER_2_SUMMARY.md      # Chapter 2 summary
├── CHAPTER_3_SUMMARY.md      # This file
└── data/
    └── the-verdict.txt       # Training data
```

---

## Key Takeaways

1. **Self-attention** makes embeddings context-aware
2. **Q, K, V matrices** transform inputs into queries, keys, and values
3. **Attention scores** measure compatibility between queries and keys
4. **Softmax** converts scores to probability distributions
5. **Context vectors** are weighted sums of values
6. **All parameters** are learned during training
7. This is the **core mechanism** that makes transformers powerful

The self-attention mechanism you've implemented is the fundamental building block that powers GPT, BERT, and all modern transformer models!

---

## Next Steps: Multi-Head Attention & Transformer Blocks

We'll extend this single attention head into:

- Multiple parallel attention heads
- Complete transformer blocks with normalization and feed-forward layers
- Causal masking for autoregressive generation
