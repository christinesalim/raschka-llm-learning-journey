# Chapter 4: Implementing the GPT Model - Summary

This document summarizes the key concepts and implementations from Chapter 4 of "Build a Large Language Model from Scratch" by Sebastian Raschka.

## Overview

Chapter 4 builds on the foundations from previous chapters to implement the actual GPT model architecture:
1. **Layer Normalization**: Stabilizes training by normalizing activations
2. **GELU Activation**: Smooth, non-linear activation function
3. **Feed-Forward Networks**: Transform representations in expanded space
4. **Shortcut Connections**: Solve vanishing gradients in deep networks
5. **Complete GPT Architecture**: Assembling all components

**Connection to Previous Chapters**:
- **Chapter 2**: Provided tokenization and embeddings
- **Chapter 3**: Introduced self-attention mechanism
- **Chapter 4**: Combines everything into a working GPT model

---

## 1. Layer Normalization

### Purpose

Normalizes activations across the embedding dimension to stabilize training and improve convergence.

### Implementation: LayerNorm ([gpt_model.py](src/raschka_llm/gpt_model.py))

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # Small constant for numerical stability
        self.scale = nn.Parameter(torch.ones(emb_dim))   # Learnable scale (gamma)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Learnable shift (beta)

    def forward(self, x):
        # Calculate mean and variance across embedding dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize: centers to mean=0, variance=1
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift: learnable parameters
        return self.scale * norm_x + self.shift
```

### How It Works

**Normalization Formula**: `(x - mean) / sqrt(variance + eps)`

**Step-by-step example**:
```python
# Original values (wide range, unstable)
x = [10, 20, 30, 40, 50]
mean = 30
variance = 200

# After normalization (mean=0, variance=1)
normalized = [-1.41, -0.71, 0.0, 0.71, 1.41]
```

### Why Normalize to Variance = 1?

**The Math**: When you divide by `c`, variance divides by `c²`

```python
# Dividing by sqrt(variance) makes variance = 1:
new_variance = variance / (sqrt(variance))²
             = variance / variance
             = 1  ✓
```

**Why it matters**:
- **Before normalization**: Different layers might have wildly different scales (0.001, 100, 5000)
- **After normalization**: All layers have consistent scale (variance = 1)
- **Result**: Gradients flow more smoothly, training is more stable

### Scale and Shift Parameters

**Purpose**: Allow the model to learn optimal mean and variance

```python
self.scale = nn.Parameter(torch.ones(emb_dim))   # Initially multiply by 1 (no effect)
self.shift = nn.Parameter(torch.zeros(emb_dim))  # Initially add 0 (no effect)
```

**Why learnable?**:
- Normalization forces mean=0, variance=1
- But maybe the optimal distribution is mean=0.5, variance=2
- `scale` and `shift` let the model learn this during training
- Formula: `output = scale * normalized + shift`

### Epsilon (eps) Explained

```python
self.eps = 1e-5  # 0.00001
norm_x = (x - mean) / torch.sqrt(var + self.eps)
                                      ↑
                        Prevents division by zero
```

**Why needed?**:
- If all values in a layer are identical: `variance = 0`
- `sqrt(0) = 0` → division by zero error!
- Adding tiny `eps` prevents this: `sqrt(0 + 0.00001) = 0.00316`

### Variance Calculation: `unbiased=False`

```python
var = x.var(dim=-1, keepdim=True, unbiased=False)
```

**Two ways to calculate variance**:

1. **Population variance** (`unbiased=False`): Divide by `n`
   - Used in LayerNorm
   - We're normalizing the actual data we have

2. **Sample variance** (`unbiased=True`, default): Divide by `n-1`
   - Used in statistics when estimating population from sample
   - Not appropriate here

**Example**:
```python
data = [1, 2, 3, 4, 5]

# Population (unbiased=False)
var = sum((x - mean)²) / 5 = 2.0

# Sample (unbiased=True)
var = sum((x - mean)²) / 4 = 2.5
```

**Important**: Always use same `unbiased` setting when checking variance!

```python
# In LayerNorm
var = x.var(dim=-1, keepdim=True, unbiased=False)

# When verifying
check_var = output.var(dim=-1, keepdim=True, unbiased=False)  # Must match!
```

---

## 2. GELU Activation Function

### What is GELU?

**Gaussian Error Linear Unit** - A smooth, non-linear activation function used in GPT models.

### Implementation: GELU ([gpt_model.py](src/raschka_llm/gpt_model.py))

```python
class GELU(nn.Module):
    def forward(self, x):
        # GELU approximation using tanh
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

### GELU vs ReLU

| Aspect | ReLU | GELU |
|--------|------|------|
| **Formula** | `max(0, x)` | `0.5 * x * (1 + tanh(...))` |
| **Shape** | Hard cutoff at 0 | Smooth curve |
| **Negative values** | All become 0 | Small negatives pass through |
| **Gradient at 0** | Undefined (sharp corner) | Smooth (has gradient everywhere) |
| **Use in transformers** | Less common | Standard (GPT-2, GPT-3, BERT) |

**Why GELU is better for transformers**:
- Smoother gradients → more stable training
- No "dead neurons" (ReLU can kill neurons if input always negative)
- Empirically performs better in language models

### Visual Comparison

```
ReLU:          GELU:
  y              y
  ↑              ↑
  │    ┌─        │      ┌─
  │   ┌          │    ╱
  │  ┌           │  ╱
  │ ┌            │ ╱
  └─────→ x      └─────→ x
  Sharp corner   Smooth curve
```

---

## 3. Feed-Forward Networks

### Purpose

Transform token representations in an expanded dimensional space, allowing the model to learn complex patterns.

### Architecture

**The 4x Expansion Pattern**: 768 → 3072 → 768

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expand: 768 → 3072
            GELU(),                                           # Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # Contract: 3072 → 768
        )

    def forward(self, x):
        return self.layers(x)
```

### Why 4x Expansion?

**The Intuition**: More space = more learning capacity

```
768 dimensions  → Limited workspace
3072 dimensions → 4x larger workspace to learn patterns
768 dimensions  → Compress insights back to compact form
```

**The Analogy**: Puzzle solving
- **Small table (768)**: Can only spread out a few pieces
- **Large table (3072)**: Spread out ALL pieces, see patterns, group by color
- **Box (768)**: Put assembled solution back in compact form

### Four Key Reasons for 4x Expansion

#### 1. Increased Representational Capacity

3072 neurons can specialize in different patterns:
- Some learn "animal + action" patterns
- Some learn "preposition + location" patterns
- Some learn subject-verb agreement
- Some learn syntax, some semantics, some grammar

Then combine all this knowledge back to 768 dimensions.

#### 2. Non-Linear Transformation Space

```python
# Without expansion:
x (768) → GELU → still 768 dimensions

# With 4x expansion:
x (768) → expand to 3072 → GELU → 3072 dimensions to transform
# More room for complex non-linear transformations!
```

#### 3. Information Mixing

```
Input (768 features)
    ↓
Linear 1: Each of 3072 neurons looks at ALL 768 inputs
    ↓
3072 intermediate features (rich combinations)
    ↓
GELU: Non-linear activation
    ↓
Linear 2: Each of 768 outputs combines ALL 3072 intermediate features
    ↓
Output (768 features - enriched with complex patterns)
```

#### 4. Parameter Count = Learning Power

**Direct transformation**:
```python
768 → 768 = 768 × 768 = 589,824 parameters
```

**With 4x expansion**:
```python
768 → 3072: 2,359,296 parameters
3072 → 768: 2,359,296 parameters
Total:      4,718,592 parameters (8x more!)
```

More parameters = more capacity to learn language patterns.

### Why Specifically 4x?

**Empirical finding** from research:
- **2x**: Not enough expansion, limited benefit
- **4x**: Sweet spot (GPT-2, BERT, most transformers use this)
- **8x or 16x**: Diminishing returns, too many parameters

The 4x ratio balances:
- ✓ Model capacity (learning power)
- ✓ Parameter efficiency (not bloated)
- ✓ Training stability
- ✓ Computational cost

### Example: Processing "The cat sat on the mat"

**Without expansion (768→768)**:
- Limited ability to simultaneously learn multiple relationships
- Can't specialize neurons for different linguistic patterns

**With 4x expansion (768→3072→768)**:
- Neurons 1-500: Learn subject-verb relationships
- Neurons 501-1000: Learn prepositional phrases
- Neurons 1001-1500: Learn semantic meaning
- Neurons 1501-2000: Learn syntax patterns
- Neurons 2001-2500: Learn long-range dependencies
- Neurons 2501-3072: Learn rare patterns

All this specialized knowledge gets compressed back to 768 dims!

### The "Thinking Space" Metaphor

```
Step 1: EXPAND (768 → 3072)
Spread information out into rich representation
Like decompressing an image to full resolution

Step 2: TRANSFORM (GELU activation)
Apply complex patterns in this wider space
Like editing the full-resolution image

Step 3: COMPRESS (3072 → 768)
Distill insights back to compact form
Like recompressing to manageable size (but now enhanced!)
```

---

## 4. PyTorch `__call__` and `forward()`

### The Pattern

```python
model = FeedForward(GPT_CONFIG_124M)
output = model(x)  # This automatically calls forward(x)
```

### How It Works

When you call `model(x)`, PyTorch's `nn.Module` implements `__call__`:

```python
class Module:
    def __call__(self, *args, **kwargs):
        # PyTorch magic: hooks, autograd tracking, etc.
        result = self.forward(*args, **kwargs)
        # More PyTorch magic
        return result
```

### Why Not Call `forward()` Directly?

```python
# ❌ Don't do this
output = model.forward(x)

# ✅ Do this instead
output = model(x)
```

**Reason**: `__call__` does important setup:
- Registers hooks for debugging
- Tracks gradients for backpropagation
- Manages training/eval mode
- Handles device placement

Calling `forward()` directly bypasses all this!

### This Pattern is Everywhere

```python
# All of these use __call__ → forward():
linear = nn.Linear(768, 3072)
out = linear(x)  # Calls linear.forward(x)

gelu = GELU()
out = gelu(x)  # Calls gelu.forward(x)

attention = SelfAttention_v2(256, 64)
out = attention(x)  # Calls attention.forward(x)
```

---

## 5. Complete GPT Architecture

### DummyGPTModel ([gpt_model.py](src/raschka_llm/gpt_model.py))

This is the skeleton of GPT-2 with 124M parameters:

```python
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 1. Token embeddings: What each token means
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Creates: (50257, 768) lookup table

        # 2. Position embeddings: Where each token appears
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # Creates: (1024, 768) lookup table

        # 3. Dropout: Regularization
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 4. Transformer blocks: 12 layers (currently dummy)
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 5. Final layer norm: Stabilize before output
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # 6. Output head: Project to vocabulary
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # Creates: (50257, 768) weight matrix

    def forward(self, in_idx):
        # in_idx: [batch_size, seq_len] - token IDs

        # Get dimensions
        batch_size, seq_len = in_idx.shape

        # Token embeddings: [batch_size, seq_len, 768]
        tok_embeds = self.tok_emb(in_idx)

        # Position embeddings: [seq_len, 768]
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Combine: [batch_size, seq_len, 768]
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # Through transformer blocks
        x = self.trf_blocks(x)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary: [batch_size, seq_len, 50257]
        logits = self.out_head(x)

        return logits
```

### GPT-2 124M Configuration

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Max sequence length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Attention heads per layer
    "n_layers": 12,         # Number of transformer blocks
    "drop_rate": 0.1,       # Dropout probability
    "qkv_bias": False       # No bias in attention projections
}
```

### Architecture Flow

```
Token IDs [batch, seq_len]
    ↓
Token Embeddings [batch, seq_len, 768]
    + Position Embeddings [seq_len, 768]
    ↓
Dropout
    ↓
12 × Transformer Blocks [batch, seq_len, 768]
  ├─ Multi-Head Attention
  ├─ Layer Norm
  ├─ Feed-Forward (768 → 3072 → 768)
  ├─ Layer Norm
  └─ Residual Connections
    ↓
Final Layer Norm
    ↓
Output Projection [batch, seq_len, 50257]
    ↓
Logits (predictions for next token)
```

### What Each Component Does

| Component | Input Shape | Output Shape | Purpose |
|-----------|------------|--------------|---------|
| Token Embedding | `[batch, seq]` | `[batch, seq, 768]` | Convert IDs to vectors |
| Position Embedding | `[seq]` | `[seq, 768]` | Add position information |
| Dropout | `[batch, seq, 768]` | `[batch, seq, 768]` | Regularization |
| Transformer Blocks | `[batch, seq, 768]` | `[batch, seq, 768]` | Process context (×12) |
| Final LayerNorm | `[batch, seq, 768]` | `[batch, seq, 768]` | Stabilize |
| Output Head | `[batch, seq, 768]` | `[batch, seq, 50257]` | Predict next token |

---

## 6. Understanding nn.Embedding and nn.Linear Internals

### nn.Embedding

```python
self.tok_emb = nn.Embedding(50257, 768)
```

**What PyTorch creates**:
- Weight matrix: shape `(50257, 768)` - learnable lookup table
- Each row is the embedding for one vocabulary token
- Initialized randomly, learned during training

**How it works**:
```python
# Input: token IDs
in_idx = torch.tensor([[15496, 11, 616]])  # [batch=1, seq=3]

# Output: embeddings (looks up rows from weight matrix)
embeddings = self.tok_emb(in_idx)  # [1, 3, 768]

# This is equivalent to:
embeddings = self.tok_emb.weight[[15496, 11, 616]]
```

**Not one-hot encoding**:
- One-hot would be: `[0, 0, ..., 1, ..., 0]` (50257 elements, one 1)
- Embedding is efficient: direct lookup, no sparse vectors
- Same result, much faster!

### nn.Linear

```python
self.out_head = nn.Linear(768, 50257, bias=False)
```

**What PyTorch creates**:
- Weight matrix: shape `(50257, 768)` - **note the transpose!**
- No bias (because `bias=False`)

**How it works**:
```python
# Forward pass (PyTorch handles transpose internally):
output = input @ weight.T

# Shape calculation:
# input:  [batch, seq, 768]
# weight: [50257, 768]
# weight.T: [768, 50257]
# output: [batch, seq, 768] @ [768, 50257] = [batch, seq, 50257]
```

**Why transpose?**:
- PyTorch stores weights as `[out_features, in_features]`
- Each row represents one output neuron
- Makes it easy to access all weights for one output
- Transpose happens automatically in `forward()`

---

## 7. Transformer Block Structure (Preview)

The `DummyTransformerBlock` is currently a placeholder. A real transformer block contains:

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Multi-head self-attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )

        # Feed-forward network
        self.ff = FeedForward(cfg)

        # Layer normalization (before each sub-layer)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Multi-head attention with residual connection
        attn_out = self.att(self.norm1(x))
        x = x + self.drop_resid(attn_out)  # Residual

        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + self.drop_resid(ff_out)  # Residual

        return x
```

**Key components**:
1. **Multi-Head Attention**: Tokens communicate (from Chapter 3)
2. **Feed-Forward Network**: Transform in expanded space
3. **Layer Normalization**: Stabilize activations (×2)
4. **Residual Connections**: Add input to output (`x = x + layer(x)`)
5. **Dropout**: Regularization (×2)

---

## 8. Shortcut Connections (Residual Connections)

### The Vanishing Gradient Problem

In deep neural networks, gradients can become extremely small as they propagate backward through many layers, making it difficult for early layers to learn effectively.

### What Are Shortcut Connections?

**Shortcut connections** (also called **residual connections** or **skip connections**) add the input of a layer directly to its output:

```python
output = input + layer(input)
```

Instead of just:
```python
output = layer(input)
```

### Implementation: ExampleDeepNeuralNetwork ([gpt_model.py](src/raschka_llm/gpt_model.py))

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut

        # Create 5 layers
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output  # Add input to output (residual connection)
            else:
                x = layer_output  # Normal layer (no shortcut)
        return x
```

### Why Shortcut Connections Are Important

**Without shortcuts** (vanishing gradients):
```
layers.4.0.weight has gradient mean of 0.00020
layers.3.0.weight has gradient mean of 0.00011
layers.2.0.weight has gradient mean of 0.00009
layers.1.0.weight has gradient mean of 0.00007
layers.0.0.weight has gradient mean of 0.00005  ← Very small!
```

**With shortcuts** (stable gradients):
```
layers.4.0.weight has gradient mean of 0.22169
layers.3.0.weight has gradient mean of 0.20694
layers.2.0.weight has gradient mean of 0.32896
layers.1.0.weight has gradient mean of 0.26250
layers.0.0.weight has gradient mean of 0.15657  ← Much larger!
```

### How Shortcut Connections Help

#### 1. Gradient Flow

**Without shortcuts**:
```
Layer 5 → Layer 4 → Layer 3 → Layer 2 → Layer 1
gradient × 0.5 × 0.5 × 0.5 × 0.5 = 0.0625 (vanishes!)
```

**With shortcuts**:
```
Layer 5 ←→ Layer 4 ←→ Layer 3 ←→ Layer 2 ←→ Layer 1
         ↑                                    ↑
         └────────── direct path ─────────────┘
```

Gradients can flow directly through the shortcuts, avoiding multiplication through many layers.

#### 2. Identity Mapping

The network can learn to "skip" layers when they're not helpful:

```python
# If a layer isn't useful, it can learn to output ~0
# Then: output = input + 0 = input (identity function)
```

This makes it easier to train very deep networks.

#### 3. Shape Constraint

```python
if self.use_shortcut and x.shape == layer_output.shape:
    x = x + layer_output
```

**Why check shapes?**
- You can only add tensors of the same shape
- If layer changes dimensions (e.g., 768 → 3072), you can't add them directly
- In transformers, most layers maintain the same shape (768 → 768), so shortcuts work

### Shortcut Connections in Transformers

In a real transformer block, shortcuts are used **twice**:

```python
def forward(self, x):
    # Shortcut around attention
    attn_out = self.att(self.norm1(x))
    x = x + self.drop_resid(attn_out)  # ← Shortcut 1

    # Shortcut around feed-forward
    ff_out = self.ff(self.norm2(x))
    x = x + self.drop_resid(ff_out)    # ← Shortcut 2

    return x
```

This allows:
- **Attention** to focus on learning relationships between tokens
- **Feed-forward** to focus on transforming individual token representations
- **Shortcuts** to preserve the original information and stabilize gradients

### Visual Representation

```
Input (x)
    ↓
    ├─────────────────┐  ← Shortcut path (identity)
    ↓                 ↓
Layer Transformation  │
    ↓                 ↓
  Output        +   Input
        └───→ Final Output
```

### Key Benefits

1. **Solves vanishing gradients**: Gradients flow through shortcuts
2. **Enables deep networks**: Can stack 100+ layers (GPT-3 has 96!)
3. **Faster training**: Better gradient flow = faster convergence
4. **Better performance**: Networks can learn when to skip layers
5. **Easier optimization**: Identity function is easy to learn as baseline

### The Mathematics

**Standard layer**:
```
y = f(x)
```

**Residual layer**:
```
y = x + f(x)
```

**Gradient flow**:
```
dy/dx = 1 + df/dx

The "+1" ensures gradient is always at least 1, preventing vanishing!
```

---

## 9. Key Concepts Summary

### Layer Normalization

- **Formula**: `(x - mean) / sqrt(variance + eps)`
- **Purpose**: Normalize activations to mean=0, variance=1
- **Scale/Shift**: Learnable parameters to find optimal distribution
- **Epsilon**: Prevents division by zero when variance=0

### GELU Activation

- **Smooth**: Has gradients everywhere (unlike ReLU)
- **Better for transformers**: Empirically outperforms ReLU
- **Formula**: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`

### Feed-Forward 4x Expansion

- **Pattern**: 768 → 3072 → 768
- **Why 4x**: Optimal balance of capacity and efficiency
- **Purpose**: Transform in expanded "thinking space"
- **Result**: 8x more parameters = more learning power

### PyTorch Internals

- **`__call__`**: Automatically calls `forward()` with PyTorch magic
- **`nn.Embedding`**: Efficient lookup table (not one-hot)
- **`nn.Linear`**: Stores weights transposed, handles it internally

---

## 9. Complete Pipeline Example

### From Text to Predictions

```python
import torch
import tiktoken
from raschka_llm.gpt_model import DummyGPTModel, GPT_CONFIG_124M

# 1. Tokenize text
tokenizer = tiktoken.get_encoding("gpt2")
text1 = "Every effort moves you"
text2 = "Every day holds a"

# 2. Encode to IDs
ids1 = tokenizer.encode(text1)  # [6109, 3626, 6100, 345]
ids2 = tokenizer.encode(text2)  # [6109, 1110, 6622, 257]

# 3. Create batch
batch = torch.tensor([ids1, ids2])  # [2, 4]

# 4. Create model
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

# 5. Forward pass
logits = model(batch)
# logits.shape: [2, 4, 50257]
#                ↑  ↑   ↑
#             batch seq vocab_size

# 6. Get predictions (argmax over vocabulary)
predictions = torch.argmax(logits, dim=-1)
# predictions.shape: [2, 4]
# Each position predicts the most likely next token
```

### What Happened?

```
Input: "Every effort moves you"
  ↓ Tokenize
Token IDs: [6109, 3626, 6100, 345]
  ↓ Token Embeddings (lookup)
[batch=1, seq=4, emb=768]
  ↓ + Position Embeddings
[batch=1, seq=4, emb=768]
  ↓ Dropout
  ↓ 12 Transformer Blocks (currently dummy)
  ↓ Final Layer Norm (currently dummy)
  ↓ Output Projection
Logits: [batch=1, seq=4, vocab=50257]
  ↓ Argmax
Predicted next tokens for each position
```

---

## 10. Project Structure

```
raschka-build-llm-from-scratch-learning-journey/
├── src/raschka_llm/
│   ├── tokenizer.py          # SimpleTokenizer (Ch 2)
│   ├── bpe_tokenizer.py      # BPE tokenization (Ch 2)
│   ├── dataloader.py         # GPTDatasetV1 (Ch 2)
│   ├── self_attention.py     # Self-attention (Ch 3)
│   ├── gpt_model.py          # GPT model (Ch 4) ← NEW
│   └── __init__.py
├── CHAPTER_2_SUMMARY.md      # Chapter 2 summary
├── CHAPTER_3_SUMMARY.md      # Chapter 3 summary
├── CHAPTER_4_SUMMARY.md      # This file
└── data/
    └── the-verdict.txt       # Training data
```

---

## 11. What's Implemented

In [src/raschka_llm/gpt_model.py](src/raschka_llm/gpt_model.py):

- ✅ `GPT_CONFIG_124M` - Configuration dictionary
- ✅ `LayerNorm` - Layer normalization with scale/shift
- ✅ `GELU` - Gaussian Error Linear Unit activation
- ✅ `FeedForward` - Feed-forward network with 4x expansion
- ✅ `ExampleDeepNeuralNetwork` - Demonstrates shortcut connections and vanishing gradients
- ✅ `DummyGPTModel` - Skeleton GPT architecture
- ✅ `DummyTransformerBlock` - Placeholder (to be implemented)
- ✅ `DummyLayerNorm` - Placeholder (to be implemented)

---

## 12. Key Takeaways

1. **Layer Normalization** stabilizes training by normalizing to mean=0, variance=1
2. **Dividing by sqrt(variance)** makes variance=1 because `var / (sqrt(var))² = 1`
3. **Scale and shift** are learnable parameters that let the model find optimal distributions
4. **GELU** is smoother than ReLU and works better in transformers
5. **4x expansion** in feed-forward networks provides "thinking space" for complex transformations
6. **Shortcut connections** (`x + f(x)`) solve vanishing gradients by providing direct gradient paths
7. **Without shortcuts**, gradients vanish in deep networks (multiply through many layers)
8. **With shortcuts**, gradients flow directly, enabling networks with 100+ layers
9. **`model(x)`** automatically calls `forward(x)` via Python's `__call__` method
10. **nn.Embedding** is an efficient lookup table, not one-hot encoding
11. **nn.Linear** stores weights transposed but handles it transparently

---

## 13. What's Next?

### In Later Sections

1. **Implement Real Transformer Block**:
   - Combine multi-head attention (Ch 3) with feed-forward network
   - Add residual connections
   - Add proper layer normalization

2. **Training the Model**:
   - Loss function (cross-entropy)
   - Optimizer (AdamW)
   - Training loop
   - Validation

3. **Text Generation**:
   - Autoregressive generation
   - Sampling strategies (greedy, top-k, nucleus)
   - Temperature control

4. **Loading Pretrained Weights**:
   - Download GPT-2 weights
   - Map to our architecture
   - Fine-tuning

The GPT model skeleton is complete - now we need to fill in the transformer blocks and train it!
