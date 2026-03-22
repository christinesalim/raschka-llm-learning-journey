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

## 7. Transformer Block - Full Implementation

### Architecture: Pre-LN Transformer

The transformer block is the core building block of GPT. GPT-2 stacks 12 of these blocks to create the 124M parameter model.

**Pre-LN Architecture Pattern**:
```
Input
  ↓
[LayerNorm → Attention → Dropout] → Add with input (residual)
  ↓
[LayerNorm → FeedForward → Dropout] → Add with input (residual)
  ↓
Output
```

### Implementation: TransformerBlock ([gpt_model.py](src/raschka_llm/gpt_model.py))

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Multi-head self-attention: Allows each token to look at all previous tokens
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],              # Input dimension: 768
            d_out=cfg["emb_dim"],             # Output dimension: 768
            context_length=cfg["context_length"],  # Max sequence length: 1024
            num_heads=cfg["n_heads"],         # Number of attention heads: 12
            dropout=cfg["drop_rate"],         # Dropout rate: 0.1
            qkv_bias=cfg["qkv_bias"]         # Bias in Q,K,V projections: False
        )

        # Feed-forward network: Processes each token independently
        # Expands to 4x dimension (768→3072) then contracts back (3072→768)
        self.ff = FeedForward(cfg)

        # Layer normalization before attention
        self.norm1 = LayerNorm(cfg["emb_dim"])

        # Layer normalization before feed-forward
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout applied to residual connections
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # First sub-layer: Multi-head self-attention with residual connection
        shortcut = x                    # Save input for residual
        x = self.norm1(x)               # Layer norm BEFORE attention
        x = self.att(x)                 # Multi-head self-attention
        x = self.drop_shortcut(x)       # Dropout
        x = x + shortcut                # Residual connection 1

        # Second sub-layer: Feed-forward network with residual connection
        shortcut = x                    # Save for second residual
        x = self.norm2(x)               # Layer norm BEFORE feed-forward
        x = self.ff(x)                  # Feed-forward (768→3072→768)
        x = self.drop_shortcut(x)       # Dropout
        x = x + shortcut                # Residual connection 2

        return x
```

### Pre-LN vs Post-LN Architecture

**Pre-LN (what we use)**:
```
x → LayerNorm → Attention → Dropout → + → x
                                       ↑
                                   Residual
```

**Post-LN (older approach)**:
```
x → Attention → Dropout → + → LayerNorm → x
                          ↑
                      Residual
```

**Why Pre-LN is better**:
- More stable training (especially for deep networks)
- Gradients flow more smoothly
- Modern approach used in GPT-2, GPT-3, and most recent transformers
- Layer norm stabilizes the input to attention, not the output

### The Two Sub-Layers

#### Sub-Layer 1: Multi-Head Attention

**Purpose**: Learn relationships between tokens
- Each token can attend to all previous tokens (causal masking)
- 12 attention heads learn different types of relationships
- Shape preserved: `(batch, seq_len, 768)` → `(batch, seq_len, 768)`

**Example**: Processing "The cat sat on the mat"
- "sat" can attend to: "The", "cat", "sat" (but not future tokens)
- Different heads might learn:
  - Head 1: Subject-verb relationships ("cat" ← "sat")
  - Head 2: Determiner-noun ("The" ← "cat")
  - Head 3: Preposition-object ("on" ← "mat")

#### Sub-Layer 2: Feed-Forward Network

**Purpose**: Transform token representations independently
- Each token processed separately (no cross-token communication)
- Expansion to 4x dimension provides "thinking space"
- Shape preserved: `(batch, seq_len, 768)` → `(batch, seq_len, 768)`

**Example**: After attention gathers context
- Token "sat" now has context: "The cat sat"
- Feed-forward enriches this with patterns like:
  - "past tense action"
  - "followed by preposition"
  - "subject performing action"

### Why Two Residual Connections?

Each residual connection serves a different purpose:

**Residual 1 (around attention)**:
```python
x = x + self.drop_shortcut(self.att(self.norm1(x)))
```
- Preserves original token embeddings
- Attention adds contextual information
- If attention isn't helpful, model can learn to output ~0
- Result: original embedding + context from other tokens

**Residual 2 (around feed-forward)**:
```python
x = x + self.drop_shortcut(self.ff(self.norm2(x)))
```
- Preserves context-aware embeddings from attention
- Feed-forward adds transformations/refinements
- If transformations aren't helpful, can output ~0
- Result: context-aware embedding + refined features

**Combined effect**:
```
Original Token → + Attention Context → + FF Transformations → Enriched Token
```

### Information Flow Through One Block

```
Input: [batch=2, seq=4, emb=768]
  ↓
Save as shortcut_1
  ↓
LayerNorm (normalize each token independently)
  ↓
Multi-Head Attention (tokens communicate)
  - Query, Key, Value projections
  - 12 attention heads compute relationships
  - Causal masking (can't see future)
  - Concatenate heads
  ↓
Dropout (regularization)
  ↓
Add shortcut_1 (residual connection)
  ↓
Save as shortcut_2
  ↓
LayerNorm (normalize again)
  ↓
Feed-Forward Network
  - Expand: 768 → 3072
  - GELU activation
  - Contract: 3072 → 768
  ↓
Dropout (regularization)
  ↓
Add shortcut_2 (residual connection)
  ↓
Output: [batch=2, seq=4, emb=768]
```

### Why Same Dropout Layer for Both Paths?

```python
self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
```

This is reused for both attention and feed-forward paths because:
- Dropout mask is generated fresh each time it's called
- No need for separate dropout layers (saves parameters)
- Same dropout rate (0.1) for both paths
- Common practice in transformer implementations

### Key Components Summary

| Component | Input Shape | Output Shape | Purpose |
|-----------|------------|--------------|---------|
| `norm1` | `[batch, seq, 768]` | `[batch, seq, 768]` | Stabilize input to attention |
| `att` | `[batch, seq, 768]` | `[batch, seq, 768]` | Learn token relationships |
| `drop_shortcut` | `[batch, seq, 768]` | `[batch, seq, 768]` | Regularization |
| Residual 1 | `[batch, seq, 768]` | `[batch, seq, 768]` | Preserve original info |
| `norm2` | `[batch, seq, 768]` | `[batch, seq, 768]` | Stabilize input to FF |
| `ff` | `[batch, seq, 768]` | `[batch, seq, 768]` | Transform representations |
| `drop_shortcut` | `[batch, seq, 768]` | `[batch, seq, 768]` | Regularization |
| Residual 2 | `[batch, seq, 768]` | `[batch, seq, 768]` | Preserve attention output |

**Notice**: Shape is preserved throughout! `[batch, seq, 768]` → `[batch, seq, 768]`

This allows us to stack 12 transformer blocks sequentially.

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

## 13. Key Concepts Summary

### Layer Normalization

- **Formula**: `(x - mean) / sqrt(variance + eps)`
- **Purpose**: Normalize activations to mean=0, variance=1
- **Scale/Shift**: Learnable parameters to find optimal distribution
- **Epsilon**: Prevents division by zero when variance=0
- **Pre-LN**: Applied BEFORE attention/feed-forward (modern approach)

### GELU Activation

- **Smooth**: Has gradients everywhere (unlike ReLU)
- **Better for transformers**: Empirically outperforms ReLU
- **Formula**: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`

### Feed-Forward 4x Expansion

- **Pattern**: 768 → 3072 → 768
- **Why 4x**: Optimal balance of capacity and efficiency
- **Purpose**: Transform in expanded "thinking space"
- **Result**: 8x more parameters = more learning power

### Transformer Block

- **Two sub-layers**: Attention (token communication) + Feed-forward (token transformation)
- **Two residual connections**: Preserve information and enable gradient flow
- **Pre-LN architecture**: LayerNorm BEFORE each sub-layer
- **Shape preservation**: `[batch, seq, 768]` throughout all blocks

### GPTModel

- **Embeddings**: Token embeddings + Position embeddings
- **Processing**: 12 stacked transformer blocks
- **Output**: Logits for next-token prediction `[batch, seq, vocab_size]`
- **Parameters**: 124 million (GPT-2 small configuration)

### PyTorch Internals

- **`__call__`**: Automatically calls `forward()` with PyTorch magic
- **`nn.Embedding`**: Efficient lookup table (not one-hot)
- **`nn.Linear`**: Stores weights transposed, handles it internally

---

## 14. Complete Pipeline Example

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

## 11. GPTModel - Full Implementation

### Complete Functional GPT-2 Model

The `GPTModel` class is a fully functional implementation of GPT-2 architecture. Unlike `DummyGPTModel`, this uses real transformer blocks with attention and feed-forward layers.

### Implementation: GPTModel ([gpt_model.py](src/raschka_llm/gpt_model.py))

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Token Embedding: Converts token IDs to dense vectors
        # Shape: (vocab_size=50257, emb_dim=768)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Position Embedding: Adds position information
        # Shape: (context_length=1024, emb_dim=768)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Dropout for embeddings
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of 12 Transformer Blocks (real ones, not dummy!)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Final Layer Normalization
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Output Projection Head
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # Convert token IDs to embeddings
        tok_embeds = self.tok_emb(in_idx)

        # Create position embeddings
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # Combine token and position embeddings
        x = tok_embeds + pos_embeds

        # Apply dropout
        x = self.drop_emb(x)

        # Pass through all 12 transformer blocks
        x = self.trf_blocks(x)

        # Final layer normalization
        x = self.final_norm(x)

        # Project to vocabulary size
        logits = self.out_head(x)

        return logits
```

### Complete Architecture Flow

```
Token IDs (integers)
  ↓
Token Embeddings + Position Embeddings (learned vectors)
  ↓
Dropout (regularization)
  ↓
12 × Transformer Blocks (attention + feed-forward)
  ↓
Final Layer Normalization
  ↓
Output Projection Head (to vocabulary size)
  ↓
Logits (scores for each vocabulary token)
```

### Detailed Step-by-Step Example

**Input**: `[[15496, 11, 616, 13]]` (batch_size=1, seq_len=4)

**Step 1: Token Embeddings**
```python
tok_embeds = self.tok_emb(in_idx)
# Shape: [1, 4, 768]
# Each token ID is converted to a 768-dim vector via lookup
```

**Step 2: Position Embeddings**
```python
pos_embeds = self.pos_emb(torch.arange(4))
# Shape: [4, 768]
# Positions [0, 1, 2, 3] each get their own 768-dim vector
```

**Step 3: Combine Embeddings**
```python
x = tok_embeds + pos_embeds
# Shape: [1, 4, 768]
# Broadcasting adds position info to each token
# Now each token knows both WHAT it is and WHERE it is
```

**Step 4: Embedding Dropout**
```python
x = self.drop_emb(x)
# Shape: [1, 4, 768]
# Randomly zeros 10% of values (training only)
```

**Step 5: Transformer Blocks (×12)**
```python
x = self.trf_blocks(x)
# Shape: [1, 4, 768] (preserved through all 12 blocks)

# Each block does:
#   1. LayerNorm → Multi-Head Attention → Dropout → Residual
#   2. LayerNorm → Feed-Forward → Dropout → Residual
```

**Step 6: Final Normalization**
```python
x = self.final_norm(x)
# Shape: [1, 4, 768]
# Stabilizes values before output projection
```

**Step 7: Output Projection**
```python
logits = self.out_head(x)
# Shape: [1, 4, 50257]
# Linear: [1, 4, 768] @ [768, 50257] = [1, 4, 50257]
# For each position, we get scores for all 50,257 vocab tokens
```

### Understanding the Output: Logits

**What are logits?**
- Unnormalized prediction scores for each vocabulary token
- Higher score = model thinks that token is more likely next
- Need softmax to convert to probabilities

**Example output interpretation**:
```python
logits.shape  # [batch=1, seq=4, vocab=50257]

# For position 0 (after "15496"):
logits[0, 0, :]  # 50,257 scores
# token 123: score = 5.2
# token 456: score = -1.3
# token 789: score = 8.7  ← Highest score!

# Apply softmax to get probabilities:
probs = torch.softmax(logits[0, 0, :], dim=-1)
# token 123: prob = 0.02
# token 456: prob = 0.0001
# token 789: prob = 0.65  ← Most likely next token!
```

### Key Differences: DummyGPTModel vs GPTModel

| Aspect | DummyGPTModel | GPTModel |
|--------|---------------|----------|
| **Transformer Blocks** | `DummyTransformerBlock` (pass-through) | `TransformerBlock` (full implementation) |
| **Layer Norm** | `DummyLayerNorm` (pass-through) | `LayerNorm` (full implementation) |
| **Functionality** | Skeleton structure only | Fully functional model |
| **Purpose** | Educational (show architecture) | Production (can train and generate) |
| **Attention** | None | Multi-head self-attention |
| **Feed-forward** | None | 768→3072→768 transformation |

### Why Final Layer Normalization?

```python
self.final_norm = LayerNorm(cfg["emb_dim"])
```

**Purpose**:
- Stabilizes values before output projection
- After 12 transformer blocks, values might have large variance
- Normalization ensures consistent scale going into final linear layer
- Helps with training stability

**Without final norm**:
```
Block 12 output: values ranging from -100 to +200 (unstable!)
  ↓
Output projection: huge logits, gradient explosion
```

**With final norm**:
```
Block 12 output: values ranging from -100 to +200
  ↓
Final LayerNorm: normalized to mean=0, variance=1
  ↓
Output projection: stable logits, smooth gradients
```

### Device Handling for Position Embeddings

```python
pos_embeds = self.pos_emb(
    torch.arange(seq_len, device=in_idx.device)
)
```

**Why `device=in_idx.device`?**
- Input tensors might be on CPU or GPU
- Position embeddings must be on the same device
- `device=in_idx.device` automatically matches the input device
- Prevents "tensors on different devices" errors

**Example**:
```python
# If input is on GPU:
in_idx.device  # cuda:0
torch.arange(4, device=in_idx.device)  # Creates tensor on cuda:0

# If input is on CPU:
in_idx.device  # cpu
torch.arange(4, device=in_idx.device)  # Creates tensor on cpu
```

### Parameter Count: 124 Million

**Where do the parameters come from?**

```python
# Token embeddings: 50257 × 768 = 38,597,376
# Position embeddings: 1024 × 768 = 786,432

# Each TransformerBlock (×12):
#   - Multi-head attention: ~2.4M parameters
#   - Feed-forward: ~4.7M parameters
#   - Layer norms: ~3,072 parameters
#   Total per block: ~7.1M
#   12 blocks: ~85M

# Final layer norm: 1,536
# Output head: 50257 × 768 = 38,597,376 (often weight-tied with tok_emb)

# Total: ~124M parameters
```

### Weight Tying (Common Optimization)

Many GPT implementations tie the token embedding and output projection weights:

```python
# Weight tying (not in our implementation, but common):
self.out_head.weight = self.tok_emb.weight
```

**Benefits**:
- Reduces parameters: 124M → ~85M
- Makes sense conceptually: same vector space for input and output
- Improves training (shared gradients)

### Using the Model

```python
# Create model
model = GPTModel(GPT_CONFIG_124M)

# Prepare input
tokenizer = tiktoken.get_encoding("gpt2")
text = "Every effort moves you"
token_ids = tokenizer.encode(text)
batch = torch.tensor([token_ids])  # [1, 4]

# Forward pass
logits = model(batch)  # [1, 4, 50257]

# Get next token predictions
next_token_logits = logits[:, -1, :]  # [1, 50257] (last position)
probs = torch.softmax(next_token_logits, dim=-1)
next_token_id = torch.argmax(probs, dim=-1)  # Most likely next token

# Decode
next_token = tokenizer.decode([next_token_id.item()])
print(f"Next token: '{next_token}'")
```

---

## 12. What's Implemented

In [src/raschka_llm/gpt_model.py](src/raschka_llm/gpt_model.py):

- ✅ `GPT_CONFIG_124M` - Configuration dictionary
- ✅ `LayerNorm` - Layer normalization with scale/shift
- ✅ `GELU` - Gaussian Error Linear Unit activation
- ✅ `FeedForward` - Feed-forward network with 4x expansion
- ✅ `ExampleDeepNeuralNetwork` - Demonstrates shortcut connections and vanishing gradients
- ✅ `TransformerBlock` - Full transformer block with attention and feed-forward
- ✅ `GPTModel` - Complete, functional GPT-2 implementation
- ✅ `DummyGPTModel` - Skeleton GPT architecture (educational)
- ✅ `DummyTransformerBlock` - Placeholder (educational)
- ✅ `DummyLayerNorm` - Placeholder (educational)

---

## 15. Key Takeaways

1. **Layer Normalization** stabilizes training by normalizing to mean=0, variance=1
2. **Dividing by sqrt(variance)** makes variance=1 because `var / (sqrt(var))² = 1`
3. **Scale and shift** are learnable parameters that let the model find optimal distributions
4. **GELU** is smoother than ReLU and works better in transformers
5. **4x expansion** in feed-forward networks provides "thinking space" for complex transformations
6. **Shortcut connections** (`x + f(x)`) solve vanishing gradients by providing direct gradient paths
7. **Without shortcuts**, gradients vanish in deep networks (multiply through many layers)
8. **With shortcuts**, gradients flow directly, enabling networks with 100+ layers
9. **Pre-LN architecture** applies LayerNorm BEFORE sub-layers (more stable than Post-LN)
10. **Transformer blocks** combine attention (token communication) with feed-forward (token transformation)
11. **Two residual connections** per block: one around attention, one around feed-forward
12. **Shape preservation**: All transformer blocks maintain `[batch, seq, emb_dim]` shape
13. **GPTModel** is fully functional: can process text and generate predictions
14. **Logits** are unnormalized scores; apply softmax to get probabilities
15. **Device handling**: Use `device=in_idx.device` for consistent tensor placement
16. **`model(x)`** automatically calls `forward(x)` via Python's `__call__` method
17. **nn.Embedding** is an efficient lookup table, not one-hot encoding
18. **nn.Linear** stores weights transposed but handles it transparently

---

## 16. What's Next?

### In Later Sections

1. **Training the Model**:
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

The complete GPT model is now implemented and ready for training!
