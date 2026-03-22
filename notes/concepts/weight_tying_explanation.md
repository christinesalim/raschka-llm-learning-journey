# Weight Tying in Neural Networks

## What is Weight Tying?

**Weight tying** is a technique where two different layers in a neural network share the same weight matrix instead of having separate, independent weight matrices.

In GPT models, this typically refers to sharing weights between:
- **Token Embedding Layer** (input): Converts token IDs → vector representations
- **Output Projection Layer** (output): Converts vector representations → token predictions

## The Question: Where Are the Weights?

When you write code like this in PyTorch:

```python
self.tok_emb = nn.Embedding(vocab_size, emb_dim)
self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)
```

You might wonder: *"Where are the weight matrices? I didn't create them!"*

**Answer**: PyTorch creates and manages them internally as `nn.Parameter` objects.

## PyTorch's Internal Weight Management

### nn.Embedding Internals

```python
# What you write:
self.tok_emb = nn.Embedding(50257, 768)

# What PyTorch creates internally:
self.tok_emb.weight = nn.Parameter(torch.randn(50257, 768))
```

**Access it**:
```python
print(model.tok_emb.weight.shape)  # torch.Size([50257, 768])
print(model.tok_emb.weight)        # The actual tensor
```

### nn.Linear Internals

```python
# What you write:
self.out_head = nn.Linear(768, 50257, bias=False)

# What PyTorch creates internally:
self.out_head.weight = nn.Parameter(torch.randn(50257, 768))
```

**Access it**:
```python
print(model.out_head.weight.shape)  # torch.Size([50257, 768])
print(model.out_head.weight)        # The actual tensor
```

### How Forward Pass Uses These Weights

**Token Embedding (Lookup)**:
```python
# Forward pass:
embedding = self.tok_emb(token_id)

# What actually happens:
embedding = self.tok_emb.weight[token_id]  # Row lookup
```

**Output Projection (Matrix Multiplication)**:
```python
# Forward pass:
logits = self.out_head(x)

# What actually happens:
logits = x @ self.out_head.weight.T  # Matrix multiplication
```

## Weight Tying: Before and After

### Before Weight Tying (Current GPTModel)

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Two separate weight matrices
        self.tok_emb = nn.Embedding(50257, 768)   # Weight: (50257, 768)
        self.out_head = nn.Linear(768, 50257, bias=False)  # Weight: (50257, 768)
```

**Result**:
- `tok_emb.weight`: 50,257 × 768 = **38,597,376 parameters**
- `out_head.weight`: 50,257 × 768 = **38,597,376 parameters**
- **Total**: **77,194,752 parameters** (just for these two layers!)

**Memory**:
- Two separate tensors in memory
- Different memory addresses
- Updating one does NOT update the other

### After Weight Tying (Like GPT-2)

```python
class GPTModelWithWeightTying(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Create token embedding
        self.tok_emb = nn.Embedding(50257, 768)

        # Create output head
        self.out_head = nn.Linear(768, 50257, bias=False)

        # Tie the weights (make them share the same matrix)
        self.out_head.weight = self.tok_emb.weight
```

**Result**:
- `tok_emb.weight`: 50,257 × 768 = **38,597,376 parameters**
- `out_head.weight`: **← Same matrix! (reused)**
- **Total**: **38,597,376 parameters** (saved 38.6M parameters!)

**Memory**:
- One tensor shared by both layers
- Same memory address
- Updating one AUTOMATICALLY updates the other

## Why Weight Tying Makes Sense

### Conceptual Symmetry

**Token Embedding (Input)**:
- Maps from **token space** → **vector space**
- "What does token ID 123 mean as a vector?"

**Output Projection (Output)**:
- Maps from **vector space** → **token space**
- "Which token does this vector represent?"

They're **inverse operations** in the same semantic space, so it makes sense to use the same mapping!

### Example: The Word "cat"

```
Token "cat" (ID: 2368)
    ↓ Embedding lookup (using tok_emb.weight)
Vector: [0.2, -0.5, 0.8, ..., 0.1] (768 dims)
    ↓ Process through transformer
Modified vector: [0.3, -0.4, 0.9, ..., 0.2]
    ↓ Output projection (using SAME weights!)
Scores for all tokens:
  - "cat": 5.2  ← High score because vector is similar to "cat" embedding
  - "dog": 4.8
  - "table": 1.2
  - ...
```

The model learns: "If the vector looks like 'cat's embedding, predict 'cat'!"

## Benefits of Weight Tying

### 1. Reduced Parameters

- **Without tying**: 163M parameters (for GPT-2 124M model)
- **With tying**: 124M parameters
- **Savings**: 39M parameters (24% reduction)

### 2. Less Memory

- Smaller model size
- Faster loading
- Lower GPU memory usage

### 3. Better Generalization

- Shared weights = shared learning
- Input and output learn consistent representations
- Reduces overfitting

### 4. Conceptual Clarity

- Input and output operate in the same semantic space
- Embeddings have the same meaning for input and output

## The Parameter Count Calculation

This is what the code in `gpt_model.py` calculates:

```python
# Count ALL parameters (no weight tying)
total_params = sum(p.numel() for p in model.parameters())
# Result: 163,037,184

# Calculate what it would be WITH weight tying
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
# Result: 124,439,808

# This shows: "If we tied weights like GPT-2, we'd have 124M parameters"
```

The "124M" in GPT-2's name comes from this weight-tied parameter count!

## Demonstration: Memory Sharing

### Before Tying

```python
tok_emb = nn.Embedding(50257, 768)
out_head = nn.Linear(768, 50257, bias=False)

print(tok_emb.weight is out_head.weight)  # False
print(id(tok_emb.weight))   # Memory address: 5091099984
print(id(out_head.weight))  # Memory address: 5091099744  ← Different!

# Total unique parameters: 77,194,752
```

### After Tying

```python
tok_emb = nn.Embedding(50257, 768)
out_head = nn.Linear(768, 50257, bias=False)

# Tie the weights
out_head.weight = tok_emb.weight

print(tok_emb.weight is out_head.weight)  # True  ← Same object!
print(id(tok_emb.weight))   # Memory address: 5091099984
print(id(out_head.weight))  # Memory address: 5091099984  ← Same!

# Total unique parameters: 38,597,376  (saved 38.6M!)
```

### Shared Updates

```python
# Modify tok_emb.weight
with torch.no_grad():
    tok_emb.weight[0, 0] = 99.99

# Check both weights
print(tok_emb.weight[0, 0])   # 99.99
print(out_head.weight[0, 0])  # 99.99  ← Also changed!

# They share the same memory!
```

## Implementation in Real GPT-2

The original GPT-2 implementation uses weight tying:

```python
class GPT2Model:
    def __init__(self, config):
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Token embeddings
        # ... other layers ...
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.wte.weight
```

This is why GPT-2 "124M" has ~124 million parameters, not ~163 million.

## Key Takeaways

1. **PyTorch manages weight matrices internally** - you don't see them in `__init__` but they exist as `.weight` attributes
2. **Weight tying shares a single matrix** between two layers instead of having separate matrices
3. **Before tying**: Two separate 38.6M parameter matrices (77.2M total)
4. **After tying**: One shared 38.6M parameter matrix (38.6M total)
5. **Memory addresses prove sharing**: Same `id()` means same object in memory
6. **GPT-2's "124M"** refers to the parameter count WITH weight tying
7. **Our current GPTModel** has ~163M parameters because we haven't tied weights yet

## Running the Demo

To see this in action, run:

```bash
python notes/concepts/weight_tying_demo.py
```

The demo shows:
- How PyTorch creates weight matrices automatically
- Weight matrices before tying (different addresses)
- Weight matrices after tying (same address)
- Parameter count reduction (77M → 38.6M)
- How updates to one weight affect the other (shared memory)
