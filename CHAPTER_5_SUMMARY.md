# Chapter 5: Pretraining on Unlabeled Data - Summary

This document summarizes the key concepts and implementations from Chapter 5 of "Build a Large Language Model from Scratch" by Sebastian Raschka.

## Overview

Chapter 5 brings the GPT model to life by training it on unlabeled text data:
1. **Text/Token Conversion**: Bridging between human-readable text and model inputs
2. **Loss Calculation**: Measuring how well the model predicts the next token
3. **Cross-Entropy Loss**: The standard loss function for language modeling
4. **Train/Validation Split**: Separating data to monitor overfitting
5. **Data Loaders**: Batching text into training samples
6. **Device Selection**: Picking CPU, CUDA, or MPS for compute
7. **Training Loop**: The full optimization process
8. **Evaluation & Generation**: Monitoring progress with metrics and text samples

**Connection to Previous Chapters**:
- **Chapter 2**: Provided tokenization and `create_dataloader_v1`
- **Chapter 3**: Built self-attention mechanisms
- **Chapter 4**: Assembled the complete `GPTModel`
- **Chapter 5**: Trains that model on real text

---

## 1. Text-to-Token and Token-to-Text Utilities

### Purpose

Translate between human-readable strings and tensors the model can process.

### Implementation ([training.py](src/raschka_llm/training.py))

```python
def text_to_token_ids(text, tokenizer):
    """
    Convert text to token IDs with batch dimension.
    "Hello" -> [15496] -> tensor([[15496]])
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back to text.
    tensor([[15496]]) -> [15496] -> "Hello"
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
```

### Why the Batch Dimension?

PyTorch models expect input shape `[batch_size, seq_len]`, even for a single example:

```python
# Single example needs batch dimension:
"Hello"            → [15496]              # No batch (won't work)
"Hello"            → [[15496]]            # With batch (works)

# unsqueeze(0) adds a batch dimension at position 0:
shape (seq_len,)   → shape (1, seq_len)
```

### The `unsqueeze` and `squeeze` Pair

| Operation | Before | After | Purpose |
|-----------|--------|-------|---------|
| `unsqueeze(0)` | `(seq_len,)` | `(1, seq_len)` | Add batch dim for model input |
| `squeeze(0)` | `(1, seq_len)` | `(seq_len,)` | Remove batch dim for tokenizer |

---

## 2. Understanding Loss: Predictions vs Targets

### The Next-Token Prediction Task

At each position, the model predicts the **next** token. So:
- **Input**: what the model sees
- **Target**: input shifted by 1 position

### Example: "every effort moves you"

```
Position:  0       1        2
Input:    [every, effort, moves]   ← Model sees these
Target:   [effort, moves, you]     ← Model should predict these
```

The target for position `i` is the token at position `i+1`.

### Code Setup

```python
# Two training examples (batch_size=2), each with 3 tokens (seq_len=3)
inputs = torch.tensor([[16833, 3626, 6100],   # "every effort moves"
                       [40, 1107, 588]])      # "I really like"

# Targets shifted by 1
targets = torch.tensor([[3626, 6100, 345],    # " effort moves you"
                        [1107, 588, 11311]])  # " really like chocolate"
```

### Forward Pass

```python
with torch.no_grad():
    logits = model(inputs)
# logits shape: (batch_size=2, seq_len=3, vocab_size=50257)
```

Each position produces **50,257 scores** — one per possible next token.

---

## 3. From Logits to Probabilities

### Softmax: Converting Scores to Probabilities

```python
probas = torch.softmax(logits, dim=-1)
# Shape: (2, 3, 50257) - probabilities sum to 1.0 across last dim
```

**Why `dim=-1`?**
- The vocab dimension is last
- Softmax normalizes scores into a probability distribution over the vocab
- Each position now has a distribution where probs sum to 1

### Argmax: Picking the Most Likely Token

```python
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# argmax returns the INDEX of the max value
# keepdim=True maintains shape (2, 3, 1) instead of (2, 3)
```

For an untrained model, predictions are essentially random — they won't match the targets.

### Visual: Logits → Probabilities → Predictions

```
Logits (raw scores):           Probabilities (softmax):       Prediction (argmax):
[2.1, -0.5, 8.7, ..., 1.3]  →  [0.001, 0.0001, 0.95, ...]  →  Token 789 (index of 0.95)
```

---

## 4. Extracting Target Probabilities

### What Probability Did the Model Assign to the CORRECT Answer?

This is the core question for measuring loss.

### Advanced Tensor Indexing

```python
text_idx = 0
# probas[0, [0,1,2], [3626, 6100, 345]] → 3 probability values
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
```

**Breaking it down**:
- `text_idx`: which batch (0 or 1)
- `[0, 1, 2]`: all 3 positions in the sequence
- `targets[text_idx]`: the 3 target token IDs for this batch

This extracts the probability the model assigned to each correct target token.

### Visual: Indexing Into the Probability Tensor

```
probas shape: [2 batches, 3 positions, 50257 vocab]

For batch 0:
  Position 0 → Pick prob of token 3626  → 0.00003
  Position 1 → Pick prob of token 6100  → 0.00002
  Position 2 → Pick prob of token 345   → 0.00001

target_probas_1 = [0.00003, 0.00002, 0.00001]
```

---

## 5. Cross-Entropy Loss

### Why Log Probabilities?

```python
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
```

Three reasons:
1. **Numerical stability**: Probabilities are tiny (e.g., 0.00002), logs are easier to work with
2. **Multiplication → addition**: Multiplying many tiny probabilities underflows; adding logs doesn't
3. **Cross-entropy uses logs**: This is the standard loss function for classification

**Interpretation**: Higher (less negative) log prob = model was more confident in the correct answer.

### The Manual Cross-Entropy Calculation

```python
# Average the log probabilities
avg_log_probas = torch.mean(log_probas)

# Negate to get cross-entropy loss
neg_avg_log_probas = avg_log_probas * -1
```

**The formula**: `cross_entropy_loss = -mean(log(p_target))`

**Why negate?**
- Log of probability is always negative (since `0 ≤ p ≤ 1`)
- We want a positive loss number where **lower is better**
- Negating flips the sign: a perfect prediction (`p=1`) gives `-log(1) = 0` loss

### PyTorch's Built-in Cross-Entropy

```python
# Flatten for PyTorch's API
logits_flat = logits.flatten(0, 1)      # (2, 3, 50257) → (6, 50257)
targets_flat = targets.flatten()        # (2, 3) → (6,)

# One-line equivalent of all manual steps
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

**PyTorch's `cross_entropy` does all the steps for you**:
1. Applies softmax internally
2. Computes log probabilities
3. Extracts target probs via efficient indexing
4. Negates and averages

### Sanity Check: Initial Loss ≈ 10.8

For an untrained GPT-2-style model on a 50,257-token vocabulary:
- Random predictions assign ~`1/50257` probability to each token
- Loss ≈ `-log(1/50257) = log(50257) ≈ 10.82`

If your initial loss is far from this, something is wrong!

### Perplexity

A related metric:
```
perplexity = exp(cross_entropy_loss)
```

**Interpretation**: "On average, the model is as confused as if it had to choose uniformly among `perplexity` tokens."
- Loss = 0 → perplexity = 1 (perfect)
- Loss = 10.82 → perplexity ≈ 50,257 (random)

---

## 6. Loading and Splitting Training Data

### Read the Training Text

```python
file_path = os.path.join(os.path.dirname(__file__), "../../data/the-verdict.txt")
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
```

**Why `os.path.dirname(__file__)`?**
- Returns the directory containing the script
- Makes the path independent of where you run the script from
- Otherwise, relative paths would break depending on the working directory

### Train/Validation Split

```python
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
```

**Why split the data?**
- **Training data**: Used to update the model's weights
- **Validation data**: Used to check generalization (held out from training)
- If train loss decreases but val loss increases → **overfitting**!

| Aspect | Training Set (90%) | Validation Set (10%) |
|--------|--------------------|-----------------------|
| **Used for** | Weight updates | Monitoring only |
| **Gradients?** | Yes | No |
| **Shuffle?** | Yes | No |
| **Drop last batch?** | Yes | No |

---

## 7. Creating Train/Val Data Loaders

### Using `create_dataloader_v1` from Chapter 2

```python
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],  # 256
    stride=GPT_CONFIG_124M["context_length"],       # 256 (no overlap)
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,   # Keep partial batches for evaluation
    shuffle=False,     # No need to shuffle validation data
    num_workers=0
)
```

### Key Parameter Choices

| Parameter | Train | Val | Why |
|-----------|-------|-----|-----|
| `shuffle` | `True` | `False` | Shuffle train for variety; val order doesn't matter |
| `drop_last` | `True` | `False` | Avoid uneven batches in training; keep all val data |
| `stride` | `context_length` | `context_length` | No overlap → each token used once |

### Why `stride == max_length`?

**No overlap between samples**:
```
Text: "the quick brown fox jumps over the lazy dog"
       └─ Sample 1 ─┘└─ Sample 2 ─┘
       (max_length=4, stride=4)
```

If `stride < max_length`, samples would overlap → more samples but more redundancy.

---

## 8. `calc_loss_batch`: Loss for One Batch

### Implementation

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss
```

### Step-by-Step

1. **Move tensors to device**: Inputs must be on same device as model
2. **Forward pass**: Get logits `[batch, seq, vocab]`
3. **Flatten for cross_entropy**:
   - Logits: `[batch, seq, vocab]` → `[batch * seq, vocab]`
   - Targets: `[batch, seq]` → `[batch * seq]`
4. **Return loss**: A scalar tensor (with autograd graph for backprop)

### Why Flatten?

PyTorch's `cross_entropy` expects:
- Predictions: `[N, num_classes]`
- Targets: `[N]`

Flattening collapses batch and sequence dimensions into one, treating each position as an independent classification problem.

---

## 9. `calc_loss_loader`: Average Loss Across Batches

### Implementation

```python
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```

### Why Both `calc_loss_batch` AND `calc_loss_loader`?

| Function | Returns | Used For |
|----------|---------|----------|
| `calc_loss_batch` | Loss for ONE batch (with autograd) | Training step (backprop) |
| `calc_loss_loader` | Average loss across MANY batches (Python float) | Evaluation / progress tracking |

### Why `.item()`?

```python
total_loss += loss.item()  # Python float, NOT tensor
```

- We're just tracking statistics, not building a computation graph
- `.item()` extracts the Python scalar, freeing the autograd graph
- Critical for memory: keeping tensors here would accumulate the entire computation history!

### Why `num_batches` Parameter?

Useful for fast checks during training:
- **Full evaluation**: `num_batches=None` → evaluate the whole dataset
- **Quick check**: `num_batches=5` → just 5 batches for a pulse-check
- Evaluating the entire validation set every few steps would be too slow

---

## 10. Device Selection: CUDA, MPS, or CPU

### Implementation

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")
model.to(device)
```

### The Three Backends

| Device | Hardware | Speed | When |
|--------|----------|-------|------|
| `"cuda"` | NVIDIA GPU | Fastest | Linux/Windows + NVIDIA card |
| `"mps"` | Apple Silicon GPU | Fast | Mac with M1/M2/M3/M4 chip |
| `"cpu"` | Any CPU | Slowest | Fallback |

### Python's Ternary Expression

```python
value = if_true if condition else if_false
```

**Chained ternaries**:
```python
"cuda" if cuda_available
else "mps" if mps_available
else "cpu"
```

Reads top-to-bottom: try CUDA, fall back to MPS, fall back to CPU.

### MPS Caveats

1. **Apple Silicon only**: Intel Macs don't have MPS
2. **Some ops unsupported**: Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for CPU fallback per-op
3. **Faster than CPU** for matrix-heavy ops (attention, embeddings)

### `model.to(device)` vs Tensor `.to(device)`

```python
model.to(device)              # Moves all model weights/buffers
input_tensor = input_tensor.to(device)  # Moves a specific tensor
```

**Critical rule**: Model and inputs must be on the **same device**, or you get a runtime error.

---

## 11. The Training Loop

### Implementation: `train_model_simple`

```python
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"EP {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen
```

### The Core 4-Step Training Pattern

For every batch, repeat:

```python
optimizer.zero_grad()          # 1. Clear old gradients
loss = calc_loss_batch(...)    # 2. Forward pass + compute loss
loss.backward()                # 3. Backward pass (compute gradients)
optimizer.step()               # 4. Update weights using gradients
```

### Why Each Step?

#### 1. `optimizer.zero_grad()` — Reset Gradients

PyTorch **accumulates** gradients by default. Without clearing them, each step would build on stale gradients from the previous batch.

```python
# Without zero_grad:
# Batch 1: gradient = g1
# Batch 2: gradient = g1 + g2  ← stale + new
# Batch 3: gradient = g1 + g2 + g3  ← worse!
```

#### 2. Forward Pass — Compute Loss

```python
loss = calc_loss_batch(input_batch, target_batch, model, device)
```

Behind the scenes, autograd builds a computation graph tracking every operation.

#### 3. `loss.backward()` — Compute Gradients

Backpropagation walks the computation graph in reverse, applying the chain rule to compute `∂loss/∂weight` for every parameter.

#### 4. `optimizer.step()` — Update Weights

The optimizer (e.g., AdamW) uses the gradients to update weights:

```
weight_new = weight_old - learning_rate * gradient  (simplified SGD)
```

AdamW is more sophisticated: it uses momentum + adaptive learning rates per-parameter.

### `model.train()` vs `model.eval()`

```python
model.train()  # Enables dropout, batchnorm uses batch stats
model.eval()   # Disables dropout, batchnorm uses running stats
```

**Why?** Some layers behave differently during training vs evaluation:
- **Dropout**: Random during training, identity during eval
- **BatchNorm**: Uses batch stats during training, running averages during eval

Forgetting this is a **common bug**. Always set the right mode!

### Tracking Progress

Three lists record metrics over time:

| Variable | Tracks |
|----------|--------|
| `train_losses` | Average loss on training set (at each eval step) |
| `val_losses` | Average loss on validation set (at each eval step) |
| `track_tokens_seen` | Cumulative tokens processed |

These are used later for plotting curves.

### `input_batch.numel()` — Token Count

```python
tokens_seen += input_batch.numel()
```

- `.numel()` = total number of elements in the tensor
- For shape `(batch=2, seq=256)` → `numel() = 512`
- Gives a hardware-agnostic measure of training progress

### Periodic Evaluation

```python
if global_step % eval_freq == 0:
    train_loss, val_loss = evaluate_model(...)
```

**Why not evaluate every step?**
- Evaluation requires a full pass over batches → slow
- Once every N steps gives smooth tracking without major overhead

**Typical values**: `eval_freq=5`, `eval_iter=5` for quick learning experiments.

### Sample Generation Between Epochs

```python
generate_and_print_sample(model, tokenizer, device, start_context)
```

**Why?** Loss numbers are abstract. Generated samples let you **see** the model improving:
- Epoch 0: gibberish
- Epoch 5: occasional real words
- Epoch 10: short coherent phrases

---

## 12. The Training Loop Visualized

```
┌─────────────────────────────────────────────────────────────────┐
│                       FOR EACH EPOCH                            │
│   ┌────────────────────────────────────────────────────────┐    │
│   │              FOR EACH BATCH IN TRAIN_LOADER            │    │
│   │                                                        │    │
│   │   ┌──────────────────┐  ┌──────────────────┐           │    │
│   │   │ optimizer        │  │  Forward pass    │           │    │
│   │   │ .zero_grad()     │→ │  calc_loss_batch │           │    │
│   │   └──────────────────┘  └──────────────────┘           │    │
│   │           ↓                       ↓                    │    │
│   │   ┌──────────────────┐  ┌──────────────────┐           │    │
│   │   │ optimizer.step() │← │  loss.backward() │           │    │
│   │   │ (update weights) │  │  (compute grads) │           │    │
│   │   └──────────────────┘  └──────────────────┘           │    │
│   │           ↓                                            │    │
│   │   ┌────────────────────────────────────┐               │    │
│   │   │ Every eval_freq steps:             │               │    │
│   │   │   - evaluate_model()               │               │    │
│   │   │   - Log losses                     │               │    │
│   │   └────────────────────────────────────┘               │    │
│   └────────────────────────────────────────────────────────┘    │
│                          ↓                                      │
│   ┌────────────────────────────────────────────────────────┐    │
│   │ End of epoch: generate_and_print_sample()              │    │
│   └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Key Concepts Summary

### Loss Calculation

- **Cross-entropy loss**: `-mean(log(p_target))`
- **Manual approach**: softmax → index targets → log → mean → negate
- **PyTorch shortcut**: `F.cross_entropy(logits_flat, targets_flat)` does it all
- **Initial loss check**: ~`ln(vocab_size)` for untrained model

### Data Pipeline

- **Train/val split**: 90/10 typical
- **DataLoader**: Yields `(input_batch, target_batch)` pairs
- **Sliding window**: Targets are inputs shifted by 1 token

### Device Management

- **Priority**: CUDA → MPS → CPU
- **Model and tensors** must be on the same device
- **`.to(device)`** moves tensors/models

### Training Loop

- **Four steps**: zero_grad → forward → backward → step
- **`model.train()` vs `model.eval()`**: Affects dropout, batchnorm
- **`optimizer.zero_grad()`**: PyTorch accumulates by default
- **`loss.backward()`**: Computes gradients via autograd
- **`optimizer.step()`**: Updates weights using gradients

### Monitoring

- **Periodic eval** to track train/val loss
- **Track tokens seen** for progress tracking
- **Generate samples** to qualitatively see improvement

---

## 14. Common Bugs and Gotchas

### 1. Forgetting `optimizer.zero_grad()`

**Symptom**: Gradients explode, loss diverges immediately.
**Fix**: Always call before `loss.backward()`.

### 2. Forgetting `model.train()` / `model.eval()`

**Symptom**: Dropout active during eval, or off during train.
**Fix**: Set the mode at the start of each loop.

### 3. Device Mismatch

**Symptom**: `RuntimeError: Expected all tensors to be on the same device`.
**Fix**: Call `.to(device)` on both model and input tensors.

### 4. Using Tensor Instead of `.item()`

**Symptom**: Memory grows unbounded during eval; OOM after many batches.
**Fix**: Use `.item()` when accumulating loss for statistics (not training).

### 5. Using `no_grad()` During Training

**Symptom**: Loss never decreases, weights don't update.
**Fix**: Only use `torch.no_grad()` for evaluation, never for the training forward pass.

### 6. Cross-Entropy Without Flatten

**Symptom**: Shape mismatch error.
**Fix**: Flatten logits to `[N, vocab]` and targets to `[N]`.

---

## 15. Project Structure

```
raschka-build-llm-from-scratch-learning-journey/
├── src/raschka_llm/
│   ├── tokenizer.py          # SimpleTokenizer (Ch 2)
│   ├── bpe_tokenizer.py      # BPE tokenization (Ch 2)
│   ├── dataloader.py         # GPTDatasetV1 (Ch 2)
│   ├── self_attention.py     # Self-attention (Ch 3)
│   ├── gpt_model.py          # GPT model (Ch 4)
│   ├── training.py           # Training pipeline (Ch 5) ← NEW
│   └── __init__.py
├── CHAPTER_2_SUMMARY.md      # Chapter 2 summary
├── CHAPTER_3_SUMMARY.md      # Chapter 3 summary
├── CHAPTER_4_SUMMARY.md      # Chapter 4 summary
├── CHAPTER_5_SUMMARY.md      # This file
└── data/
    └── the-verdict.txt       # Training data
```

---

## 16. What's Implemented

In [src/raschka_llm/training.py](src/raschka_llm/training.py):

- ✅ `text_to_token_ids` - Encode text with batch dimension
- ✅ `token_ids_to_text` - Decode token IDs back to text
- ✅ Manual loss calculation walkthrough (softmax, log, mean, negate)
- ✅ Cross-entropy loss using `torch.nn.functional.cross_entropy`
- ✅ Train/validation data split (90/10)
- ✅ Train and validation DataLoaders
- ✅ `calc_loss_batch` - Loss for a single batch
- ✅ `calc_loss_loader` - Average loss across many batches
- ✅ Device selection (CUDA / MPS / CPU)
- ✅ `train_model_simple` - Training loop with periodic evaluation

### Still To Implement

- ⏳ `evaluate_model` - Compute train+val loss in eval mode
- ⏳ `generate_and_print_sample` - Generate text between epochs
- ⏳ Plotting losses vs tokens seen
- ⏳ Sampling strategies (temperature, top-k)
- ⏳ Loading pretrained GPT-2 weights

---

## 17. Complete Pipeline Example

### From Untrained Model to Trained Model

```python
import torch
import tiktoken
from raschka_llm.gpt_model import GPTModel, GPT_CONFIG_124M
from raschka_llm.dataloader import create_dataloader_v1

# 1. Setup model
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
tokenizer = tiktoken.get_encoding("gpt2")

# 2. Load text data
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

# 3. Split train/val
split_idx = int(0.90 * len(text_data))
train_data, val_data = text_data[:split_idx], text_data[split_idx:]

# 4. Create dataloaders
train_loader = create_dataloader_v1(
    train_data, batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True, shuffle=True, num_workers=0
)
val_loader = create_dataloader_v1(
    val_data, batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False, shuffle=False, num_workers=0
)

# 5. Pick device + move model
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model.to(device)

# 6. Initial loss check (should be ~10.8)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print(f"Initial train loss: {train_loss:.3f}, val loss: {val_loss:.3f}")

# 7. Train
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=10, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer
)
```

### What Happens?

```
Initial state:
  - Loss ≈ 10.8 (random predictions over 50,257 vocab)
  - Generated text: gibberish

After training:
  - Loss decreases (model learns patterns in the text)
  - Generated text: starts to resemble training data
  - Eventually: model overfits the small dataset (val loss rises)
```

---

## 18. Key Takeaways

1. **Cross-entropy loss** is the standard loss for language modeling — it measures how surprised the model is by the correct answer
2. **Initial loss ≈ ln(vocab_size)** is a sanity check for an untrained model
3. **Targets are inputs shifted by 1** — this is the essence of next-token prediction
4. **PyTorch's `cross_entropy`** combines softmax + log + indexing + negation into one fast op
5. **Train/val split** lets you monitor overfitting — train loss decreasing while val loss rises is the warning sign
6. **DataLoaders** batch data efficiently; use `shuffle=True` and `drop_last=True` for training
7. **Device selection** should prefer GPU (CUDA or MPS) over CPU for speed
8. **Model and tensors** must be on the same device, or PyTorch raises an error
9. **`.item()`** extracts a Python scalar from a tensor — use it for stats, not for training
10. **The 4-step training pattern**: `zero_grad` → forward → `backward` → `step`
11. **`optimizer.zero_grad()`** is critical — PyTorch accumulates gradients by default
12. **`model.train()` / `model.eval()`** switch dropout and batchnorm behavior
13. **`torch.no_grad()`** during evaluation saves memory and speed
14. **Periodic evaluation** during training tracks both loss curves without slowing training too much
15. **Generating samples between epochs** gives qualitative insight that loss numbers can't
16. **AdamW** is the standard optimizer for transformers (adaptive learning rate + weight decay)
17. **Tracking tokens seen** is hardware-agnostic progress measurement
18. **`loss.backward()`** uses autograd to compute gradients via backpropagation
19. **`optimizer.step()`** applies the gradients to update weights
20. **The training loop** is the bridge between architecture (Ch 4) and a useful model

---

## 19. What's Next?

### Chapter 5 Continued / Chapter 6

1. **Better Generation**:
   - Temperature scaling
   - Top-k and nucleus (top-p) sampling
   - Avoiding repetition

2. **Plotting Training Curves**:
   - matplotlib visualization
   - Train vs val loss
   - Spotting overfitting

3. **Loading Pretrained Weights**:
   - Download OpenAI's GPT-2 weights
   - Map weight names to our architecture
   - Use pretrained model as starting point

4. **Fine-tuning** (Chapter 6+):
   - Classification fine-tuning
   - Instruction following
   - Reinforcement Learning from Human Feedback (RLHF)

With Chapter 5 complete, you can now train your GPT model on real text!
