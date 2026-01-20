# Understanding transpose(1, 2) in Multi-Head Attention

## Quick Summary

`transpose(1, 2)` reorganizes tensor dimensions to enable parallel processing of all attention heads.

- **Before**: `[batch, tokens, heads, features]` - organized by tokens
- **After**: `[batch, heads, tokens, features]` - organized by heads

This allows PyTorch to compute attention for all heads simultaneously using batched matrix operations.

---

## Detailed Example Output

```
======================================================================
UNDERSTANDING transpose(1, 2) IN MULTI-HEAD ATTENTION
======================================================================

BEFORE transpose(1, 2):
Shape: torch.Size([1, 3, 2, 4]) = [batch, tokens, heads, features]

Data organized by TOKENS first:
tensor([[[[ 0,  1,  2,  3],
          [ 4,  5,  6,  7]],

         [[ 8,  9, 10, 11],
          [12, 13, 14, 15]],

         [[16, 17, 18, 19],
          [20, 21, 22, 23]]]])

──────────────────────────────────────────────────────────────────────

AFTER transpose(1, 2):
Shape: torch.Size([1, 2, 3, 4]) = [batch, heads, tokens, features]

Data reorganized by HEADS first:
tensor([[[[ 0,  1,  2,  3],
          [ 8,  9, 10, 11],
          [16, 17, 18, 19]],

         [[ 4,  5,  6,  7],
          [12, 13, 14, 15],
          [20, 21, 22, 23]]]])

======================================================================
WHY THIS MATTERS FOR ATTENTION
======================================================================

Scenario 1: WITHOUT transpose (WRONG)
──────────────────────────────────────────────────────────────────────
queries shape: torch.Size([1, 3, 2, 4])
keys shape: torch.Size([1, 3, 2, 4])

❌ Can't easily compute attention per head!
   All heads are mixed together in the same dimension


Scenario 2: WITH transpose (CORRECT)
──────────────────────────────────────────────────────────────────────
queries shape: torch.Size([1, 2, 3, 4]) = [batch, heads, tokens, features]
keys shape: torch.Size([1, 2, 3, 4])

Attention scores shape: torch.Size([1, 2, 3, 3]) = [batch, heads, tokens, tokens]

✅ Each head has its own 3x3 attention matrix!

Attention scores:
tensor([[[[  14,   62,  110],
          [  62,  366,  670],
          [ 110,  670, 1230]],

         [[ 126,  302,  478],
          [ 302,  734, 1166],
          [ 478, 1166, 1854]]]])

======================================================================
VISUAL BREAKDOWN
======================================================================

Before transpose - organized by TOKEN:
  Token 0: [Head0_data, Head1_data]
  Token 1: [Head0_data, Head1_data]
  Token 2: [Head0_data, Head1_data]

After transpose - organized by HEAD:
  Head 0: [Token0_data, Token1_data, Token2_data]
  Head 1: [Token0_data, Token1_data, Token2_data]

💡 Key insight:
   With heads separated, we can compute attention for ALL heads
   in parallel using batched matrix multiplication!
```

---

## Key Takeaways

1. **transpose(1, 2) swaps dimensions 1 and 2**
   - Position 1: tokens ↔ heads
   - Keeps batch (0) and features (3) unchanged

2. **Why we need it**
   - PyTorch treats leading dimensions as "batch" dimensions
   - With `[batch, heads, tokens, features]`, both batch and heads are processed in parallel
   - This means all heads compute attention simultaneously!

3. **The workflow**
   ```python
   # Step 1: Project to d_out
   keys = self.W_key(x)  # [batch, tokens, d_out]

   # Step 2: Split into heads
   keys = keys.view(b, tokens, num_heads, head_dim)  # [batch, tokens, heads, head_dim]

   # Step 3: Transpose to enable parallel processing
   keys = keys.transpose(1, 2)  # [batch, heads, tokens, head_dim]

   # Step 4: Compute attention (all heads in parallel!)
   attn_scores = queries @ keys.transpose(2, 3)  # [batch, heads, tokens, tokens]
   ```

4. **Real-world example**
   - 8 heads, each with 6 tokens and 64 features
   - Before: `[batch, 6, 8, 64]` - tokens are grouped, heads mixed
   - After: `[batch, 8, 6, 64]` - heads are separated, can process in parallel
   - Result: 8 independent 6×6 attention matrices computed simultaneously!

---

## Related Files

- Demo script: `transpose_demo.py`
- Implementation: `src/raschka_llm/self_attention.py` (MultiHeadAttention class, line 508)
