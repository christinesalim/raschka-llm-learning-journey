"""Demo to understand transpose(1, 2) in multi-head attention."""
import torch

print("=" * 70)
print("UNDERSTANDING transpose(1, 2) IN MULTI-HEAD ATTENTION")
print("=" * 70)

# Small example for clarity
batch_size = 1      # Just 1 batch to keep it simple
num_tokens = 3      # 3 tokens (words)
num_heads = 2       # 2 attention heads
head_dim = 4        # 4 features per head

# Create example tensor (before transpose)
# Shape: [batch, tokens, heads, features]
tensor_before = torch.arange(batch_size * num_tokens * num_heads * head_dim).reshape(
    batch_size, num_tokens, num_heads, head_dim
)

print(f"\nBEFORE transpose(1, 2):")
print(f"Shape: {tensor_before.shape} = [batch, tokens, heads, features]")
print(f"\nData organized by TOKENS first:")
print(tensor_before)

print("\n" + "─" * 70)

# Apply transpose(1, 2)
tensor_after = tensor_before.transpose(1, 2)

print(f"\nAFTER transpose(1, 2):")
print(f"Shape: {tensor_after.shape} = [batch, heads, tokens, features]")
print(f"\nData reorganized by HEADS first:")
print(tensor_after)

print("\n" + "=" * 70)
print("WHY THIS MATTERS FOR ATTENTION")
print("=" * 70)

# Simulate the attention computation
queries = tensor_before  # [1, 3, 2, 4]
keys = tensor_before     # [1, 3, 2, 4]

print("\nScenario 1: WITHOUT transpose (WRONG)")
print("─" * 70)
print(f"queries shape: {queries.shape}")
print(f"keys shape: {keys.shape}")
print("\n❌ Can't easily compute attention per head!")
print("   All heads are mixed together in the same dimension")

print("\n\nScenario 2: WITH transpose (CORRECT)")
print("─" * 70)
queries_transposed = queries.transpose(1, 2)  # [1, 2, 3, 4]
keys_transposed = keys.transpose(1, 2)        # [1, 2, 3, 4]

print(f"queries shape: {queries_transposed.shape} = [batch, heads, tokens, features]")
print(f"keys shape: {keys_transposed.shape}")

# Now compute attention scores
attn_scores = queries_transposed @ keys_transposed.transpose(2, 3)
print(f"\nAttention scores shape: {attn_scores.shape} = [batch, heads, tokens, tokens]")
print(f"\n✅ Each head has its own {num_tokens}x{num_tokens} attention matrix!")

print("\nAttention scores:")
print(attn_scores)

print("\n" + "=" * 70)
print("VISUAL BREAKDOWN")
print("=" * 70)

print("\nBefore transpose - organized by TOKEN:")
print("  Token 0: [Head0_data, Head1_data]")
print("  Token 1: [Head0_data, Head1_data]")
print("  Token 2: [Head0_data, Head1_data]")
print("\nAfter transpose - organized by HEAD:")
print("  Head 0: [Token0_data, Token1_data, Token2_data]")
print("  Head 1: [Token0_data, Token1_data, Token2_data]")

print("\n💡 Key insight:")
print("   With heads separated, we can compute attention for ALL heads")
print("   in parallel using batched matrix multiplication!")
print("=" * 70)
