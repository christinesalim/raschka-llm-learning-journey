"""
Demonstration: Weight matrices are managed internally by PyTorch
"""
import torch
import torch.nn as nn

print("=" * 70)
print("WEIGHT MATRICES ARE MANAGED INTERNALLY BY PYTORCH")
print("=" * 70)

# Simple example with nn.Embedding
print("\n" + "=" * 70)
print("EXAMPLE 1: nn.Embedding")
print("=" * 70)

embedding = nn.Embedding(num_embeddings=100, embedding_dim=16)

print(f"\nCreated: nn.Embedding(100, 16)")
print(f"PyTorch automatically created: embedding.weight")
print(f"Weight shape: {embedding.weight.shape}")
print(f"Weight type: {type(embedding.weight)}")
print(f"\nFirst 3 token embeddings (first 5 dimensions):")
print(embedding.weight[:3, :5])

# How it's used
print(f"\n--- How Forward Pass Works ---")
token_ids = torch.tensor([5, 12, 99])
result = embedding(token_ids)
manual_lookup = embedding.weight[[5, 12, 99]]
print(f"embedding([5, 12, 99]) == embedding.weight[[5, 12, 99]]? {torch.equal(result, manual_lookup)}")

# Simple example with nn.Linear
print("\n" + "=" * 70)
print("EXAMPLE 2: nn.Linear")
print("=" * 70)

linear = nn.Linear(in_features=768, out_features=50257, bias=False)

print(f"\nCreated: nn.Linear(768, 50257, bias=False)")
print(f"PyTorch automatically created: linear.weight")
print(f"Weight shape: {linear.weight.shape}")
print(f"Weight type: {type(linear.weight)}")
print(f"\nFirst 3 rows (first 5 dimensions):")
print(linear.weight[:3, :5])

# How it's used
print(f"\n--- How Forward Pass Works ---")
x = torch.randn(2, 4, 768)  # [batch=2, seq=4, emb=768]
result = linear(x)
manual_matmul = x @ linear.weight.T
print(f"linear(x).shape: {result.shape}")
print(f"(x @ linear.weight.T).shape: {manual_matmul.shape}")
print(f"linear(x) == x @ linear.weight.T? {torch.allclose(result, manual_matmul)}")

# Weight Tying Demonstration
print("\n" + "=" * 70)
print("EXAMPLE 3: Weight Tying (Sharing Weight Matrices)")
print("=" * 70)

vocab_size = 50257
emb_dim = 768

# Create two separate layers (like in your GPTModel)
tok_emb = nn.Embedding(vocab_size, emb_dim)
out_head = nn.Linear(emb_dim, vocab_size, bias=False)

print(f"\nBefore weight tying:")
print(f"  tok_emb.weight.shape: {tok_emb.weight.shape}")
print(f"  out_head.weight.shape: {out_head.weight.shape}")
print(f"  Are they the same object? {tok_emb.weight is out_head.weight}")
print(f"  Memory addresses:")
print(f"    tok_emb.weight:  {id(tok_emb.weight)}")
print(f"    out_head.weight: {id(out_head.weight)}")

# Count parameters
params_before = tok_emb.weight.numel() + out_head.weight.numel()
print(f"  Total parameters: {params_before:,}")

# Tie the weights (make them share the same matrix)
print(f"\n--- Tying the weights ---")
out_head.weight = tok_emb.weight

print(f"\nAfter weight tying:")
print(f"  tok_emb.weight.shape: {tok_emb.weight.shape}")
print(f"  out_head.weight.shape: {out_head.weight.shape}")
print(f"  Are they the same object? {tok_emb.weight is out_head.weight}")
print(f"  Memory addresses:")
print(f"    tok_emb.weight:  {id(tok_emb.weight)}")
print(f"    out_head.weight: {id(out_head.weight)}")

# Count unique parameters
unique_params = {id(p): p for p in [tok_emb.weight, out_head.weight]}
params_after = sum(p.numel() for p in unique_params.values())
print(f"  Unique parameters: {params_after:,}")

print(f"\n  Parameter savings: {params_before - params_after:,}")

# Demonstrate shared updates
print(f"\n--- Shared Updates ---")
print(f"Before modification:")
print(f"  tok_emb.weight[0, 0] = {tok_emb.weight[0, 0]:.4f}")
print(f"  out_head.weight[0, 0] = {out_head.weight[0, 0]:.4f}")

tok_emb.weight[0, 0] = 99.99

print(f"\nAfter modifying tok_emb.weight[0, 0] = 99.99:")
print(f"  tok_emb.weight[0, 0] = {tok_emb.weight[0, 0]:.4f}")
print(f"  out_head.weight[0, 0] = {out_head.weight[0, 0]:.4f}")
print(f"  ✓ Both changed! They share the same memory!")

print("\n" + "=" * 70)
print("KEY TAKEAWAY:")
print("=" * 70)
print("""
1. PyTorch automatically creates weight matrices when you instantiate layers
2. You can access them via .weight attribute (e.g., model.tok_emb.weight)
3. They're nn.Parameter objects (trainable tensors)
4. Weight tying makes two layers share the SAME weight matrix in memory
5. Updating one automatically updates the other (they're the same object)
""")
