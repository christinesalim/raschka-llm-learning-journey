import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')

from raschka_llm.gpt_model import GPTModel, GPT_CONFIG_124M

# Create model
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

print("=" * 70)
print("WEIGHT MATRICES ARE MANAGED INTERNALLY BY PYTORCH")
print("=" * 70)

# 1. Token Embedding Weight Matrix
print("\n1. TOKEN EMBEDDING WEIGHT MATRIX:")
print(f"   Location: model.tok_emb.weight")
print(f"   Shape: {model.tok_emb.weight.shape}")
print(f"   Type: {type(model.tok_emb.weight)}")
print(f"   First 3 rows (tokens 0, 1, 2), first 5 dimensions:")
print(model.tok_emb.weight[:3, :5])  # First 3 tokens, first 5 dimensions
print(f"   ...")

# 2. Output Head Weight Matrix
print("\n2. OUTPUT HEAD WEIGHT MATRIX:")
print(f"   Location: model.out_head.weight")
print(f"   Shape: {model.out_head.weight.shape}")
print(f"   Type: {type(model.out_head.weight)}")
print(f"   First 3 rows, first 5 dimensions:")
print(model.out_head.weight[:3, :5])  # First 3 outputs, first 5 dimensions
print(f"   ...")

# 3. Check if they're different matrices (no weight tying yet)
print("\n3. ARE THEY THE SAME MATRIX? (Weight Tying Check)")
print(f"   Same object? {model.tok_emb.weight is model.out_head.weight}")
print(f"   Same values? {torch.equal(model.tok_emb.weight, model.out_head.weight)}")

# 4. How the forward pass uses these weights
print("\n4. HOW FORWARD PASS USES THESE WEIGHTS:")
print("\n   Token Embedding (lookup):")
token_id = torch.tensor([[123]])  # Token ID 123
embedding = model.tok_emb(token_id)
manual_lookup = model.tok_emb.weight[123]
print(f"   model.tok_emb(123) == model.tok_emb.weight[123]? {torch.equal(embedding[0, 0], manual_lookup)}")

print("\n   Output Projection (matrix multiplication):")
x = torch.randn(1, 4, 768)  # Fake transformer output
logits = model.out_head(x)
manual_matmul = x @ model.out_head.weight.T
print(f"   model.out_head(x) == x @ model.out_head.weight.T? {torch.allclose(logits, manual_matmul)}")

# 5. All trainable parameters
print("\n5. ALL TRAINABLE PARAMETERS IN THE MODEL:")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\n   First few parameter tensors:")
for i, (name, param) in enumerate(model.named_parameters()):
    if i < 5:
        print(f"   - {name}: shape {param.shape}")
    elif i == 5:
        print(f"   ... (and {sum(1 for _ in model.named_parameters()) - 5} more)")
        break

# 6. Demonstrate weight tying (what GPT-2 actually does)
print("\n6. WEIGHT TYING DEMONSTRATION (what GPT-2 does):")
print("   Before tying:")
print(f"   - tok_emb.weight is out_head.weight? {model.tok_emb.weight is model.out_head.weight}")
print(f"   - Total params: {sum(p.numel() for p in model.parameters()):,}")

# Tie the weights
model.out_head.weight = model.tok_emb.weight

print("\n   After tying:")
print(f"   - tok_emb.weight is out_head.weight? {model.tok_emb.weight is model.out_head.weight}")
print(f"   - Total params: {sum(p.numel() for p in model.parameters()):,}")
print("\n   Now they share the SAME weight matrix in memory!")

print("\n" + "=" * 70)
