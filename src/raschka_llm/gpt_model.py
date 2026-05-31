"""
GPT Model Implementation

ENVIRONMENT SETUP REMINDER:
---------------------------
If you get "ModuleNotFoundError: No module named 'torch'":

1. Check which Python you're using:
   which python

2. If it shows .venv/bin/python, install dependencies:
   pip install -r requirements.txt

3. If VSCode isn't using .venv:
   - Cmd+Shift+P → "Python: Select Interpreter"
   - Choose: .venv/bin/python

4. Verify installation:
   python -c "import torch; print(torch.__version__)"

The .venv (virtual environment) keeps this project's dependencies isolated.

MEMORY TIP:
-----------
To remember what total_params means:
- total_params = sum(p.numel() for p in model.parameters())
- Counts every trainable number (weight/bias) in the model
- For FeedForward (768→3072→768):
  * Layer 1: (3072×768) weights + 3072 bias = 2,362,368 params
  * Layer 2: (768×3072) weights + 768 bias = 2,360,064 params
  * Total: 4,722,432 parameters in ONE feedforward block
- GPT-124M has 12 blocks × 4.7M ≈ 56M params just in feedforward!
- Each param is 4 bytes (float32) → multiply by 4 to get MB
"""

import torch
import torch.nn as nn
import tiktoken

# Import from same package - works when run directly or as module
try:
    from .self_attention import MultiHeadAttention
except ImportError:
    # Fallback for running file directly
    from self_attention import MultiHeadAttention

#This configuration has 124 million parameters - it matches GPT-2's smallest model
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size (number of unique tokens)
    "context_length": 1024, # Context length (can process up to 1024 tokens at once)
    "emb_dim": 768,         # Embedding dimension (each token becomes 768 dimensional vector)
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        #Token Embedding Layer: Converts token IDs into 768-dim vectors
        #Input: token ID (integer from 0 to 50,256)
        #Output: 768-dimensional vector
        #
        #nn.Embedding(50257, 768) internally creates:
        #  - weight matrix: shape (50257, 768) - learnable lookup table
        #  - Each row is the embedding for one vocabulary token
        #  - Forward pass: looks up the row corresponding to each token ID
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        #Positional Embedding Layer: Adds position information
        #Input: position (0, 1, 2, ..., up to 1023)
        #Output: 768-dimensional vector
        #
        #nn.Embedding(1024, 768) internally creates:
        #  - weight matrix: shape (1024, 768) - learnable lookup table
        #  - Each position (0 to 1023) gets its own learnable 768-dim vector
        #  - Allows model to learn that position matters (word order)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        #Dropout Layer: Randomly zeros out 10% of values during training
        #Helps prevent overfitting
        #Only active during training, turned off during inference
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        #Transformer Blocks Stack: Creates 12 transformer blocks and chains them
        #Currently "Dummy" - they just pass data through unchanged
        #nn.Sequential chains the 12 transformer blocks: output of block 1 -> input to block 2
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )

        #Final Layer Normalization: Normalizes the output before prediction
        #Stabilizes the final layer's input values
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        #Output Head: Maps 768-dim embeddings back to vocabulary probabilities
        #Input: 768-dimensional vector
        #Output: 50,257-dimensional vector (one score per vocabulary token)
        #No bias term (bias=False) following GPT-2 architecture
        #
        #nn.Linear(768, 50257) internally creates:
        #  - weight matrix: shape (50257, 768) - one row per vocab token
        #  - During forward pass: output = input @ weight.T
        #  - Result: (batch, seq, 768) @ (768, 50257) = (batch, seq, 50257)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward (self, in_idx):
        """
        Forward pass through the GPT model.

        Args:
            in_idx: Input tensor of token IDs, shape (batch_size, seq_len)
                   Example: [[15496, 11, 616], [314, 1842, 345]] for batch_size=2, seq_len=3

        Returns:
            logits: Predicted scores for each vocab token, shape (batch_size, seq_len, vocab_size)
                   Each position predicts the next token
        """
        # Extract dimensions from input
        batch_size, seq_len = in_idx.shape

        # Convert token IDs to embeddings: (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)

        # Create position embeddings for each position in the sequence
        # torch.arange creates [0, 1, 2, ..., seq_len-1] on the same device as input
        # Result shape: (seq_len, emb_dim)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # Combine token and position embeddings (broadcasting adds pos_embeds to each batch item)
        # Shape: (batch_size, seq_len, emb_dim)
        x = tok_embeds + pos_embeds

        # Apply dropout for regularization (only during training)
        x = self.drop_emb(x)

        # Pass through all 12 transformer blocks sequentially
        x = self.trf_blocks(x)

        # Apply final layer normalization
        x = self.final_norm(x)

        # Project to vocabulary size to get logits (unnormalized probabilities)
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.out_head(x)

        return logits
    
class DummyTransformerBlock(nn.Module):
    """
    Placeholder transformer block that doesn't modify the input.

    In a real GPT model, each transformer block contains:
    1. Multi-head self-attention mechanism (learns relationships between tokens)
    2. Feed-forward neural network (processes each position independently)
    3. Layer normalization (before each sub-layer)
    4. Residual connections (adds input to output of each sub-layer)

    Currently: Just passes data through unchanged for skeleton structure.
    """
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        # TODO: Implement actual transformer block with attention and feed-forward layers
        return x
    
class DummyLayerNorm(nn.Module):
    """
    Placeholder layer normalization that doesn't modify the input.

    Real layer normalization:
    - Normalizes values across the embedding dimension for each token
    - Formula: (x - mean) / sqrt(variance + eps)
    - Then scales and shifts with learnable parameters (gamma, beta)
    - Helps stabilize training and allows deeper networks

    Args:
        normalized_shape: The dimension to normalize over (emb_dim = 768)
        eps: Small constant for numerical stability (default: 1e-5)

    Currently: Just passes data through unchanged for skeleton structure.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        # TODO: Implement actual layer normalization
        return x


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print (batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape: ", logits.shape)
print(logits)

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normaized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)


class LayerNorm(nn.Module):
    """
    Layer Normalization with learnable scale and shift parameters.

    Scale and shift:
    - In addition to performing normalization (subtracting mean and dividing by variance),
      we add two trainable parameters: `scale` and `shift`
    - The initial `scale` (multiplying by 1) and `shift` (adding 0) values don't have
      any effect on the normalized output initially
    - However, `scale` and `shift` are trainable parameters that the LLM automatically
      adjusts during training if it is determined that doing so would improve the model's
      performance on its training task
    - This allows the model to learn appropriate scaling and shifting that best suit
      the data it is processing

    Epsilon (eps):
    - We add a small value (`eps`) before computing the square root of the variance
    - This is to avoid division-by-zero errors if the variance is 0
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # Small constant for numerical stability
        self.scale = nn.Parameter(torch.ones(emb_dim))   # Learnable scale (gamma)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Learnable shift (beta)
        
    def forward(self, x):
        # Calculate mean and variance across the embedding dimension (last dimension)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize: Subtracting the mean and dividing by the square-root of the variance
        # (standard deviation) centers the inputs to have a mean of 0 and a variance of 1
        # across the column (feature) dimension
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift: Allow the model to learn optimal mean and variance
        return self.scale * norm_x + self.shift
    
    
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)

mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)  # Must match LayerNorm's calculation!

print("*Mean:\n", mean)
print("*Variance:\n", var)


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    GELU is a smooth, non-linear activation function used in GPT models.
    It's similar to ReLU but smoother, allowing small negative values through.

    Formula: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    Why GELU?
    - Smoother than ReLU (has gradients everywhere)
    - Performs better than ReLU in transformer models
    - Used in GPT-2, GPT-3, BERT
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU approximation using tanh
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        
class FeedForward(nn.Module):
    """
    Feed-Forward Network used in transformer blocks.

    Architecture:
    1. Linear layer: Expands from emb_dim (768) to 4*emb_dim (3072)
    2. GELU activation: Non-linear transformation
    3. Linear layer: Projects back down to emb_dim (768)

    This expansion and contraction allows the model to learn complex patterns.

    Flow example with GPT-2 config:
    Input:  (batch, seq_len, 768)
       ↓ Linear (768 → 3072)
    (batch, seq_len, 3072)  ← 4x wider, more capacity
       ↓ GELU activation
    (batch, seq_len, 3072)  ← Non-linear transformation
       ↓ Linear (3072 → 768)
    Output: (batch, seq_len, 768)  ← Back to original size
    """
    def __init__(self, cfg):
        super().__init__()
        # Sequential: chains layers together (output of layer1 → input to layer2)
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expand: 768 → 3072
            GELU(),                                           # Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # Contract: 3072 → 768
        )

    def forward(self, x):
        # Simply pass through the sequential layers
        # Calling self.layers(x) automatically calls forward() on nn.Sequential
        return self.layers(x)
    
# ============================================================================
# Testing FeedForward Network
# ============================================================================

# Create a FeedForward instance with GPT-2 124M config
ffn = FeedForward(GPT_CONFIG_124M)

# Create sample input: [batch_size, num_tokens, emb_size]
# - batch_size=2: processing 2 sequences at once
# - num_tokens=3: each sequence has 3 tokens
# - emb_size=768: each token is represented as a 768-dimensional vector
x = torch.rand(2, 3, 768)

# Pass through feed-forward network
# ffn(x) automatically calls ffn.forward(x) via __call__
# The network expands to 3072, applies GELU, then contracts back to 768
out = ffn(x)

print("FeedForward output shape:", out.shape)
# Expected: torch.Size([2, 3, 768]) - same shape as input!


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        
        #Implement 5 layers
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
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
            

class TransformerBlock(nn.Module):
    """
    A single Transformer Block - the core building block of GPT.

    GPT-2 stacks 12 of these blocks (GPT_CONFIG_124M has n_layers=12).
    Each block performs two main operations in sequence:
    1. Multi-head self-attention (learns relationships between tokens)
    2. Feed-forward network (processes each token independently)

    Both operations use:
    - Layer normalization (applied BEFORE each sub-layer)
    - Residual connections (adds input to output)
    - Dropout (for regularization)

    Architecture pattern (Pre-LN Transformer):
        Input
          ↓
        [LayerNorm → Attention → Dropout] → Add with input (residual)
          ↓
        [LayerNorm → FeedForward → Dropout] → Add with input (residual)
          ↓
        Output
    """
    def __init__(self, cfg):
        super().__init__()

        # Multi-head self-attention: Allows each token to look at all previous tokens
        # Input/Output: (batch, seq_len, emb_dim=768)
        # The attention mechanism learns which tokens are relevant to each other
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],              # Input dimension: 768
            d_out=cfg["emb_dim"],             # Output dimension: 768 (same as input)
            context_length=cfg["context_length"],  # Max sequence length: 1024
            num_heads=cfg["n_heads"],         # Number of attention heads: 12
            dropout=cfg["drop_rate"],         # Dropout rate: 0.1
            qkv_bias=cfg["qkv_bias"]         # Whether to use bias in Q,K,V projections: False
        )

        # Feed-forward network: Processes each token position independently
        # Expands to 4x dimension (768→3072) then contracts back (3072→768)
        self.ff = FeedForward(cfg)

        # Layer normalization before attention
        # Normalizes across the embedding dimension for stable training
        self.norm1 = LayerNorm(cfg["emb_dim"])

        # Layer normalization before feed-forward
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout applied to residual connections
        # Same dropout layer used for both attention and feed-forward paths
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Forward pass through one transformer block.

        Args:
            x: Input tensor, shape (batch_size, seq_len, emb_dim)
               Example: (2, 4, 768) for batch_size=2, seq_len=4

        Returns:
            Output tensor, shape (batch_size, seq_len, emb_dim)
            Same shape as input - transformer blocks preserve dimensions
        """
        # ========================================================================
        # First sub-layer: Multi-head self-attention with residual connection
        # ========================================================================

        # Save the input for the residual connection
        # Residual connections help gradients flow during backpropagation
        shortcut = x

        # Apply layer normalization BEFORE attention (Pre-LN architecture)
        # This stabilizes training and is the modern approach
        x = self.norm1(x)

        # Apply multi-head self-attention
        # Each token attends to all previous tokens (causal masking applied inside)
        x = self.att(x)

        # Apply dropout for regularization
        x = self.drop_shortcut(x)

        # Add the residual connection: output = attention_output + original_input
        # This allows the model to learn incremental refinements
        x = x + shortcut  # Residual connection 1

        # ========================================================================
        # Second sub-layer: Feed-forward network with residual connection
        # ========================================================================

        # Save the output from attention layer for the second residual connection
        shortcut = x

        # Apply layer normalization BEFORE feed-forward
        x = self.norm2(x)

        # Apply feed-forward network: 768 → 3072 → 768
        # Processes each token independently with non-linear transformations
        x = self.ff(x)

        # Apply dropout for regularization
        x = self.drop_shortcut(x)

        # Add the second residual connection: output = ff_output + attention_output
        x = x + shortcut  # Residual connection 2

        return x


class GPTModel(nn.Module):
    """
    Full GPT (Generative Pre-trained Transformer) Model.

    This is a complete, functional implementation matching GPT-2 architecture.
    The model transforms token IDs into probability distributions over the vocabulary
    to predict the next token in a sequence.

    Architecture Overview:
        Token IDs (integers)
          ↓
        Token Embeddings + Position Embeddings (learned vectors)
          ↓
        Dropout (regularization)
          ↓
        12 Transformer Blocks (attention + feed-forward layers)
          ↓
        Final Layer Normalization
          ↓
        Output Projection Head (to vocabulary size)
          ↓
        Logits (scores for each vocabulary token)

    With GPT_CONFIG_124M, this creates a 124 million parameter model.

    Args:
        cfg: Configuration dictionary with keys:
            - vocab_size: Number of tokens in vocabulary (50,257)
            - context_length: Maximum sequence length (1024)
            - emb_dim: Embedding dimension (768)
            - n_heads: Number of attention heads (12)
            - n_layers: Number of transformer blocks (12)
            - drop_rate: Dropout probability (0.1)
            - qkv_bias: Whether to use bias in attention (False)
    """
    def __init__(self, cfg):
        super().__init__()

        # Token Embedding: Converts token IDs to dense vectors
        # Shape: (vocab_size=50257, emb_dim=768)
        # Each of the 50,257 possible tokens gets its own learnable 768-dim vector
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Position Embedding: Adds position information to each token
        # Shape: (context_length=1024, emb_dim=768)
        # Each position (0 to 1023) gets its own learnable 768-dim vector
        # This helps the model understand token order and relative positions
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Dropout for embeddings: Prevents overfitting
        # Randomly zeros 10% of embedding values during training
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of 12 Transformer Blocks
        # Each block contains multi-head attention and feed-forward layers
        # nn.Sequential chains them: output of block[i] → input of block[i+1]
        # The * unpacks the list of blocks created by the list comprehension
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Final Layer Normalization
        # Applied after all transformer blocks, before the output projection
        # Stabilizes the values going into the final linear layer
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Output Projection Head: Maps embeddings back to vocabulary logits
        # Input: (batch, seq_len, emb_dim=768)
        # Output: (batch, seq_len, vocab_size=50257)
        # No bias term (bias=False) following GPT-2 design
        # Each position outputs scores for all 50,257 possible next tokens
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        """
        Forward pass: Convert token IDs to next-token prediction logits.

        Args:
            in_idx: Input token IDs, shape (batch_size, seq_len)
                   Example: [[15496, 11, 616, 13]] for batch_size=1, seq_len=4
                   Each integer is a token ID from 0 to 50,256

        Returns:
            logits: Unnormalized prediction scores, shape (batch_size, seq_len, vocab_size)
                   Example shape: (2, 4, 50257)
                   - For each position, model outputs 50,257 scores (one per vocab token)
                   - Higher score = model thinks that token is more likely to come next
                   - Apply softmax to convert to probabilities: softmax(logits, dim=-1)
        """
        # Extract dimensions from input shape
        batch_size, seq_len = in_idx.shape

        # Convert token IDs to embeddings
        # Input: (batch_size, seq_len) with integer token IDs
        # Output: (batch_size, seq_len, emb_dim=768)
        tok_embeds = self.tok_emb(in_idx)

        # Create position embeddings for the sequence
        # torch.arange(seq_len) creates [0, 1, 2, ..., seq_len-1]
        # device=in_idx.device ensures embeddings are on same device (CPU/GPU)
        # Output: (seq_len, emb_dim=768)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # Combine token and position embeddings
        # Broadcasting: pos_embeds (seq_len, 768) is added to each batch item
        # Result: (batch_size, seq_len, emb_dim=768)
        # Now each token has both its semantic meaning AND positional information
        x = tok_embeds + pos_embeds

        # Apply dropout to combined embeddings (only active during training)
        x = self.drop_emb(x)

        # Pass through all 12 transformer blocks sequentially
        # Each block refines the representations using attention and feed-forward layers
        # Shape remains: (batch_size, seq_len, emb_dim=768)
        x = self.trf_blocks(x)

        # Apply final layer normalization
        # Normalizes the output from the transformer blocks
        # Shape unchanged: (batch_size, seq_len, emb_dim=768)
        x = self.final_norm(x)

        # Project to vocabulary size to get logits (unnormalized probabilities)
        # Linear transformation: (batch, seq_len, 768) @ (768, 50257) = (batch, seq_len, 50257)
        # For each token position, we get a score for every possible next token
        logits = self.out_head(x)

        return logits
    
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate new tokens one at a time using the GPT model.

    This function implements autoregressive text generation:
    - Start with some initial tokens (prompt)
    - Predict the next token
    - Add it to the sequence
    - Repeat

    Args:
        model: The GPT model to use for generation
        idx: Starting token IDs, shape (batch_size, seq_len)
             Example: tensor([[15496, 11, 616]]) = "Hello, I"
        max_new_tokens: How many new tokens to generate
        context_size: Maximum context length the model can handle (e.g., 1024)

    Returns:
        idx: Extended sequence with generated tokens, shape (batch_size, seq_len + max_new_tokens)

    INDEXING EXPLAINED:
    -------------------
    idx[:, -context_size:]  → Takes last N tokens
        : = all batches
        -context_size: = last 'context_size' tokens
        Example: if idx has 2000 tokens but context_size=1024,
                 this takes tokens [976:2000] (the last 1024)

    logits[:, -1, :]  → Gets predictions for the LAST token position
        : = all batches
        -1 = last position in sequence
        : = all vocabulary scores (50,257 values)
        Example shape: (batch_size, vocab_size) = (2, 50257)

    torch.argmax(..., dim=-1, keepdim=True)  → Picks token with highest score
        dim=-1 = across vocabulary dimension
        keepdim=True = keep as shape (batch, 1) instead of (batch,)

    torch.cat((idx, idx_next), dim=1)  → Append new token to sequence
        dim=1 = concatenate along sequence dimension
        idx: (batch, seq_len) + idx_next: (batch, 1) → (batch, seq_len+1)
    """
    # Generate max_new_tokens tokens, one at a time
    for _ in range(max_new_tokens):

        # STEP 1: Crop context if sequence is too long
        # idx[:, -context_size:] means "take last context_size tokens from all batches"
        #
        # Why? The model can only handle up to context_size tokens (e.g., 1024)
        # If we've generated 2000 tokens, we can only use the last 1024
        #
        # Example:
        #   idx shape: (2, 2000) - 2 sequences, each with 2000 tokens
        #   idx[:, -1024:] → (2, 1024) - last 1024 tokens from each sequence
        idx_cond = idx[:, -context_size:]

        # STEP 2: Get model predictions (no gradient needed for generation)
        # with torch.no_grad() saves memory - we're not training, just inferring
        with torch.no_grad():
            # logits shape: (batch_size, seq_len, vocab_size)
            # Example: (2, 1024, 50257) - scores for next token at each position
            logits = model(idx_cond)

        # STEP 3: Extract predictions for the LAST position only
        # logits[:, -1, :] means "from all batches, take the last position, all vocab scores"
        #
        # Why -1? The model predicts the NEXT token after each position.
        # We only care about what comes after the last token.
        #
        # Before: (batch, seq_len, vocab_size) = (2, 1024, 50257)
        # After:  (batch, vocab_size) = (2, 50257)
        #
        # Indexing breakdown:
        #   [:, -1, :]
        #   │   │   └── all 50,257 vocabulary scores
        #   │   └────── last position in sequence (position 1023 if seq_len=1024)
        #   └────────── all batches
        logits = logits[:, -1, :]

        # STEP 4: Convert logits to probabilities using softmax
        # Softmax turns raw scores into probabilities that sum to 1.0
        # dim=-1 means normalize across the vocabulary dimension
        # Shape stays: (batch_size, vocab_size) = (2, 50257)
        probas = torch.softmax(logits, dim=-1)

        # STEP 5: Pick the token with highest probability (greedy decoding)
        # torch.argmax finds the index of the maximum value
        #
        # dim=-1: find max across vocabulary dimension
        # keepdim=True: keep shape as (batch, 1) instead of (batch,)
        #
        # Example:
        #   probas = [[0.01, 0.02, 0.95, 0.02], [0.3, 0.6, 0.05, 0.05]]
        #   argmax →  [[2], [1]]  ← indices of highest probabilities
        #
        # Shape: (batch_size, 1) = (2, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # STEP 6: Append the new token to the sequence
        # torch.cat concatenates tensors along a dimension
        # dim=1 means concatenate along the sequence dimension
        #
        # Before:
        #   idx:      (batch, seq_len)   = (2, 4)  = [[15496, 11, 616, 13], ...]
        #   idx_next: (batch, 1)         = (2, 1)  = [[345], [42]]
        # After:
        #   idx:      (batch, seq_len+1) = (2, 5)  = [[15496, 11, 616, 13, 345], ...]
        #
        # Now the sequence is 1 token longer!
        idx = torch.cat((idx, idx_next), dim=1)

    # After the loop, we've added max_new_tokens to the original sequence
    # Return the full sequence including both original and generated tokens
    return idx


# Test code - only runs when file is executed directly
if __name__ == "__main__":
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])

    torch.manual_seed(123)
    # Vanishing gradients as we progress to each layer
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )
    print_gradients(model_without_shortcut, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True
    )
    print_gradients(model_with_shortcut, sample_input)
    
    #Test the GPT Model
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape: ", out.shape)
    print(out)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
    
    total_params_gpt2 = (
        total_params - sum(p.numel()
                           for p in model.out_head.parameters())
        
    )
    print(f"Number of trainable parameters "
          f"considering weight tying: {total_params_gpt2:,}")
    
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model in bytes: {total_size_mb:.2f} MB")
    
    #Test the generate_text_simple function
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    #Add batch dimension at position 0 since GPT model expects input shape (batch_size, seq_len)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
    print("encoded_tensor.shape:", encoded_tensor.shape)
    
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)