import torch
import torch.nn as nn
import tiktoken

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
            

layer_sizes = [3, 3, 3, 3, 3, 1]  

sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
#Vanishing gradients as we progress to each layer
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
print_gradients(model_without_shortcut, sample_input)



torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)