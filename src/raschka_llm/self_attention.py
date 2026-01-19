"""Self-attention mechanism implementation for transformer models."""
import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    """Self-attention mechanism for transformers.

    Computes context-aware representations by allowing each token to attend to
    all other tokens in the sequence. Uses Query, Key, and Value projections
    to compute attention weights and aggregate information.

    Args:
        d_in: Input embedding dimension
        d_out: Output dimension for queries, keys, and values

    Example:
        >>> attention = SelfAttention_v1(d_in=256, d_out=64)
        >>> x = torch.randn(8, 4, 256)  # [batch, seq_len, d_in]
        >>> output = attention(x)  # [8, 4, 64]
    """

    def __init__(self, d_in, d_out):
        """Initialize the self-attention layer.

        Creates three learnable weight matrices for Query, Key, and Value projections.
        These matrices are randomly initialized and registered as model parameters.

        Args:
            d_in: Dimension of input embeddings
            d_out: Dimension of output embeddings (Q, K, V projection size)
        """
        super().__init__()

        # nn.Parameter wraps the tensor of random values and tells PyTorch that it is a learnable
        # parameter so PyTorch tracks gradients during backpropagation. These tensors get registered
        # as part of the model's parameters.
        self.W_query  = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key    = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value  = nn.Parameter(torch.rand(d_in, d_out))


    def forward(self, x):
        """Compute self-attention for input embeddings.

        Steps:
            1. Project inputs to queries, keys, and values using learned weights
            2. Compute attention scores via dot product of queries and keys
            3. Scale scores by sqrt(d_out) for numerical stability
            4. Apply softmax to get attention weights (probabilities)
            5. Compute weighted sum of values using attention weights

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_in]

        Returns:
            Context vectors of shape [batch_size, seq_len, d_out]
            Each output token contains information from all input tokens,
            weighted by attention scores.
        """
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(
                attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """Improved self-attention using nn.Linear layers.

    This version replaces nn.Parameter matrices with nn.Linear layers, which:
    - Provides better weight initialization (Xavier/Kaiming instead of random)
    - Optionally includes bias terms for more expressiveness
    - Is more consistent with PyTorch conventions
    - Handles batched matrix multiplication more efficiently

    Differences from v1:
    - v1: Uses nn.Parameter(torch.rand(...)) - random initialization [0, 1]
    - v2: Uses nn.Linear(...) - proper initialization (Xavier uniform by default)
    - v2: Can optionally add bias terms to Q, K, V projections

    Args:
        d_in: Input embedding dimension
        d_out: Output dimension for queries, keys, and values
        qkv_bias: Whether to include bias in Q, K, V projections (default: False)
                  Setting to True adds learnable bias vectors to each projection

    Example:
        >>> # Without bias (more common for attention)
        >>> attention = SelfAttention_v2(d_in=256, d_out=64, qkv_bias=False)
        >>> x = torch.randn(8, 4, 256)  # [batch, seq_len, d_in]
        >>> output = attention(x)  # [8, 4, 64]
        >>>
        >>> # With bias (optional, adds more parameters)
        >>> attention_biased = SelfAttention_v2(d_in=256, d_out=64, qkv_bias=True)
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        """Initialize self-attention with nn.Linear layers.

        Creates three linear transformation layers for Q, K, V projections.
        nn.Linear provides:
        - Better weight initialization (Xavier uniform by default)
        - Optional bias terms
        - Efficient batched operations

        Args:
            d_in: Dimension of input embeddings
            d_out: Dimension of output embeddings (Q, K, V projection size)
            qkv_bias: If True, add learnable bias to Q, K, V projections
                      If False (default), no bias terms (more common in transformers)
        """
        super().__init__()

        # nn.Linear(d_in, d_out) creates a learnable linear transformation:
        # - Weight matrix: [d_out, d_in] (transposed compared to nn.Parameter)
        # - Bias vector: [d_out] (if bias=True)
        # - Initialization: Xavier uniform for weights, zeros for bias
        # - Forward: output = input @ weight.T + bias
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """Compute self-attention using nn.Linear projections.

        Identical computation to v1, but uses nn.Linear instead of manual
        matrix multiplication. nn.Linear automatically handles:
        - Proper weight transpose: x @ W.T
        - Optional bias addition: x @ W.T + b
        - Batched operations across batch dimension

        Steps:
            1. Project inputs to Q, K, V using nn.Linear (handles transpose internally)
            2. Compute attention scores via dot product of queries and keys
            3. Scale scores by sqrt(d_out) for numerical stability
            4. Apply softmax to get attention weights (probabilities)
            5. Compute weighted sum of values using attention weights

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_in]

        Returns:
            Context vectors of shape [batch_size, seq_len, d_out]
            Each output token contains information from all input tokens,
            weighted by attention scores.

        Note:
            The only difference from v1 is using self.W_query(x) instead of
            x @ self.W_query. Both produce the same result, but nn.Linear
            provides better initialization and optional bias.
        """
        # nn.Linear automatically computes x @ weight.T (+ bias if enabled)
        keys = self.W_key(x)        # [batch, seq_len, d_out]
        queries = self.W_query(x)   # [batch, seq_len, d_out]
        values = self.W_value(x)    # [batch, seq_len, d_out]

        # Rest is identical to v1
        attn_scores = queries @ keys.T  # [batch, seq_len, seq_len]
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values  # [batch, seq_len, d_out]
        return context_vec
    
    

if __name__ == "__main__":
    # Example usage
    inputs = torch.tensor(
      [[0.43, 0.15, 0.89],  # Your     (x^1)
       [0.55, 0.87, 0.66],  # journey  (x^2)
       [0.57, 0.85, 0.64],  # starts   (x^3)
       [0.22, 0.58, 0.33],  # with     (x^4)
       [0.77, 0.25, 0.10],  # one      (x^5)
       [0.05, 0.80, 0.55]]  # step     (x^6)
    )

    d_in = inputs.shape[1]  # the input embedding size, d=3
    d_out = 2  # the output embedding size, d=2
    torch.manual_seed(789)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))


    #version 2 usage
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))