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
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))
