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
    

class CausalAttention(nn.Module):
    """Causal self-attention with masking for autoregressive language models.

    Implements causal (masked) self-attention where each token can only attend to
    previous tokens and itself, preventing information flow from future positions.
    This is essential for autoregressive models like GPT that generate text
    one token at a time.

    Key differences from SelfAttention_v2:
    - Adds causal masking to prevent attending to future tokens
    - Includes dropout for regularization during training
    - Uses register_buffer for the mask (not a learnable parameter)
    - Handles variable sequence lengths up to context_length

    Args:
        d_in: Input embedding dimension
        d_out: Output dimension for queries, keys, and values
        context_length: Maximum sequence length (size of causal mask)
        dropout: Dropout probability for attention weights (0.0 to 1.0)
        qkv_bias: Whether to include bias in Q, K, V projections (default: False)

    Example:
        >>> attention = CausalAttention(d_in=256, d_out=64,
        ...                             context_length=1024, dropout=0.1)
        >>> x = torch.randn(8, 50, 256)  # [batch, seq_len, d_in]
        >>> output = attention(x)  # [8, 50, 64]
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """Initialize causal self-attention with masking and dropout.

        Creates the Q, K, V projection layers and registers a causal mask buffer.
        The mask is registered as a buffer (not a parameter) so it:
        - Moves to GPU/CPU with the model
        - Is saved in model state_dict
        - Does NOT receive gradient updates

        Args:
            d_in: Dimension of input embeddings
            d_out: Dimension of output embeddings (Q, K, V projection size)
            context_length: Maximum sequence length for the causal mask
            dropout: Dropout rate (e.g., 0.1 = 10% dropout)
            qkv_bias: If True, add learnable bias to Q, K, V projections
        """
        super().__init__()
        self.d_out = d_out

        # Q, K, V projection layers (same as SelfAttention_v2)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout layer for regularization
        # Applied to attention weights to randomly zero out some connections
        self.dropout = nn.Dropout(dropout)

        # Register causal mask as a buffer (not a trainable parameter)
        # torch.triu creates upper triangular matrix with 1s above diagonal
        # This marks future positions that should be masked out
        self.register_buffer(
            'mask',  # Name of the buffer
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)  # diagonal=1: keep diagonal as 0 (can attend to self)
        )

    def forward(self, x):
        """Compute causal self-attention with masking and dropout.

        Steps:
            1. Project inputs to Q, K, V
            2. Compute attention scores (Q @ K^T)
            3. Apply causal mask: set future positions to -inf
            4. Scale and apply softmax to get attention weights
            5. Apply dropout for regularization
            6. Compute weighted sum of values

        Args:
            x: Input tensor of shape [batch_size, num_tokens, d_in]

        Returns:
            Context vectors of shape [batch_size, num_tokens, d_out]
            Each output token contains information ONLY from current and
            previous tokens, never from future tokens.
        """
        # Extract batch size and sequence length
        b, num_tokens, d_in = x.shape

        # Project inputs to queries, keys, and values
        # Shape: [batch_size, num_tokens, d_out]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Compute attention scores: Q @ K^T
        # transpose(1,2) swaps the sequence and feature dimensions for batched matmul
        # Shape: [batch_size, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(1, 2)

        # Apply causal mask: set future positions to -inf
        # self.mask[:num_tokens, :num_tokens] handles variable sequence lengths
        # masked_fill_ modifies tensor in-place for efficiency
        # -torch.inf ensures softmax gives these positions weight of 0
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        # Scale by sqrt(d_out) for numerical stability, then apply softmax
        # Softmax converts -inf to 0, so future positions have zero attention
        # dim=-1: normalize across last dimension (each row sums to 1)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        # Apply dropout to attention weights
        # During training: randomly zeros some attention connections
        # During eval: does nothing (dropout is automatically disabled)
        attn_weights = self.dropout(attn_weights)

        # Compute weighted sum of values using attention weights
        # Shape: [batch_size, num_tokens, d_out]
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """Simple multi-head attention using multiple independent attention heads.

    This is a straightforward implementation that creates multiple CausalAttention
    heads, processes the input through each head independently, and concatenates
    the outputs. Each head learns different attention patterns.

    Note: This is a simplified wrapper for learning purposes. Production transformers
    typically use a more efficient implementation (MultiHeadAttention) that splits
    a single large projection into multiple heads rather than using separate heads.

    How it works:
        1. Creates num_heads independent CausalAttention modules
        2. Each head processes the same input with different learned weights
        3. Concatenates all head outputs along the feature dimension
        4. Output dimension = num_heads * d_out

    Args:
        d_in: Input embedding dimension
        d_out: Output dimension for EACH head (not total output size)
        context_length: Maximum sequence length for causal masking
        dropout: Dropout probability for attention weights
        num_heads: Number of independent attention heads
        qkv_bias: Whether to include bias in Q, K, V projections

    Example:
        >>> # 2 heads, each outputs 64 features -> total output is 128 features
        >>> mha = MultiHeadAttentionWrapper(d_in=512, d_out=64,
        ...                                 context_length=1024, dropout=0.1,
        ...                                 num_heads=2)
        >>> x = torch.randn(8, 50, 512)  # [batch, seq_len, d_in]
        >>> output = mha(x)  # [8, 50, 128] = 2 heads * 64 features each
    """

    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        """Initialize multi-head attention with independent heads.

        Creates num_heads separate CausalAttention modules, each with its own
        learnable parameters. nn.ModuleList is used so PyTorch properly tracks
        all the parameters and moves them to GPU when needed.

        Args:
            d_in: Dimension of input embeddings
            d_out: Dimension of output for EACH head (total output = num_heads * d_out)
            context_length: Maximum sequence length for causal masking
            dropout: Dropout rate for attention weights
            num_heads: Number of parallel attention heads
            qkv_bias: If True, add learnable bias to Q, K, V projections
        """
        super().__init__()

        # Create a list of independent attention heads
        # nn.ModuleList registers all modules so their parameters are tracked
        # Each head has its own W_query, W_key, W_value weights
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        """Process input through all heads and concatenate outputs.

        Each head independently:
        - Takes the same input x
        - Uses its own learned weights (different from other heads)
        - Produces output of shape [batch_size, seq_len, d_out]

        All outputs are concatenated along the last dimension (features).

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_in]

        Returns:
            Concatenated outputs of shape [batch_size, seq_len, num_heads * d_out]

        Example:
            If num_heads=2 and d_out=64:
            - head[0](x) outputs [batch, seq_len, 64]
            - head[1](x) outputs [batch, seq_len, 64]
            - concatenated result: [batch, seq_len, 128]
        """
        # Process input through each head and concatenate results
        # [head(x) for head in self.heads] creates a list of outputs
        # torch.cat(..., dim=-1) concatenates along the feature dimension
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
    
class MultiHeadAttention(nn.Module):
    """Efficient multi-head attention implementation used in production transformers.

    This is the standard implementation used in models like GPT and BERT. Instead of
    creating separate attention modules for each head (like MultiHeadAttentionWrapper),
    this implementation:
    1. Projects input to a large Q/K/V space (size d_out)
    2. Splits the projection into num_heads smaller subspaces
    3. Computes attention for all heads in parallel
    4. Combines heads with an output projection

    This approach is more memory and compute efficient than using separate modules.

    Key differences from MultiHeadAttentionWrapper:
    - Single large Q/K/V projection (d_in -> d_out) instead of num_heads small ones
    - Splits d_out into num_heads of size head_dim each
    - Processes all heads in parallel using tensor reshaping
    - Adds output projection layer to combine heads
    - Much more memory and compute efficient

    Args:
        d_in: Input embedding dimension
        d_out: Total output dimension (must be divisible by num_heads)
        context_length: Maximum sequence length for causal masking
        dropout: Dropout probability for attention weights
        num_heads: Number of attention heads
        qkv_bias: Whether to include bias in Q, K, V projections

    Example:
        >>> # 8 heads * 64 features per head = 512 total output dimension
        >>> mha = MultiHeadAttention(d_in=512, d_out=512,
        ...                          context_length=1024, dropout=0.1,
        ...                          num_heads=8)
        >>> x = torch.randn(4, 50, 512)  # [batch, seq_len, d_in]
        >>> output = mha(x)  # [4, 50, 512]
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """Initialize efficient multi-head attention.

        Creates single large Q/K/V projections that will be split into multiple heads,
        plus an output projection to combine the heads after attention.

        Args:
            d_in: Dimension of input embeddings
            d_out: Total output dimension (split across all heads)
            context_length: Maximum sequence length for causal masking
            dropout: Dropout rate for attention weights
            num_heads: Number of parallel attention heads
            qkv_bias: If True, add learnable bias to Q, K, V projections

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()

        # Ensure d_out can be evenly split across heads
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Each head processes a slice of size head_dim from the d_out projection
        self.head_dim = d_out // num_heads  # e.g., 512 / 8 = 64 per head

        # Single large projections (more efficient than separate projections per head)
        # These project d_in -> d_out, then we'll split d_out into num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection to combine information from all heads
        # This is crucial for allowing heads to interact and share information
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        # Register causal mask as a buffer (same as CausalAttention)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        """Compute multi-head attention efficiently.

        Steps:
            1. Project input to Q, K, V (shape: [batch, seq_len, d_out])
            2. Reshape to split into heads (shape: [batch, seq_len, num_heads, head_dim])
            3. Transpose for parallel processing (shape: [batch, num_heads, seq_len, head_dim])
            4. Compute attention scores for all heads in parallel
            5. Apply causal mask and softmax
            6. Apply dropout and compute context vectors
            7. Reshape to merge heads back together
            8. Apply output projection

        Args:
            x: Input tensor of shape [batch_size, num_tokens, d_in]

        Returns:
            Context vectors of shape [batch_size, num_tokens, d_out]
            with information from all attention heads combined
        """
        # Extract dimensions
        b, num_tokens, d_in = x.shape

        # Step 1: Project inputs to queries, keys, and values
        # Shape: [batch_size, num_tokens, d_out]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Step 2: Reshape to split d_out into (num_heads, head_dim)
        # Before view(): self.W_keys(x) has shape [b, num_tokens, d_out]
        # view() creates the multiple "heads" by reshaping the last dimension
        # Shape: [batch_size, num_tokens, num_heads, head_dim]
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Step 3: Transpose to bring heads dimension before sequence dimension
        # This allows us to process all heads in parallel using batch operations
        # Shape: [batch_size, num_heads, num_tokens, head_dim]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Step 4: Compute attention scores for all heads in parallel
        # keys.transpose(2,3) swaps num_tokens and head_dim dimensions
        # Shape: [batch_size, num_heads, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(2, 3)

        # Step 5: Apply causal mask to prevent attending to future tokens
        # Slice mask to match current sequence length
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Step 6: Scale and apply softmax to get attention weights
        # Scaling by sqrt(head_dim) for numerical stability
        # Shape: [batch_size, num_heads, num_tokens, num_tokens]
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # Step 7: Compute context vectors and transpose back
        # (attn_weights @ values): [batch, num_heads, num_tokens, head_dim]
        # After transpose: [batch, num_tokens, num_heads, head_dim]
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Step 8: Merge all heads back together
        # contiguous() ensures tensor is stored contiguously in memory (required for view)
        # view() reshapes from [batch, num_tokens, num_heads, head_dim]
        #              back to [batch, num_tokens, d_out]
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Step 9: Apply output projection to combine information from all heads
        # This allows the heads to interact and integrate their different perspectives
        context_vec = self.out_proj(context_vec)
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
    
    #Causal Attention v1
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    print(attn_weights)
    
    print(attn_scores.shape)
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)
    masked_simple = attn_weights * mask_simple
    print(masked_simple)
    #renormalize weights to sum to 1 across columns
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print(masked_simple_norm)
    
    
    #Causal attention v2
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    print(mask)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)
    attn_weights = torch.softmax(masked /keys.shape[-1]**0.5, dim=1)
    print(attn_weights)
    
    #Masking Attention Weights with dropout
    #Dropout is a deep learning techinique where randomly selected hidden layer units are ignored during training
    
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5) #Dropout rate of 50%
    example = torch.ones(6,6)
    print(dropout (example))
    
    torch.manual_seed(123)
    print(dropout(attn_weights))

    #Duplicate inputs to to simulate multiple batches
    batch = torch.stack((inputs,inputs), dim=0)
    print(batch.shape)
    print(batch)
    
    
    #Use CausalAttention class
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)
    
    #MultiHeadAttentionWrapper
    torch.manual_seed(123)
    context_length = batch.shape[1] #number of tokens
    d_in, d_out = 3,1
    mha = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, 0.0, num_heads=2
    )    
    context_vecs = mha(batch)
    print (batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    
    #MultiheadAttention
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_length = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    
        
    #Exercise 3.3: count size of multi-head attention module 
    torch.manual_seed(123)
    context_length = 1024
    d_in, d_out = 768, 768
    num_heads = 12

    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(mha))