"""
Training utilities for GPT model - Chapter 5

This module contains functions for:
- Converting between text and token IDs
- Calculating loss (batch and loader level)
- Evaluating model performance
- Training loop implementation
- Plotting training progress
"""

import torch
import matplotlib.pyplot as plt

# Import from same package
try:
    from .gpt_model import generate_text_simple
except ImportError:
    from gpt_model import generate_text_simple




# Test code - only runs when file is executed directly
if __name__ == "__main__":
    import tiktoken
    import os
    import requests

    from gpt_model import GPTModel

    # Configuration
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,  # Shortened for faster training
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # Set random seed for reproducibility
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Load GPT-2 tokenizer (50,257 vocabulary size)
    tokenizer = tiktoken.get_encoding("gpt2")

    def text_to_token_ids(text, tokenizer):
        """
        Convert text to token IDs with batch dimension.

        "Hello" -> [15496] -> tensor([[15496]])

        unsqueeze(0) adds batch dimension: shape (seq_len,) -> (1, seq_len)
        """
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    def token_ids_to_text(token_ids, tokenizer):
        """
        Convert token IDs back to text.

        tensor([[15496]]) -> [15496] -> "Hello"

        squeeze(0) removes batch dimension: shape (1, seq_len) -> (seq_len,)
        tolist() converts tensor to Python list for the tokenizer
        """
        flat = token_ids.squeeze(0)
        return tokenizer.decode(flat.tolist())

    # =========================================================================
    # Test text generation with untrained model
    # =========================================================================
    start_context = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # =========================================================================
    # Calculating text generation loss
    # =========================================================================
    #
    # To understand how well the model predicts, we compare:
    # - inputs: what the model sees
    # - targets: what the model should predict (inputs shifted by 1 position)
    #
    # Example: "every effort moves you"
    #   Input:  [every, effort, moves]  -> Model sees these
    #   Target: [effort, moves, you]    -> Model should predict these
    #
    # The target for position i is the token at position i+1
    # =========================================================================

    # Two training examples (batch_size=2), each with 3 tokens (seq_len=3)
    inputs = torch.tensor([[16833, 3626, 6100],  # "every effort moves"
                           [40, 1107, 588]])     # "I really like"

    # Targets are shifted by 1: predict the NEXT token at each position
    targets = torch.tensor([[3626, 6100, 345],    # " effort moves you"
                            [1107, 588, 11311]])  # " really like chocolate"

    # Use context manager to temporarily disable gradients since we are not training
    with torch.no_grad():
        # logits shape: (batch_size=2, seq_len=3, vocab_size=50257)
        # Each position outputs 50,257 scores (one per possible next token)
        logits = model(inputs)

    # Convert logits to probabilities using softmax
    # dim=-1 means softmax across the vocabulary dimension
    # Each position now has a probability distribution over all 50,257 tokens
    # probas shape: (2, 3, 50257) - probabilities sum to 1.0 across last dim
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)

    # Find the token with the highest probability at each position
    # argmax returns the INDEX of the max value (i.e., the predicted token ID)
    # keepdim=True maintains shape (2, 3, 1) instead of (2, 3)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)

    # Compare what the model predicted vs what it should have predicted
    # With an untrained model, predictions will be essentially random
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1:"
          f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    # =========================================================================
    # Extract probabilities for the TARGET tokens
    # =========================================================================
    #
    # We want to know: what probability did the model assign to the CORRECT
    # next token at each position?
    #
    # probas shape: (2, 3, 50257) - 2 batches, 3 positions, 50257 vocab probs
    # targets shape: (2, 3) - the correct token IDs we want to predict
    #
    # Advanced indexing: probas[batch_idx, positions, token_ids]
    #   - text_idx: which batch (0 or 1)
    #   - [0, 1, 2]: all 3 positions in sequence
    #   - targets[text_idx]: the 3 target token IDs for this batch
    #
    # This extracts the probability the model assigned to each correct token
    # =========================================================================

    text_idx = 0
    # For batch 0, get probability of correct token at each of 3 positions
    # probas[0, [0,1,2], [3626, 6100, 345]] → 3 probability values
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    # For batch 1, get probability of correct token at each of 3 positions
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    # =========================================================================
    # Convert to log probabilities
    # =========================================================================
    #
    # Why log probabilities?
    # 1. Probabilities are tiny (e.g., 0.00002) - logs are easier to work with
    # 2. Multiplying probabilities → adding log probabilities (numerically stable)
    # 3. Cross-entropy loss uses log probabilities
    #
    # torch.cat combines the 6 probabilities (3 from each batch) into one tensor
    # torch.log converts probabilities to log probabilities
    #
    # Higher (less negative) log prob = model was more confident in correct answer
    # =========================================================================

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)
    
    
    
    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)
    
    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)
    
    
    print("Logits shape: ", logits.shape)
    print("Targets shape:", targets.shape)
    
    logits_flat = logits.flatten(0,1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    
    #Pytorch's cross_entropy function will take care of all manual steps
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)
    
    
    file_path = os.path.join(os.path.dirname(__file__), "../../data/the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
        
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens: ", total_tokens)
    
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    try:
        from .dataloader import create_dataloader_v1
    except ImportError:
        from dataloader import create_dataloader_v1
    
    
    torch.manual_seed(123)
    
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
        
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)    
        
        
    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        logits = model (input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
            
        )
        return loss
        
    # =========================================================================
    # Compute average loss across multiple batches from a DataLoader
    # =========================================================================
    #
    # Why do we need this?
    # - calc_loss_batch gives us loss for a SINGLE batch
    # - To evaluate the model fairly, we need the average loss across the
    #   entire training or validation set (many batches)
    # - This gives us a single number summarizing how well the model is doing
    #
    # Parameters:
    # - data_loader: yields (input_batch, target_batch) pairs
    # - model: the GPT model to evaluate
    # - device: "cpu" or "cuda" - where tensors should live
    # - num_batches: optionally limit how many batches to evaluate
    #                (useful for quick checks during training to save time)
    # =========================================================================
    def calc_loss_loader(data_loader, model, device, num_batches=None):
        # Accumulator for summing losses across batches
        total_loss = 0.

        # Edge case: empty data loader → can't compute mean, return NaN
        if len(data_loader) == 0:
            return float("nan")
        # If caller didn't specify, evaluate ALL batches in the loader
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Safety clamp: can't evaluate more batches than actually exist
            # e.g., asking for 100 batches when loader only has 5
            num_batches = min(num_batches, len(data_loader))

        # Iterate through batches and accumulate loss
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                # Compute cross-entropy loss for this batch
                loss = calc_loss_batch(
                    input_batch, target_batch, model, device
                )
                # .item() converts a scalar tensor to a Python float
                # We use float (not tensor) because we're just tracking stats,
                # not building a computation graph for backprop
                total_loss += loss.item()
            else:
                # Stop early once we've processed num_batches
                break

        # Return the AVERAGE loss per batch
        return total_loss / num_batches
    
    
    # =========================================================================
    # Set up compute device and evaluate initial loss
    # =========================================================================
    #
    # Why pick a device?
    # - PyTorch tensors and models live on a specific device (CPU or GPU)
    # - Operations are MUCH faster on a GPU when one is available
    # - We pick the best available option: CUDA → MPS (Apple Silicon) → CPU
    # =========================================================================

    # Choose the best available hardware:
    #   - "cuda": NVIDIA GPU (Linux/Windows with NVIDIA card)
    #   - "mps":  Apple Silicon GPU via Metal Performance Shaders (M1/M2/M3/M4)
    #   - "cpu":  Fallback when no GPU is available
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu")
    
    print("Device:", device)

    # Move the model's parameters and buffers to the chosen device.
    # All input tensors must also be on the same device to avoid runtime errors.
    model.to(device)

    # torch.no_grad() disables gradient tracking — we're only EVALUATING here,
    # not training, so we don't need to build a computation graph.
    # This saves memory and speeds up the forward pass.
    with torch.no_grad():
        # Compute average loss across the entire training set
        # (a baseline for how well the untrained model performs)
        train_loss = calc_loss_loader(train_loader, model, device)
        # Compute average loss across the validation set
        # (should be similar to train_loss for an untrained model)
        val_loss = calc_loss_loader(val_loader, model, device)

    # Both losses should be ~10.8 for an untrained GPT-2-style model on this
    # vocabulary (≈ ln(50257), since random predictions over 50,257 tokens
    # give a uniform 1/50257 probability → -log(1/50257) ≈ 10.82)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
        
    # =========================================================================
    # Main training loop
    # =========================================================================
    #
    # This is the heart of model training. The pattern is:
    #   for each epoch:
    #     for each batch:
    #       1. Zero out previous gradients
    #       2. Forward pass + compute loss
    #       3. Backward pass (compute gradients)
    #       4. Optimizer step (update weights)
    #       5. Periodically evaluate + log progress
    #     After epoch: generate a sample to see qualitative progress
    #
    # Parameters:
    # - model: the GPT model being trained
    # - train_loader / val_loader: iterables yielding (input, target) batches
    # - optimizer: e.g., AdamW — updates model weights based on gradients
    # - device: where to run computation ("cuda", "mps", or "cpu")
    # - num_epochs: how many full passes through the training data
    # - eval_freq: evaluate every N steps (don't evaluate every step — too slow)
    # - eval_iter: how many batches to sample when evaluating
    # - start_context: seed text for generating samples between epochs
    # - tokenizer: needed by generate_and_print_sample
    # =========================================================================
    def train_model_simple(model, train_loader, val_loader,
                           optimizer, device, num_epochs,
                           eval_freq, eval_iter, start_context, tokenizer):

        # Lists to track progress across training — used later for plotting
        train_losses, val_losses, track_tokens_seen = [], [], []
        # Counters: total tokens processed and total optimizer steps taken
        # global_step starts at -1 so the first increment makes it 0
        tokens_seen, global_step = 0, -1

        # Outer loop: one full pass through the dataset = one epoch
        for epoch in range(num_epochs):
            # Switch to training mode — enables dropout, batch norm updates, etc.
            # (We previously called model.eval() during the loss-calc demo)
            model.train()

            # Inner loop: iterate over batches from the training set
            for input_batch, target_batch in train_loader:
                # Clear gradients from the previous step.
                # PyTorch ACCUMULATES gradients by default, so we must
                # reset them or each step would build on stale gradients.
                optimizer.zero_grad()

                # Forward pass: compute loss for this batch
                loss = calc_loss_batch(
                    input_batch, target_batch, model, device
                )

                # Backward pass: compute gradients of loss w.r.t. weights
                # via backpropagation (the chain rule, automated by autograd)
                loss.backward()

                # Update weights using the gradients (e.g., AdamW step)
                optimizer.step()

                # Track how many tokens we've trained on so far
                # .numel() = total number of elements in the tensor
                tokens_seen += input_batch.numel()
                global_step += 1

                # Periodic evaluation: every eval_freq steps, check losses
                # on both train and val sets to monitor overfitting
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter
                    )
                    # Record losses for plotting later
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    # Log current progress to console
                    print(f"EP {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, "
                          f"Val loss {val_loss:.3f}")

            # End of epoch: generate a text sample so we can SEE the model
            # qualitatively improving (loss numbers alone are abstract)
            generate_and_print_sample(model, tokenizer, device, start_context)

        # Return tracked metrics for plotting/analysis after training
        return train_losses, val_losses, track_tokens_seen

