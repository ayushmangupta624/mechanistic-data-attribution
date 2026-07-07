"""Synthetic induction-task data generation."""

import torch


def generate_batch(batch_size, vocab_size, prefix_len=8, device="cpu"):
    """Each sequence = [prefix, prefix], no extra tokens."""
    seq_len = 2 * prefix_len
    # Random prefix for each example
    prefix = torch.randint(0, vocab_size, (batch_size, prefix_len), device=device)
    # Full sequence: prefix followed by its copy
    tokens = torch.cat([prefix, prefix], dim=1)       # (batch_size, seq_len)
    x = tokens[:, :-1]                                 # input
    y = tokens[:, 1:]                                  # target
    return x, y, prefix_len
