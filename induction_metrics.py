"""Utilities for measuring induction-head strength across layers/heads."""

import torch

from data import generate_batch
from config import device


def compute_induction_scores(model, n_layers, n_heads, prefix_len=8, n_seqs=100):
    model.eval()
    x_test, _, _ = generate_batch(n_seqs, model.cfg.d_vocab, prefix_len=prefix_len,
                                   device=device)
    with torch.no_grad():
        _, cache = model.run_with_cache(x_test, prepend_bos=False)

    scores = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head]
            ind = [pattern[b, prefix_len+i, i+1].item()
                   for b in range(n_seqs) for i in range(prefix_len-2)]
            scores[(layer, head)] = sum(ind) / len(ind)

    model.train()
    return scores
