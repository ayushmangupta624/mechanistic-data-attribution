"""Score training samples by their influence on the probe behavior."""

import numpy as np
import torch
import torch.nn.functional as F

from config import device, VOCAB_SIZE, PREFIX_LEN
from data import generate_batch


def score_training_samples(model, ekfac, p_qk, p_v, p_o, layer, head, n_batches):
    """
    For each training sequence, compute influence score:
        s = -(g_train_qk · p_qk + g_train_v · p_v + g_train_o · p_o)
    A high positive score = this sample promotes induction head behavior.
    """
    attn = model.blocks[layer].attn
    for p in model.parameters():
        p.requires_grad_(False)
    attn.W_Q.requires_grad_(True)
    attn.W_K.requires_grad_(True)
    attn.W_V.requires_grad_(True)
    attn.W_O.requires_grad_(True)

    model.eval()
    scores   = []
    metadata = []   # store (is_induction, prefix_len) for ground-truth validation

    print(f"Scoring {n_batches} batches...")

    for batch_idx in range(n_batches):
        # Generate one sequence at a time for per-sample gradients
        x, y, _ = generate_batch(1, VOCAB_SIZE, PREFIX_LEN, device)
        is_ind   = True   # all examples from generate_batch are induction examples

        with torch.enable_grad():
            logits = model(x, prepend_bos=False)
            loss   = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE).float(),
                y.reshape(-1),
                reduction="sum"
            )
            grads = torch.autograd.grad(
                loss,
                inputs=[attn.W_Q, attn.W_K, attn.W_V, attn.W_O],
                retain_graph=False, create_graph=False
            )

        gQ = grads[0][head].float()
        gK = grads[1][head].float()
        gV = grads[2][head].float()
        gO = grads[3][head].float()

        g_qk = torch.cat([gQ, gK], dim=-1)
        s    = (torch.sum(g_qk * p_qk) + torch.sum(gV * p_v) + torch.sum(gO * p_o)) #note: no negative sign
        scores.append(s.item())
        metadata.append({'batch_idx': batch_idx, 'is_induction': is_ind})

        if (batch_idx + 1) % 100 == 0:
            print(f"  scored {batch_idx+1}/{n_batches}")

    for p in model.parameters():
        p.requires_grad_(False)

    return np.array(scores), metadata
