"""Stage 1: fit the EK-FAC approximation (A/S covariances, then Lambda) for a head."""

import torch
import torch.nn.functional as F

from config import device, VOCAB_SIZE, PREFIX_LEN, BATCH_SIZE, D_MODEL, D_HEAD
from data import generate_batch
from .hooks import QKVOActivationCache, setup_qkvo_hooks


def run_ekfac_stage1(model, ekfac, layer, head, n_batches):
    """Stage 1A + 1B combined: accumulate A/S then fit Lambda."""
    cache = QKVOActivationCache()
    attn  = model.blocks[layer].attn

    # Ensure target parameters require grad
    for p in model.parameters():
        p.requires_grad_(False)
    attn.W_Q.requires_grad_(True)
    attn.W_K.requires_grad_(True)
    attn.W_V.requires_grad_(True)
    attn.W_O.requires_grad_(True)

    print("Stage 1A: accumulating A/S covariances...")
    hooks = setup_qkvo_hooks(model, layer, cache)
    model.eval()

    for batch_idx in range(n_batches):
        cache.clear()
        x, y, _ = generate_batch(BATCH_SIZE, VOCAB_SIZE, PREFIX_LEN, device)

        with torch.enable_grad():
            logits = model(x, prepend_bos=False)
            # Use pseudo-labels (sample from model) as per EK-FAC standard
            V = logits.shape[-1]
            probs = torch.softmax(logits.reshape(-1, V).float(), dim=-1)
            pseudo_labels = torch.multinomial(probs, num_samples=1).squeeze(-1)
            loss = F.cross_entropy(logits.reshape(-1, V).float(), pseudo_labels, reduction="sum")

            grads = torch.autograd.grad(
                loss,
                inputs=[cache.Q, cache.K, cache.V, cache.result],
                retain_graph=False, create_graph=False
            )

        dQ = grads[0][:, :, head, :].reshape(-1, D_HEAD).float().detach()
        dK = grads[1][:, :, head, :].reshape(-1, D_HEAD).float().detach()
        dV = grads[2][:, :, head, :].reshape(-1, D_HEAD).float().detach()
        dR = grads[3][:, :, head, :].reshape(-1, D_MODEL).float().detach()
        X_flat = cache.X.reshape(-1, D_MODEL).float().detach()
        Z_flat = cache.Z[:, :, head, :].reshape(-1, D_HEAD).float().detach()

        ekfac.accumulate_AS(X_flat, dQ, dK, dV, Z_flat, dR)

        if (batch_idx + 1) % 50 == 0:
            print(f"  A/S batch {batch_idx+1}/{n_batches}")

    model.reset_hooks()
    ekfac.finalize_eigendecomposition()
    print("Stage 1A done.")

    print("Stage 1B: fitting Lambda...")
    hooks = setup_qkvo_hooks(model, layer, cache)
    total_weight = 0.0

    for batch_idx in range(n_batches):
        cache.clear()
        x, y, _ = generate_batch(BATCH_SIZE, VOCAB_SIZE, PREFIX_LEN, device)
        B = x.shape[0]

        with torch.enable_grad():
            logits = model(x, prepend_bos=False)
            V = logits.shape[-1]
            probs = torch.softmax(logits.reshape(-1, V).float(), dim=-1)
            pseudo_labels = torch.multinomial(probs, num_samples=1).squeeze(-1)
            loss = F.cross_entropy(logits.reshape(-1, V).float(), pseudo_labels, reduction="sum")

            grads = torch.autograd.grad(
                loss,
                inputs=[cache.Q, cache.K, cache.V, cache.result],
                retain_graph=False, create_graph=False
            )

        dQ = grads[0][:, :, head, :].reshape(-1, D_HEAD).float().detach()
        dK = grads[1][:, :, head, :].reshape(-1, D_HEAD).float().detach()
        dV = grads[2][:, :, head, :].reshape(-1, D_HEAD).float().detach()
        dR = grads[3][:, :, head, :].reshape(-1, D_MODEL).float().detach()
        X_flat = cache.X.reshape(-1, D_MODEL).float().detach()
        Z_flat = cache.Z[:, :, head, :].reshape(-1, D_HEAD).float().detach()

        ekfac.fit_lambda(X_flat, dQ, dK, dV, Z_flat, dR, weight=float(B))
        total_weight += float(B)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Lambda batch {batch_idx+1}/{n_batches}")

    model.reset_hooks()
    ekfac.finalize_lambda(total_weight)
    print("Stage 1B done.")

    # Re-freeze everything
    for p in model.parameters():
        p.requires_grad_(False)
