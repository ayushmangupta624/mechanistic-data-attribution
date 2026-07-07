"""Probe gradient for the synthetic induction-copying behavior."""

import torch
import torch.nn.functional as F

from config import device, VOCAB_SIZE, PREFIX_LEN, D_MODEL, D_HEAD


def compute_probe_gradient(model, layer, head, n_samples=200):
    """
    f_probe: on synthetic repeated sequences [prefix, prefix],
    sum the log-prob of the induction target (token after previous occurrence)
    at each position in the second half.
    This is the copy_target_synthetic probe from their code, adapted for our
    toy setup where every sequence IS [prefix, prefix].
    """
    attn = model.blocks[layer].attn
    for p in model.parameters():
        p.requires_grad_(False)
    attn.W_Q.requires_grad_(True)
    attn.W_K.requires_grad_(True)
    attn.W_V.requires_grad_(True)
    attn.W_O.requires_grad_(True)

    model.eval()

    v_qk = torch.zeros(D_MODEL, 2 * D_HEAD, device=device, dtype=torch.float32)
    v_v  = torch.zeros(D_MODEL, D_HEAD,     device=device, dtype=torch.float32)
    v_o  = torch.zeros(D_HEAD,  D_MODEL,    device=device, dtype=torch.float32)

    for _ in range(n_samples):
        prefix = torch.randint(0, VOCAB_SIZE, (PREFIX_LEN,), device=device)
        tokens = torch.cat([prefix, prefix]).unsqueeze(0)  # (1, 2*PREFIX_LEN)
        x      = tokens[:, :-1]   # input
        y      = tokens[:, 1:]    # targets

        with torch.enable_grad():
            logits    = model(x, prepend_bos=False)  # (1, 2*PREFIX_LEN-1, V)
            log_probs = F.log_softmax(logits[0].float(), dim=-1)

            # Sum log-prob of induction targets in the second half
            # At position PREFIX_LEN + i in x (0-indexed), the induction target
            # is prefix[i] (the token that followed prefix[i-1] in the first copy)
            loss = torch.tensor(0.0, device=device)
            seq  = tokens[0]
            for t in range(1, x.shape[1]):
                # Find previous occurrence of seq[t] before position t
                key  = seq[t].item()
                prev = (seq[:t] == key).nonzero(as_tuple=True)[0]
                if prev.numel() == 0:
                    continue
                match_pos  = int(prev[-1].item())
                target_tok = seq[match_pos + 1].item() if match_pos + 1 < len(seq) else -1
                if target_tok == -1:
                    continue
                loss = loss + log_probs[t - 1, target_tok]

            grads = torch.autograd.grad(
                loss,
                inputs=[attn.W_Q, attn.W_K, attn.W_V, attn.W_O],
                retain_graph=False, create_graph=False, allow_unused=False
            )

        v_qk.add_(torch.cat([grads[0][head].float(), grads[1][head].float()], dim=-1))
        v_v.add_(grads[2][head].float())
        v_o.add_(grads[3][head].float())

    v_qk /= n_samples
    v_v  /= n_samples
    v_o  /= n_samples

    for p in model.parameters():
        p.requires_grad_(False)

    return v_qk, v_v, v_o
