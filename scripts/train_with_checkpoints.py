"""Train the toy induction model with dense checkpointing every 5 steps.

Produces /content/working_model.pt containing the final model state dict,
config, per-step checkpoints, and the training loss log.
"""

import random

import torch
import torch.nn.functional as F

from config import cfg, device
from data import generate_batch
from induction_metrics import compute_induction_scores


def main():
    torch.manual_seed(0)
    random.seed(0)

    from transformer_lens import HookedTransformer

    model = HookedTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    checkpoints = {}
    training_log = []

    for step in range(1000):
        model.train()
        x, y, _ = generate_batch(256, 16, prefix_len=8, device=device)
        logits = model(x, prepend_bos=False)
        loss = F.cross_entropy(logits.reshape(-1, 16), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_log.append({"step": step, "loss": loss.item()})

        # Dense checkpointing — every 5 steps
        if step % 5 == 0:
            checkpoints[step] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        if step % 50 == 0:
            scores = compute_induction_scores(model, 2, 4, prefix_len=8, n_seqs=100)
            best = max(scores, key=scores.get)
            print(f"step {step:>4d} | loss {loss.item():.4f} "
                  f"| best L{best[0]}H{best[1]} = {scores[best]:.3f}")

    print(f"Done. {len(checkpoints)} checkpoints saved.")

    # Save everything
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": cfg,
        "checkpoints": checkpoints,
        "training_log": training_log,
    }, "/content/working_model.pt")
    print("Saved.")

    return model, checkpoints, training_log


if __name__ == "__main__":
    main()
