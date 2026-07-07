"""Train the toy induction model and log induction accuracy / attention score."""

import torch
import torch.nn.functional as F

from config import device, VOCAB_SIZE, PREFIX_LEN, BATCH_SIZE, N_STEPS
from data import generate_batch
from model import build_model


def main():
    model = build_model()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    for step in range(N_STEPS):
        model.train()
        x, y, _ = generate_batch(BATCH_SIZE, VOCAB_SIZE, prefix_len=PREFIX_LEN, device=device)
        logits = model(x, prepend_bos=False)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == N_STEPS - 1:
            model.eval()
            with torch.no_grad():
                x_test, y_test, _ = generate_batch(100, VOCAB_SIZE, prefix_len=PREFIX_LEN, device=device)
                logits_test = model(x_test, prepend_bos=False)

                # Induction accuracy: only on the second copy (positions PREFIX_LEN .. 2*PREFIX_LEN-1)
                ind_mask = torch.zeros_like(y_test, dtype=torch.bool)
                ind_mask[:, PREFIX_LEN:] = True                   # all second-half positions
                # The last token in the second copy has no induction target (no next token), so ignore it
                ind_mask[:, -1] = False

                preds = logits_test.argmax(-1)
                correct = (preds[ind_mask] == y_test[ind_mask]).float().mean().item()

                # Induction score: attention from pos L+i to i+1 (head 0 in layer 0)
                _, cache = model.run_with_cache(x_test, prepend_bos=False)
                pattern = cache["blocks.0.attn.hook_pattern"][:, 0, :, :]   # (batch, seq, seq)
                scores = []
                for b in range(100):
                    for i in range(PREFIX_LEN - 1):
                        scores.append(pattern[b, PREFIX_LEN + i, i + 1].item())
                ind_score = sum(scores) / len(scores)
            model.train()
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Ind Acc: {correct:.3f} | Ind Score: {ind_score:.3f}")

    print("Training complete.")
    return model


if __name__ == "__main__":
    main()
