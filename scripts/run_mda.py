"""Run the full Mechanistic Data Attribution (MDA) pipeline on a trained toy model.

Assumes a trained model checkpoint is available (see scripts/train.py or
scripts/train_with_checkpoints.py). Adjust `load_model()` to point at your
checkpoint of choice.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import (
    device, TARGET_LAYER, TARGET_HEAD, D_MODEL, D_HEAD,
    DAMPING, DAMPING_ALPHA, N_EKFAC_BATCHES, N_PROBE_SAMPLES,
    N_SCORING_BATCHES,
)
from mda.ekfac import EKFACQKVOHead
from mda.stage1 import run_ekfac_stage1
from mda.probe import compute_probe_gradient
from mda.scoring import score_training_samples


def load_model(checkpoint_path="/content/working_model.pt", checkpoint_step=None):
    """Load a trained model.

    Args:
        checkpoint_path: path to the .pt file saved by scripts/train_with_checkpoints.py.
        checkpoint_step: if None, loads the final `model_state_dict`. If an int,
            loads the intermediate checkpoint saved at that training step
            (from the `checkpoints` dict, which is keyed every 5 steps).
    """
    from model import build_model
    model = build_model()

    # map_location ensures this works even if the checkpoint was saved on a
    # different device (e.g. trained on GPU, later loaded on CPU).
    ckpt = torch.load(checkpoint_path, map_location=device)

    if checkpoint_step is None:
        state_dict = ckpt["model_state_dict"]
    else:
        if checkpoint_step not in ckpt["checkpoints"]:
            available_steps = sorted(ckpt["checkpoints"].keys())
            raise ValueError(
                f"No checkpoint saved at step {checkpoint_step}. "
                f"Available steps: {available_steps}"
            )
        state_dict = ckpt["checkpoints"][checkpoint_step]

    model.load_state_dict(state_dict)
    model.to(device)
    return model


def main():
    model = load_model()

    print("=" * 60)
    print(f"Running MDA for L{TARGET_LAYER}H{TARGET_HEAD}")
    print("=" * 60)

    # 5a. EK-FAC
    ekfac = EKFACQKVOHead(D_MODEL, D_HEAD, DAMPING, DAMPING_ALPHA)
    run_ekfac_stage1(model, ekfac, TARGET_LAYER, TARGET_HEAD, N_EKFAC_BATCHES)

    # 5b. Probe gradient
    print("\nComputing probe gradient...")
    v_qk, v_v, v_o = compute_probe_gradient(model, TARGET_LAYER, TARGET_HEAD, N_PROBE_SAMPLES)
    print(f"  probe grad norms — QK: {v_qk.norm():.4f}, V: {v_v.norm():.4f}, O: {v_o.norm():.4f}")

    # 5c. IHVP: p = H^{-1} v
    print("\nComputing IHVP...")
    p_qk = ekfac.inverse_hvp(0, v_qk)
    p_v  = ekfac.inverse_hvp(1, v_v)
    p_o  = ekfac.inverse_hvp(2, v_o)
    print(f"  IHVP norms — QK: {p_qk.norm():.4f}, V: {p_v.norm():.4f}, O: {p_o.norm():.4f}")

    # 5d. Score training samples
    print("\nScoring training samples...")
    scores, metadata = score_training_samples(
        model, ekfac, p_qk, p_v, p_o,
        TARGET_LAYER, TARGET_HEAD, N_SCORING_BATCHES
    )
    print(f"Score stats — mean: {scores.mean():.4f}, std: {scores.std():.4f}, "
          f"max: {scores.max():.4f}, min: {scores.min():.4f}")

    # ── Step 6: Validate and visualize ──────────────────────────────────────

    # Sort by influence score
    ranked_idx = np.argsort(scores)[::-1]

    print(f"\nTop-10 most influential samples (should all be induction examples):")
    for i in range(10):
        idx = ranked_idx[i]
        print(f"  rank {i+1}: score={scores[idx]:.4f}, meta={metadata[idx]}")

    print(f"\nBottom-10 least influential (negative influence):")
    for i in range(1, 11):
        idx = ranked_idx[-i]
        print(f"  rank {-i}: score={scores[idx]:.4f}, meta={metadata[idx]}")

    # Plot influence score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(scores, bins=50, color='steelblue', edgecolor='white')
    axes[0].axvline(0, color='red', linestyle='--', label='zero')
    axes[0].set_xlabel('Influence score')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Influence score distribution\nL{TARGET_LAYER}H{TARGET_HEAD}')
    axes[0].legend()

    axes[1].plot(np.sort(scores)[::-1], color='crimson')
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set_xlabel('Rank')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Ranked influence scores (power-law shape expected)')

    plt.tight_layout()
    plt.savefig('/content/mda_influence_scores.png', dpi=150)
    plt.show()

    # Save results
    torch.save({
        'scores':   scores,
        'metadata': metadata,
        'p_qk':     p_qk.cpu(),
        'p_v':      p_v.cpu(),
        'p_o':      p_o.cpu(),
        'ekfac_QA': [q.cpu() for q in ekfac.Q_A],
        'ekfac_QS': [q.cpu() for q in ekfac.Q_S],
        'ekfac_L':  [l.cpu() for l in ekfac.Lambda],
    }, '/content/mda_results.pt')
    print("\nResults saved to /content/mda_results.pt")


if __name__ == "__main__":
    main()