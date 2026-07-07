"""Toy 2-layer attention-only model builder."""

from transformer_lens import HookedTransformer

from config import cfg, device


def build_model():
    """Instantiate a fresh HookedTransformer using the project config."""
    model = HookedTransformer(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model
