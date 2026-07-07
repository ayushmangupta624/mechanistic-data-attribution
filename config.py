"""Shared configuration constants for the toy induction model and MDA pipeline."""

import torch
from transformer_lens import HookedTransformerConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Task / data ────────────────────────────────────────────────────────────
VOCAB_SIZE = 16
PREFIX_LEN = 8
SEQ_LEN = 2 * PREFIX_LEN
BATCH_SIZE = 256
N_STEPS = 1000

# ── Model ──────────────────────────────────────────────────────────────────
cfg = HookedTransformerConfig(
    n_layers=2,
    n_heads=4,
    d_model=64,
    d_head=64,
    d_mlp=None,
    n_ctx=SEQ_LEN,
    d_vocab=VOCAB_SIZE,
    act_fn="relu",
    normalization_type="LN",         # no LayerNorm
    init_mode="gpt2",
    initializer_range=0.02,
    device=device,
    default_prepend_bos=False,
)

# ── MDA pipeline ───────────────────────────────────────────────────────────
TARGET_LAYER = 1      # the layer containing our induction heads
TARGET_HEAD  = 1      # L1H1
D_MODEL      = cfg.d_model    # 64
D_HEAD       = cfg.d_head     # 64
DAMPING      = 1e-5
DAMPING_ALPHA = 0.1
N_EKFAC_BATCHES   = 200   # batches for A/S accumulation and Lambda fitting
N_PROBE_SAMPLES   = 200   # sequences for probe gradient
N_SCORING_BATCHES = 500   # training sequences to score
TOP_K        = 50
