# Mechanistic Data Attribution (MDA) 

This is a minimal reproduction of Mechanistic Data Attribution paper on a toy 2-layer attention-only transformer trained on a synthetic induction task. See the original paper [here.](https://arxiv.org/pdf/2601.21996)

## Layout

```
mda_project/
├── config.py                    # shared constants (task, model, EK-FAC hyperparams)
├── data.py                      # generate_batch: synthetic [prefix, prefix] task
├── model.py                     # build_model: HookedTransformer builder
├── induction_metrics.py         # compute_induction_scores: per-head induction strength
├── mda/
│   ├── hooks.py                 # QKVOActivationCache + setup_qkvo_hooks
│   ├── ekfac.py                 # EKFACQKVOHead: A/S accumulation, eigendecomp, IHVP
│   ├── stage1.py                # run_ekfac_stage1: fits EK-FAC (A/S, then Lambda)
│   ├── probe.py                 # compute_probe_gradient: induction-copying probe
│   └── scoring.py               # score_training_samples: influence scoring
└── scripts/
    ├── train.py                       # basic training loop w/ induction acc/score logging
    ├── train_with_checkpoints.py      # training loop with dense checkpointing
    └── run_mda.py                     # full pipeline: EK-FAC → probe → IHVP → scoring → plots
```

## Usage

Train a model:

```bash
python scripts/train_with_checkpoints.py
```

Run the MDA pipeline (edit `load_model()` in `scripts/run_mda.py` to load your
trained checkpoint):

```bash
python scripts/run_mda.py
```

<img width="590" height="490" alt="image" src="https://github.com/user-attachments/assets/ba6052e4-f971-4686-82f8-2a95e4187b20" />
