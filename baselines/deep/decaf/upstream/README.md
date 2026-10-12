DECAF upstream
==============

Original code: https://github.com/carankt/DECAF.
Commit: see `decaf/GIT_COMMIT`.
(The repo linked from the paper, https://github.com/JHU-LCAP/DECAF, contains
only a README that forwards to the above.)

Vendored in full except `.git` and `paper_figures/`.  The model is **used
directly, not re-implemented**: `../models_upstream.py` binds this tree under a
private `decaf_upstream` package namespace — its top-level `src` would
otherwise collide with this repository's own `src` package — and instantiates
`src/models/two_branch_model_recursive.py::TwoBranchModelRecursive`.

The configuration is the one the authors trained (`train_models_3.sh`):

    python train_model.py --model two_branch_recursive_context_3.0sec_dynamic \
                          --loss_function pearson_l1

which resolves through `src/models/registry.py` and
`configs/two_branch_recursive_config.py` to `d_model=128`, `d_inner=1024`,
`n_head=2`, `n_layers=8`, FFN kernels `(9, 1)`, `dropout=0.6`,
`fusion_strategy='dynamic'`, `rnn_type='gru'`, `context_seconds=3.0`,
`prediction_seconds=3.5` — 11 405 986 parameters, i.e. the paper's 11.4 M.

Notes taken from the code that the paper does not state:

* Windows are **3.5 s present + 3.5 s past** (7 s total), not the "3-second
  windows" of §3.1 — see `create_filelist_recursive.py` and the
  `filelist_recursive/..._7.0s_past3.5s_present3.5s.csv` names.
* `pearson_l1` is `(1 - r) + 0.2 * L1` (`src/train/losses.py`); the paper's
  Eq. 5 puts the weights the other way round.
* Training uses `teacher_forcing_ratio = 1.0` throughout and **validates**
  through `forward_teacher_forcing`, i.e. model selection sees the
  ground-truth past envelope (`src/train/trainer.py`).
* The learning rate is the Noam schedule (`d_model**-0.5`, 3 warm-up epochs,
  stepped per batch), which overrides `learning_rate: 3e-4`.
* Inputs are z-scored per window and channel; the envelope target is min-max
  scaled to [0, 1] to match the model's sigmoid head
  (`src/data/eeg_dataloader.py`).

Reference: K. Thakkar, M. Elhilali, "DECAF: Dynamic Envelope Context-Aware
Fusion for Speech-Envelope Reconstruction from EEG", ICASSP 2026
(arXiv:2602.19395).
