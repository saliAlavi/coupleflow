"""DECAF — the upstream model, loaded verbatim from ``upstream/decaf``.

Upstream: ``github.com/carankt/DECAF`` (commit in
``upstream/decaf/GIT_COMMIT``), the training code advertised by

    Thakkar & Elhilali (2026).  "DECAF: Dynamic Envelope Context-Aware
    Fusion for Speech-Envelope Reconstruction from EEG."  ICASSP 2026
    (arXiv:2602.19395).

Nothing here re-implements the architecture: ``build_decaf`` instantiates
``src.models.two_branch_model_recursive.TwoBranchModelRecursive`` with the
values from ``configs/two_branch_recursive_config.py`` and the model name the
authors actually trained (``two_branch_recursive_context_3.0sec_dynamic``,
see ``train_models_3.sh``).  That gives, matching the paper's Fig. 2 and
Table 1:

* **EEG-to-Envelope**: HappyQuoka — Conv1d(64->128, k=7) then 8 pre-LN FFT
  blocks (``d_model`` 128, ``d_inner`` 1024, 2 heads, FFN Conv1d kernels
  (9, 1)), dropout 0.6, sigmoid output.  11.1 M params, exactly the
  "HappyQuokka" row of Table 1.
* **Envelope Forecaster**: Conv1d(1->128, k=7) -> 2-layer unidirectional GRU
  (hidden 128) -> 4-head self-attention -> MLP head emitting the whole
  3.5 s (224-sample) horizon.
* **Dynamic Fusion**: Conv1d 2->16->8->1 with kernels 5, 3, 1, ReLU and a
  final sigmoid producing a per-time-step gate; the output is the convex
  combination of the two branches.

Total 11 405 986 params = the paper's 11.4 M.

The upstream package cannot simply be put on ``sys.path`` because its
top-level ``src`` collides with this repository's own ``src`` package, so the
modules are bound under a private ``decaf_upstream`` namespace instead; their
relative imports resolve inside it unchanged.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

_UPSTREAM = Path(__file__).resolve().parent / "upstream" / "decaf"

# Config values from configs/two_branch_recursive_config.py (+ its
# BaselineConfig parent) for the variant in train_models_3.sh.
DECAF_MODEL_KWARGS = dict(
    in_channel=64,
    d_model=128,
    d_inner=1024,
    n_head=2,
    n_layers=8,
    fft_conv1d_kernel=(9, 1),
    fft_conv1d_padding=(4, 0),
    dropout=0.6,
    fusion_strategy="dynamic",
    rnn_type="gru",
    context_seconds=3.0,        # forecaster consumes the last 3.0 s of context
    prediction_seconds=3.5,     # ... and emits the full 3.5 s present window
)
PRESENT_S = 3.5                 # filelist_recursive/..._present3.5s
PAST_S = 3.5                    # filelist_recursive/..._past3.5s
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
WARMUP_EPOCHS = 3
NUM_EPOCHS = 10
PATIENCE = 3
BATCH_SIZE = 64                 # paper §3.1 ("Adam optimizer (batch size of 64)")
PEARSON_L1_ALPHA = 0.2          # LossFactory "pearson_l1"
EEG_DELAY_S = 0.5               # paper §3.1: "A 500ms delay was applied to the EEG"


def _bind_namespace() -> None:
    """Expose ``upstream/decaf`` as the ``decaf_upstream`` package tree."""
    if "decaf_upstream" in sys.modules:
        return
    for name, path in (
        ("decaf_upstream", _UPSTREAM),
        ("decaf_upstream.src", _UPSTREAM / "src"),
        ("decaf_upstream.src.models", _UPSTREAM / "src" / "models"),
        ("decaf_upstream.src.train", _UPSTREAM / "src" / "train"),
    ):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]          # type: ignore[attr-defined]
        mod.__package__ = name
        sys.modules[name] = mod


def load_upstream():
    """Import and return the upstream model and loss modules."""
    _bind_namespace()
    models = importlib.import_module(
        "decaf_upstream.src.models.two_branch_model_recursive")
    losses = importlib.import_module("decaf_upstream.src.train.losses")
    return models, losses


def build_decaf(**overrides) -> nn.Module:
    """Instantiate the upstream ``TwoBranchModelRecursive`` as DECAF.

    Args:
        **overrides: Any of :data:`DECAF_MODEL_KWARGS`.

    Returns:
        The upstream module, untouched.
    """
    models, _ = load_upstream()
    kwargs = {**DECAF_MODEL_KWARGS, **overrides}
    return models.TwoBranchModelRecursive(**kwargs)


class PearsonL1Loss(nn.Module):
    """Upstream ``pearson_l1``: ``(1 - r) + alpha * L1``.

    This is the objective the released code trains with
    (``LossFactory.create("pearson_l1")``, ``train_models_3.sh``).  The paper
    writes it as ``lambda1 * L1 - lambda2 * r`` with ``lambda1 = 1``,
    ``lambda2 = 0.2``; the code weights the *L1* term by 0.2 instead.  Up to
    an additive constant both are "Pearson plus L1", and we follow the code.

    Args:
        alpha: Weight on the L1 term.
    """

    def __init__(self, alpha: float = PEARSON_L1_ALPHA):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: ``(B, T, 1)`` tensors. Returns: scalar loss.

        The guard sits inside the square root — see
        :func:`coupleflow.envelope_common.pearson` for why adding it
        afterwards (as upstream does) makes one constant window NaN the whole
        run.
        """
        pm = pred.mean(dim=1, keepdim=True)
        tm = target.mean(dim=1, keepdim=True)
        pc, tc = pred - pm, target - tm
        num = (pc * tc).sum(dim=1)
        den = (pc.pow(2).sum(dim=1)
               * tc.pow(2).sum(dim=1)).clamp_min(1e-16).sqrt()
        r = num / den
        l1 = (pred - target).abs().mean(dim=1)
        return ((1.0 - r) + self.alpha * l1).mean()


class NoamSchedule:
    """Upstream ``OptimizerScheduler(scheduler_type="transformer")``.

    ``lr = d_model**-0.5 * min(step**-0.5, warmup**-1.5 * step)`` stepped once
    per optimiser step, with ``warmup = warmup_epochs * steps_per_epoch``.

    Args:
        optimizer: Optimiser whose param groups get the new lr.
        d_model: Model width used for the scale factor.
        warmup_steps: Steps of linear warmup.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int,
                 warmup_steps: int):
        self.optimizer = optimizer
        self.init_lr = float(d_model) ** -0.5
        self.warmup_steps = max(1, int(warmup_steps))
        self.n_steps = 0
        self.step()

    def step(self) -> None:
        """Advance one optimiser step and set the new learning rate."""
        self.n_steps += 1
        scale = min(self.n_steps ** -0.5,
                    self.warmup_steps ** -1.5 * self.n_steps)
        lr = self.init_lr * scale
        for g in self.optimizer.param_groups:
            g["lr"] = lr


__all__ = ["DECAF_MODEL_KWARGS", "EEG_DELAY_S", "NoamSchedule",
           "PAST_S", "PRESENT_S", "PearsonL1Loss", "build_decaf",
           "load_upstream"]
