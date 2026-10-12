"""Load a trained CoupleFlow run without its dataset.

    from coupleflow.checkpoint import load_run
    model, cfg, subjects = load_run("runs/final/kuleuven_fold0")

``best.pt`` stores the state dict and the listener list; ``config.yaml`` next to it the
configuration the run was trained with.  The feature-bank widths and the EEG channel count,
which training reads from the data, are recovered here from the configuration and checked
against the state dict.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from coupleflow.data import n_audio_channels
from coupleflow.model import CoupleFlow
from coupleflow.trainer import build_model, feature_banks

# Channels per candidate in each feature bank of the paper's bundles (gammatone sub-bands).
BANK_DIMS = {"sb": 8, "sb28": 28}


def load_run(run_dir: str | Path, device: str = "cpu") -> tuple[CoupleFlow, DictConfig, list[str]]:
    """Return ``(model in eval mode, run config, listener ids in the model's order)``."""
    run = Path(run_dir)
    cfg = OmegaConf.load(run / "config.yaml")
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    sd = ck["model"]
    banks = feature_banks(cfg)
    unknown = [b for b in banks if b not in BANK_DIMS]
    if unknown:
        raise ValueError(f"unknown feature bank(s) {unknown}; add their width to BANK_DIMS")
    bank_dims = [BANK_DIMS[b] for b in banks]
    n_extra = len(cfg.data.get("cand_extra", None) or [])
    n_aud = n_audio_channels(n_extra, bool(cfg.data.get("glimpse", False)), bank_dims)
    if sd["aud_enc.net.0.weight"].shape[1] != n_aud:
        raise ValueError(f"{run}: audio input width {sd['aud_enc.net.0.weight'].shape[1]} "
                         f"!= {n_aud} implied by banks {banks}")
    n_channels = int(sd["U"].shape[1]) // (4 if bool(cfg.data.get("eeg_bands", False)) else 1)
    model = build_model(cfg, len(ck["subjects"]), bank_dims, n_channels=n_channels)
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg, list(ck["subjects"])
