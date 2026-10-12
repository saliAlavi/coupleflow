"""Per-dataset settings shared by CoupleFlow's paper runs and every baseline / check script.

These mirror the CoupleFlow paper-run configs (``runs/coupleflow/final*/<ds>_fold*/config.yaml``)
so that baselines, diagnostics and the split audit see exactly the data and splits CoupleFlow
was trained and tested on.
"""

from __future__ import annotations

# Validation settings for ``split_fold``: Fuglsang'20 trials last ~50 s, so the default tail
# rule (last 15% of trials >= 200 s) would leave it without validation data.
SPLIT_KW: dict[str, dict] = {"fuglsang2020": dict(val_frac=0.2, tail_min_s=30.0)}

# Input bundle: MEG-MASC uses all 208 sensors (no per-listener PCA).
STREAM: dict[str, str] = {"megmasc": "unified_native.pt"}


def split_kw(ds: str) -> dict:
    """Keyword arguments for ``split_fold`` (content folds and LOSO alike)."""
    return {"val_mode": "auto", "val_frac": 0.15, "tail_min_s": 200.0, "seed": 42} | SPLIT_KW.get(ds, {})


def stream(ds: str) -> str:
    """Bundle file name for dataset ``ds``."""
    return STREAM.get(ds, "unified_env64.pt")


def check_name(ds: str) -> str:
    """File name of the per-listener tracking check for the paper's bundle."""
    return stream(ds).replace(".pt", ".check.json")
