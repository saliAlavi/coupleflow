#!/usr/bin/env python
"""Sanity check for a unified bundle: alignment and envelope tracking.

Per subject, a ridge backward model (EEG lags 0..250 ms after the envelope)
is fitted with leave-trials-out CV and scored on held-out trials:

* ``r_att``   -- held-out r with the attended envelope (must be > 0; for 64-ch
  EEG typically 0.05-0.20)
* ``r_unatt`` -- r with the competing envelope (two-talker data; < r_att)
* ``r_shift`` -- r with the attended envelope shifted by +-{1..4} s (chance
  reference; a correctly aligned bundle has r_att >> r_shift)
* ``acc30``   -- fraction of 30-s windows where r_att > r_unatt

    python scripts/checks/check.py --dataset kuleuven [--subjects 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.preprocessing.core import BUNDLE, CACHE_ROOT  # noqa: E402

LAGS = np.arange(0, 17)


def lagged(eeg: np.ndarray) -> np.ndarray:
    C, T = eeg.shape
    out = np.zeros((T, C * len(LAGS)), np.float32)
    for i, lag in enumerate(LAGS):
        out[: T - lag, i * C:(i + 1) * C] = eeg[:, lag:].T
    return out


def z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(z(a) * z(b)))


def check_subject(trials: list[dict], lam: float = 1e-2, n_folds: int = 4) -> dict:
    """Leave-trials-out backward model; ``lam`` is relative to the mean eigenvalue."""
    X = [lagged(t["eeg"].float().numpy()) for t in trials]
    A = [z(t["env_att"].numpy()) for t in trials]
    U = [z(t["env_unatt"].numpy()) if t["has_unattended"] else None for t in trials]
    G = [x.T @ x for x in X]
    V = [x.T @ a for x, a in zip(X, A)]
    idx = np.arange(len(trials))
    folds = np.array_split(np.random.default_rng(0).permutation(idx), min(n_folds, len(idx)))
    ra, ru, rs, wins = [], [], [], []
    for te in folds:
        tr = np.setdiff1d(idx, te)
        if len(tr) == 0:
            continue
        XtX = sum(G[i] for i in tr)
        Xty = sum(V[i] for i in tr)
        w = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]) * np.trace(XtX) / XtX.shape[0], Xty)
        for i in te:
            p = X[i] @ w
            ra.append(corr(p, A[i]))
            for s in (64, 128, 256):
                if len(p) > 2 * s:
                    rs.append(corr(p[s:], A[i][:-s]))
            if U[i] is not None:
                ru.append(corr(p, U[i]))
                W = 30 * 64
                for s in range(0, len(p) - W + 1, W):
                    wins.append(corr(p[s:s + W], A[i][s:s + W]) > corr(p[s:s + W], U[i][s:s + W]))
    return {"r_att": float(np.mean(ra)), "r_unatt": float(np.mean(ru)) if ru else None,
            "r_shift": float(np.mean(rs)) if rs else None,
            "acc30": float(np.mean(wins)) if wins else None, "n_trials": len(trials)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", default=BUNDLE)
    ap.add_argument("--subjects", type=int, default=6)
    args = ap.parse_args()
    b = torch.load(CACHE_ROOT / args.dataset / args.name, map_location="cpu", weights_only=False)
    by: dict[str, list] = {}
    for t in b["trials"]:
        by.setdefault(t["subject"], []).append(t)
    res = {}
    for s in sorted(by)[: args.subjects]:
        if len(by[s]) < 2:
            continue
        res[s] = check_subject(by[s])
        print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[s].items()},
              flush=True)
    keys = ("r_att", "r_unatt", "r_shift", "acc30")
    summ = {k: float(np.mean([r[k] for r in res.values() if r[k] is not None]))
            for k in keys if any(r[k] is not None for r in res.values())}
    print("MEAN", {k: round(v, 4) for k, v in summ.items()})
    out = CACHE_ROOT / args.dataset / args.name.replace(".pt", ".check.json")
    out.write_text(json.dumps({"per_subject": res, "mean": summ}, indent=1))


if __name__ == "__main__":
    main()
