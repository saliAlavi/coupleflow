#!/usr/bin/env python
"""E6 (rebuttal): linear backward model on MAESTRO, 4-way candidates, release vs corrected timing.

Same ridge recipe as ``baselines/linear/linear_baseline.py`` (within listeners: one decoder
per listener, lambda on the fold's validation trials, refit on train + val) and
``baselines/linear/linear_loso.py`` (unseen listeners: one pooled decoder, lambda on
validation), on CoupleFlow's MAESTRO folds and 4-way candidate sets (see ``common.py``).

Lag sets (EEG after the envelope sample):
* ``250``  -- 0..250 ms, every sample (the paper's linear baseline);
* ``1000`` -- 0..969 ms, every second sample (CoupleFlow's 0-ms lag window, 0..984 ms).

    python scripts/rebuttal/e6/linear_maestro.py --timing corrected --protocol within --lags 250
Writes /fs/scratch/PAS2301/alialavi/rebuttal/e6/linear/<timing>_<protocol>_lag<lags>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_ROOT, Collector, load, n_folds, split  # noqa: E402

LAMS = (1e-2, 1e-1, 1.0, 1e1, 1e2)          # as baselines/linear/linear_baseline.py
LAG_SETS = {"250": np.arange(0, 17), "1000": np.arange(0, 63, 2)}


def lagged(eeg: np.ndarray, lags: np.ndarray) -> np.ndarray:
    C, T = eeg.shape
    out = np.zeros((T, C * len(lags)), np.float32)
    for i, lag in enumerate(lags):
        out[: T - lag, i * C:(i + 1) * C] = eeg[:, lag:].T
    return out


def zc(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)


def rcorr(a, b) -> float:
    return float(np.mean(zc(a) * zc(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", choices=["corrected", "release"], required=True)
    ap.add_argument("--protocol", choices=["within", "loso"], required=True)
    ap.add_argument("--lags", choices=list(LAG_SETS), default="250")
    ap.add_argument("--max-folds", type=int, default=None, help="smoke test")
    a = ap.parse_args()
    t0 = time.time()
    lags = LAG_SETS[a.lags]
    data = load(a.timing)
    tr = data["trials"]
    print(f"loaded {len(tr)} trials in {time.time() - t0:.0f}s", flush=True)

    cache: dict[int, tuple] = {}          # whole-trial X'X, X'y (all segments are whole trials)

    def xtx(seg):
        n = tr[seg.trial]["eeg"].shape[1]
        if seg.start == 0 and seg.end == n:
            if seg.trial not in cache:
                x = lagged(tr[seg.trial]["eeg"].float().numpy(), lags)
                y = zc(tr[seg.trial]["env_att"].numpy())
                cache[seg.trial] = ((x.T @ x).astype(np.float32), (x.T @ y).astype(np.float64))
            return cache[seg.trial]
        x = lagged(tr[seg.trial]["eeg"][:, seg.start:seg.end].float().numpy(), lags)
        y = zc(tr[seg.trial]["env_att"][seg.start:seg.end].numpy())
        return x.T @ x, x.T @ y

    def sums(segs):
        A, b = 0.0, 0.0
        for s in segs:
            p, q = xtx(s)
            A = A + p.astype(np.float64)
            b = b + q
        return A, b

    def val_r(w, segs):
        return float(np.mean([rcorr(lagged(tr[s.trial]["eeg"][:, s.start:s.end].float().numpy(),
                                           lags) @ w,
                                    zc(tr[s.trial]["env_att"][s.start:s.end].numpy()))
                              for s in segs]))

    col = Collector()
    lam_used = []
    nf = n_folds(data, a.protocol)
    if a.max_folds:
        nf = min(nf, a.max_folds)
    for f in range(nf):
        sp = split(data, a.protocol, f)
        if a.protocol == "within":
            by: dict[str, dict] = {}
            for k in ("train", "val", "test"):
                for s in sp[k]:
                    by.setdefault(tr[s.trial]["subject"], {"train": [], "val": [], "test": []})[k].append(s)
            jobs = [(sub, d["train"], d["val"], d["test"]) for sub, d in by.items()
                    if d["test"] and d["train"] and d["val"]]
            assert len(jobs) == len(by), "a listener without train/val/test data"
        else:
            jobs = [(sp["test_subject"], sp["train"], sp["val"], sp["test"])]
        for sub, trs, vas, tes in jobs:
            A, b = sums(trs)
            sc = np.trace(A) / A.shape[0]
            best, lam = -9.0, LAMS[0]
            for lm in LAMS:
                w = np.linalg.solve(A + lm * sc * np.eye(len(A)), b)
                r = val_r(w, vas)
                if r > best:
                    best, lam = r, lm
            if a.protocol == "within":        # refit on train + val (linear_baseline.py)
                Av, bv = sums(vas)
                A, b = A + Av, b + bv
            w = np.linalg.solve(A + lam * sc * np.eye(len(A)), b)
            lam_used.append(lam)
            for s in tes:
                p = lagged(tr[s.trial]["eeg"].float().numpy(), lags) @ w
                col.add_recon(data, s.trial, p, s.start, s.end)
        print(f"fold {f}: {len(jobs)} decoder(s), {time.time() - t0:.0f}s", flush=True)
    res = {"model": f"linear_lag{a.lags}", "timing": a.timing, "protocol": a.protocol,
           "lags_samples": lags.tolist(), "n_folds": nf, "lambdas": lam_used,
           **col.summary(), "minutes": (time.time() - t0) / 60}
    out = OUT_ROOT / "linear" / f"{a.timing}_{a.protocol}_lag{a.lags}{'_smoke' if a.max_folds else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("windows", "trial_sum5", "trial_whole")}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
