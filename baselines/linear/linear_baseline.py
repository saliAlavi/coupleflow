#!/usr/bin/env python
"""Linear backward-model (mTRF-style) baseline on the CoupleFlow test folds.

Same stimulus-disjoint folds (``split_fold``, split_seed 42), same bundle, same
non-overlapping windows (hop = window, CoupleFlow's margin of context on both
sides discarded so the window positions coincide) as the CoupleFlow runs.

Per listener: ridge decoder EEG lags 0..250 ms -> attended envelope, fitted on
that listener's training trials (train + val segments of the fold), lambda
chosen on the fold's val segments from a grid.  Decision per window:

* competing   -- r(recon, attended) > r(recon, competing)
* same_talker -- r(recon, attended) > r(recon, attended envelope of a random
  non-overlapping window of the same trial)

Trial accuracy: correlations over the whole test trial (competing only).

    python baselines/linear/linear_baseline.py --dataset kuleuven [--single]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.data import load_dataset, make_folds, split_fold  # noqa: E402
from coupleflow.preprocessing.core import CACHE_ROOT  # noqa: E402
from coupleflow.preprocessing.run_config import split_kw, stream  # noqa: E402

LAGS = np.arange(0, 17)
MARGIN = 100
WINDOWS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
LAMS = (1e-2, 1e-1, 1.0, 1e1, 1e2)


def lagged(eeg: np.ndarray) -> np.ndarray:
    C, T = eeg.shape
    out = np.zeros((T, C * len(LAGS)), np.float32)
    for i, lag in enumerate(LAGS):
        out[: T - lag, i * C:(i + 1) * C] = eeg[:, lag:].T
    return out


def zc(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)


def rcorr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(zc(a) * zc(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stream-name", default=None, help="default: the paper bundle (run_config)")
    ap.add_argument("--groups-name", default="unified_groups.json")
    ap.add_argument("--single", action="store_true", help="single-talker data")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-name", default="linear_baseline.json")
    args = ap.parse_args()
    args.stream_name = args.stream_name or stream(args.dataset)
    data = load_dataset(str(CACHE_ROOT), args.dataset, stream_name=args.stream_name,
                        groups_name=args.groups_name)
    trials = data["trials"]
    rng = np.random.default_rng(0)
    hits = {w: {"competing": [], "same_talker": []} for w in WINDOWS}
    trial_hits = []
    n_no_val = [0]                        # listener-folds whose lambda came from inner CV
    per_sub: dict[str, list] = {}         # 5-s competing hits per listener
    n_folds = None
    n_folds = len(make_folds(data["groups"], trials, args.n_folds, 42))
    for fold in range(n_folds):
        sp = split_fold(data, fold, args.n_folds, MARGIN, **split_kw(args.dataset))
        by_sub: dict[str, dict[str, list]] = {}
        for split in ("train", "val", "test"):
            for s in sp[split]:
                by_sub.setdefault(trials[s.trial]["subject"], {"train": [], "val": [],
                                                               "test": []})[split].append(s)
        for sub, d in by_sub.items():
            if not d["test"] or not (d["train"] or d["val"]):
                continue

            def xy(segs):
                X, y = [], []
                for s in segs:
                    t = trials[s.trial]
                    X.append(lagged(t["eeg"][:, s.start:s.end].float().numpy()))
                    y.append(zc(t["env_att"][s.start:s.end].numpy()))
                return X, y
            Xtr, ytr = xy(d["train"] or d["val"])
            XtX = sum(x.T @ x for x in Xtr)
            Xty = sum(x.T @ y for x, y in zip(Xtr, ytr))
            sc = np.trace(XtX) / XtX.shape[0]
            best, lam_best = -9.0, LAMS[0]
            if d["val"] and d["train"]:
                Xv, yv = xy(d["val"])
                for lam in LAMS:
                    w = np.linalg.solve(XtX + lam * sc * np.eye(len(XtX)), Xty)
                    r = float(np.mean([rcorr(x @ w, y) for x, y in zip(Xv, yv)]))
                    if r > best:
                        best, lam_best = r, lam
            else:
                # No validation data for this listener (its validation groups hold none of its
                # trials): choose lambda by leave-one-segment-out over its training segments.
                n_no_val[0] += 1
                parts = [(x.T @ x, x.T @ y) for x, y in zip(Xtr, ytr)]
                for lam in LAMS:
                    rs = []
                    for k, (x, y) in enumerate(zip(Xtr, ytr)):
                        if len(Xtr) < 2:
                            break
                        A = XtX - parts[k][0]
                        w = np.linalg.solve(A + lam * (np.trace(A) / len(A)) * np.eye(len(A)),
                                            Xty - parts[k][1])
                        rs.append(rcorr(x @ w, y))
                    r = float(np.mean(rs)) if rs else -9.0
                    if r > best:
                        best, lam_best = r, lam
            if d["val"] and d["train"]:       # refit on train + val with the chosen lambda
                XtX = XtX + sum(x.T @ x for x in Xv)
                Xty = Xty + sum(x.T @ y for x, y in zip(Xv, yv))
            bw = np.linalg.solve(XtX + lam_best * sc * np.eye(len(XtX)), Xty)
            for s in d["test"]:
                t = trials[s.trial]
                p = lagged(t["eeg"].float().numpy()) @ bw
                a = t["env_att"].numpy()
                u = t["env_unatt"].numpy() if t["has_unattended"] else None
                n = len(p)
                if u is not None and not args.single:
                    trial_hits.append(rcorr(p, a) > rcorr(p, u))
                for W in WINDOWS:
                    win = int(W * 64)
                    L = win + 2 * MARGIN
                    for st in range(0, n - L + 1, win):
                        c = slice(st + MARGIN, st + MARGIN + win)
                        ra = rcorr(p[c], a[c])
                        if u is not None and not args.single:
                            hits[W]["competing"].append(ra > rcorr(p[c], u[c]))
                            if W == 5.0:
                                per_sub.setdefault(sub, []).append(hits[W]["competing"][-1])
                        ok = [q for q in range(0, n - L + 1, max(1, win // 2))
                              if q + L <= st or q >= st + L]
                        if ok:
                            q = int(rng.choice(ok)) + MARGIN
                            hits[W]["same_talker"].append(ra > rcorr(p[c], a[q:q + win]))
        print(f"fold {fold} done", flush=True)
    res = {"dataset": args.dataset, "n_folds": n_folds, "windows": {},
           "per_subject_5s_competing": {k: [float(np.mean(v)), len(v)] for k, v in per_sub.items()}}
    for W in WINDOWS:
        res["windows"][f"{W:g}s"] = {k: (float(np.mean(v)) if v else None, len(v))
                                     for k, v in hits[W].items()}
    res["n_listener_folds_inner_cv"] = n_no_val[0]
    res["stream"], res["split_kw"] = args.stream_name, split_kw(args.dataset)
    res["trial_competing"] = (float(np.mean(trial_hits)) if trial_hits else None, len(trial_hits))
    print(json.dumps(res, indent=1))
    out = CACHE_ROOT / args.dataset / args.out_name
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
