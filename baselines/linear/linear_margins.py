#!/usr/bin/env python
"""Per-window margins of the linear backward-model baseline, for confidence analyses.

Same decoder, folds, lambda grid and window positions as ``linear_baseline.py``;
instead of hit/miss it stores the margin ``r(recon, attended) - r(recon, competing)``
of every window, for the test split (final decoder, fitted on train + val) and for
the val split (decoder fitted on train only, so val margins are out-of-sample and
can be used to calibrate).  Competing-talker datasets only.

Windows: 1 s with hop 1 / 0.5 / 0.25 s, 2 s and 5 s with hop = window.

    python baselines/linear/linear_margins.py --dataset kuleuven
Writes ``<cache>/<dataset>/linear_margins.npz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.data import load_dataset, make_folds, split_fold  # noqa: E402
from coupleflow.preprocessing.run_config import split_kw, stream  # noqa: E402
from coupleflow.preprocessing.core import CACHE_ROOT  # noqa: E402
from baselines.linear.linear_baseline import LAMS, MARGIN, lagged, rcorr, zc  # noqa: E402

CELLS = ((1.0, 1.0), (1.0, 0.5), (1.0, 0.25), (2.0, 2.0), (5.0, 5.0))


def key(split: str, w: float, hop: float) -> str:
    return f"{split}_{w:g}s" + ("" if hop == w else f"_h{hop:g}")


def solve(X: list, y: list, lam: float) -> np.ndarray:
    XtX = sum(x.T @ x for x in X)
    sc = np.trace(XtX) / XtX.shape[0]
    return np.linalg.solve(XtX + lam * sc * np.eye(len(XtX)), sum(x.T @ t for x, t in zip(X, y)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stream-name", default=None, help="default: the paper bundle (run_config)")
    ap.add_argument("--groups-name", default="unified_groups.json")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()
    args.stream_name = args.stream_name or stream(args.dataset)
    data = load_dataset(str(CACHE_ROOT), args.dataset, stream_name=args.stream_name,
                        groups_name=args.groups_name)
    trials = data["trials"]
    out: dict[str, list] = {}
    n_folds = len(make_folds(data["groups"], trials, args.n_folds, 42))
    for fold in range(n_folds):
        sp = split_fold(data, fold, args.n_folds, MARGIN, **split_kw(args.dataset))
        by_sub: dict[str, dict[str, list]] = {}
        for split in ("train", "val", "test"):
            for s in sp[split]:
                by_sub.setdefault(trials[s.trial]["subject"],
                                  {"train": [], "val": [], "test": []})[split].append(s)
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
            has_val = bool(d["val"] and d["train"])
            Xv, yv = xy(d["val"]) if has_val else (Xtr, ytr)
            best, lam = -9.0, LAMS[0]
            for lm in LAMS:
                w = solve(Xtr, ytr, lm)
                r = float(np.mean([rcorr(x @ w, y) for x, y in zip(Xv, yv)]))
                if r > best:
                    best, lam = r, lm
            w_train = solve(Xtr, ytr, lam)                       # for out-of-sample val margins
            w_final = solve(Xtr + Xv, ytr + yv, lam) if has_val else w_train
            jobs = [("test", s, w_final) for s in d["test"]]
            if has_val:
                jobs += [("val", s, w_train) for s in d["val"]]
            for split, s, w in jobs:
                t = trials[s.trial]
                if not t["has_unattended"]:
                    continue
                p = lagged(t["eeg"].float().numpy()) @ w
                a, u = t["env_att"].numpy(), t["env_unatt"].numpy()
                for W, H in CELLS:
                    win, hop = int(W * 64), int(H * 64)
                    L = win + 2 * MARGIN
                    lo, hi = (0, len(p)) if split == "test" else (s.start, s.end)
                    for st in range(lo, hi - L + 1, hop):
                        c = slice(st + MARGIN, st + MARGIN + win)
                        k = key(split, W, H)
                        out.setdefault(f"{k}__coupling", []).append(rcorr(p[c], a[c])
                                                                    - rcorr(p[c], u[c]))
                        out.setdefault(f"{k}__trial", []).append(fold * 100000 + s.trial)
                        out.setdefault(f"{k}__start", []).append(st)
                        out.setdefault(f"{k}__fold", []).append(fold)
        print(f"fold {fold} done", flush=True)
    dst = CACHE_ROOT / args.dataset / "linear_margins.npz"
    np.savez_compressed(dst, **{k: np.asarray(v) for k, v in out.items()})
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
