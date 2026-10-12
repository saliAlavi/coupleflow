#!/usr/bin/env python
"""Scan a global EEG-audio offset to detect stimulus misalignment in a cache.

For each offset delta, fit a ridge backward model (EEG lags 0..250 ms after
the envelope) per subject on 80 % of trials and report held-out Pearson r
with the attended envelope shifted by delta.  A correctly aligned dataset
peaks at delta ~ 0; a constant misalignment shows up as a peak elsewhere.

    python scripts/checks/offset_scan.py --dataset dtu --subjects 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.envelope_common import DEFAULT_CACHE_ROOT, load_streams  # noqa: E402

SR = 64
LAGS = np.arange(0, 17)          # 0..250 ms, EEG after envelope


def lagged(eeg: np.ndarray) -> np.ndarray:
    """(C,T) -> (T, C*L) with EEG[t + lag]."""
    C, T = eeg.shape
    out = np.zeros((T, C * len(LAGS)), np.float32)
    for i, lag in enumerate(LAGS):
        out[: T - lag, i * C:(i + 1) * C] = eeg[:, lag:].T
    return out


def scan_subject(trials: list[dict], offsets: np.ndarray, lam: float = 1e3) -> np.ndarray:
    """Held-out r for each offset.  The design matrix is fixed (edges trimmed by
    the largest offset), so only the target moves and X'X is factorised once."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(trials))
    n_te = max(1, len(idx) // 5)
    te, tr = idx[:n_te], idx[n_te:]
    M = int(np.abs(offsets).max())

    def build(ids):
        X, E = [], []
        for i in ids:
            env = trials[i]["env_att"].numpy()
            env = (env - env.mean()) / (env.std() + 1e-8)
            T = len(env)
            if T <= 2 * M + 64:
                continue
            X.append(lagged(trials[i]["eeg"].numpy())[M:T - M])
            # column k: env[t - offsets[k]] for t in [M, T - M)
            E.append(np.stack([env[M - d:T - M - d] for d in offsets], 1))
        return np.concatenate(X), np.concatenate(E)

    Xtr, Ytr = build(tr)
    Xte, Yte = build(te)
    A = Xtr.T @ Xtr + lam * len(Xtr) / 1e4 * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)                    # (F, n_offsets)
    P = Xte @ W
    P = (P - P.mean(0)) / (P.std(0) + 1e-8)
    Yc = (Yte - Yte.mean(0)) / (Yte.std(0) + 1e-8)
    return (P * Yc).mean(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subjects", type=int, default=6)
    ap.add_argument("--max-s", type=float, default=4.0)
    ap.add_argument("--step-s", type=float, default=0.125)
    ap.add_argument("--stream-name", default=None)
    args = ap.parse_args()
    b = load_streams(DEFAULT_CACHE_ROOT, args.dataset, stream_name=args.stream_name)
    subs = sorted({t["subject"] for t in b["trials"]},
                  key=lambda s: int("".join(c for c in s if c.isdigit())))[: args.subjects]
    offsets = np.arange(-args.max_s, args.max_s + 1e-9, args.step_s)
    offs = np.round(offsets * SR).astype(int)
    allr = []
    for s in subs:
        tr = [t for t in b["trials"] if t["subject"] == s]
        r = scan_subject(tr, offs)
        allr.append(r)
        k = int(np.argmax(r))
        print(f"[{args.dataset}] {s}: peak r {r[k]:+.4f} at offset {offsets[k]:+.3f} s | "
              f"r at 0: {r[np.argmin(np.abs(offsets))]:+.4f}", flush=True)
    m = np.mean(allr, 0)
    k = int(np.argmax(m))
    print(f"[{args.dataset}] MEAN: peak r {m[k]:+.4f} at offset {offsets[k]:+.3f} s | "
          f"r at 0: {m[np.argmin(np.abs(offsets))]:+.4f}")
    out = Path("results/offset_scan") / f"{args.dataset}{'_' + args.stream_name if args.stream_name else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"offsets_s": offsets.tolist(), "subjects": subs,
                               "r": [x.tolist() for x in allr]}, indent=1))


if __name__ == "__main__":
    main()
