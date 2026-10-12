#!/usr/bin/env python
"""Subject-independent linear backward model under CoupleFlow's LOSO folds (diagnostic).

Same LOSO folds as ``split_mode=loso`` (held-out listener x held-out content). One ridge
decoder (EEG lags 0..250 ms -> attended envelope) pooled over all training listeners,
lambda chosen on the fold's validation segments, scored on the held-out listener's test
trials with 5-s windows: correct when r(recon, attended) > r(recon, competing).

    python baselines/linear/linear_loso.py --dataset broderick_cp
Writes <cache>/<ds>/linear_loso.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.data import load_dataset, split_fold  # noqa: E402
from coupleflow.preprocessing.core import CACHE_ROOT  # noqa: E402
from coupleflow.preprocessing.run_config import split_kw, stream  # noqa: E402
from baselines.linear.linear_baseline import LAMS, MARGIN, lagged, rcorr, zc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--window", type=float, default=5.0)
    a = ap.parse_args()
    data = load_dataset(str(CACHE_ROOT), a.dataset, stream_name=stream(a.dataset),
                        groups_name="unified_groups.json")
    data["cand_idx"] = []
    tr = data["trials"]
    win = int(a.window * 64)
    hits, per = [], {}
    for f in range(len(data["subjects"])):
        sp = split_fold(data, f, 5, MARGIN, mode="loso", **split_kw(a.dataset))

        def acc(segs):
            XtX, Xty = 0, 0
            for s in segs:
                x = lagged(tr[s.trial]["eeg"][:, s.start:s.end].float().numpy())
                y = zc(tr[s.trial]["env_att"][s.start:s.end].numpy())
                XtX = XtX + x.T @ x
                Xty = Xty + x.T @ y
            return XtX, Xty
        A, b = acc(sp["train"])
        sc = np.trace(A) / A.shape[0]
        best, lam = -9, LAMS[0]
        for lm in LAMS:
            w = np.linalg.solve(A + lm * sc * np.eye(len(A)), b)
            r = np.mean([rcorr(lagged(tr[s.trial]["eeg"][:, s.start:s.end].float().numpy()) @ w,
                               zc(tr[s.trial]["env_att"][s.start:s.end].numpy()))
                         for s in sp["val"][:40]])
            if r > best:
                best, lam = r, lm
        w = np.linalg.solve(A + lam * sc * np.eye(len(A)), b)
        h = []
        for s in sp["test"]:
            t = tr[s.trial]
            p = lagged(t["eeg"].float().numpy()) @ w
            at, un = t["env_att"].numpy(), t["env_unatt"].numpy()
            for st in range(0, len(p) - win - 2 * MARGIN + 1, win):
                c = slice(st + MARGIN, st + MARGIN + win)
                h.append(rcorr(p[c], at[c]) > rcorr(p[c], un[c]))
        per[sp["test_subject"]] = [float(np.mean(h)) if h else None, len(h)]
        hits += h
        print(f"fold {f} {sp['test_subject']}: {np.mean(h) if h else float('nan'):.3f} (n={len(h)})",
              flush=True)
    out = {"dataset": a.dataset, "window_s": a.window, "acc": float(np.mean(hits)),
           "n_windows": len(hits), "per_listener": per}
    print(json.dumps({k: v for k, v in out.items() if k != "per_listener"}))
    (CACHE_ROOT / a.dataset / "linear_loso.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
