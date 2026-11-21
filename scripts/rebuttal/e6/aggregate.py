#!/usr/bin/env python
"""E6 (rebuttal): MAESTRO baselines at release vs corrected timing, next to CoupleFlow.

Reads the E6 outputs (/fs/scratch/PAS2301/alialavi/rebuttal/e6/{linear,deep}), CoupleFlow's
numbers from paper/tables/maestro_main.json, and the published MERIT / benchmark numbers
from the REF dict of scripts/tables/maestro_paper.py (parsed, not executed: running that
script rewrites the paper tables).  Prints markdown tables (4-way, chance 25%).

    python scripts/rebuttal/e6/aggregate.py [--set cands_matched|cands]
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E6 = Path("/fs/scratch/PAS2301/alialavi/rebuttal/e6")
PROTOS = ("within", "loso")
NFOLD = {"within": 5, "loso": 16}


def published() -> dict:
    src = (ROOT / "scripts/tables/maestro_paper.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "REF" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("REF not found in maestro_paper.py")


def pool(files: list[Path], cs: str) -> dict:
    """Window- / trial-weighted pooling over folds (as maestro_paper.pool)."""
    acc = {"w5": [0.0, 0], "w5_null": [0.0, 0], "sum5": [0.0, 0], "sum5_null": [0.0, 0],
           "whole": [0.0, 0]}
    for f in files:
        r = json.loads(f.read_text())
        w = r["windows"].get("5s", {}).get(cs)
        if w:
            acc["w5"][0] += w["acc"] * w["n_windows"]; acc["w5"][1] += w["n_windows"]
            acc["w5_null"][0] += w["null"] * w["n_windows"]; acc["w5_null"][1] += w["n_windows"]
        t = r["trial_sum5"].get(cs)
        if t:
            acc["sum5"][0] += t["acc"] * t["n_trials"]; acc["sum5"][1] += t["n_trials"]
            acc["sum5_null"][0] += t["null"] * t["n_trials"]; acc["sum5_null"][1] += t["n_trials"]
        t = r["trial_whole"].get(cs)
        if t:
            acc["whole"][0] += t["acc"] * t["n_trials"]; acc["whole"][1] += t["n_trials"]
    return {k: (s / n if n else None) for k, (s, n) in acc.items()} | {"n_files": len(files)}


def pct(x, sd=None):
    if x is None:
        return "–"
    return f"{100 * x:.1f}" + (f" ± {100 * sd:.1f}" if sd is not None else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="cands_matched", choices=["cands_matched", "cands"])
    a = ap.parse_args()
    cf = json.loads((ROOT / "paper/tables/maestro_main.json").read_text())
    ref = published()
    print(f"### MAESTRO, 4-way {'quantile-matched' if a.set == 'cands_matched' else 'as-released'} "
          "candidates (chance 25%)\n")
    print("Per trial = sum of the trial's 5-s window scores (CoupleFlow's rule); whole trial = one "
          "decision over the ~30-s trial (the benchmark's 30-s windows). Null = across-trial EEG "
          "permutation, per trial.\n")
    hdr = ("| Model | Timing | Within: 5 s | Within: per trial | Within: whole trial | Within: null "
           "| Unseen: 5 s | Unseen: per trial | Unseen: whole trial | Unseen: null | seeds |")
    print(hdr)
    print("|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
    rows = []
    # CoupleFlow (paper runs; matched candidates only)
    for timing, sfx in (("release", "_pkg"), ("corrected", "")):
        cells = []
        for p in PROTOS:
            m = cf[f"{p}{sfx}"]["mean_sd"]
            cells += [pct(*m["w5"]), pct(*m["trial"]), "–",
                      pct(m["trial"][0] - m["contrib"][0])]
        rows.append(["CoupleFlow (paper)", timing] + cells + ["3 + 3"])
    for name, (a_w, c_w) in ref["within"].items():
        a_l, c_l = list(ref["loso"].values())[list(ref["within"]).index(name)]
        rows.append([f"{name.replace(chr(92) + ',', ' ')} (published)", "release",
                     "–", pct(a_w), "–", "–", "–", pct(a_l), "–", "–", "published"])
    models = [("linear, lags 0–250 ms", "linear", "lag250"), ("linear, lags 0–969 ms", "linear", "lag1000"),
              ("VLAAI", "deep/vlaai", None), ("dilated MM", "deep/mm", None)]
    for label, sub, lag in models:
        for timing in ("release", "corrected"):
            cells, nf = [], []
            for p in PROTOS:
                if lag:      # deterministic: one job pools all folds
                    per_seed = [[Path(f) for f in glob.glob(str(E6 / sub / f"{timing}_{p}_{lag}.json"))]]
                else:        # seeds 0, 1, 2; a seed counts only with all its folds
                    per_seed = []
                    for sd in (0, 1, 2):
                        sfx = f"_s{sd}" if sd else ""
                        fs = [E6 / sub / f"{timing}_{p}_fold{k}{sfx}.json" for k in range(NFOLD[p])]
                        if all(f.exists() for f in fs):
                            per_seed.append(fs)
                rs = [pool(fs, a.set) for fs in per_seed if fs]
                for k in ("w5", "sum5", "whole", "sum5_null"):
                    v = [r[k] for r in rs if r[k] is not None]
                    if not v:
                        cells.append("–")
                    elif len(v) == 1 or k == "sum5_null":
                        cells.append(pct(float(np.mean(v))))
                    else:
                        cells.append(pct(float(np.mean(v)), float(np.std(v, ddof=1))))
                nf.append(f"{len(rs)}")
            rows.append([label, timing] + cells + [" + ".join(nf)])
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print("\nCoupleFlow null = per-trial accuracy minus contribution (EEG-shuffle null), "
          "from maestro_main.json. Published rows sit in the per-trial column, as in the paper's "
          "Table 4 (one decision per trial). Deep rows: mean ± s.d. over complete seeds "
          "(within + unseen); linear rows are deterministic (one fit).")


if __name__ == "__main__":
    main()
