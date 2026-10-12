#!/usr/bin/env python
"""MAESTRO ablation: accuracy vs what the brain contributes, for every component.

Within-listener content-disjoint folds, 0-ms lag window, 4-way candidates. Two candidate
sets: UNMATCHED (the attended talker is the loudest in every trial -- an audio shortcut is
available) and MATCHED (quantile-matched, shortcut removed). Per variant: 5-s accuracy,
across-trial null, contribution (acc - null), circular-shift null (time-locking only),
per-trial accuracy and contribution, flip rate under EEG shuffle, embedding collapse.

    python scripts/tables/maestro_ablation.py   # -> tables/maestro_ablation.{tex,json}
"""
import glob
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402

import numpy as np

R = str(paths.RUNS)
T = Path("paper/tables")
VARIANTS = [("maestro_within_lag0", "full model"), ("mabl_noflow", "no flow head"),
            ("mabl_norank", "no listener correction"), ("mabl_nomatch", "unmatched training"),
            ("mabl_noglimpse", "no glimpse features"), ("mabl_nobanks", "no sub-band features"),
            ("mabl_noadv", "no audio adversary"), ("mabl_nohinge", "no null hinge"),
            ("mabl_nocon", "no InfoNCE"), ("mabl_selacc", "select on accuracy"),
            ("mabl_noside", "no side head"), ("mabl_noaux", "no env/VICReg aux."),
            ("mabl_before", "lag control (EEG before sound)")]


def load(tag):
    out = []
    for d in sorted(glob.glob(f"{R}/{tag}/maestro_fold*")):
        for f in ("results.json", "results_val.json"):
            if os.path.exists(f"{d}/{f}"):
                out.append(json.load(open(f"{d}/{f}"))["results"])
                break
    return out


def pool(rs, cell, key, w="n_windows"):
    v = [(r[cell][key], r[cell].get(w, 1)) for r in rs if cell in r and r[cell].get(key) is not None]
    return float(np.average([a for a, _ in v], weights=[n for _, n in v])) if v else None


res, rows = {}, []
for tag, name in VARIANTS:
    rs = load(tag)
    d = {"folds": len(rs)}
    for cs in ("cands", "cands_matched"):
        c = f"{cs}_5s"
        acc, nul = pool(rs, c, "acc_coupling"), pool(rs, c, "null_coupling")
        d[cs] = {"acc": acc, "null": nul, "contrib": None if acc is None else acc - nul,
                 "circ": pool(rs, c, "null_circ_coupling"),
                 "trial": pool(rs, c, "trial_acc_coupling", "n_trial_decisions"),
                 "trial_null": pool(rs, c, "trial_null_coupling", "n_trial_decisions"),
                 "flip": pool(rs, c, "flip_rate"), "collapse": pool(rs, c, "collapse")}
    res[tag] = d
    u, m = d["cands"], d["cands_matched"]
    f = lambda x: "--" if x is None else f"{100 * x:.1f}"  # noqa: E731
    g = lambda x: "--" if x is None else f"{100 * x:+.1f}"  # noqa: E731
    tc = None if m["trial"] is None else m["trial"] - m["trial_null"]
    col = "--" if m["collapse"] is None else f"{m['collapse']:.2f}"
    rows.append(f"{name} & {d['folds']} & {f(u['acc'])} & {g(u['contrib'])} & {f(m['acc'])} & "
                f"{f(m['null'])} & {g(m['contrib'])} & {f(m['circ'])} & {f(m['trial'])} & "
                f"{g(tc)} & {col} \\\\")
(T / "maestro_ablation.json").write_text(json.dumps(res, indent=1))
(T / "maestro_ablation.tex").write_text(
    "\\begin{tabular}{lc|cc|cccccc|c}\\toprule\n & & \\multicolumn{2}{c|}{unmatched (5\\,s)} & "
    "\\multicolumn{6}{c|}{matched} & \\\\\nVariant & folds & acc. & contrib. & acc. & null & "
    "contrib. & circ.\\ null & trial & trial contrib. & collapse \\\\\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print(open(T / "maestro_ablation.tex").read())
