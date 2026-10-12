#!/usr/bin/env python
"""MAESTRO table + numbers for the paper.

4-way MATCHED candidates (chance 0.25), EEG only. CoupleFlow = mean +- s.d. over 3 seeds of
pooled per-trial accuracy and per-trial contribution (accuracy - EEG-shuffle null), plus
5-s window accuracy. Reference rows are the numbers reported by MERIT (EEG alone) and by the
dataset benchmark (EEG only, 30-s windows = one decision per trial), under their protocols.

    python scripts/tables/maestro_paper.py   # -> tables/maestro_main.{tex,json}
"""
import glob
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = str(paths.RUNS)
T = Path("paper/tables")
SEEDS = {"within": ["maestro_within_lag0", "maestro_within_lag0_s1", "maestro_within_lag0_s2"],
         "loso": ["maestro_loso_lag0", "maestro_loso_lag0_s1", "maestro_loso_lag0_s2"]}
# the release's timing (anchor at the earliest playback timestamp, no latency correction),
# as used by MERIT and the benchmark: the like-for-like comparison with their numbers
PKG = {"within": ["maestro_pkg_within", "maestro_pkg_within_s1", "maestro_pkg_within_s2"],
       "loso": ["maestro_pkg_loso", "maestro_pkg_loso_s1", "maestro_pkg_loso_s2"]}
REF = {"within": {"MERIT (EEG alone)": (0.6915, 0.3971), "Benchmark (EEG, 30\\,s)": (0.591, 0.320)},
       "loso": {"MERIT (EEG alone)": (0.6000, 0.3132), "Benchmark (EEG, 30\\,s)": (0.619, 0.363)}}


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


def stats(tag, cs="cands_matched"):
    rs = load(tag)
    tr = pool(rs, f"{cs}_5s", "trial_acc_coupling", "n_trial_decisions")
    trn = pool(rs, f"{cs}_5s", "trial_null_coupling", "n_trial_decisions")
    return {"folds": len(rs), "trial": tr, "contrib": None if tr is None else tr - trn,
            "w5": pool(rs, f"{cs}_5s", "acc_coupling"), "w20": pool(rs, f"{cs}_20s", "acc_coupling"),
            "st2_5": pool(rs, "same_talker_5s", "acc_coupling"),
            "st3_5": pool(rs, "same_talker3_5s", "acc_coupling")}


def agg(tags):
    st = [stats(t) for t in tags]
    return st, {k: (float(np.mean([x[k] for x in st])), float(np.std([x[k] for x in st], ddof=1)))
                for k in ("trial", "contrib", "w5", "w20", "st2_5", "st3_5")}


def cell(m, k, plus=False):
    return (f"{'+' if plus else ''}{100 * m[k][0]:.1f}{{\\scriptsize$\\pm${100 * m[k][1]:.1f}}}")


res = {}
rows = []
for proto in ("within", "loso"):
    s_c, m_c = agg(SEEDS[proto])
    s_p, m_p = agg(PKG[proto])
    res[proto] = {"seeds": s_c, "mean_sd": m_c}
    res[f"{proto}_pkg"] = {"seeds": s_p, "mean_sd": m_p}
    name = "within listeners" if proto == "within" else "unseen listeners (LOSO)"
    rows.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{name}}}}} \\\\")
    rows.append(f"CoupleFlow, release timing & {cell(m_p, 'w5')} & {cell(m_p, 'trial')} & "
                f"{cell(m_p, 'contrib', True)} \\\\")
    for rn, (a, c) in REF[proto].items():
        rows.append(f"{rn}, release timing & -- & {100 * a:.1f} & +{100 * c:.1f} \\\\")
    rows.append(f"CoupleFlow, latency-corrected (causal) & {cell(m_c, 'w5')} & {cell(m_c, 'trial')} & "
                f"{cell(m_c, 'contrib', True)} \\\\")
(T / "maestro_main.tex").write_text(
    "\\begin{tabular}{lccc}\\toprule\nModel & 5-s window & per trial & contribution \\\\\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# alignment comparison (within, seed 0) and pre-sound window
res["align"] = {"compensated": stats("maestro_within_lag0"),
                "package_exact": stats("maestro_pkg_within"),
                "compensated_pre_window": stats("maestro_within_pre50")}
# accuracy vs attended-talker SNR (within, seed 0)
# per-trial SNR from the release metadata; without the release, its two-column extract
# <data>/maestro/trial_snr.csv (shipped with the repository; CC BY-NC-SA 4.0)
src = paths.raw("maestro") / "metadata" / "trials.csv"
if not src.exists():
    src = paths.DATA / "maestro" / "trial_snr.csv"
tr = pd.read_csv(src).set_index("trial_id")
pts = []
for p in glob.glob(f"{R}/maestro_within_lag0/maestro_fold*/results*.json"):
    for t in json.load(open(p))["results"]["cands_matched_5s"]["per_trial"]:
        pts.append((tr.loc[t["trial"].split("_", 1)[1], "snr_db"], t["acc_coupling"]))
a = np.array(pts)
rho, pv = spearmanr(a[:, 0], a[:, 1])
res["snr"] = {"rho": float(rho), "p": float(pv), "n": len(a)}
(T / "maestro_main.json").write_text(json.dumps(res, indent=1))
print(open(T / "maestro_main.tex").read())
print(json.dumps({k: v for k, v in res.items() if k != "within" and k != "loso"}, indent=1)[:2500])
