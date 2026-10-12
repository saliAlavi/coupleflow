#!/usr/bin/env python
"""Test Prop. 3 quantitatively: does per-sample tracking predict the accumulation-law kappa?

Inputs: <ds>/unified_env64.check.json (per-listener backward-model r with the attended and
competing envelopes, all listeners) and the kappa fits of the linear decoder
(snr_law_linear.json) and of CoupleFlow (snr_law_final.json).

Predictions (Fisher-z, small rho):
  match-mismatch (negative = same talker, other time, rho_neg = 0):  Dz_mm   = atanh(r_att)
  competing talker:                                                    Dz_comp = atanh(r_att) - atanh(r_unatt)
  => kappa = Dz * sqrt(f / 2), one effective rate f shared by all datasets and tasks
  => kappa_comp / kappa_mm = 1 - gamma,  gamma = r_unatt / r_att  (ignored-talker tracking)

    python scripts/figures/kappa_theory.py   # -> paper/tables/kappa_theory.{json,tex}
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.preprocessing.run_config import check_name  # noqa: E402

C = paths.DATA
F = Path("figures")
T = Path("paper/tables")
TWO = ["kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp"]
ONE = ["sparrkulee", "megmasc"]
NAMES = {"kuleuven": "KU Leuven", "dtu": "DTU", "fuglsang2020": "Fuglsang'20", "esaa": "ESAA",
         "uhd255": "KUL-255", "broderick_cp": "Broderick", "sparrkulee": "SparrKULee",
         "megmasc": "MEG-MASC"}
lin = json.loads((F / "snr_law_linear.json").read_text())
cf = json.loads((F / "snr_law_final.json").read_text())

track = {}
for ds in TWO + ONE:
    ps = json.loads((C / ds / check_name(ds)).read_text())["per_subject"]
    ra = float(np.mean([v["r_att"] for v in ps.values()]))
    ru = float(np.mean([v["r_unatt"] for v in ps.values()])) if ds in TWO else 0.0
    track[ds] = {"r_att": ra, "r_unatt": ru, "gamma": ru / ra, "n": len(ps)}

# 1. task effect: kappa_comp / kappa_mm vs 1 - gamma
task = {}
for src, K in (("linear", lin), ("coupleflow", cf)):
    pred = np.array([1 - track[d]["gamma"] for d in TWO])
    obs = np.array([K[f"{d}/comp"]["kappa"] / K[f"{d}/same"]["kappa"] for d in TWO])
    r, p = pearsonr(pred, obs)
    task[src] = {"pred": pred.tolist(), "obs": obs.tolist(), "mae": float(np.abs(pred - obs).mean()),
                 "r": float(r), "p": float(p)}

# 2. absolute kappa of the linear decoder vs Delta-zeta (one shared f, least squares through 0)
X, Y, lab = [], [], []
for d in TWO + ONE:
    za = np.arctanh(track[d]["r_att"])
    X.append(za), Y.append(lin[f"{d}/same"]["kappa"]), lab.append(f"{d}/same")
    if d in TWO:
        X.append(za - np.arctanh(track[d]["r_unatt"])), Y.append(lin[f"{d}/comp"]["kappa"])
        lab.append(f"{d}/comp")
X, Y = np.array(X), np.array(Y)
slope = float((X @ Y) / (X @ X))                 # kappa = slope * Dz, slope = sqrt(f/2)
f_eff = 2 * slope ** 2
pred = slope * X
r_abs, p_abs = pearsonr(X, Y)
absolute = {"f_eff": f_eff, "slope": slope, "r": float(r_abs), "p": float(p_abs),
            "mape": float(np.mean(np.abs(pred - Y) / Y)),
            "points": {l: {"dz": float(x), "kappa": float(y), "pred": float(q)}
                       for l, x, y, q in zip(lab, X, Y, pred)}}

# 3. MEG vs EEG, same task (match-mismatch), same listening condition (single talker)
meg, eeg = track["megmasc"], track["sparrkulee"]
decomp = {
    "r_ratio_meg_over_sparr": meg["r_att"] / eeg["r_att"],
    "dz_ratio_pred": float(np.arctanh(meg["r_att"]) / np.arctanh(eeg["r_att"])),
    "kappa_lin_ratio_obs": lin["megmasc/same"]["kappa"] / lin["sparrkulee/same"]["kappa"],
    "kappa_cf_ratio_obs": cf["megmasc/same"]["kappa"] / cf["sparrkulee/same"]["kappa"],
    "task_factor_mean": float(np.mean([1 / (1 - track[d]["gamma"]) for d in TWO])),
    "task_factor_range": [float(min(1 / (1 - track[d]["gamma"]) for d in TWO)),
                          float(max(1 / (1 - track[d]["gamma"]) for d in TWO))],
    "two_talker_r_att_range": [min(track[d]["r_att"] for d in TWO),
                               max(track[d]["r_att"] for d in TWO)],
}
res = {"tracking": track, "task_effect": task, "absolute": absolute, "meg_vs_eeg": decomp}
(T / "kappa_theory.json").write_text(json.dumps(res, indent=1))

rows = []
for i, d in enumerate(TWO):
    rows.append(f"{NAMES[d]} & {track[d]['r_att']:.3f} & {track[d]['r_unatt']:.3f} & "
                f"{track[d]['gamma']:.2f} & {1 - track[d]['gamma']:.2f} & "
                f"{task['linear']['obs'][i]:.2f} & {task['coupleflow']['obs'][i]:.2f} \\\\")
for d in ONE:
    rows.append(f"{NAMES[d]} & {track[d]['r_att']:.3f} & -- & -- & -- & -- & -- \\\\")
(T / "kappa_theory.tex").write_text(
    "\\begin{tabular}{lcccc|cc}\\toprule\n & & & & pred. & \\multicolumn{2}{c}{obs.\\ "
    "$\\kappa_{\\rm comp}/\\kappa_{\\rm mm}$} \\\\\nDataset & $r_{\\rm att}$ & $r_{\\rm comp}$ & "
    "$\\gamma$ & $1-\\gamma$ & linear & CoupleFlow \\\\\\midrule\n" + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\n")
print(json.dumps({k: v for k, v in res.items() if k != "absolute"}, indent=1))
print("absolute:", {k: v for k, v in absolute.items() if k != "points"})
for l, v in absolute["points"].items():
    print(f"  {l:22s} dz {v['dz']:.3f} kappa {v['kappa']:.3f} pred {v['pred']:.3f}")
