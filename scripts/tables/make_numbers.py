#!/usr/bin/env python
"""Every number quoted in the paper text, as LaTeX macros (paper/tables/numbers.tex).

Run after make_tables.py, snr_law.py --tag final, leak_table.py, spatial_table.py:
    python scripts/tables/make_numbers.py
The text uses only these macros, so prose and tables cannot disagree.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402

import numpy as np

T = Path("paper/tables")
F = Path("figures")
tab = json.loads((T / "tables.json").read_text())
law = json.loads((F / "snr_law_final.json").read_text())
lawn = json.loads((F / "snr_law_unified_native.json").read_text())
law64 = json.loads((F / "snr_law_unified.json").read_text())
leak = json.loads((T / "leak.json").read_text())
spat = json.loads((T / "spatial.json").read_text())

M: dict[str, str] = {}
p = lambda x: f"{100 * x:.1f}"  # noqa: E731
TWO = ["kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp"]
ALL = TWO + ["sparrkulee", "megmasc"]
KEY = {"kuleuven": "KUL", "dtu": "DTU", "fuglsang2020": "Fug", "esaa": "ESAA",
       "uhd255": "UHD", "broderick_cp": "Brod", "sparrkulee": "Sparr", "megmasc": "MEG"}

# main table margins vs strongest baseline
margins, wins = {}, 0
for ds in ALL:
    r = tab["main"][ds]
    best = max(v for k, v in r.items() if k in ("linear", "vlaai", "mm") and v is not None)
    d = r["cf"] - best
    margins[ds] = d
    wins += d > 0
    M[f"cf{KEY[ds]}"] = p(r["cf"])
    M[f"mg{KEY[ds]}"] = f"{100 * d:+.1f}"
    M[f"best{KEY[ds]}"] = p(best)
M["nWins"] = {6: "six", 7: "seven", 8: "eight", 5: "five"}[wins]
M["maxMargin"] = f"{100 * max(margins.values()):.1f}"
two = [tab["main"][d]["cf"] for d in TWO]
M["twoMin"], M["twoMax"] = f"{100 * min(two):.0f}", f"{100 * max(two):.0f}"
sds = [tab["main"][d]["cf_sd"] for d in ALL if tab["main"][d]["cf_sd"] is not None]
M["sdMin"], M["sdMax"] = f"{100 * min(sds):.1f}", f"{100 * max(sds):.1f}"
trials = [tab["main"][d]["trial"] for d in TWO]
M["trialNge"] = str(sum(t >= 0.93 for t in trials))
M["trialMinNT"] = f"{100 * min(t for t in trials if t >= 0.93):.0f}"
M["trialMax"] = f"{100 * max(trials):.0f}"

# accumulation law (final configuration)
ks = {k: v["kappa"] for k, v in law.items()}
eeg = [v for k, v in ks.items() if not k.startswith("megmasc")]
M["kEEGmin"], M["kEEGmax"] = f"{min(eeg):.2f}", f"{max(eeg):.2f}"
M["kMEG"] = f"{ks['megmasc/same']:.2f}"
t95 = {k: v["T95_s"] for k, v in law.items()}
t95e = [v for k, v in t95.items() if not k.startswith("megmasc")]
M["tEEGmin"], M["tEEGmax"] = f"{min(t95e):.0f}", f"{max(t95e):.0f}"
M["tMEG"] = f"{t95['megmasc/same']:.1f}"
M["lawMaeMin"] = f"{100 * min(v['mae'] for v in law.values()):.1f}"
M["lawMaeMax"] = f"{100 * max(v['mae'] for v in law.values()):.1f}"
M["lawLooMin"] = f"{100 * min(v['loo_mae'] for v in law.values()):.1f}"
M["lawLooMax"] = f"{100 * max(v['loo_mae'] for v in law.values()):.1f}"
M["nCurves"] = str(len(law))
# native channels (125-ms window runs, compared with the same-window 64-ch runs)
for ds, k in (("broderick_cp", "Brod"), ("uhd255", "UHD")):
    a, b = law64[f"{ds}/comp"], lawn[f"{ds}/comp"]
    M[f"ksf{k}"], M[f"kN{k}"] = f"{a['kappa']:.3f}", f"{b['kappa']:.3f}"
    M[f"tsf{k}"], M[f"tN{k}"] = f"{a['T95_s']:.0f}", f"{b['T95_s']:.0f}"

# flow rule
fl = tab["flow"]
gaps = [fl[d]["acc_flow_pmi"] - fl[d]["acc_flow_cond_only"] for d in fl]
M["pmiGapMin"], M["pmiGapMax"] = f"{100 * min(gaps):.0f}", f"{100 * max(gaps):.0f}"

# ablation
ab = tab["ablation"]
M["ablRankKUL"] = f"{-100 * ab['norank/kuleuven']['delta']:.1f}"
M["ablRankDTU"] = f"{-100 * ab['norank/dtu']['delta']:.1f}"
M["ablRankMEG"] = f"{-100 * ab['norank/megmasc']['delta']:.1f}"
others = [abs(v["delta"]) for k, v in ab.items()
          if not k.split("/")[0] in ("norank", "before", "lag0") and v["delta"] is not None]
M["ablOtherMax"] = f"{100 * max(others):.1f}"
for ds, k in (("kuleuven", "KUL"), ("dtu", "DTU"), ("megmasc", "MEG")):
    M[f"lagCtl{k}"] = f"{-100 * ab[f'before/{ds}']['delta']:.1f}"
    M[f"lagCtlAcc{k}"] = p(ab[f"before/{ds}"]["acc"])
    M[f"lagZero{k}"] = f"{100 * ab[f'lag0/{ds}']['delta']:+.1f}"

# leakage
lk = [v[0] - v[1] for d in leak.values() for v in d.values()]   # [trial split, content-disjoint]
M["leakMin"], M["leakMax"] = f"{100 * min(lk):+.1f}", f"{100 * max(lk):+.1f}"

# spatial classifiers (subject-dependent, content-disjoint)
def spat_range(dss, key):
    v = [spat[f"{d}/{m}/{w}"][key] for d in dss for m in ("darnet", "dbpnet", "listennet")
         for w in ("1s", "5s")]
    return f"{min(v):.0f}", f"{max(v):.0f}"
M["spAmin"], M["spAmax"] = spat_range(["dtu", "esaa", "uhd255"], "A_dep")
M["spFugMin"], M["spFugMax"] = spat_range(["fuglsang2020"], "A_dep")
kb = [spat[f"kuleuven/{m}/1s"]["B"] for m in ("darnet", "dbpnet", "listennet")]
M["spKULBmin"], M["spKULBmax"] = f"{min(kb):.1f}", f"{max(kb):.1f}"
M["cfFugOne"] = f"{100 * spat['fuglsang2020/coupleflow']['1s']:.0f}"
M["cfFugFive"] = f"{100 * spat['fuglsang2020/coupleflow']['5s']:.0f}"

# PMI attention effect by modulation band (final configuration, seed 0)
import glob
import os
BANDS = ["delta_0.5-4", "theta_4-8", "alpha_8-13", "beta_13-32"]
share = {b: [] for b in BANDS}
for ds in ALL:
    cell = "competing_5s" if ds in TWO else "same_talker_5s"
    v = {b: [] for b in BANDS}
    ws = []
    for d in glob.glob(f"{paths.RUNS}/final/{ds}_fold*"):
        f = [q for q in (d + "/results.json", d + "/results_val.json") if os.path.exists(q)]
        if not f:
            continue
        c = json.loads(Path(f[0]).read_text())["results"][cell]
        ws.append(c["n_windows"])
        for b in BANDS:
            v[b].append(c["flow"]["PMI_by_band"][b]["att"] - c["flow"]["PMI_by_band"][b]["un"])
    m = {b: max(float(np.average(v[b], weights=ws)), 0.0) for b in BANDS}
    tot = sum(m.values())
    for b in BANDS:
        share[b].append(m[b] / tot)
M["deltaMin"], M["deltaMax"] = f"{100 * min(share[BANDS[0]]):.0f}", f"{100 * max(share[BANDS[0]]):.0f}"
M["thetaMin"], M["thetaMax"] = f"{100 * min(share[BANDS[1]]):.0f}", f"{100 * max(share[BANDS[1]]):.0f}"
M["abMax"] = f"{100 * max(a + b for a, b in zip(share[BANDS[2]], share[BANDS[3]])):.0f}"

# kappa theory (scripts/figures/kappa_theory.py)
kt = json.loads((T / "kappa_theory.json").read_text())
M["fEff"] = f"{kt['absolute']['f_eff']:.1f}"
M["thR"] = f"{kt['absolute']['r']:.3f}"
M["thMape"] = f"{100 * kt['absolute']['mape']:.1f}"
M["taskMaeLin"] = f"{kt['task_effect']['linear']['mae']:.2f}"
M["taskRLin"] = f"{kt['task_effect']['linear']['r']:.2f}"
M["taskPLin"] = f"{kt['task_effect']['linear']['p']:.3f}"
M["taskMaeCF"] = f"{kt['task_effect']['coupleflow']['mae']:.2f}"
M["taskRCF"] = f"{kt['task_effect']['coupleflow']['r']:.2f}"
mv = kt["meg_vs_eeg"]
M["megRratio"] = f"{mv['r_ratio_meg_over_sparr']:.2f}"
M["megDzPred"] = f"{mv['dz_ratio_pred']:.2f}"
M["megKlinObs"] = f"{mv['kappa_lin_ratio_obs']:.2f}"
M["megKcfObs"] = f"{mv['kappa_cf_ratio_obs']:.2f}"
M["taskFacMin"], M["taskFacMax"] = (f"{mv['task_factor_range'][0]:.2f}", f"{mv['task_factor_range'][1]:.2f}")
M["taskFacMean"] = f"{mv['task_factor_mean']:.2f}"
M["rTwoMin"], M["rTwoMax"] = (f"{mv['two_talker_r_att_range'][0]:.2f}", f"{mv['two_talker_r_att_range'][1]:.2f}")
M["rSparr"] = f"{kt['tracking']['sparrkulee']['r_att']:.2f}"
M["rMEG"] = f"{kt['tracking']['megmasc']['r_att']:.2f}"
M["kLinMEG"] = f"{json.loads((F / 'snr_law_linear.json').read_text())['megmasc/same']['kappa']:.2f}"

# MAESTRO (scripts/tables/maestro_paper.py, maestro_ablation.py)
mp = json.loads((T / "maestro_main.json").read_text())
for proto, k in (("within", "W"), ("loso", "L")):
    ms = mp[proto]["mean_sd"]
    M[f"mae{k}Trial"], M[f"mae{k}TrialSd"] = p(ms["trial"][0]), p(ms["trial"][1])
    M[f"mae{k}Con"], M[f"mae{k}ConSd"] = p(ms["contrib"][0]), p(ms["contrib"][1])
    M[f"mae{k}Five"] = p(ms["w5"][0])
    M[f"mae{k}Twenty"] = p(ms["w20"][0])
    M[f"mae{k}StTwo"], M[f"mae{k}StThree"] = p(ms["st2_5"][0]), p(ms["st3_5"][0])
for proto, k in (("within_pkg", "PW"), ("loso_pkg", "PL")):
    ms = mp[proto]["mean_sd"]
    M[f"mae{k}Trial"], M[f"mae{k}TrialSd"] = p(ms["trial"][0]), p(ms["trial"][1])
    M[f"mae{k}Con"] = p(ms["contrib"][0])
al = mp["align"]
M["maePkgTrial"] = p(al["package_exact"]["trial"])
M["maeCompTrial"] = p(al["compensated"]["trial"])
M["maePreTrial"] = p(al["compensated_pre_window"]["trial"])
M["maeSnrRho"] = f"{mp['snr']['rho']:.2f}"
M["maeSnrP"] = f"{mp['snr']['p']:.2f}"
ma = json.loads((T / "maestro_ablation.json").read_text())
full = ma["maestro_within_lag0"]["cands_matched"]
def mdel(tag, key="trial"):
    return 100 * (ma[tag]["cands_matched"][key] - full[key])
M["maAblRank"] = f"{mdel('mabl_norank'):+.1f}"
M["maAblNce"] = f"{mdel('mabl_nocon'):+.1f}"
M["maAblBanks"] = f"{mdel('mabl_nobanks'):+.1f}"
M["maAblGlimpse"] = f"{mdel('mabl_noglimpse'):+.1f}"
M["maAblAux"] = f"{mdel('mabl_noaux'):+.1f}"
M["maBeforeTrial"] = p(ma["mabl_before"]["cands_matched"]["trial"])
M["maBeforeFive"] = p(ma["mabl_before"]["cands_matched"]["acc"])
safe = ["mabl_noflow", "mabl_noadv", "mabl_nohinge", "mabl_nomatch", "mabl_selacc", "mabl_noside"]
M["maSafeMax"] = f"{max(abs(mdel(t)) for t in safe):.1f}"
M["maSafeMaxFive"] = f"{max(abs(mdel(t, 'acc')) for t in safe):.1f}"
nul = [v[c]["null"] for v in ma.values() for c in ("cands", "cands_matched") if v[c]["null"] is not None]
M["maNullMin"], M["maNullMax"] = f"{100 * min(nul):.1f}", f"{100 * max(nul):.1f}"
um = [abs(v["cands"]["acc"] - v["cands_matched"]["acc"]) for v in ma.values() if v["cands"]["acc"] is not None]
M["maUnmatchedGap"] = f"{100 * max(um):.1f}"

lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
(T / "numbers.tex").write_text("% generated by scripts/tables/make_numbers.py -- do not edit\n"
                               + "\n".join(lines) + "\n")
print("\n".join(lines))
