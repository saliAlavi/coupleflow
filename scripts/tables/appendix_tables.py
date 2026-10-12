#!/usr/bin/env python
"""Appendix tables: CoupleFlow and every baseline at every window, CoupleFlow variants, side classifiers.

    python scripts/tables/appendix_tables.py      # writes paper/tables/app_*.tex
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.analysis.aggregate import CACHE, WINDOWS, load, pooled  # noqa: E402

T = Path("paper/tables")
TWO = ["kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp"]
ONE = ["sparrkulee", "megmasc"]
NAMES = {"kuleuven": "KU Leuven", "dtu": "DTU", "fuglsang2020": "Fuglsang'20", "esaa": "ESAA",
         "uhd255": "KUL-255", "broderick_cp": "Broderick", "sparrkulee": "SparrKULee",
         "megmasc": "MEG-MASC"}
f = lambda x: "--" if x is None else f"{100 * x:.1f}"  # noqa: E731
task = lambda ds: "same_talker" if ds in ONE else "competing"  # noqa: E731

# 1. CoupleFlow (mean of 3 seeds, validation-chosen lag) and every baseline at every window
BL = [("linear_baseline", "Linear"), ("baseline_vlaai", "VLAAI"), ("baseline_mm", "Dilated MM"),
      ("baseline_decaf", "DECAF"), ("baseline_decaf_eeg", "DECAF (EEG branch)")]
rows = []
for ds in TWO + ONE:
    for i, (fn, name) in enumerate(BL):
        p = CACHE / ds / f"{fn}.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        accs = [j["windows"].get(w, {}).get(task(ds), [None])[0] for w in WINDOWS]
        tr = j.get("trial_competing", [None])[0] if ds in TWO else None
        rows.append(f"{NAMES[ds] if i == 0 else ''} & {name} & " + " & ".join(f(a) for a in accs)
                    + f" & {f(tr)} \\\\")
    cf = [load(t).get(ds, []) for t in ("final", "final_s1", "final_s2")]

    def cf3(cell: str, key: str = "acc_coupling", wkey: str = "n_windows") -> float | None:
        v = [pooled(r, cell, key, wkey) for r in cf]
        return sum(v) / len(v) if all(x is not None for x in v) else None
    accs = [cf3(f"{task(ds)}_{w}") for w in WINDOWS]
    tr = cf3("competing_5s", "trial_acc_coupling", "n_trial_decisions") if ds in TWO else None
    rows.append(" & CoupleFlow & " + " & ".join(f(a) for a in accs) + f" & {f(tr)} \\\\")
    if ds in TWO:  # match-mismatch on the same two-talker recordings (no baseline row)
        accs = [cf3(f"same_talker_{w}") for w in WINDOWS]
        rows.append(" & CoupleFlow, mm & " + " & ".join(f(a) for a in accs) + " & -- \\\\")
    rows.append("\\midrule")
(T / "app_baselines.tex").write_text(
    "\\begin{tabular}{ll" + "c" * (len(WINDOWS) + 1) + "}\\toprule\nDataset & model & "
    + " & ".join(w.replace("s", "\\,s") for w in WINDOWS) + " & trial \\\\\\midrule\n"
    + "\n".join(rows[:-1]) + "\n\\bottomrule\\end{tabular}\n")

# 2. CoupleFlow variants at 5 s (seed 0)
VAR = [("unified", "125\\,ms (base)"), ("esaa_lag0", "0\\,ms"), ("unified_envonly", "env.\\ only"),
       ("esaa_rank8", "rank 8"), ("esaa_sft", "listener FT"), ("unified_native", "native ch.")]
# Cells where the variant cannot differ from the base: envelope-only releases already use the
# broadband envelope alone; 64-channel EEG caps have no channels beyond the canonical 64.
NA = {("unified_envonly", "fuglsang2020"), ("unified_envonly", "broderick_cp")} | {
    ("unified_native", ds) for ds in ["kuleuven", "dtu", "fuglsang2020", "esaa", "sparrkulee",
                                      "megmasc"]}
rows = []
for ds in TWO + ONE:
    cells = []
    for tag, _ in VAR:
        runs = load(tag).get(ds, [])
        cells.append(f(pooled(runs, f"{task(ds)}_5s", "acc_coupling")) if runs
                     else "n/a" if (tag, ds) in NA else "--")
    rows.append(f"{NAMES[ds]} & " + " & ".join(cells) + " \\\\")
(T / "app_variants.tex").write_text(
    "\\begin{tabular}{l" + "c" * len(VAR) + "}\\toprule\nDataset & "
    + " & ".join(n for _, n in VAR) + " \\\\\\midrule\n" + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\n")

# 3. side classifiers, all protocols
MOD = [("darnet", "DARNet"), ("dbpnet", "DBPNet"), ("listennet", "ListenNet")]
rows = []
for ds in ["kuleuven", "dtu", "esaa", "fuglsang2020", "uhd255"]:
    for i, (m, name) in enumerate(MOD):
        j = json.loads((CACHE / ds / f"spatial_{m}.json").read_text())
        c = []
        for key in ("protocol_B", "protocol_A_subject_dep", "protocol_A_subject_indep"):
            c += [("--" if j[key].get(w) is None else f"{j[key][w]:.1f}") for w in ("1s", "2s", "5s")]
        rows.append(f"{NAMES[ds] if i == 0 else ''} & {name} & " + " & ".join(c) + " \\\\")
    rows.append("\\midrule")
(T / "app_spatial.tex").write_text(
    "\\begin{tabular}{llccc|ccc|ccc}\\toprule\n & & \\multicolumn{3}{c|}{their split} & "
    "\\multicolumn{3}{c|}{content-disj., subj.-dep.} & \\multicolumn{3}{c}{content-disj., subj.-indep.}"
    " \\\\\nDataset & model & 1\\,s & 2\\,s & 5\\,s & 1\\,s & 2\\,s & 5\\,s & 1\\,s & 2\\,s & 5\\,s "
    "\\\\\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\\end{tabular}\n")
print("wrote", [p.name for p in T.glob("app_*.tex")])
