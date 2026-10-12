#!/usr/bin/env python
"""Generate every LaTeX table of the AISTATS paper from run outputs.

    python scripts/tables/make_tables.py --out paper/tables

Tables (each also dumped to JSON next to it):
  main.tex      CoupleFlow (mean ± s.d. over seeds) vs linear / deep baselines, 5 s + trial
  windows.tex   CoupleFlow accuracy vs window, every dataset and task
  ablation.tex  change in 5-s accuracy when one component is removed (KUL, DTU, MEG)
  channels.tex  64 canonical vs native 128 / 253 channels
  flow.tex      flow-head decision rules (PMI vs conditional vs prior) at 5 s
Missing runs are printed as "--" and listed on stdout, never silently filled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.analysis.aggregate import CACHE, WINDOWS, load, pooled  # noqa: E402

TWO = ["kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp"]
ONE = ["sparrkulee", "megmasc"]
NAMES = {"kuleuven": "KU Leuven", "dtu": "DTU", "fuglsang2020": "Fuglsang'20 (NH+HI)",
         "esaa": "ESAA (zh)", "uhd255": "KUL-255ch", "broderick_cp": "Broderick",
         "sparrkulee": "SparrKULee", "megmasc": "MEG-MASC (MEG)"}
SEED_TAGS = ["final", "final_s1", "final_s2"]          # validation-chosen lag window
BASE_TAGS = ["unified", "unified_s1", "unified_s2"]      # 125-ms window: ablation base
ABL = [("lag0", "lag window from 0\\,ms"), ("noflow", "no flow head"), ("norank", "no listener correction"),
       ("nomatch", "no matched candidates"), ("noglimpse", "no glimpse features"),
       ("noadv", "no audio adversary"), ("nohinge", "no null hinge"),
       ("nocon", "no InfoNCE"), ("selacc", "select on accuracy"),
       ("noside", "no side head"), ("noaux", "no env/VICReg aux."),
       ("before", "lag control (EEG before sound)")]
ABL_DS = ["kuleuven", "dtu", "megmasc"]
BASELINES = [("linear", "Linear"), ("vlaai", "VLAAI"),
             ("mm", "Dilated MM")]
MISSING: list[str] = []


def task_of(ds: str) -> str:
    return "same_talker" if ds in ONE else "competing"


def cf_acc(tag: str, ds: str, cell: str, key: str = "acc_coupling",
           wkey: str = "n_windows") -> float | None:
    runs = load(tag).get(ds)
    if not runs:
        MISSING.append(f"{tag}/{ds}")
        return None
    return pooled(runs, cell, key, wkey)


def base_acc(model: str, ds: str, w: str = "5s", trial: bool = False) -> float | None:
    p = CACHE / ds / ("linear_baseline.json" if model == "linear" else f"baseline_{model}.json")
    if not p.exists():
        MISSING.append(str(p))
        return None
    j = json.loads(p.read_text())
    if trial:
        return j.get("trial_competing", [None])[0]
    return j["windows"].get(w, {}).get(task_of(ds), [None])[0]


def fmt(x: float | None, bold: bool = False) -> str:
    if x is None:
        return "--"
    s = f"{100 * x:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def mean_sd(vals: list[float | None]) -> tuple[float | None, float | None, int]:
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None), len(v)


def table_main(out: Path) -> dict:
    rows, res = [], {}
    for ds in TWO + ONE:
        cell = f"{task_of(ds)}_5s"
        m, sd, n = mean_sd([cf_acc(t, ds, cell) for t in SEED_TAGS])
        tr = (mean_sd([cf_acc(t, ds, "competing_5s", "trial_acc_coupling", "n_trial_decisions")
                       for t in SEED_TAGS])[0] if ds in TWO else None)
        bs = {b: base_acc(b, ds) for b, _ in BASELINES}
        best_b = max([v for v in bs.values() if v is not None], default=None)
        cf_s = fmt(m, bold=(m is not None and best_b is not None and m > best_b))
        if sd is not None:
            cf_s += f"{{\\scriptsize$\\pm${100 * sd:.1f}}}"
        res[ds] = {"cf": m, "cf_sd": sd, "n_seeds": n, "trial": tr, **bs}
        rows.append(f"{NAMES[ds]} & {'comp.' if ds in TWO else 'mm'} & "
                    + " & ".join(fmt(bs[b]) for b, _ in BASELINES)
                    + f" & {cf_s} & {fmt(tr)} \\\\")
    hdr = ("\\begin{tabular}{llccc|cc}\\toprule\n"
           "Dataset & task & " + " & ".join(n for _, n in BASELINES)
           + " & CoupleFlow & trial \\\\\\midrule\n")
    (out / "main.tex").write_text(hdr + "\n".join(rows[:len(TWO)]) + "\n\\midrule\n"
                                  + "\n".join(rows[len(TWO):]) + "\n\\bottomrule\\end{tabular}\n")
    return res


def table_windows(out: Path) -> dict:
    rows, res = [], {}
    for ds in TWO + ONE:
        for task in (["competing", "same_talker"] if ds in TWO else ["same_talker"]):
            accs = [cf_acc("final", ds, f"{task}_{w}") for w in WINDOWS]
            res[f"{ds}/{task}"] = accs
            rows.append(f"{NAMES[ds]} & {'comp.' if task == 'competing' else 'mm'} & "
                        + " & ".join(fmt(a) for a in accs) + " \\\\")
    hdr = ("\\begin{tabular}{ll" + "c" * len(WINDOWS) + "}\\toprule\n"
           "Dataset & task & " + " & ".join(w.replace("s", "\\,s") for w in WINDOWS)
           + " \\\\\\midrule\n")
    (out / "windows.tex").write_text(hdr + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
    return res


def table_ablation(out: Path) -> dict:
    res, rows = {}, []
    full = {ds: mean_sd([cf_acc(t, ds, f"{task_of(ds)}_5s") for t in BASE_TAGS])
            for ds in ABL_DS}
    rows.append("full model & " + " & ".join(
        fmt(full[ds][0]) + (f"{{\\scriptsize$\\pm${100 * full[ds][1]:.1f}}}"
                            if full[ds][1] is not None else "") for ds in ABL_DS) + " \\\\\\midrule")
    for tag, name in ABL:
        cells = []
        for ds in ABL_DS:
            a = cf_acc("esaa_lag0" if tag == "lag0" else f"abl_{tag}", ds, f"{task_of(ds)}_5s")
            f0 = full[ds][0]
            d = None if (a is None or f0 is None) else a - f0
            res[f"{tag}/{ds}"] = {"acc": a, "delta": d}
            cells.append("--" if d is None else f"{100 * d:+.1f}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    hdr = ("\\begin{tabular}{l" + "c" * len(ABL_DS) + "}\\toprule\n"
           "Variant & " + " & ".join(NAMES[d] for d in ABL_DS) + " \\\\\\midrule\n")
    (out / "ablation.tex").write_text(hdr + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
    return res


def table_channels(out: Path) -> dict:
    res, rows = {}, []
    for ds, n in (("broderick_cp", 128), ("uhd255", 253)):
        a64 = cf_acc("unified", ds, "competing_5s")
        an = cf_acc("unified_native", ds, "competing_5s")
        t64 = cf_acc("unified", ds, "competing_5s", "trial_acc_coupling", "n_trial_decisions")
        tn = cf_acc("unified_native", ds, "competing_5s", "trial_acc_coupling",
                    "n_trial_decisions")
        res[ds] = {"64": a64, "native": an, "trial64": t64, "trial_native": tn}
        rows.append(f"{NAMES[ds]} & 64 / {n} & {fmt(a64)} / {fmt(an)} & {fmt(t64)} / {fmt(tn)} \\\\")
    (out / "channels.tex").write_text(
        "\\begin{tabular}{lccc}\\toprule\nDataset & channels & 5\\,s & trial \\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
    return res


def table_flow(out: Path) -> dict:
    res, rows = {}, []
    for ds in TWO + ONE:
        cell = f"{task_of(ds)}_5s"
        runs = load("final").get(ds, [])
        vals: dict[str, list] = {}
        ws = []
        for r in runs:
            c = r["results"].get(cell, {})
            f = c.get("flow")
            if not f:
                continue
            ws.append(c.get("n_windows", 1))
            for k in ("acc_flow_pmi", "null_flow_pmi", "acc_flow_cond_only",
                      "acc_flow_prior_only", "R_c_att", "R_c_un", "R_u_att", "R_u_un"):
                vals.setdefault(k, []).append(f.get(k, np.nan))
        if not ws:
            continue
        m = {k: float(np.average(np.array(v, float), weights=ws)) for k, v in vals.items()}
        res[ds] = m
        rows.append(f"{NAMES[ds]} & {fmt(m['acc_flow_pmi'])} & {fmt(m['acc_flow_cond_only'])} & "
                    f"{fmt(m['acc_flow_prior_only'])} & {fmt(m['null_flow_pmi'])} & "
                    f"{m['R_c_un'] - m['R_c_att']:+.3f} & {m['R_u_un'] - m['R_u_att']:+.3f} \\\\")
    (out / "flow.tex").write_text(
        "\\begin{tabular}{lcccccc}\\toprule\n"
        "Dataset & PMI & $\\log p(a|x)$ & $\\log p(a)$ & null & "
        "$\\Delta R_{\\rm cond}$ & $\\Delta R_{\\rm unc}$ \\\\\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\\end{tabular}\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allres = {"main": table_main(out), "windows": table_windows(out),
              "ablation": table_ablation(out), "channels": table_channels(out),
              "flow": table_flow(out)}
    (out / "tables.json").write_text(json.dumps(allres, indent=1))
    miss = sorted(set(MISSING))
    print(f"wrote tables to {out}; missing inputs ({len(miss)}):")
    for m in miss:
        print("  ", m)


if __name__ == "__main__":
    main()
