#!/usr/bin/env python
"""Rebuttal E1: per-window agreement of the coupling and PMI (flow) decision rules.

Reads the per-window margins that training already saved next to every run
(``test_windows.npz``: ``<cell>__margin_coupling``, ``<cell>__margin_flow_pmi``,
attended minus the strongest competitor; a window is correct when its margin is > 0,
as in ``coupleflow.evaluate._acc``).  No inference, no fitting, nothing selected on test.

Runs: the seed-0 ``final`` runs (repo ``runs/final`` symlinks name the tag of each
dataset), 5-s test windows, the candidate set of Table 3 (``competing`` for two-talker
data, ``same_talker`` for SparrKULee / MEG-MASC); MAESTRO: ``maestro_within_lag0``,
``cands_matched_5s`` (4-way).  Before any statistic, the accuracies recomputed from the
margins are checked against the runs' ``results*.json`` and against
``paper/tables/tables.json`` (Table 3, PMI) / ``maestro_main.json``.

    python scripts/rebuttal/e1/agreement.py --out /fs/scratch/PAS2301/alialavi/rebuttal/e1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu, spearmanr

REPO = Path(__file__).resolve().parents[3]
RUNS = Path(os.environ.get("COUPLEFLOW_RUNS", "/fs/scratch/PAS2301/alialavi/runs/coupleflow"))
TWO = ["kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp"]
ONE = ["sparrkulee", "megmasc"]
NAMES = {"kuleuven": "KU Leuven", "dtu": "DTU", "fuglsang2020": "Fuglsang'20",
         "esaa": "ESAA", "uhd255": "KUL-255ch", "broderick_cp": "Broderick",
         "sparrkulee": "SparrKULee", "megmasc": "MEG-MASC", "maestro": "MAESTRO (4-way)"}


def final_dirs(ds: str) -> list[Path]:
    """Scratch run directories of the seed-0 ``final`` runs of ``ds``."""
    out = []
    for link in sorted((REPO / "runs" / "final").glob(f"{ds}_fold*")):
        tag = Path(os.readlink(link)).parent.name          # ../<tag>/<ds>_fold<k>
        out.append(RUNS / tag / link.name)
    return out


def results_of(d: Path) -> dict:
    for name in ("results.json", "results_val.json"):
        if (d / name).exists():
            return json.loads((d / name).read_text())["results"]
    raise FileNotFoundError(d)


def load(ds: str, dirs: list[Path], cell: str) -> dict:
    """Pool margins over folds; check per-fold accuracies against results.json."""
    mc, mp, subj, trial, fold, checks = [], [], [], [], [], []
    for k, d in enumerate(dirs):
        z = np.load(d / "test_windows.npz")
        c = z[f"{cell}__margin_coupling"].astype(np.float64)
        p = z[f"{cell}__margin_flow_pmi"].astype(np.float64)
        r = results_of(d)[cell]
        checks.append({"run": f"{d.parent.name}/{d.name}", "n": len(c), "n_json": r["n_windows"],
                       "acc_coupling": float((c > 0).mean()), "json_coupling": r["acc_coupling"],
                       "acc_pmi": float((p > 0).mean()), "json_pmi": r["flow"]["acc_flow_pmi"],
                       "trial_coupling_json": r.get("trial_acc_coupling"),
                       "trial_pmi_json": r["flow"].get("trial_acc_flow_pmi")})
        mc.append(c / c.std())                 # per-run scale, for margin-magnitude analysis
        mp.append(p / p.std())
        subj.append(np.char.add(f"{k}:", z[f"{cell}__subject"].astype(str)))
        trial.append(z[f"{cell}__trial"] + 100000 * k)
        fold.append(np.full(len(c), k))
        # per-trial decisions for K = 2 (sum of window margins = sum of logit differences)
        if not cell.startswith("same_talker") and ds != "maestro":
            t = z[f"{cell}__trial"]
            tc = np.array([c[t == u].sum() > 0 for u in np.unique(t)])
            tp = np.array([p[t == u].sum() > 0 for u in np.unique(t)])
            checks[-1]["trial_coupling"] = float(tc.mean())
            checks[-1]["trial_pmi"] = float(tp.mean())
    return {"c": np.concatenate(mc), "p": np.concatenate(mp), "subj": np.concatenate(subj),
            "trial": np.concatenate(trial), "fold": np.concatenate(fold), "checks": checks}


def table(cc: np.ndarray, pc: np.ndarray) -> dict:
    """2x2 correctness table and agreement statistics (cc, pc: bool per unit)."""
    n = len(cc)
    a = int((cc & pc).sum())
    b = int((cc & ~pc).sum())          # coupling right, PMI wrong
    c = int((~cc & pc).sum())          # PMI corrects a coupling error
    d = int((~cc & ~pc).sum())
    acc_c, acc_p = (a + b) / n, (a + c) / n
    agree = (a + d) / n
    exp = acc_c * acc_p + (1 - acc_c) * (1 - acc_p)
    den = np.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
    return {"n": n, "both": a, "coupling_only": b, "pmi_only": c, "neither": d,
            "acc_coupling": acc_c, "acc_pmi": acc_p, "agree": agree, "agree_indep": exp,
            "kappa": (agree - exp) / (1 - exp) if exp < 1 else float("nan"),
            "phi": (a * d - b * c) / den if den > 0 else float("nan"),
            "mcnemar_p": binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0,
            "p_pmi_right_given_coupling_wrong": c / (c + d) if c + d else float("nan"),
            "p_coupling_right_given_pmi_wrong": b / (b + d) if b + d else float("nan"),
            "coupling_share_of_disagreements": b / (b + c) if b + c else float("nan")}


def cluster_ci(cc, pc, groups, rng, n_boot=2000):
    """95% listener-bootstrap interval of acc(coupling) - acc(PMI)."""
    u = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in u}
    diffs = []
    for _ in range(n_boot):
        s = np.concatenate([idx[g] for g in rng.choice(u, len(u))])
        diffs.append(cc[s].mean() - pc[s].mean())
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def margin_analysis(x: dict) -> dict:
    """Does margin size say which rule is right when the two disagree?"""
    cc, pc = x["c"] > 0, x["p"] > 0
    dis = cc != pc
    out = {"spearman_margins": float(spearmanr(x["c"], x["p"]).statistic)}
    if dis.sum() < 10:
        return out
    yc = cc[dis]                                         # True: coupling is the right one
    ac, ap = np.abs(x["c"][dis]), np.abs(x["p"][dis])
    def auc(score, y):
        if y.all() or (~y).all():
            return float("nan")
        return float(mannwhitneyu(score[y], score[~y]).statistic / (y.sum() * (~y).sum()))
    out["auc_abs_coupling_margin"] = auc(ac, yc)        # > 0.5: large |coupling| -> coupling right
    out["auc_abs_pmi_margin"] = auc(ap, ~yc)            # > 0.5: large |PMI| -> PMI right
    out["auc_rel"] = auc(ac - ap, yc)                   # larger relative margin wins?
    # agreement as a function of coupling-margin size (quartiles of |z| over all windows)
    q = np.quantile(np.abs(x["c"]), [0.25, 0.5, 0.75])
    bins = np.digitize(np.abs(x["c"]), q)
    out["agree_by_abs_coupling_quartile"] = [float((cc[bins == i] == pc[bins == i]).mean())
                                             for i in range(4)]
    out["acc_coupling_by_quartile"] = [float(cc[bins == i].mean()) for i in range(4)]
    out["acc_pmi_by_quartile"] = [float(pc[bins == i].mean()) for i in range(4)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/fs/scratch/PAS2301/alialavi/rebuttal/e1")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = json.loads((REPO / "paper/tables/tables.json").read_text())
    mae = json.loads((REPO / "paper/tables/maestro_main.json").read_text())
    rng = np.random.default_rng(0)
    res = {}
    jobs = [(ds, final_dirs(ds), "same_talker_5s" if ds in ONE else "competing_5s")
            for ds in TWO + ONE]
    jobs.append(("maestro", sorted((RUNS / "maestro_within_lag0").glob("maestro_fold*")),
                 "cands_matched_5s"))
    for ds, dirs, cell in jobs:
        x = load(ds, dirs, cell)
        cc, pc = x["c"] > 0, x["p"] > 0
        r = {"cell": cell, "folds": len(dirs), "window": table(cc, pc), "checks": x["checks"]}
        r["window"]["acc_diff_ci_listener_boot"] = cluster_ci(cc, pc, x["subj"], rng)
        r["margins"] = margin_analysis(x)
        # reproduction against the tables the paper is built from
        if ds in tables["flow"]:
            r["paper_acc_pmi"] = tables["flow"][ds]["acc_flow_pmi"]
        if ds == "maestro":
            r["paper_w5_coupling_seed0"] = mae["within"]["seeds"][0]["w5"]
            r["paper_trial_coupling_seed0"] = mae["within"]["seeds"][0]["trial"]
        # per-trial (K = 2, competing set only)
        if "trial_coupling" in x["checks"][0]:
            tc, tp = [], []
            for k, d in enumerate(dirs):
                z = np.load(d / "test_windows.npz")
                t = z[f"{cell}__trial"]
                c = z[f"{cell}__margin_coupling"]
                p = z[f"{cell}__margin_flow_pmi"]
                for u in np.unique(t):
                    tc.append(c[t == u].sum() > 0)
                    tp.append(p[t == u].sum() > 0)
            r["trial"] = table(np.array(tc), np.array(tp))
        res[ds] = r
    (out / "agreement.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(res, out / "agreement.md")
    print((out / "agreement.md").read_text())


def write_md(res: dict, path: Path) -> None:
    f = lambda v: f"{100 * v:.1f}"
    L = ["## Reproduction (pooled seed-0 test windows, 5 s)", "",
         "| dataset | n | coupling (margins) | coupling (json) | PMI (margins) | PMI (json) | "
         "PMI (Table 3) |", "|---|---|---|---|---|---|---|"]
    for ds, r in res.items():
        ch = r["checks"]
        w = np.array([c["n"] for c in ch], float)
        jc = np.average([c["json_coupling"] for c in ch], weights=w)
        jp = np.average([c["json_pmi"] for c in ch], weights=w)
        L.append(f"| {NAMES[ds]} | {int(w.sum())} | {f(r['window']['acc_coupling'])} | {f(jc)} | "
                 f"{f(r['window']['acc_pmi'])} | {f(jp)} | "
                 f"{f(r['paper_acc_pmi']) if 'paper_acc_pmi' in r else '–'} |")
    L += ["", "## Agreement of the coupling and PMI rules, 5-s test windows", "",
          "Counts are windows. 'PMI only' = the PMI rule corrects a coupling error. "
          "Expected agreement assumes independent errors at the two accuracies. "
          "McNemar is exact and treats windows as independent; the CI is a listener bootstrap "
          "of acc(coupling) - acc(PMI).", "",
          "| dataset | n | both | coupling only | PMI only | neither | agree | agree if indep. | "
          "κ | φ | P(PMI right \\| coupling wrong) | McNemar p | Δacc [95% CI] | ρ(margins) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in res.items():
        t = r["window"]
        ci = t["acc_diff_ci_listener_boot"]
        L.append(f"| {NAMES[ds]} | {t['n']} | {t['both']} | {t['coupling_only']} | {t['pmi_only']} | "
                 f"{t['neither']} | {f(t['agree'])} | {f(t['agree_indep'])} | {t['kappa']:.2f} | "
                 f"{t['phi']:.2f} | {f(t['p_pmi_right_given_coupling_wrong'])} | "
                 f"{t['mcnemar_p']:.1e} | {100 * (t['acc_coupling'] - t['acc_pmi']):+.1f} "
                 f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] | {r['margins']['spearman_margins']:.2f} |")
    L += ["", "## When they disagree: does margin size tell which rule is right?", "",
          "AUC over disagreement windows (0.5 = no information). Margins are scaled by each "
          "run's s.d.", "",
          "| dataset | coupling right among disagreements | AUC \\|coupling\\| | AUC \\|PMI\\| | "
          "agree by \\|coupling\\| quartile (low→high) |", "|---|---|---|---|---|"]
    for ds, r in res.items():
        m, t = r["margins"], r["window"]
        q = " / ".join(f(v) for v in m.get("agree_by_abs_coupling_quartile", []))
        L.append(f"| {NAMES[ds]} | {f(t['coupling_share_of_disagreements'])} | "
                 f"{m.get('auc_abs_coupling_margin', float('nan')):.2f} | "
                 f"{m.get('auc_abs_pmi_margin', float('nan')):.2f} | {q} |")
    tr = {ds: r for ds, r in res.items() if "trial" in r}
    if tr:
        L += ["", "## Per-trial decisions (two-talker, competing set; window margins summed)", "",
              "| dataset | trials | both | coupling only | PMI only | neither | agree | "
              "acc coupling | acc PMI | max fold \\|Δ\\| vs json (coupling / PMI) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for ds, r in tr.items():
            t, ch = r["trial"], r["checks"]
            dc = max(abs(c["trial_coupling"] - c["trial_coupling_json"]) for c in ch)
            dp = max(abs(c["trial_pmi"] - c["trial_pmi_json"]) for c in ch)
            L.append(f"| {NAMES[ds]} | {t['n']} | {t['both']} | {t['coupling_only']} | "
                     f"{t['pmi_only']} | {t['neither']} | {f(t['agree'])} | "
                     f"{f(t['acc_coupling'])} | {f(t['acc_pmi'])} | {dc:.1e} / {dp:.1e} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
