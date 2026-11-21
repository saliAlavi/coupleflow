#!/usr/bin/env python
"""Rebuttal E3: the simpler model (decoding terms only) next to the full model and ablations.

    source scripts/env/osc.sh            # read the runs on scratch ($COUPLEFLOW_RUNS)
    python scripts/rebuttal/e3/aggregate.py [--json out.json]

KU Leuven, DTU, MEG-MASC: 5-s coupling accuracy (competing for the two-talker sets,
same-talker match-mismatch for MEG-MASC), pooled over test windows of all folds exactly as
scripts/tables/make_tables.py (cf_acc, mean_sd; ablation table base = unified/_s1/_s2),
plus the across-trial null (null_coupling) and the circular-shift null (null_circ_coupling).

MAESTRO (within-listener content-disjoint folds, 0-ms window, matched 4-way candidates): per-trial
accuracy and 5-s accuracy as scripts/tables/maestro_paper.py / maestro_ablation.py (cell
cands_matched_5s, trial_acc_coupling weighted by n_trial_decisions), plus the same nulls.

Seeds are averaged per seed (mean, s.d. with ddof=1 over seeds), as the paper's tables do.
A cell whose runs are incomplete is marked with the number of finished folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "tables"))
import make_tables as mt  # noqa: E402  (readers of the paper's tables)
from functools import lru_cache  # noqa: E402

from coupleflow.analysis import aggregate as _agg  # noqa: E402
from coupleflow.analysis.aggregate import pooled  # noqa: E402

# every tag is read once (results.json files hold per-trial lists and are large)
load = lru_cache(maxsize=None)(_agg.load)
mt.load = load

NFOLD = {"kuleuven": 4, "dtu": 5, "megmasc": 5, "maestro": 5}
SEEDS3 = ["", "_s1", "_s2"]

# (label, tags) for KU Leuven / DTU / MEG-MASC
EEG_ROWS = ([("full model", mt.BASE_TAGS),
             ("**core: CE + InfoNCE only**", [f"reb_core{s}" for s in SEEDS3]),
             ("**core, CE only**", ["reb_ce"])]
            + [(f"abl: {name}", ["esaa_lag0" if t == "lag0" else f"abl_{t}"])
               for t, name in mt.ABL])
MAE_ABL = [("noflow", "no flow head"), ("norank", "no listener correction"),
           ("nomatch", "unmatched training"), ("noglimpse", "no glimpse features"),
           ("nobanks", "no sub-band features"), ("noadv", "no audio adversary"),
           ("nohinge", "no null hinge"), ("nocon", "no InfoNCE"),
           ("selacc", "select on accuracy"), ("noside", "no side head"),
           ("noaux", "no env/VICReg aux."), ("before", "lag control (EEG before sound)")]
MAE_ROWS = ([("full model", [f"maestro_within_lag0{s}" for s in SEEDS3]),
             ("full model, seed 0 (tab:mabl base)", ["maestro_within_lag0"]),
             ("**core: CE + InfoNCE only**", [f"reb_mcore{s}" for s in SEEDS3]),
             ("**core, CE only**", ["reb_mce"])]
            + [(f"abl: {name}", [f"mabl_{t}"]) for t, name in MAE_ABL])


def per_seed(tags: list[str], ds: str, cell: str, key: str, wkey: str = "n_windows"):
    """Pooled value per finished tag, plus the fold count of each tag."""
    vals, folds = [], []
    for t in tags:
        runs = load(t).get(ds, [])
        folds.append(len(runs))
        vals.append(pooled(runs, cell, key, wkey) if runs else None)
    return vals, folds


def fmt(m, sd, n, nseed, folds, nf) -> str:
    if m is None:
        return "–"
    s = f"{100 * m:.1f}"
    if sd is not None:
        s += f" ± {100 * sd:.1f}"
    notes = []
    if n < nseed:
        notes.append(f"{n}/{nseed} seeds")
    if any(0 < f < nf for f in folds):
        notes.append("folds " + "/".join(str(f) for f in folds) + f" of {nf}")
    return s + (f" ({', '.join(notes)})" if notes else "")


def fmt1(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write the numbers here")
    args = ap.parse_args()
    out: dict = {}

    print("## KU Leuven, DTU, MEG-MASC: 5-s accuracy (%), 125-ms window\n")
    print("Accuracy: mean ± s.d. over seeds. Nulls: pooled, mean over seeds. "
          "Δ = accuracy − full-model mean.\n")
    print("| variant | " + " | ".join(
        f"{mt.NAMES[d]} acc | Δ | null | circ." for d in mt.ABL_DS) + " |")
    print("|---|" + "---|---|---|---|" * len(mt.ABL_DS))
    full = {}
    for label, tags in EEG_ROWS:
        cells = []
        for ds in mt.ABL_DS:
            cell = f"{mt.task_of(ds)}_5s"
            acc, folds = per_seed(tags, ds, cell, "acc_coupling")
            m, sd, n = mt.mean_sd(acc)
            nul = mt.mean_sd(per_seed(tags, ds, cell, "null_coupling")[0])[0]
            circ = mt.mean_sd(per_seed(tags, ds, cell, "null_circ_coupling")[0])[0]
            if label == "full model":
                full[ds] = m
            d = None if (m is None or full.get(ds) is None) else m - full[ds]
            out.setdefault("eeg", {}).setdefault(label, {})[ds] = {
                "acc": m, "sd": sd, "n_seeds": n, "per_seed": acc, "folds": folds,
                "delta": d, "null": nul, "circ": circ}
            cells += [fmt(m, sd, n, len(tags), folds, NFOLD[ds]),
                      "" if label == "full model" or d is None else f"{100 * d:+.1f}",
                      fmt1(nul), fmt1(circ)]
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n## MAESTRO: within listeners, 0-ms window, matched 4-way candidates (%)\n")
    print("Δ: rows with 3 seeds vs the 3-seed full-model mean (Table 4); single-seed rows vs "
          "seed 0 of the full model, as tab:mabl.\n")
    print("| variant | per-trial acc | Δ | trial null | 5-s acc | null | circ. |")
    print("|---|---|---|---|---|---|---|")
    full_tr, full_tr0 = None, None
    for label, tags in MAE_ROWS:
        c = "cands_matched_5s"
        tr, folds = per_seed(tags, "maestro", c, "trial_acc_coupling", "n_trial_decisions")
        m, sd, n = mt.mean_sd(tr)
        trn = mt.mean_sd(per_seed(tags, "maestro", c, "trial_null_coupling",
                                  "n_trial_decisions")[0])[0]
        w5 = mt.mean_sd(per_seed(tags, "maestro", c, "acc_coupling")[0])
        nul = mt.mean_sd(per_seed(tags, "maestro", c, "null_coupling")[0])[0]
        circ = mt.mean_sd(per_seed(tags, "maestro", c, "null_circ_coupling")[0])[0]
        if label == "full model":
            full_tr = m
        elif label.startswith("full model, seed 0"):
            full_tr0 = m
        ref = full_tr if len(tags) > 1 else full_tr0
        d = None if (m is None or ref is None or label.startswith("full")) else m - ref
        out.setdefault("maestro", {})[label] = {
            "trial": m, "trial_sd": sd, "n_seeds": n, "per_seed": tr, "folds": folds,
            "delta": d, "trial_null": trn, "w5": w5[0], "w5_sd": w5[1], "null": nul,
            "circ": circ}
        print(f"| {label} | {fmt(m, sd, n, len(tags), folds, NFOLD['maestro'])} | "
              f"{'' if label == 'full model' or d is None else f'{100 * d:+.1f}'} | "
              f"{fmt1(trn)} | {fmt(w5[0], w5[1], w5[2], len(tags), folds, NFOLD['maestro'])} | "
              f"{fmt1(nul)} | {fmt1(circ)} |")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
