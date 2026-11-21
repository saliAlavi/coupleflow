#!/usr/bin/env python
"""Rebuttal E7: CoupleFlow with the frontal channels zeroed vs. the full-channel runs.

Reads the runs' results.json (results_val.json if absent) with coupleflow.analysis.aggregate,
pooled over folds as the paper tables do (window-weighted; per-trial accuracy weighted by the
number of trial decisions).  Nothing is typed in by hand.

  KU Leuven, DTU  (competing talker, 125-ms lag window, seed 0):
      reb_nofrontal vs unified                       5-s accuracy, across-trial and circular
                                                     nulls, per-trial accuracy
  MAESTRO (4-way matched candidates, 0-ms lag window, seeds 0-2, mean +- s.d.):
      reb_mnofrontal{,_s1,_s2} vs maestro_within_lag0{,_s1,_s2}
                                                     per-trial accuracy (Table 4 metric), its
                                                     across-trial null, 5-s accuracy, 5-s nulls

    python scripts/rebuttal/e7/aggregate.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from coupleflow.analysis.aggregate import load, pooled  # noqa: E402

TWO = {"kuleuven": "KU Leuven", "dtu": "DTU"}
MAE_FULL = ["maestro_within_lag0", "maestro_within_lag0_s1", "maestro_within_lag0_s2"]
MAE_NOF = ["reb_mnofrontal", "reb_mnofrontal_s1", "reb_mnofrontal_s2"]
NFOLD = {"kuleuven": 4, "dtu": 5, "maestro": 5}


def cells(tag: str, ds: str, cs: str) -> dict | None:
    rows = load(tag).get(ds)
    if not rows:
        return None
    c = f"{cs}_5s"
    return {"folds": len(rows),
            "acc5": pooled(rows, c, "acc_coupling"),
            "null5": pooled(rows, c, "null_coupling"),
            "circ5": pooled(rows, c, "null_circ_coupling"),
            "trial": pooled(rows, c, "trial_acc_coupling", "n_trial_decisions"),
            "trial_null": pooled(rows, c, "trial_null_coupling", "n_trial_decisions")}


def pct(x) -> str:
    return "  --  " if x is None else f"{100 * x:5.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out: dict = {}
    print("Two-talker EEG, competing talker, seed 0, 125-ms lag window")
    print(f"{'dataset':10s} {'model':18s} folds  5-s acc  across-null  circ-null  per-trial")
    for ds, name in TWO.items():
        for lab, tag in (("all channels", "unified"), ("frontal zeroed", "reb_nofrontal")):
            c = cells(tag, ds, "competing")
            out[f"{ds}/{tag}"] = c
            if c is None:
                print(f"{name:10s} {lab:18s}  missing")
                continue
            print(f"{name:10s} {lab:18s} {c['folds']}/{NFOLD[ds]}   {pct(c['acc5'])}    {pct(c['null5'])}"
                  f"      {pct(c['circ5'])}     {pct(c['trial'])}")
        f, n = out.get(f"{ds}/unified"), out.get(f"{ds}/reb_nofrontal")
        if f and n and f["acc5"] is not None and n["acc5"] is not None:
            print(f"{'':10s} {'difference':18s}       {100 * (n['acc5'] - f['acc5']):+5.1f}")
    print("\nMAESTRO, within listeners, 4-way matched candidates (chance 25%), 0-ms lag window")
    print(f"{'model':18s} seeds  per-trial      trial-null   5-s acc       5-s across-null  5-s circ-null")
    for lab, tags in (("all channels", MAE_FULL), ("frontal zeroed", MAE_NOF)):
        cs = [cells(t, "maestro", "cands_matched") for t in tags]
        cs = [c for c in cs if c is not None and c["folds"] == NFOLD["maestro"]]
        out[f"maestro/{lab}"] = cs
        if not cs:
            print(f"{lab:18s} missing")
            continue

        def ms(k):
            v = [c[k] for c in cs if c[k] is not None]
            if not v:
                return "   --       "
            sd = np.std(v, ddof=1) if len(v) > 1 else float("nan")
            return f"{100 * np.mean(v):5.1f} ± {100 * sd:3.1f}"
        print(f"{lab:18s} {len(cs)}      {ms('trial')}   {ms('trial_null')}  {ms('acc5')}   "
              f"{ms('null5')}     {ms('circ5')}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
