#!/usr/bin/env python
"""Choose the EEG lag window per dataset on VALIDATION contribution only (never test).

For each dataset, compare the default window (+125..+1109 ms, tag `unified`) with the
0-ms window (0..+984 ms, tag `esaa_lag0`) by the mean over folds of `best_crit`
(validation coupling contribution at the selected epoch). Because runs re-evaluated from a
checkpoint lack `best_crit`, the criterion actually used is the validation coupling
accuracy at the selected epoch, parsed from each run's train.log for BOTH windows (all nulls
are ~0.50, so this ranks windows the same way as contribution). Test accuracy is printed only
for reporting after the choice.

    python scripts/results/select_lag.py
"""
import json
import sys
from pathlib import Path

import re

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.analysis.aggregate import load, pooled  # noqa: E402

ROOT = paths.RUNS


def val_at_best(tag: str, ds: str) -> list[float]:
    """Validation coupling accuracy at the selected epoch, one value per fold."""
    out = []
    for d in sorted((ROOT / tag).glob(f"{ds}_fold*")):
        log = (d / "train.log").read_text() if (d / "train.log").exists() else ""
        m = re.search(r"best epoch (\d+)", log) or re.search(r"\(best (\d+)\)", log)
        if not m:
            continue
        best = int(m.group(1))
        ep = re.search(rf"^ep\s+{best} \|.*? coup ([0-9.]+)", log, re.M)
        if ep:
            out.append(float(ep.group(1)))
    return out


TWO = {"kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp"}
out = {}
for ds in ["kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp", "sparrkulee",
           "megmasc"]:
    cell = "competing_5s" if ds in TWO else "same_talker_5s"
    r = {}
    for tag, lab in (("unified", "125ms"), ("esaa_lag0", "0ms")):
        runs = load(tag).get(ds, [])
        crit = val_at_best(tag, ds)
        r[lab] = {"val_crit": float(np.mean(crit)) if crit else None, "n": len(crit),
                  "test": pooled(runs, cell, "acc_coupling")}
    ok = all(v["val_crit"] is not None and v["n"] == len(load("unified").get(ds, []))
             for v in r.values())
    pick = (max(r, key=lambda k: r[k]["val_crit"]) if ok else None)
    out[ds] = {"choice": pick, **r}
    f = lambda v: "--" if v is None else f"{v:.4f}"  # noqa: E731
    print(f"{ds:14s} val 125ms {f(r['125ms']['val_crit'])} 0ms {f(r['0ms']['val_crit'])} -> "
          f"{pick} | test 125ms {f(r['125ms']['test'])} 0ms {f(r['0ms']['test'])}")
Path("paper/tables/lag_selection.json").write_text(json.dumps(out, indent=1))
