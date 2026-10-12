#!/usr/bin/env python
"""Assemble tags `final`, `final_s1`, `final_s2` from the validation-chosen lag window.

Each dataset's fold directories are symlinked from the runs of the window that
`scripts/results/select_lag.py` chose on VALIDATION accuracy (never test):
125 ms -> unified / unified_s1 / unified_s2;  0 ms -> esaa_lag0 / esaa_lag0_s1 / esaa_lag0_s2.

    python scripts/results/select_lag.py && python scripts/results/make_final.py
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402

ROOT = paths.RUNS
SRC = {"125ms": ["unified", "unified_s1", "unified_s2"],
       "0ms": ["esaa_lag0", "esaa_lag0_s1", "esaa_lag0_s2"]}
sel = json.loads(Path("paper/tables/lag_selection.json").read_text())
for i, dst in enumerate(["final", "final_s1", "final_s2"]):
    (ROOT / dst).mkdir(exist_ok=True)
    for ds, v in sel.items():
        if v["choice"] is None:
            print(f"{ds}: no choice yet, skipped")
            continue
        src = ROOT / SRC[v["choice"]][i]
        for d in sorted(src.glob(f"{ds}_fold*")):
            link = ROOT / dst / d.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path("..") / src.name / d.name)   # relative: survives a clone
    print(dst, "->", {ds: v["choice"] for ds, v in sel.items()})
