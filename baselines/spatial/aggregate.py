#!/usr/bin/env python
"""Aggregate spatial-classifier results -> per-dataset JSON + RESULTS tables.

Writes ``$COUPLEFLOW_DATA/<ds>/spatial_<model>.json`` and prints
Markdown tables (pooled accuracy and mean +- sd over listeners).

    python baselines/spatial/aggregate.py > /tmp/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import CACHE_ROOT, DATASETS, MODELS, N_FOLDS, OUT_ROOT, WINDOWS, task_list  # noqa: E402

KEYS = {"A_indep": "protocol_A_subject_indep", "A_dep": "protocol_A_subject_dep",
        "B": "protocol_B"}


def collect(model: str, ds: str) -> dict:
    out = {"n_folds": N_FOLDS[ds], "missing": []}
    for p, key in KEYS.items():
        pooled, lmean, lsd, per_sub_all, n_win = {}, {}, {}, {}, {}
        for w in WINDOWS:
            per_sub: dict[str, list[int]] = {}
            folds = range(N_FOLDS[ds]) if p != "B" else [0]
            ok = True
            for f in folds:
                fp = OUT_ROOT / model / f"{ds}_{p}_{w}s_fold{f}.json"
                if not fp.exists():
                    out["missing"].append(fp.name)
                    ok = False
                    continue
                r = json.loads(fp.read_text())
                for s, (h, n) in r["per_subject"].items():
                    a = per_sub.setdefault(s, [0, 0])
                    a[0] += h
                    a[1] += n
            if not per_sub or not ok:
                continue
            H = sum(h for h, _ in per_sub.values())
            N = sum(n for _, n in per_sub.values())
            accs = [100 * h / n for h, n in per_sub.values() if n]
            pooled[f"{w}s"] = round(100 * H / N, 2)
            lmean[f"{w}s"] = round(float(np.mean(accs)), 2)
            lsd[f"{w}s"] = round(float(np.std(accs)), 2)
            n_win[f"{w}s"] = N
            per_sub_all[f"{w}s"] = {s: round(100 * h / n, 2) for s, (h, n) in per_sub.items()}
        out[key] = pooled
        out[key + "_listener_mean"] = lmean
        out[key + "_listener_sd"] = lsd
        out[key + "_n_windows"] = n_win
        out[key + "_per_listener"] = per_sub_all
    return out


def main() -> None:
    allres = {}
    for ds in DATASETS:
        for m in MODELS:
            r = collect(m, ds)
            allres[(m, ds)] = r
            (CACHE_ROOT / ds / f"spatial_{m}.json").write_text(json.dumps(r, indent=1))
    n_missing = sum(len(r["missing"]) for r in allres.values())
    print(f"<!-- {len(task_list()) - n_missing}/{len(task_list())} task files present -->\n")
    for key, title in [("protocol_B", "Protocol B (papers' within-trial split, subject-dependent)"),
                       ("protocol_A_subject_dep", "Protocol A, subject-dependent"),
                       ("protocol_A_subject_indep", "Protocol A, subject-independent")]:
        print(f"### {title}: accuracy % (pooled windows | listener mean ± sd)\n")
        print("| dataset | model | " + " | ".join(f"{w} s" for w in WINDOWS) + " |")
        print("|---|---|" + "---|" * len(WINDOWS))
        for ds in DATASETS:
            for m in MODELS:
                r = allres[(m, ds)]
                cells = []
                for w in WINDOWS:
                    k = f"{w}s"
                    if k in r[key]:
                        cells.append(f"{r[key][k]:.1f} \\| {r[key + '_listener_mean'][k]:.1f}"
                                     f" ± {r[key + '_listener_sd'][k]:.1f}")
                    else:
                        cells.append("–")
                print(f"| {ds} | {m} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
