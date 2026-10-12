#!/usr/bin/env python
"""Leakage control: 5-s competing accuracy under a trial split vs the content-disjoint split.

    python scripts/tables/leak_table.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.analysis.aggregate import CACHE, load, pooled  # noqa: E402

LEAK = paths.BASELINES / "results_leak"
NAMES = {"kuleuven": "KU Leuven", "dtu": "DTU"}


def base_leak(model: str, ds: str, cell: str = "5s", key: str = "competing") -> float | None:
    hit = n = 0
    for p in sorted((LEAK / model).glob(f"{ds}_fold*.json")):
        w = json.loads(p.read_text())["results"][model]["windows"][cell][key]
        hit, n = hit + w[0], n + w[1]
    return hit / n if n else None


def base_disjoint(model: str, ds: str) -> float | None:
    name = "linear_baseline.json" if model == "linear" else f"baseline_{model}.json"
    return json.loads((CACHE / ds / name).read_text())["windows"]["5s"]["competing"][0]


def main() -> None:
    rows, res = [], {}
    for ds in ("kuleuven", "dtu"):
        lin_leak = json.loads((CACHE / ds / "linear_baseline_trialsplit.json").read_text())
        vals = {
            "Linear": (lin_leak["windows"]["5s"]["competing"][0], base_disjoint("linear", ds)),
            "VLAAI": (base_leak("vlaai", ds), base_disjoint("vlaai", ds)),
            "Dilated MM": (base_leak("mm", ds), base_disjoint("mm", ds)),
            "CoupleFlow": (pooled(load("leak_trial")[ds], "competing_5s", "acc_coupling"),
                           pooled(load("unified")[ds], "competing_5s", "acc_coupling")),
        }
        res[ds] = vals
        for i, (m, (a, b)) in enumerate(vals.items()):
            rows.append(f"{NAMES[ds] if i == 0 else ''} & {m} & {100 * a:.1f} & {100 * b:.1f} & "
                        f"{100 * (a - b):+.1f} \\\\")
        rows.append("\\midrule" if ds == "kuleuven" else "")
    tex = ("\\begin{tabular}{llccc}\\toprule\nDataset & model & trial split & content-disj. & "
           "inflation \\\\\\midrule\n" + "\n".join(r for r in rows if r) + "\n\\bottomrule\\end{tabular}\n")
    Path("paper/tables/leak.tex").write_text(tex)
    Path("paper/tables/leak.json").write_text(json.dumps(res, indent=1))
    print(tex)


if __name__ == "__main__":
    main()
