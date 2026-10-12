#!/usr/bin/env python
"""Aggregate unified-bundle CoupleFlow runs (all datasets x windows) to markdown + JSON.

Windows are pooled across folds (folds are disjoint, so every test window is
counted once); per-listener means are also reported (mean over listeners of
the listener's pooled accuracy) with their spread.

    python coupleflow/analysis/aggregate.py --tag unified [--out paper_review/unified_results.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from coupleflow import paths  # noqa: E402

ROOT = paths.RUNS
CACHE = paths.DATA
WINDOWS = ("1s", "2s", "5s", "10s", "20s", "30s", "60s")


def load(tag: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for d in sorted((ROOT / tag).glob("*_fold*")):
        p = d / "results.json"
        if not p.exists():
            p = d / "results_val.json"        # evaluated from best.pt by the validator
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        out.setdefault(r["dataset"], []).append(r)
    return out


def pooled(rows: list[dict], cell: str, key: str, wkey: str = "n_windows") -> float | None:
    v, w = [], []
    for r in rows:
        c = r["results"].get(cell)
        if not c or c.get(key) is None:
            continue
        v.append(c[key])
        w.append(c.get(wkey, 1))
    return float(np.average(v, weights=w)) if v else None


def per_listener(rows: list[dict], cell: str) -> tuple[float, float, int] | None:
    acc: dict[str, list] = {}
    for r in rows:
        c = r["results"].get(cell)
        if not c:
            continue
        for s, d in c.get("per_subject", {}).items():
            acc.setdefault(s, []).append((d["coupling"], d["n"]))
    if not acc:
        return None
    m = [np.average([a for a, _ in v], weights=[n for _, n in v]) for v in acc.values()]
    return float(np.mean(m)), float(np.std(m)), len(m)


def f(x: float | None, pct: bool = True) -> str:
    return "–" if x is None else (f"{100 * x:.1f}" if pct else f"{x:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="unified")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = load(args.tag)
    summary: dict = {}
    lines = [f"## CoupleFlow, unified preprocessing (tag `{args.tag}`)", "",
             "Stimulus-disjoint folds, coupling-only decisions, % correct.  "
             "`comp` = attended vs competing talker; `same` = attended vs the same "
             "talker at another time (match-mismatch; the only task for single-talker "
             "data).  Pooled over all test windows of all folds.", ""]
    hdr = "| dataset | folds | listeners | task | " + " | ".join(WINDOWS) + " | trial | null 5s |"
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for ds in sorted(runs):
        rows = runs[ds]
        n_l = per_listener(rows, "same_talker_5s")
        summary[ds] = {"folds": len(rows)}
        for task, pre in (("comp", "competing"), ("same", "same_talker")):
            accs = [pooled(rows, f"{pre}_{w}", "acc_coupling") for w in WINDOWS]
            if all(a is None for a in accs):
                continue
            trial = (pooled(rows, f"{pre}_5s", "trial_acc_coupling", "n_trial_decisions")
                     if pre == "competing" else None)
            null = pooled(rows, f"{pre}_5s", "null_coupling")
            summary[ds][task] = {"windows": dict(zip(WINDOWS, accs)), "trial": trial,
                                 "null_5s": null,
                                 "per_listener_5s": per_listener(rows, f"{pre}_5s")}
            lines.append(f"| {ds} | {len(rows)} | {n_l[2] if n_l else '–'} | {task} | "
                         + " | ".join(f(a) for a in accs) + f" | {f(trial)} | {f(null)} |")
    lines += ["", "### Linear backward-model baseline (same folds / windows)", "",
              "| dataset | task | " + " | ".join(WINDOWS) + " | trial |",
              "|" + "---|" * (len(WINDOWS) + 3)]
    for ds in sorted(runs):
        p = CACHE / ds / "linear_baseline.json"
        if not p.exists():
            continue
        lb = json.loads(p.read_text())
        summary[ds]["linear"] = lb
        for task, key in (("comp", "competing"), ("same", "same_talker")):
            accs = [lb["windows"].get(w, {}).get(key, [None])[0] for w in WINDOWS]
            if all(a is None for a in accs):
                continue
            tr = lb["trial_competing"][0] if task == "comp" else None
            lines.append(f"| {ds} | {task} | " + " | ".join(f(a) for a in accs) + f" | {f(tr)} |")
    lines += ["", "### Per-listener 5-s accuracy (mean ± s.d. over listeners)", "",
              "| dataset | comp | same |", "|---|---|---|"]
    for ds in sorted(summary):
        c = summary[ds].get("comp", {}).get("per_listener_5s")
        s = summary[ds].get("same", {}).get("per_listener_5s")
        g = lambda t: "–" if not t else f"{100 * t[0]:.1f} ± {100 * t[1]:.1f} (n={t[2]})"  # noqa: E731
        lines.append(f"| {ds} | {g(c)} | {g(s)} |")
    txt = "\n".join(lines)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
