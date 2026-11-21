#!/usr/bin/env python
"""Summarise the E8 outputs (/fs/scratch/PAS2301/alialavi/rebuttal/e8/*.json) as markdown.

CoupleFlow's lag-control accuracies are read from paper/tables/tables.json ("before/<ds>", 5 s,
competing on KUL/DTU, same_talker on MEG-MASC) and paper/tables/maestro_ablation.json
("mabl_before", cands_matched), never typed.

    python scripts/rebuttal/e8/summarize.py
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import spearmanr, ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[3]
E8 = Path("/fs/scratch/PAS2301/alialavi/rebuttal/e8")
DS = ["kuleuven", "dtu", "sparrkulee", "megmasc", "maestro"]
TASK = {"kuleuven": "competing", "dtu": "competing", "sparrkulee": "same_talker",
        "megmasc": "same_talker", "maestro": "cands"}
CH = {"competing": 0.5, "same_talker": 0.5, "cands": 0.25}


def load(ds: str, mode: str, setting: str) -> dict | None:
    p = E8 / f"{ds}_{mode}_{setting}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell(r: dict | None, task: str, w: str = "5s") -> str:
    if r is None:
        return "—"
    v = r["windows"][w].get(task)
    return "—" if not v or v[0] is None else f"{100 * v[0]:.1f}"


def listener_test(r: dict | None, task: str) -> str:
    """Listener-level test of 5-s accuracy against chance: mean, k/N above, one-sided p."""
    if r is None or "per_listener_5s" not in r:
        return "—"
    v = [d[task][0] for d in r["per_listener_5s"].values() if task in d and d[task][1] >= 5]
    if len(v) < 3:
        return "—"
    c = CH[task]
    p_t = ttest_1samp(v, c, alternative="greater").pvalue
    p_w = wilcoxon([x - c for x in v], alternative="greater").pvalue
    return (f"{100 * sum(v) / len(v):.1f}; {sum(x > c for x in v)}/{len(v)} > chance; "
            f"t p={p_t:.1g}, W p={p_w:.1g}")


def main() -> None:
    tab = json.loads((ROOT / "paper/tables/tables.json").read_text())["ablation"]
    ma = json.loads((ROOT / "paper/tables/maestro_ablation.json").read_text())
    cf = {ds: tab[f"before/{ds}"]["acc"] for ds in ("kuleuven", "dtu", "megmasc")}
    cf["maestro"] = ma["mabl_before"]["cands_matched"]["acc"]
    gaps = ["gap125", "gap250", "gap500", "gap1000"]
    print("## (a) Stimulus-only control: attended envelope over the lag-control window\n")
    print("5-s accuracy (%), task as in CoupleFlow's lag control; chance 50 (MAESTRO 4-way: 25). "
          "Δr = r(pred, attended) − r(pred, best other), 5 s.\n")
    print("| Dataset | task | n (5 s) | " + " | ".join(g[3:] + " ms" for g in gaps)
          + " | Δr @125 | 1 s @125 | CoupleFlow lag ctl (5 s) |")
    print("|" + "---|" * (6 + len(gaps)))
    stim125 = {}
    for ds in DS:
        task = TASK[ds]
        rs = [load(ds, "stim", g) for g in gaps]
        r0 = rs[0]
        n = r0["windows"]["5s"][task][1] if r0 else "—"
        drv = r0["delta_r"]["5s"].get(task) if r0 else None
        if r0 and r0["windows"]["5s"][task][0] is not None:
            stim125[ds] = r0["windows"]["5s"][task][0]
        print(f"| {ds} | {task} | {n} | " + " | ".join(cell(r, task) for r in rs)
              + f" | {'—' if drv is None else f'{drv:.3f}'} | {cell(r0, task, '1s')} | "
              + (f"{100 * cf[ds]:.1f}" if ds in cf else "not run") + " |")
    common = [d for d in ("kuleuven", "dtu", "megmasc", "maestro") if d in stim125]
    if len(common) >= 3:
        norm = lambda d, a: (a - CH[TASK[d]]) / (1 - CH[TASK[d]])  # noqa: E731
        two = [d for d in common if d != "maestro"]
        r2 = spearmanr([stim125[d] for d in two], [cf[d] for d in two])[0]
        r4 = spearmanr([norm(d, stim125[d]) for d in common], [norm(d, cf[d]) for d in common])[0]
        print(f"\nSpearman ρ, stimulus-only (125 ms) vs CoupleFlow lag control: {r2:.2f} over "
              f"{', '.join(two)} (n={len(two)}); {r4:.2f} over {', '.join(common)} with chance-"
              f"normalised accuracy (n={len(common)}).")
    print("\n## (b) Linear EEG lag control: per-listener ridge, 17-lag (250-ms) EEG window "
          "ending g ms before the candidate\n")
    allg = ["after", "gap0"] + gaps
    print("| Dataset | task | " + " | ".join("after (paper lags)" if g == "after" else g[3:] + " ms"
                                           for g in allg) + " | n (5 s) |")
    print("|" + "---|" * (3 + len(allg)))
    for ds in DS:
        task = TASK[ds]
        rs = [load(ds, "eeg", g) for g in allg]
        n = next((r["windows"]["5s"][task][1] for r in rs if r), "—")
        print(f"| {ds} | {task} | " + " | ".join(cell(r, task) for r in rs) + f" | {n} |")
    print("\nListener-level tests (5 s; mean of listener accuracies; one-sided vs chance):\n")
    for mode, gl in (("stim", gaps), ("eeg", allg)):
        for ds in DS:
            task = TASK[ds]
            for g in gl:
                r = load(ds, mode, g)
                if r is not None:
                    print(f"- {mode} {ds} {g}: {listener_test(r, task)}")
    print("\nTrial-level (MAESTRO 4-way, ~30 s; two-talker competing):")
    for ds in DS:
        for mode, gl in (("stim", gaps), ("eeg", allg)):
            vals = []
            for g in gl:
                r = load(ds, mode, g)
                if r and r["trial"][0] is not None:
                    vals.append(f"{g} {100 * r['trial'][0]:.1f} (n={r['trial'][1]})")
            if vals:
                print(f"- {ds} {mode}: " + "; ".join(vals))
    if (ROOT / "paper/tables").exists():
        print(f"\nCoupleFlow MAESTRO lag control per trial: "
              f"{100 * ma['mabl_before']['cands_matched']['trial']:.1f}")


if __name__ == "__main__":
    main()
