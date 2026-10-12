#!/usr/bin/env python
"""Pool per-fold baseline results -> <ds>/baseline_<model>.json + RESULTS.md.

Folds are pooled by summing hits and window counts (count-weighted), as in
``baselines/linear/linear_baseline.py``.  A dataset is written only when every
fold is present; missing folds are listed as failures.

    python baselines/deep/aggregate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402

CACHE_ROOT = paths.DATA
RES_ROOT = paths.BASELINES / "results"
DATASETS = ("kuleuven", "dtu", "esaa", "uhd255", "fuglsang2020", "broderick_cp",
            "sparrkulee", "megmasc")
N_FOLDS = {"kuleuven": 4, "uhd255": 2, "fuglsang2020": 4}
TRAINERS = {"vlaai": ["vlaai"], "decaf": ["decaf", "decaf_eeg"], "mm": ["mm"]}
WINDOWS = ("1s", "2s", "5s", "10s", "20s", "30s", "60s")


def fmt(v: list) -> str:
    return "–" if not v or v[0] is None else f"{100 * v[0]:.1f}"


def main() -> None:
    md = ["# Deep AAD baselines on the unified bundles (CoupleFlow stimulus-disjoint folds)",
          "",
          "One subject-independent model per fold (all training listeners), early-stopped on",
          "the fold's val segments, scored on held-out stimulus groups with the windowing,",
          "margins and same-talker negatives of `baselines/linear/linear_baseline.py` (identical",
          "rng draws). Accuracy in %, folds pooled by window count. `comp` = attended vs",
          "competing talker, `same` = attended vs another window of the same attended",
          "envelope, `trial` = whole-trial competing accuracy. Linear = ridge mTRF",
          "(`linear_baseline.json`, per listener).", ""]
    failures = []
    for trainer, variants in TRAINERS.items():
        per_fold: dict[str, list[dict]] = {}
        for ds in DATASETS:
            nf = N_FOLDS.get(ds, 5)
            files = [RES_ROOT / trainer / f"{ds}_fold{f}.json" for f in range(nf)]
            miss = [f.name for f in files if not f.exists()]
            if miss:
                failures.append(f"{trainer}: {ds} missing {', '.join(miss)}")
                continue
            per_fold[ds] = [json.loads(f.read_text()) for f in files]
        for var in variants:
            md += [f"## {var}", "",
                   "| dataset | " + " | ".join(f"{w} comp | {w} same" for w in WINDOWS)
                   + " | trial | test r | best ep (per fold) |",
                   "|---|" + "---|" * (2 * len(WINDOWS) + 3)]
            for ds, folds in per_fold.items():
                res = {"dataset": ds, "n_folds": len(folds), "windows": {}}
                for w in WINDOWS:
                    res["windows"][w] = {}
                    for rule in ("competing", "same_talker"):
                        k = sum(f["results"][var]["windows"][w][rule][0] for f in folds)
                        n = sum(f["results"][var]["windows"][w][rule][1] for f in folds)
                        res["windows"][w][rule] = (k / n if n else None, n)
                k = sum(f["results"][var]["trial_competing"][0] for f in folds)
                n = sum(f["results"][var]["trial_competing"][1] for f in folds)
                res["trial_competing"] = (k / n if n else None, n)
                (CACHE_ROOT / ds / f"baseline_{var}.json").write_text(json.dumps(res, indent=1))
                rs = [f["results"][var].get("test_r_att") for f in folds]
                r = (f"{sum(rs) / len(rs):.3f}" if all(x is not None for x in rs) else "–")
                eps = ",".join(str(f["best_epoch"]) for f in folds)
                md.append(f"| {ds} | " + " | ".join(
                    f"{fmt(res['windows'][w]['competing'])} | {fmt(res['windows'][w]['same_talker'])}"
                    for w in WINDOWS) + f" | {fmt(res['trial_competing'])} | {r} | {eps} |")
            md.append("")
    md += ["## Linear reference (linear_baseline.json)", "",
           "| dataset | " + " | ".join(f"{w} comp | {w} same" for w in WINDOWS) + " | trial |",
           "|---|" + "---|" * (2 * len(WINDOWS) + 1)]
    for ds in DATASETS:
        p = CACHE_ROOT / ds / "linear_baseline.json"
        if p.exists():
            res = json.loads(p.read_text())
            md.append(f"| {ds} | " + " | ".join(
                f"{fmt(res['windows'][w]['competing'])} | {fmt(res['windows'][w]['same_talker'])}"
                for w in WINDOWS) + f" | {fmt(res['trial_competing'])} |")
    md += ["", "## Notes", "",
           "- Recipes: VLAAI paper recipe (5-s windows, 80 % overlap, -r, Adam 1e-3, patience 5);",
           "  DECAF upstream recipe with 5-s present / 3.5-s past / 0.5-s delay, 10 epochs, patience 3,",
           "  selected on the *recursive* val r (oracle uses the attended envelope, so it is not a",
           "  decoder); `decaf_eeg` = its EEG branch alone, same checkpoint; MM = dilation model trained",
           "  on 5-s windows vs a same-trial mismatch window, selected on val competing accuracy",
           "  (same-talker on single-talker data). Epochs capped at 1500 steps; 95-min training budget",
           "  (never reached).",
           "- DECAF (recursive and EEG branch) sits at test r ~ 0 on all datasets, matching the",
           "  earlier strict replication; its below-chance fuglsang2020 / sparrkulee cells come from a",
           "  near-zero-r output tiled in per-window min-max-scaled 5-s chunks, not from a real signal.",
           "- broderick_cp: all subject-independent deep models are at chance for `competing` while the",
           "  per-listener linear decoder reaches 84 % trial accuracy.",
           "- test r = mean whole-trial r(recon, attended) over test trials."]
    md += ["", "## Failures / missing folds", ""]
    md += [f"- {x}" for x in failures] or ["- none"]
    out = Path(__file__).resolve().parent / "RESULTS.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
