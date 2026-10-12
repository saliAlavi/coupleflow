#!/usr/bin/env python
"""Does a set of folds leak content?  Rebuild the folds exactly as the runs did (from
``--groups``), then check every test trial against a second, reference grouping
(``--ref``): a test trial leaks if its reference group also has a training or validation
trial in the same fold.

    python scripts/checks/check_fold_leak.py --dataset sparrkulee \
        --groups unified_groups.json --ref unified_groups_v2.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.data import load_dataset, split_fold  # noqa: E402

CACHE = str(paths.DATA)
SR = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--groups", default="unified_groups.json")
    ap.add_argument("--ref", default="unified_groups_v2.json")
    ap.add_argument("--stream", default="unified_env64.pt")
    a = ap.parse_args()
    data = load_dataset(CACHE, a.dataset, stream_name=a.stream, groups_name=a.groups)
    ref = load_dataset(CACHE, a.dataset, stream_name=a.stream, groups_name=a.ref)["groups"]
    n_folds = None
    tot_trials = tot_leak = 0
    tot_s = leak_s = 0.0
    for fold in range(5):
        sp = split_fold(data, fold, 5, 100, val_mode="auto", val_frac=0.15,
                        tail_min_s=200.0, seed=42)
        n_folds = sp["n_folds"]
        if fold >= n_folds:
            break
        seen = {ref[s.trial] for part in ("train", "val") for s in sp[part]}
        test = {s.trial for s in sp["test"]}
        leak = [t for t in test if ref[t] in seen]
        secs = sum(data["trials"][t]["eeg"].shape[1] for t in test) / SR
        lsecs = sum(data["trials"][t]["eeg"].shape[1] for t in leak) / SR
        tot_trials += len(test)
        tot_leak += len(leak)
        tot_s += secs
        leak_s += lsecs
        print(f"fold {fold}: {len(leak)}/{len(test)} test trials share content with "
              f"train/val ({100 * lsecs / max(secs, 1):.1f}% of test time)")
    print(f"ALL: {tot_leak}/{tot_trials} test trials, {100 * leak_s / max(tot_s, 1):.1f}% of "
          f"test time")


if __name__ == "__main__":
    main()
