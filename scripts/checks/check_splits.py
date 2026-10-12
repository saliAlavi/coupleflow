#!/usr/bin/env python
"""Audit every fold of a dataset: trial-, content-, stimulus-name- and sample-disjointness.

For the content folds (``split_fold``) and, with ``--loso``, every leave-one-listener-out
fold (``split_mode=loso``), checks that

1. no trial occurs in test and in train/val (trial-disjoint);
2. no content group of a test trial occurs in train/val (content-disjoint, audio groups);
3. no stimulus file name of a test trial occurs in train/val (independent, from metadata;
   Fuglsang'20 names are per listener, so this check is weaker there);
4. validation tails cut from training trials share no samples with the training part;
5. LOSO: the test listener has no training/validation data and the test set is non-empty.

    python scripts/checks/check_splits.py --dataset kuleuven [--loso]
Appends one JSON line per dataset/mode to paper/tables/split_audit.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.data import load_dataset, make_folds, split_fold  # noqa: E402
from coupleflow.preprocessing.run_config import split_kw, stream  # noqa: E402

CACHE = str(paths.DATA)
OUT = Path("paper/tables/split_audit.jsonl")
MARGIN = 100


def names(t: dict) -> set:
    return {str(t[k]) for k in ("att_name", "unatt_name") if t.get(k)}


def audit(data: dict, sp: dict, loso: bool) -> dict:
    tr = data["trials"]
    g = data["groups"]
    test = {s.trial for s in sp["test"]}
    seen = {s.trial for p in ("train", "val") for s in sp[p]}
    r = {"n_test_trials": len(test)}
    r["trial_overlap"] = len(test & seen)
    r["content_overlap"] = len({g[i] for i in test} & {g[i] for i in seen})
    r["name_overlap"] = len(set().union(*[names(tr[i]) for i in test] or [set()])
                            & set().union(*[names(tr[i]) for i in seen] or [set()]))
    # sample overlap between train and val parts of the same trial
    segs = {}
    for p in ("train", "val"):
        for s in sp[p]:
            segs.setdefault(s.trial, []).append((s.start, s.end))
    bad = 0
    for iv in segs.values():
        iv.sort()
        bad += sum(1 for (a0, a1), (b0, b1) in zip(iv, iv[1:]) if b0 < a1)
    r["sample_overlap"] = bad
    if loso:
        ts = sp["test_subject"]
        r["test_subject_in_train"] = int(any(tr[i]["subject"] == ts for i in seen))
        r["empty_test"] = int(len(test) == 0)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--loso", action="store_true")
    a = ap.parse_args()
    data = load_dataset(CACHE, a.dataset, stream_name=stream(a.dataset),
                        groups_name="unified_groups.json")
    data["cand_idx"] = []
    res = []
    if a.loso:
        for f in range(len(data["subjects"])):
            sp = split_fold(data, f, 5, MARGIN, mode="loso", **split_kw(a.dataset))
            res.append({"fold": f, **audit(data, sp, True), "val_mode": sp["val_mode"]})
    else:
        n = len(make_folds(data["groups"], data["trials"], 5, 42))
        for f in range(n):
            sp = split_fold(data, f, 5, MARGIN, **split_kw(a.dataset))
            res.append({"fold": f, **audit(data, sp, False), "val_mode": sp["val_mode"]})
    keys = [k for k in res[0] if k not in ("fold", "n_test_trials", "val_mode")]
    summary = {"dataset": a.dataset, "mode": "loso" if a.loso else "content_folds",
               "n_folds": len(res), "min_test_trials": min(r["n_test_trials"] for r in res),
               **{k: sum(r[k] for r in res) for k in keys},
               "val_modes": sorted({r["val_mode"] for r in res})}
    ok = all(summary[k] == 0 for k in keys)
    summary["PASS"] = ok
    print(json.dumps(summary))
    with open(OUT, "a") as fh:
        fh.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
