#!/usr/bin/env python
"""Build a unified bundle for one dataset (see ``coupleflow/preprocessing/core.py``).

    python coupleflow/preprocessing/build.py --dataset kuleuven --workers 8
    python coupleflow/preprocessing/build.py --dataset kuleuven --limit 1      # smoke test

Readers live in ``coupleflow/preprocessing/readers/<dataset>.py`` and expose
``subjects() -> list[str]`` and ``read(subject) -> list[RawTrial]``; optional
``project(subject, trials)`` returns a MEG sensor projection.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.preprocessing.core import BUNDLE, process_trial, save_bundle  # noqa: E402


def _one(args: tuple) -> list[dict]:
    ds, subject = args
    import torch
    torch.set_num_threads(2)
    mod = importlib.import_module(f"coupleflow.preprocessing.readers.{ds}")
    t0 = time.time()
    try:
        raw = mod.read(subject)
        proj = mod.project(subject, raw) if hasattr(mod, "project") else None
        out = [process_trial(rt, proj) for rt in raw]
    except Exception:
        traceback.print_exc()
        print(f"[{ds}] {subject} FAILED", flush=True)
        return []
    print(f"[{ds}] {subject}: {len(out)} trials ({time.time() - t0:.0f}s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N subjects only")
    ap.add_argument("--name", default=BUNDLE)
    args = ap.parse_args()
    mod = importlib.import_module(f"coupleflow.preprocessing.readers.{args.dataset}")
    subs = mod.subjects()[: args.limit]
    jobs = [(args.dataset, s) for s in subs]
    trials: list[dict] = []
    if args.workers <= 1:
        for j in jobs:
            trials += _one(j)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(_one, jobs):
                trials += r
    meta = getattr(mod, "META", {})
    name = args.name if args.limit is None else args.name.replace(".pt", ".smoke.pt")
    save_bundle(args.dataset, trials, meta, name)


if __name__ == "__main__":
    main()
