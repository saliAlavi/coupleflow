#!/usr/bin/env python
"""Accumulation-law kappa for the per-listener linear decoder (same model as check.py's r).

    python scripts/figures/kappa_linear.py   # -> figures/snr_law_linear.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from scripts.figures.snr_law import fit_kappa  # noqa: E402

C = paths.DATA
out = {}
for ds in ["kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp", "sparrkulee",
           "megmasc"]:
    j = json.loads((C / ds / "linear_baseline.json").read_text())
    for task, key in (("comp", "competing"), ("same", "same_talker")):
        T, A, N = [], [], []
        for w, v in j["windows"].items():
            a, n = v.get(key, [None, 0])
            if a is not None and n:
                T.append(float(w[:-1])), A.append(a), N.append(n)
        if len(T) >= 3:
            out[f"{ds}/{task}"] = {"kappa": fit_kappa(np.array(T), np.array(A), np.array(N))}
Path("figures/snr_law_linear.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"{k:22s} kappa_lin {v['kappa']:.3f}")
