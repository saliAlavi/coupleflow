#!/usr/bin/env python
"""How close do *different* stimuli come to the 0.95 link threshold?  (App. C)

For each dataset: every non-overlapping 10-s chunk of every stream of listener A (the
query) is correlated, at every 1/8-s offset, with every 10-s window of every stream of a
second listener B, at 8 Hz (the stage at which ``coupleflow/preprocessing/stimulus_groups.py`` searches),
excluding windows from the query's own content group.  Reported: the median, 99th
percentile and maximum over queries of the best such match.  Listeners are the two with
the most trials, so their streams cover the dataset's content.

    python scripts/tables/groups_null.py [--datasets kuleuven,dtu,...]
Writes paper/tables/groups_null.json.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.envelope_common import load_streams  # noqa: E402
from coupleflow.preprocessing.stimulus_groups import DS, WIN, _coarse, _znorm_rows  # noqa: E402

CACHE = str(paths.DATA)
OUT = Path("paper/tables/groups_null.json")
ALL = ["kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp", "sparrkulee",
       "megmasc", "maestro"]


def windows(streams):
    rows, grp, q, qg, src, pos, qsrc, qpos = [], [], [], [], [], [], [], []
    for si, (env, g) in enumerate(streams):
        c = _coarse(env)
        if len(c) < WIN:
            continue
        w = _znorm_rows(c.unfold(0, WIN, 1))
        rows.append(w)
        grp.append(torch.full((w.shape[0],), g))
        src.append(torch.full((w.shape[0],), si))
        pos.append(torch.arange(w.shape[0]))
        idx = torch.arange(0, w.shape[0], WIN)
        q.append(w[idx])
        qg.append(torch.full((len(idx),), g))
        qsrc.append(torch.full((len(idx),), si))
        qpos.append(idx)
    cat = torch.cat
    return (cat(rows), cat(grp), cat(q), cat(qg), cat(src), cat(pos), cat(qsrc), cat(qpos))


def r64(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r of two equal-length 64-Hz segments."""
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def run(ds: str) -> dict:
    b = load_streams(CACHE, ds, stream_name="unified_env64.pt")
    gp = Path(f"{CACHE}/{ds}/unified_groups_v2.json")
    gp = gp if gp.exists() else Path(f"{CACHE}/{ds}/unified_groups.json")
    g = json.loads(gp.read_text())["groups"]
    by_s = collections.defaultdict(list)
    for t in b["trials"]:
        gid = g[f"{t['subject']}|{t['trial']}"]
        by_s[t["subject"]].append((t["env_att"].numpy(), gid))
        if t.get("has_unattended", True) and t.get("env_unatt") is not None:
            by_s[t["subject"]].append((t["env_unatt"].numpy(), gid))
    sa, sb = sorted(by_s, key=lambda s: -len(by_s[s]))[:2]
    A, B = by_s[sa], by_s[sb]
    _, _, Q, Qg, _, _, Qs, Qp = windows(A)
    W, Wg, _, _, Ws, Wp, _, _ = windows(B)
    best, arg = [], []
    for s in range(0, Q.shape[0], 256):
        sim = Q[s:s + 256] @ W.T
        sim = sim.masked_fill(Qg[s:s + 256, None] == Wg[None, :], -2.0)
        v, i = sim.max(1)
        best.append(v)
        arg.append(i)
    best, arg = torch.cat(best), torch.cat(arg)
    ok = best > -2
    # The decisive stage: the 64-Hz correlation of each query's best other-group match,
    # with the same +-1/8-s offset jitter as coupleflow/preprocessing/stimulus_groups.py.
    L = WIN * DS
    b64 = []
    for qi in torch.nonzero(ok).flatten().tolist():
        qa = A[int(Qs[qi])][0][int(Qp[qi]) * DS:int(Qp[qi]) * DS + L]
        wb = B[int(Ws[arg[qi]])][0]
        p0 = int(Wp[arg[qi]]) * DS
        cand = [r64(qa, wb[p0 + j:p0 + j + L]) for j in range(-DS, DS + 1)
                if p0 + j >= 0 and p0 + j + L <= len(wb) and len(qa) == L]
        if cand:
            b64.append(max(cand))
    best, b64 = best[ok], torch.tensor(b64)
    r = {"listeners": [sa, sb], "n_queries": int(len(best)),
         "median": float(best.median()), "p99": float(best.quantile(0.99)),
         "max": float(best.max()),
         "median64": float(b64.median()), "p99_64": float(b64.quantile(0.99)),
         "max64": float(b64.max())}
    print(ds, r, flush=True)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(ALL))
    args = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for ds in args.datasets.split(","):
        res[ds] = run(ds)
        OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
