#!/usr/bin/env python
"""Group trials that share stimulus content, from the audio itself.

A split that holds out whole *trials* can still put the same *audio* in train
and test: KU Leuven replays every story pair five times per subject (trials 1,
5, 9, 13, 17 share one pair, with the ears swapped in trial 5), and all
datasets play the same stories to every subject.  A decoder with a learned
audio pathway can memorise that audio, so honest evaluation must hold out
stimulus *content*.

The datasets do not ship a usable stimulus key in our 64-Hz stream cache, so
we recover it from the envelopes:

1. Coarse pass (8 Hz): cut every stream (attended and unattended envelope of
   every trial) into non-overlapping 10-s query chunks and correlate each
   against every 10-s window, at every 1/8-s offset, of every other stream.
2. Verify pass (64 Hz): each coarse hit (r > ``coarse_thr``) implies an
   offset; re-correlate the full overlap at 64 Hz and keep the link only if
   r > ``verify_thr``.  Real repeats of one waveform give r ~ 1.0; chance
   matches between different speech do not survive at full rate.
3. Union-find over trials: two trials are in one group if any of their
   streams share content.  ``group`` is the stimulus key for splitting.

Output: ``<cache_root>/<dataset>/stimulus_groups.json`` with
``{"groups": {"<subject>|<trial>": gid}, "stats": {...}}``.

Usage:
    python coupleflow/preprocessing/stimulus_groups.py --dataset kuleuven
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.envelope_common import DEFAULT_CACHE_ROOT, load_streams  # noqa: E402

SR = 64
DS = 8                   # coarse-pass downsample factor: 64 Hz -> 8 Hz
WIN = 80                 # coarse window: 10 s at 8 Hz


def _znorm_rows(x: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-norm rows. ``x``: ``(N, L)``."""
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _coarse(env: np.ndarray) -> torch.Tensor:
    """64 Hz -> 8 Hz by block averaging."""
    n = len(env) // DS
    return torch.from_numpy(env[: n * DS].reshape(n, DS).mean(1).astype(np.float32))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _verify(a: np.ndarray, b: np.ndarray, off: int, min_len: int) -> float:
    """Pearson r at 64 Hz between ``a[t]`` and ``b[t - off]`` over their overlap."""
    lo = max(0, off)
    hi = min(len(a), len(b) + off)
    if hi - lo < min_len:
        return 0.0
    x = a[lo:hi].astype(np.float64)
    y = b[lo - off:hi - off].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else 0.0


def group_dataset(dataset: str, cache_root: Path, coarse_thr: float = 0.9,
                  verify_thr: float = 0.95, block: int = 4096,
                  stream_name: str | None = None) -> dict:
    """Compute stimulus groups for one dataset. See module docstring."""
    bundle = load_streams(cache_root, dataset, stream_name=stream_name)
    trials = bundle["trials"]
    streams: list[np.ndarray] = []
    owner: list[int] = []
    for ti, t in enumerate(trials):
        for key in ("env_att", "env_unatt"):
            if key == "env_unatt" and not t.get("has_unattended", True):
                continue
            streams.append(t[key].numpy())
            owner.append(ti)

    # Sliding 10-s windows (stride 1 at 8 Hz) of every stream, and the
    # non-overlapping query chunks.
    win_rows, win_src, win_pos = [], [], []
    q_rows, q_src, q_pos = [], [], []
    for si, env in enumerate(streams):
        c = _coarse(env)
        if len(c) < WIN:
            continue
        w = c.unfold(0, WIN, 1)                           # (n_pos, WIN)
        win_rows.append(_znorm_rows(w))
        win_src.append(torch.full((w.shape[0],), si))
        win_pos.append(torch.arange(w.shape[0]))
        qs = torch.arange(0, len(c) - WIN + 1, WIN)
        q_rows.append(_znorm_rows(w[qs]))
        q_src.append(torch.full((len(qs),), si))
        q_pos.append(qs)
    W = torch.cat(win_rows)
    Wsrc, Wpos = torch.cat(win_src), torch.cat(win_pos)
    Q = torch.cat(q_rows)
    Qsrc, Qpos = torch.cat(q_src), torch.cat(q_pos)
    print(f"[{dataset}] {len(trials)} trials, {len(streams)} streams, "
          f"{Q.shape[0]} queries x {W.shape[0]} windows", flush=True)

    # Coarse pass: every (query stream, window stream, offset) above threshold.
    t0 = time.time()
    Qsrc_n, Qpos_n = Qsrc.numpy(), Qpos.numpy()
    Wsrc_n, Wpos_n = Wsrc.numpy(), Wpos.numpy()
    hit_arrays = []
    for s in range(0, Q.shape[0], block):
        sim = Q[s:s + block] @ W.T                # (b, n_win)
        qi, wi = torch.nonzero(sim > coarse_thr, as_tuple=True)
        qi, wi = qi.numpy() + s, wi.numpy()
        sa, sb = Qsrc_n[qi], Wsrc_n[wi]
        keep = sa != sb
        off = (Qpos_n[qi] - Wpos_n[wi]) * DS
        hit_arrays.append(np.stack([sa[keep], sb[keep], off[keep]], 1))
    if hit_arrays:
        # A true copy is supported by every chunk at the same offset; chance hits scatter.
        # Rank each stream pair's offsets by their number of supporting chunks, so the
        # three verification attempts below go to the best-supported offsets (sorting by
        # offset instead could spend them on chance hits and miss a real copy).
        allh = np.concatenate(hit_arrays)
        allh[:, 2] = np.round(allh[:, 2] / DS).astype(int) * DS
        hits, votes = np.unique(allh, axis=0, return_counts=True)
        hits = hits[np.lexsort((-votes, hits[:, 1], hits[:, 0]))]
    else:
        hits = np.zeros((0, 3), int)
    print(f"[{dataset}] coarse pass {time.time() - t0:.0f}s, "
          f"{len(hits)} candidate (pair, offset) hits", flush=True)

    # Verify pass at 64 Hz, allowing +-DS samples of offset jitter.  Pairs whose
    # trials are already linked need no check, and each stream pair is tried at
    # no more than three offsets.
    uf = _UnionFind(len(trials))
    verified: dict[tuple[int, int], float] = {}
    tried: dict[tuple[int, int], int] = {}
    for sa, sb, off in hits.tolist():
        ta, tb = owner[sa], owner[sb]
        if uf.find(ta) == uf.find(tb):
            continue
        key = (min(sa, sb), max(sa, sb))
        if tried.get(key, 0) >= 3:
            continue
        tried[key] = tried.get(key, 0) + 1
        best = max(_verify(streams[sa], streams[sb], off + j, WIN * DS)
                   for j in range(-DS, DS + 1))
        if best > verify_thr:
            verified[key] = best
            uf.union(ta, tb)

    roots = [uf.find(i) for i in range(len(trials))]
    rid = {r: g for g, r in enumerate(sorted(set(roots)))}
    groups = {f"{t['subject']}|{t['trial']}": rid[r] for t, r in zip(trials, roots)}

    per_subj = collections.defaultdict(set)
    for t, r in zip(trials, roots):
        per_subj[t["subject"]].add(rid[r])
    sizes = collections.Counter(rid[r] for r in roots)
    stats = {
        "n_trials": len(trials),
        "n_groups": len(rid),
        "n_verified_links": len(verified),
        "group_size_hist": dict(collections.Counter(sizes.values())),
        "groups_per_subject": {s: len(g) for s, g in sorted(per_subj.items())},
        "min_verified_r": min(verified.values()) if verified else None,
        "coarse_thr": coarse_thr,
        "verify_thr": verify_thr,
    }
    return {"dataset": dataset, "groups": groups, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--coarse-thr", type=float, default=0.9)
    ap.add_argument("--verify-thr", type=float, default=0.95)
    ap.add_argument("--stream-name", default=None)
    ap.add_argument("--out-name", default="stimulus_groups.json")
    ap.add_argument("--block", type=int, default=4096,
                    help="queries per similarity block (memory ~ block x windows x 4 B)")
    args = ap.parse_args()
    out = group_dataset(args.dataset, Path(args.cache_root),
                        args.coarse_thr, args.verify_thr, block=args.block,
                        stream_name=args.stream_name)
    path = Path(args.cache_root) / args.dataset / args.out_name
    path.write_text(json.dumps(out, indent=1))
    s = out["stats"]
    print(f"[{args.dataset}] {s['n_groups']} stimulus groups over {s['n_trials']} trials; "
          f"group sizes {s['group_size_hist']}; groups/subject "
          f"{sorted(set(s['groups_per_subject'].values()))}; min verified r "
          f"{s['min_verified_r']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
