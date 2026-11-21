#!/usr/bin/env python
"""Linear lag controls on the CoupleFlow test folds (E8, rebuttal; CPU).

Two predictors of the candidate envelope at audio time t, both scored exactly like
``baselines/linear/linear_baseline.py`` (same content-disjoint folds, split_seed 42, the paper
bundle from run_config, non-overlapping windows with CoupleFlow's 100-sample margin discarded,
correlation decision):

* ``--mode stim`` (stimulus-only control): the input is the *attended* talker's own envelope
  (the only talker on single-talker sets) over env[t-g-62 .. t-g], i.e. exactly the 63-sample
  window that CoupleFlow's lag control reads from EEG (model.py, lag_mode="before":
  t-70 .. t-8 at g = 125 ms).  No neural data.  One pooled ridge model per fold (audio only),
  lambda on the fold's validation segments, refit on train + val.  This measures how well the
  speech that precedes a candidate predicts it across a gap g, i.e. the speech-rhythm route.
* ``--mode eeg`` (linear EEG lag control): the paper's per-listener ridge backward model with
  its 17-lag (250-ms) EEG window moved to end g ms before the candidate sample:
  eeg[t-g-16 .. t-g].  ``--after`` instead uses the paper's lags t .. t+16 (reproduces
  linear_baseline.json; sanity check).

Decisions: competing (attended vs competing talker, two-talker sets), same_talker (attended vs
the attended envelope of a random non-overlapping window of the same trial, as in
linear_baseline.py), and on MAESTRO the 4-way ``cands`` set (attended + the other three
attendable talkers, data.cand_idx=[0,1,2,3]) per window and per trial.

    python scripts/rebuttal/e8/lagctl.py --dataset megmasc --mode stim --gaps 125 250 500 1000
    python scripts/rebuttal/e8/lagctl.py --dataset megmasc --mode eeg --gaps 125
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from coupleflow.data import load_dataset, make_folds, split_fold  # noqa: E402
from coupleflow.preprocessing.core import CACHE_ROOT  # noqa: E402
from coupleflow.preprocessing.run_config import split_kw, stream  # noqa: E402

MARGIN = 100
WINDOWS = (1.0, 2.0, 5.0, 10.0, 20.0)
LAMS = (1e-2, 1e-1, 1.0, 1e1, 1e2)
SR = 64
OUT = Path("/fs/scratch/PAS2301/alialavi/rebuttal/e8")
SINGLE = {"sparrkulee", "megmasc"}
MAESTRO_CANDS = [0, 1, 2, 3]


def zc(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)


def rcorr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(zc(a) * zc(b)))


def lagged(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """``x`` (C, T) -> (T, C*len(lags)); column block i holds x[:, t + lags[i]] (0 outside)."""
    C, T = x.shape
    out = np.zeros((T, C * len(lags)), np.float32)
    for i, lag in enumerate(lags):
        blk = slice(i * C, (i + 1) * C)
        if lag >= 0:
            out[: T - lag, blk] = x[:, lag:].T
        else:
            out[-lag:, blk] = x[:, : T + lag].T
    return out


def att_env(t: dict) -> np.ndarray:
    return t["env_att"].float().numpy()


def inputs(t: dict, mode: str, a: int, b: int) -> np.ndarray:
    """Raw input channels (C, b-a) of trial ``t`` for samples [a, b)."""
    if mode == "stim":
        return att_env(t)[None, a:b]
    return t["eeg"][:, a:b].float().numpy()


def ridge_fit(XtX: np.ndarray, Xty: np.ndarray, lam: float) -> np.ndarray:
    sc = np.trace(XtX) / XtX.shape[0]
    return np.linalg.solve(XtX + lam * sc * np.eye(len(XtX)), Xty)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", choices=("stim", "eeg"), required=True)
    ap.add_argument("--gaps", type=float, nargs="*", default=[125.0], help="ms")
    ap.add_argument("--after", action="store_true", help="eeg mode: paper lags 0..250 ms")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--groups-name", default="unified_groups.json")
    ap.add_argument("--max-subjects", type=int, default=None, help="smoke test only")
    args = ap.parse_args()
    ds = args.dataset
    t0 = time.time()
    data = load_dataset(str(CACHE_ROOT), ds, stream_name=stream(ds), groups_name=args.groups_name)
    trials = data["trials"]
    maestro = "env_cands" in trials[0]
    single = ds in SINGLE
    n_folds = len(make_folds(data["groups"], trials, args.n_folds, 42))
    splits = [split_fold(data, f, args.n_folds, MARGIN, **split_kw(ds)) for f in range(n_folds)]
    subjects = sorted({t["subject"] for t in trials})
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    print(f"{ds}: {len(trials)} trials, {n_folds} folds, {len(subjects)} listeners, "
          f"load {time.time() - t0:.0f}s", flush=True)
    settings = [("after", None)] if args.after else [(f"gap{int(g)}", g) for g in args.gaps]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, g in settings:
        if args.after:
            lags = np.arange(0, 17)
        else:
            gs = int(round(g * SR / 1000))
            n_l = 63 if args.mode == "stim" else 17
            lags = -(gs + np.arange(n_l))             # t-g, t-g-1, ..., t-g-(n_l-1)
        res = run(data, splits, subjects, args.mode, lags, single, maestro, ds)
        res.update(dataset=ds, mode=args.mode, setting=name, gap_ms=g,
                   lags=[int(lags.min()), int(lags.max())], n_lags=len(lags), n_folds=n_folds,
                   stream=stream(ds), split_kw=split_kw(ds), minutes=(time.time() - t0) / 60)
        sfx = "_smoke" if args.max_subjects else ""
        out = OUT / f"{ds}_{args.mode}_{name}{sfx}.json"
        out.write_text(json.dumps(res, indent=1))
        w5 = res["windows"]["5s"]
        print(f"{ds} {args.mode} {name}: 5s " + ", ".join(f"{k} {v[0]:.3f} (n={v[1]})"
              for k, v in w5.items() if v[1]) + f"; {(time.time() - t0) / 60:.1f} min", flush=True)


def run(data: dict, splits: list[dict], subjects: list[str], mode: str, lags: np.ndarray,
        single: bool, maestro: bool, ds: str) -> dict:
    trials = data["trials"]
    rng = np.random.default_rng(0)
    sets = (["same_talker"] if single else ["competing", "same_talker"]) if not maestro \
        else ["cands", "same_talker"]
    hits = {w: {k: [] for k in sets} for w in WINDOWS}
    dr = {w: {k: [] for k in sets} for w in WINDOWS}      # r(att) - max r(other)
    ratt = {w: [] for w in WINDOWS}
    trial_hits: list[bool] = []
    sub_hits: dict = {}                                    # listener -> set -> 5-s hits
    cache: dict = {}                                       # (trial, a, b) -> (XtX, Xty)
    n_noval = [0]                                          # fits with a fixed lambda (no val)

    def stats(seg) -> tuple[np.ndarray, np.ndarray]:
        key = (seg.trial, seg.start, seg.end)
        if key not in cache:
            t = trials[seg.trial]
            X = lagged(inputs(t, mode, seg.start, seg.end), lags)
            y = zc(att_env(t)[seg.start:seg.end]).astype(np.float32)
            cache[key] = (X.T @ X, X.T @ y)
        return cache[key]

    def val_r(segs, ws: list[np.ndarray]) -> list[float]:
        """Mean validation correlation of each weight vector (one lagged build per segment)."""
        rs = [[] for _ in ws]
        Wm = np.stack(ws, 1)
        for s in segs:
            t = trials[s.trial]
            P = lagged(inputs(t, mode, s.start, s.end), lags) @ Wm
            y = att_env(t)[s.start:s.end]
            for j in range(len(ws)):
                rs[j].append(rcorr(P[:, j], y))
        return [float(np.mean(r)) for r in rs]

    def fit(train, val) -> np.ndarray:
        XtX = sum(stats(s)[0] for s in train)
        Xty = sum(stats(s)[1] for s in train)
        lam_best = LAMS[2]          # listener-folds without validation data: fixed lambda
        if not val:
            n_noval[0] += 1
        if val:
            rv = val_r(val, [ridge_fit(XtX, Xty, lam) for lam in LAMS])
            lam_best = LAMS[int(np.argmax(rv))]
            XtX = XtX + sum(stats(s)[0] for s in val)
            Xty = Xty + sum(stats(s)[1] for s in val)
        return ridge_fit(XtX, Xty, lam_best)

    def score(test_segs, w) -> None:
        for s in test_segs:
            t = trials[s.trial]
            p = lagged(inputs(t, mode, 0, t["eeg"].shape[1]), lags) @ w
            a = att_env(t)
            n = len(p)
            if maestro:
                ai = int(t["att_idx"])
                others = [t["env_cands"][i].float().numpy() for i in MAESTRO_CANDS if i != ai]
            elif not single and t.get("has_unattended", True):
                others = [t["env_unatt"].float().numpy()]
            else:
                others = []
            if others:
                trial_hits.append(rcorr(p, a) > max(rcorr(p, o) for o in others))
            for W in WINDOWS:
                win = int(W * SR)
                L = win + 2 * MARGIN
                for st in range(0, n - L + 1, win):
                    c = slice(st + MARGIN, st + MARGIN + win)
                    ra = rcorr(p[c], a[c])
                    ratt[W].append(ra)
                    if others:
                        ro = max(rcorr(p[c], o[c]) for o in others)
                        k = "cands" if maestro else "competing"
                        hits[W][k].append(ra > ro)
                        dr[W][k].append(ra - ro)
                        if W == 5.0:
                            sub_hits.setdefault(t["subject"], {}).setdefault(k, []).append(ra > ro)
                    ok = [q for q in range(0, n - L + 1, max(1, win // 2))
                          if q + L <= st or q >= st + L]
                    if ok:
                        q = int(rng.choice(ok)) + MARGIN
                        ro = rcorr(p[c], a[q:q + win])
                        hits[W]["same_talker"].append(ra > ro)
                        dr[W]["same_talker"].append(ra - ro)
                        if W == 5.0:
                            sub_hits.setdefault(t["subject"], {}).setdefault(
                                "same_talker", []).append(ra > ro)

    if mode == "stim":                                     # one pooled audio-only model per fold
        for f, sp in enumerate(splits):
            keep = set(subjects)
            sel = {k: [s for s in sp[k] if trials[s.trial]["subject"] in keep]
                   for k in ("train", "val", "test")}
            w = fit(sel["train"] or sel["val"], sel["val"] if sel["train"] else [])
            score(sel["test"], w)
            print(f"  fold {f} done", flush=True)
    else:                                                  # per listener, as linear_baseline.py
        for si, sub in enumerate(subjects):
            cache.clear()
            for f, sp in enumerate(splits):
                d = {k: [s for s in sp[k] if trials[s.trial]["subject"] == sub]
                     for k in ("train", "val", "test")}
                if not d["test"] or not (d["train"] or d["val"]):
                    continue
                w = fit(d["train"] or d["val"], d["val"] if d["train"] else [])
                score(d["test"], w)
            print(f"  listener {si + 1}/{len(subjects)} done", flush=True)
    res = {"windows": {}, "delta_r": {}, "r_att": {}}
    for W in WINDOWS:
        res["windows"][f"{W:g}s"] = {k: (float(np.mean(v)) if v else None, len(v))
                                     for k, v in hits[W].items()}
        res["delta_r"][f"{W:g}s"] = {k: (float(np.mean(v)) if v else None) for k, v in dr[W].items()}
        res["r_att"][f"{W:g}s"] = float(np.mean(ratt[W])) if ratt[W] else None
    res["trial"] = (float(np.mean(trial_hits)) if trial_hits else None, len(trial_hits))
    res["n_fits_fixed_lambda"] = n_noval[0]
    res["per_listener_5s"] = {sub: {k: [float(np.mean(v)), len(v)] for k, v in d.items()}
                              for sub, d in sub_hits.items()}
    res["chance"] = {"competing": 0.5, "same_talker": 0.5, "cands": 0.25}
    return res


if __name__ == "__main__":
    main()
