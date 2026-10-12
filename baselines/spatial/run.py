#!/usr/bin/env python
"""Published spatial AAD classifiers (DARNet, ListenNet, DBPNet) under two protocols.

The models classify the attended *side* (L/R) from an EEG window alone.  Model
definitions are imported unmodified from the authors' repositories, cloned under
``baselines/spatial/thirdparty/`` by ``fetch_thirdparty.sh`` (not copied here: the
repositories carry no licence).  Training recipes follow the authors' code:

============  ==========================================================================
DARNet        CSP (64 comps, csp_space, concat cov, trace-normalised) fitted on training
              data; Adam lr 5e-4 wd 3e-4; batch 32; <= 100 epochs; stop after > 10
              epochs without val-loss improvement; training order shuffled once.
ListenNet     per-window max-abs normalisation + Euclidean alignment computed per set
              (train / val / test, as in ``dep/main.py``); AdamW lr 5e-4 wd 1e-4
              (subject-independent: lr 1e-3, ``indep/main.py``); max-norm constraint;
              batch 32; <= 100 epochs; patience 20; reshuffled every epoch.
DBPNet        temporal branch: CSP (32 comps) of the window; frequency branch: log band
              power (delta..gamma, FFT of the window) interpolated to 5 x 32 x 32
              topographic images (Clough-Tocher = ``griddata(method='cubic')``), z-scored
              per band over the set; AdamW lr 3e-3 wd 3e-5, cosine warm restarts
              (T0=10, Tmult=2, eta_min=3e-4); batch 32; <= 200 epochs; patience 15;
              training order shuffled once.
============  ==========================================================================

All windows overlap by 50 % for training (the papers' setting); the loss is class-balanced
cross-entropy (see RESULTS.md, deviations).  Input: the unified
bundle (64 canonical channels, 0.5-32 Hz, 64 Hz, z-scored per trial), upsampled to
128 Hz with a polyphase filter so that every kernel keeps the temporal extent it has
in the papers (KUL 128 Hz; DTU 128 Hz in ListenNet/DBPNet).

Protocols
---------
* ``A_indep`` -- CoupleFlow's content-disjoint fold (``split_fold``); one model on all
  training listeners, early-stopped on the fold's val segments, scored on
  non-overlapping windows (hop = window, same positions as CoupleFlow's
  ``eval_windows`` with margin 100) of the held-out stimulus groups.
* ``A_dep`` -- same fold, one model per listener (that listener's train / val
  segments; if the listener has no val segment, a random 10 % of its training
  windows, as in the papers).
* ``B`` -- the papers' within-trial split (DARNet's corrected protocol, "DARNet2"):
  each trial is cut at 90 % of its samples; windows (50 % overlap) of the first
  part are randomly split 9:1 into train / val, windows of the last part are the
  test set; CSP is fitted on the training part.  Test windows come from the same
  trials (same talkers, same story, same side) as the training windows.  KU
  Leuven: only the eight 6-min trials, as in the papers.

    python baselines/spatial/run.py --task 0
    python baselines/spatial/run.py --model darnet --dataset dtu \
        --protocol A_dep --window 1 --fold 0
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.signal import resample_poly

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from coupleflow import paths  # noqa: E402
from coupleflow.preprocessing.channels import STANDARD_64_CHANNELS  # noqa: E402
from coupleflow.data import eval_windows, load_dataset, split_fold  # noqa: E402

CACHE_ROOT = paths.DATA
THIRD = paths.THIRDPARTY
OUT_ROOT = paths.BASELINES / "spatial_results"
SR_IN, SR = 64, 128
MARGIN = 100
N_FOLDS_REQ = 5
SPLIT_KW = {"fuglsang2020": dict(tail_min_s=30.0, val_frac=0.2)}
MODELS = ("darnet", "listennet", "dbpnet")
DATASETS = ("kuleuven", "dtu", "esaa", "fuglsang2020", "uhd255")
WINDOWS = (1, 2, 5)
PROTOCOLS = ("A_indep", "A_dep", "B")
N_FOLDS = {"kuleuven": 4, "dtu": 5, "esaa": 5, "fuglsang2020": 4, "uhd255": 2}


def task_list() -> list[dict]:
    """Deterministic SLURM task table (B has a single "fold")."""
    out = []
    for m in MODELS:
        for ds in DATASETS:
            for w in WINDOWS:
                for p in PROTOCOLS:
                    for f in (range(N_FOLDS[ds]) if p != "B" else [0]):
                        out.append(dict(model=m, dataset=ds, window=w, protocol=p, fold=f))
    return out


# =============================================================================
# Third-party model import (from the scratch clones, unmodified)
# =============================================================================

def _load(name: str, path: Path, extra_path: Path | None = None):
    if extra_path is not None and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DBPNetWrap(nn.Module):
    """DBPNet top level (``DBPNet/model.py``): TAB(4) || FRB(4) -> Linear(8, 2).

    Re-stated here because ``model.py`` imports unrelated training dependencies
    (dotmap); the two branches are imported from ``model_module.py``.
    """

    def __init__(self, mm, in_channels: int) -> None:
        super().__init__()
        self.fre = mm.FRBNet()
        self.tem = mm.TABNet(in_channels)
        self.linear = nn.Linear(8, 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat((self.tem(x1), self.fre(x2)), dim=1))


def build_model(name: str, n_ch: int, L: int) -> nn.Module:
    if name == "darnet":
        mod = _load("darnet_model", THIRD / "darnet" / "model.py")
        return mod.DARNet({"Data_shape": (0, n_ch, L)})
    if name == "listennet":
        d = THIRD / "ListenNet" / "dep"
        mod = _load("listennet_model", d / "model.py", extra_path=d)
        if torch.cuda.is_available():
            return mod.ListenNet(num_classes=2, chans=n_ch, samples=L, kernel=8, depth=16,
                                 avepool=8)
        orig = torch.Tensor.cuda            # upstream calls .cuda() in __init__
        torch.Tensor.cuda = lambda self, *a, **k: self
        try:
            return mod.ListenNet(num_classes=2, chans=n_ch, samples=L, kernel=8, depth=16,
                                 avepool=8)
        finally:
            torch.Tensor.cuda = orig
    if name == "dbpnet":
        mm = _load("dbpnet_modules", THIRD / "DBPNet" / "model_module.py")
        return DBPNetWrap(mm, n_ch)
    raise KeyError(name)


# =============================================================================
# Pre-processing (fitted on training data only)
# =============================================================================

def fit_csp(trials: list[dict], regions: list[tuple[int, int, int]], n_comp: int,
            y_of: dict[int, int]) -> np.ndarray:
    """MNE CSP as in DARNet / DBPNet, fitted on non-overlapping 1-s chunks of ``regions``.

    ``cov_est='concat'`` makes chunking equivalent to fitting on the whole regions.
    Returns the (n_comp, 64) spatial filters.
    """
    import mne
    from mne.decoding import CSP
    mne.set_log_level("WARNING")
    X, y = [], []
    for ti, a, b in regions:
        e = trials[ti]["eeg128"]
        for st in range(a, b - SR + 1, SR):
            X.append(e[:, st:st + SR])
            y.append(y_of[ti])
    X = np.asarray(X, np.float64)
    # reg: tiny shrinkage.  The bundle is average-referenced and has zeroed bad
    # channels, so class covariances are rank-deficient; unregularised CSP then
    # amplifies null-space noise by orders of magnitude.
    csp = CSP(n_components=n_comp, reg=1e-3, log=None, cov_est="concat",
              transform_into="csp_space", norm_trace=True)
    csp.fit(X, np.asarray(y))
    return csp.filters_[:n_comp].astype(np.float32)


def max_abs_norm(x: np.ndarray) -> np.ndarray:
    """ListenNet ``data_norm``: each window divided by its max |value|."""
    return x / (np.abs(x).reshape(len(x), -1).max(1)[:, None, None] + 1e-12)


def ea_matrix(x: np.ndarray) -> np.ndarray:
    """ListenNet Euclidean alignment ``R^{-1/2}`` from windows ``(N, C, L)``.

    Eigenvalues are floored at 1e-3 x their mean (zeroed channels / average
    reference make ``R`` singular, and ``inv(sqrtm(R))`` would be infinite).
    """
    R = np.zeros((x.shape[1], x.shape[1]))
    for s in range(0, len(x), 4096):
        xs = x[s:s + 4096].astype(np.float64)
        R += np.einsum("ijk,ilk->jl", xs, xs, optimize=True)
    R /= len(x)
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, 1e-3 * w.mean())
    return ((V / np.sqrt(w)) @ V.T).astype(np.float32)


def _electrode_2d() -> np.ndarray:
    """Azimuthal-equidistant projection (DBPNet ``azim_proj``) of the canonical layout."""
    import mne
    mont = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    alias = {"TP9": "P9", "TP10": "P10"}       # BioSemi P9/P10 fill these slots
    out = []
    for c in STANDARD_64_CHANNELS:
        x, y, z = mont[alias.get(c, c)]
        elev = math.atan2(z, math.sqrt(x * x + y * y))
        az = math.atan2(y, x)
        rho = math.pi / 2 - elev
        out.append((rho * math.cos(az), rho * math.sin(az)))
    return np.asarray(out)


BANDS = ((1, 3), (4, 7), (8, 13), (14, 30), (31, 50))   # DBPNet delta..gamma (Hz)


def band_power(x: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """DBPNet ``to_alpha*``: log2 band power per channel, ``(N, C, L) -> (N, 5, C)``.

    Channels that are zero in a window (bad channels) get the mean over the
    window's good channels instead of log2(0).  The bundle is low-passed at 32 Hz,
    so the gamma band (31-50 Hz) carries only the 31-32 Hz edge.
    """
    N, C, L = x.shape
    res = SR / L
    P = np.abs(np.fft.fft(x, n=L, axis=-1)) ** 2
    out = np.empty((N, len(BANDS), C), np.float32)
    for k, (lo, hi) in enumerate(BANDS):
        a, b = math.ceil(lo / res), math.ceil(hi / res) + 1
        out[:, k] = np.log2(P[..., a:b].sum(-1) / L + 1e-12)
    m = masks[:, None, :]
    good_mean = (out * m).sum(-1, keepdims=True) / np.maximum(m.sum(-1, keepdims=True), 1)
    return np.where(m, out, good_mean).astype(np.float32)


def topo_images(bp: np.ndarray, size: int = 32) -> np.ndarray:
    """``(N, 5, 64)`` band powers -> ``(N, 5, 32, 32)`` images (NaN outside the hull)."""
    pos = _electrode_2d()
    gx, gy = np.mgrid[pos[:, 0].min():pos[:, 0].max():size * 1j,
                      pos[:, 1].min():pos[:, 1].max():size * 1j]
    N, B, C = bp.shape
    vals = bp.reshape(N * B, C).T                              # (64, N*B)
    out = np.empty((N * B, size, size), np.float32)
    step = 20000
    for s in range(0, N * B, step):
        it = CloughTocher2DInterpolator(pos, vals[:, s:s + step], fill_value=np.nan)
        out[s:s + step] = np.moveaxis(it(gx, gy), -1, 0)
    return out.reshape(N, B, size, size)


def image_stats(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean / std over all non-NaN pixels of a set (DBPNet ``scale``)."""
    mu = np.array([np.nanmean(img[:, k]) for k in range(img.shape[1])])
    sd = np.array([np.nanstd(img[:, k]) for k in range(img.shape[1])])
    return mu, sd


def apply_image_stats(img: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    img = (img - mu[None, :, None, None]) / (sd[None, :, None, None] + 1e-12)
    return np.nan_to_num(img, nan=0.0).astype(np.float32)


# =============================================================================
# Windows
# =============================================================================

def cut(trials: list[dict], wins: list[tuple[int, int]], L: int) -> tuple[np.ndarray, ...]:
    """``wins`` = [(trial, start@128Hz)] -> EEG ``(N, 64, L)``, labels, channel masks."""
    X = np.empty((len(wins), 64, L), np.float32)
    y = np.empty(len(wins), np.int64)
    M = np.empty((len(wins), 64), bool)
    for i, (ti, st) in enumerate(wins):
        t = trials[ti]
        X[i] = t["eeg128"][:, st:st + L]
        y[i] = t["y"]
        M[i] = t["mask"]
    return X, y, M


def sliding(ti: int, a: int, b: int, L: int, hop: int) -> list[tuple[int, int]]:
    return [(ti, st) for st in range(a, b - L + 1, hop)]


class Prep:
    """Model-specific input pipeline, fitted on training windows only."""

    def __init__(self, model: str, trials: list[dict], train_regions: list[tuple[int, int, int]],
                 train_wins: list[tuple[int, int]], L: int) -> None:
        self.model, self.trials, self.L = model, trials, L
        y_of = {ti: trials[ti]["y"] for ti, _, _ in train_regions}
        if model == "darnet":
            self.W = fit_csp(trials, train_regions, 64, y_of)
        elif model == "dbpnet":
            self.W = fit_csp(trials, train_regions, 32, y_of)
        self.img_stats = None

    def __call__(self, wins: list[tuple[int, int]], set_name: str
                 ) -> tuple[list[np.ndarray], np.ndarray]:
        X, y, M = cut(self.trials, wins, self.L)
        if self.model == "darnet":
            return [np.matmul(self.W, X)], y
        if self.model == "listennet":
            Xn = max_abs_norm(X)
            A = ea_matrix(Xn)                       # EA per set, as in dep/main.py
            return [np.matmul(A, Xn)[:, None]], y
        # dbpnet
        seq = np.matmul(self.W, X)[:, None]         # (N, 1, 32, L)
        img = topo_images(band_power(X, M))
        if set_name == "test" or self.img_stats is None:
            # upstream: images of the train set and of the test set are each z-scored
            # with their own statistics; val is carved from train (train statistics).
            stats = image_stats(img)
            if set_name == "train":
                self.img_stats = stats
        else:
            stats = self.img_stats
        img = apply_image_stats(img, *stats)[:, None]              # (N, 1, 5, 32, 32)
        return [seq, img], y


# =============================================================================
# Training
# =============================================================================

RECIPES = {
    "darnet": dict(opt="adam", lr=5e-4, wd=3e-4, epochs=100, patience=10, reshuffle=False),
    "listennet": dict(opt="adamw", lr=5e-4, wd=1e-4, epochs=100, patience=20, reshuffle=True),
    "dbpnet": dict(opt="adamw", lr=3e-3, wd=3e-5, epochs=200, patience=15, reshuffle=False),
}


def max_norm_(model: nn.Module) -> None:
    """ListenNet ``MaxNormDefaultConstraint`` (top-level children only, as upstream)."""
    last = None
    for _, m in model.named_children():
        if hasattr(m, "weight") and not m.__class__.__name__.startswith("BatchNorm"):
            m.weight.data = torch.renorm(m.weight.data, 2, 0, maxnorm=2)
            last = m.weight
    if last is not None:
        last.data = torch.renorm(last.data, 2, 0, maxnorm=0.5)


def logits_of(model: nn.Module, name: str, xs: list[torch.Tensor]) -> torch.Tensor:
    out = model(*xs)
    return out[1] if name == "listennet" else out


@torch.no_grad()
def predict(model: nn.Module, name: str, xs: list[torch.Tensor], bs: int = 512
            ) -> torch.Tensor:
    model.eval()
    out = []
    for s in range(0, len(xs[0]), bs):
        out.append(logits_of(model, name, [x[s:s + bs] for x in xs]).float())
    return torch.cat(out)


def train_eval(name: str, tr: tuple, va: tuple, te: tuple, indep: bool, seed: int,
               dev: torch.device) -> tuple[np.ndarray, int]:
    """Train with the paper recipe, early-stop on val CE, return test predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rec = dict(RECIPES[name])
    if name == "listennet" and indep:
        rec["lr"] = 1e-3                                           # indep/main.py
    to = lambda arrs: [torch.from_numpy(a).to(dev) for a in arrs]  # noqa: E731
    xtr, ytr = to(tr[0]), torch.from_numpy(tr[1]).to(dev)
    xva, yva = to(va[0]), torch.from_numpy(va[1]).to(dev)
    xte = to(te[0])
    n_ch, L = tr[0][0].shape[-2], tr[0][0].shape[-1]
    model = build_model(name, n_ch, L).to(dev)
    if name == "listennet":
        model.channel_weight2.data = model.channel_weight2.data.to(dev)
    opt_cls = torch.optim.Adam if rec["opt"] == "adam" else torch.optim.AdamW
    opt = opt_cls(model.parameters(), lr=rec["lr"], weight_decay=rec["wd"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2,
                                                                  eta_min=3e-3 / 10)
             if name == "dbpnet" else None)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(ytr)
    # Class-balanced CE (deviation, see RESULTS.md): under content-disjoint folds the
    # training prior of a listener can anti-correlate with its test label (KU Leuven:
    # every listener-fold test set is single-class), so an unweighted loss rewards a
    # prior that is guaranteed wrong on test.  No-op for balanced splits.
    cnt = torch.bincount(ytr, minlength=2).float().clamp(min=1)
    cw = (cnt.sum() / (2 * cnt)).to(dev)
    cnt_v = torch.bincount(yva, minlength=2).float().clamp(min=1)
    cw_v = (cnt_v.sum() / (2 * cnt_v)).to(dev)
    order = torch.randperm(n, generator=g)                        # shuffled once
    bs = 32
    best, best_state, stale, ep_done = float("inf"), None, 0, 0
    for ep in range(rec["epochs"]):
        model.train()
        if rec["reshuffle"]:
            order = torch.randperm(n, generator=g)
        for s in range(0, n - bs + 1, bs):                        # drop_last=True
            idx = order[s:s + bs].to(dev)
            loss = F.cross_entropy(logits_of(model, name, [x[idx] for x in xtr]), ytr[idx],
                                   weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if name == "listennet":
                max_norm_(model)
        if sched is not None:
            sched.step()
        vl = F.cross_entropy(predict(model, name, xva), yva, weight=cw_v).item()
        ep_done = ep + 1
        if vl < best:
            best, stale = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale > rec["patience"]:
                break
    model.load_state_dict(best_state)
    pred = predict(model, name, xte).argmax(1).cpu().numpy()
    return pred, ep_done


# =============================================================================
# Protocols
# =============================================================================

def listener_split(data: dict, sp: dict, subjects: list[str] | None) -> dict:
    """Segments of ``sp`` restricted to ``subjects`` (None = all)."""
    keep = None if subjects is None else set(subjects)
    return {k: [s for s in sp[k] if keep is None or data["trials"][s.trial]["subject"] in keep]
            for k in ("train", "val", "test")}


def run_A(args, data: dict, dev: torch.device) -> dict:
    trials = data["trials"]
    L, hop = args.window * SR, args.window * SR // 2
    kw = SPLIT_KW.get(args.dataset, {})
    sp = split_fold(data, args.fold, N_FOLDS_REQ, MARGIN, val_mode="auto",
                    val_frac=kw.get("val_frac", 0.15), seed=42,
                    **{k: v for k, v in kw.items() if k != "val_frac"})
    # Listeners scored: those with test data and some training data (CoupleFlow rule).
    by_sub: dict[str, dict] = {}
    for k in ("train", "val", "test"):
        for s in sp[k]:
            by_sub.setdefault(trials[s.trial]["subject"], {"train": [], "val": [], "test": []}
                              )[k].append(s)
    scored = sorted(s for s, d in by_sub.items() if d["test"] and (d["train"] or d["val"]))
    groups = [None] if args.protocol == "A_indep" else [[s] for s in scored]
    res = {"per_subject": {}, "epochs": [], "val_fallback": [], "val_mode": sp["val_mode"],
           "n_folds": sp["n_folds"]}
    for grp in groups:
        seg = listener_split(data, sp, grp)
        seg["test"] = [s for s in seg["test"] if trials[s.trial]["subject"] in scored]
        train_regions = [(s.trial, 2 * s.start, 2 * s.end) for s in seg["train"]]
        tr_w = [w for ti, a, b in train_regions for w in sliding(ti, a, b, L, hop)]
        va_w = [w for s in seg["val"] for w in sliding(s.trial, 2 * s.start, 2 * s.end, L, hop)]
        if not tr_w:
            continue
        rng = np.random.default_rng(42)
        if not va_w:                                   # papers: random 10 % of train windows
            perm = rng.permutation(len(tr_w))
            nv = max(1, len(tr_w) // 10)
            va_w = [tr_w[i] for i in perm[:nv]]
            tr_w = [tr_w[i] for i in perm[nv:]]
            res["val_fallback"].append(grp[0] if grp else "all")
        te_w = [(w["trial"], 2 * (w["start"] + MARGIN))
                for w in eval_windows(data, seg["test"], args.window * SR_IN, MARGIN)]
        if not te_w:
            continue
        t1 = time.time()
        prep = Prep(args.model, trials, train_regions, tr_w, L)
        tr = prep(tr_w, "train")
        va = prep(va_w, "val")
        te = prep(te_w, "test")
        t2 = time.time()
        pred, ep = train_eval(args.model, tr, va, te, args.protocol == "A_indep", 42, dev)
        print(f"    prep {t2 - t1:.0f} s, train {time.time() - t2:.0f} s", flush=True)
        res["epochs"].append(ep)
        for (ti, _), p, yy in zip(te_w, pred, te[1]):
            h = res["per_subject"].setdefault(trials[ti]["subject"], [0, 0])
            h[0] += int(p == yy)
            h[1] += 1
        print(f"  {grp[0] if grp else 'all'}: n_tr={len(tr_w)} n_va={len(va_w)} "
              f"n_te={len(te_w)} ep={ep} acc={np.mean(pred == te[1]):.3f}", flush=True)
    return res


def run_B(args, data: dict, dev: torch.device) -> dict:
    trials = data["trials"]
    L, hop = args.window * SR, args.window * SR // 2
    res = {"per_subject": {}, "epochs": []}
    for subj in data["subjects"]:
        tis = [i for i, t in enumerate(trials) if t["subject"] == subj]
        if args.dataset == "kuleuven":             # the papers use the eight 6-min trials
            tis = [i for i in tis if trials[i]["eeg128"].shape[1] >= 300 * SR]
        train_regions, tr_w, te_w = [], [], []
        for ti in tis:
            n = trials[ti]["eeg128"].shape[1]
            cut_at = int(n * 0.9)
            train_regions.append((ti, 0, cut_at))
            tr_w += sliding(ti, 0, cut_at, L, hop)
            te_w += sliding(ti, cut_at, n, L, hop)
        if not tr_w or not te_w:
            continue
        perm = np.random.default_rng(42).permutation(len(tr_w))
        nv = int(round(0.1 * len(tr_w)))
        va_w = [tr_w[i] for i in perm[:nv]]
        tr_w = [tr_w[i] for i in perm[nv:]]
        t1 = time.time()
        prep = Prep(args.model, trials, train_regions, tr_w, L)
        tr = prep(tr_w, "train")
        va = prep(va_w, "val")
        te = prep(te_w, "test")
        t2 = time.time()
        pred, ep = train_eval(args.model, tr, va, te, False, 42, dev)
        print(f"    prep {t2 - t1:.0f} s, train {time.time() - t2:.0f} s", flush=True)
        res["epochs"].append(ep)
        res["per_subject"][subj] = [int((pred == te[1]).sum()), int(len(pred))]
        print(f"  {subj}: n_tr={len(tr_w)} n_va={len(va_w)} n_te={len(te_w)} ep={ep} "
              f"acc={np.mean(pred == te[1]):.3f}", flush=True)
    return res


def prepare(dataset: str, subjects: set[str] | None = None) -> dict:
    data = load_dataset(str(CACHE_ROOT), dataset, stream_name="unified_env64.pt",
                        groups_name="unified_groups.json")
    if subjects:                                   # debug subset
        idx = [i for i, t in enumerate(data["trials"]) if t["subject"] in subjects]
        data = {"trials": [data["trials"][i] for i in idx],
                "groups": [data["groups"][i] for i in idx], "subjects": sorted(subjects)}
    for t in data["trials"]:
        t["eeg128"] = resample_poly(t["eeg"].float().numpy(), SR, SR_IN, axis=-1
                                    ).astype(np.float32)
        t["y"] = {"L": 0, "R": 1}[t["attended_ear"]]
        t["mask"] = t["channel_mask"].numpy().astype(bool)
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--protocol", choices=PROTOCOLS)
    ap.add_argument("--window", type=int, choices=WINDOWS)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--subjects", type=str, default=None, help="debug: comma list")
    ap.add_argument("--max-epochs", type=int, default=None, help="debug override")
    ap.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    args = ap.parse_args()
    tasks = task_list()
    if args.list:
        print(len(tasks))
        return
    if args.task is not None:
        for k, v in tasks[args.task].items():
            setattr(args, k, v)
    if args.max_epochs:
        for r in RECIPES.values():
            r["epochs"] = args.max_epochs
    out = Path(args.out_root) / args.model / (
        f"{args.dataset}_{args.protocol}_{args.window}s_fold{args.fold}.json")
    if out.exists() and args.subjects is None:
        print(f"{out} exists")
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"=== {args.model} {args.dataset} {args.protocol} {args.window}s fold {args.fold} "
          f"({dev}) ===", flush=True)
    data = prepare(args.dataset, set(args.subjects.split(",")) if args.subjects else None)
    print(f"data ready {time.time() - t0:.0f} s", flush=True)
    res = run_B(args, data, dev) if args.protocol == "B" else run_A(args, data, dev)
    res.update(model=args.model, dataset=args.dataset, protocol=args.protocol,
               window=args.window, fold=args.fold, seconds=time.time() - t0)
    hits = sum(h for h, _ in res["per_subject"].values())
    tot = sum(n for _, n in res["per_subject"].values())
    print(f"pooled acc = {hits / max(tot, 1):.4f} ({hits}/{tot}), {res['seconds']:.0f} s")
    if args.subjects is None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
