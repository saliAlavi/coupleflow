#!/usr/bin/env python
"""Deep AAD baselines on the unified bundles, under CoupleFlow's split regime.

One model per (dataset, fold), subject-independent: trained on every training
listener of the fold, early-stopped on the fold's val segments, scored once on
the fold's held-out stimulus groups.

Models
------
* ``vlaai``  -- VLAAI (Accou et al. 2023), backward envelope reconstruction.
  Model = ``paper_review/vlaai/models_upstream.py`` (Keras port), paper recipe:
  5-s windows, 80 % overlap, -Pearson loss, Adam 1e-3, batch 64, patience 5.
* ``decaf``  -- DECAF (Thakkar & Elhilali 2026), upstream module via
  ``paper_review/decaf/models_upstream.py``.  Recipe as in
  ``paper_review/decaf/train.py`` with a 5-s present window, 3.5-s past
  context, 500 ms EEG delay, (1-r)+0.2*L1, Noam schedule, 10 epochs,
  patience 3.  Selected and scored in the fully recursive configuration (past
  envelope predicted from past EEG); the oracle variant uses the attended
  envelope and is therefore not an AAD decoder.  The EEG branch alone
  (HappyQuokka) is scored as ``decaf_eeg`` from the same checkpoint.
* ``mm``     -- match/mismatch dilation model (Accou et al. 2021; ICASSP 2023
  auditory-EEG challenge baseline): 1x1 spatial conv (8) + 3 dilated convs
  (16 filters, k=3, dilation 1/3/9, valid, ReLU) on EEG, the same dilated
  stack (shared across candidates) on the envelope, cosine-similarity matrix
  over time (16x16), linear read-out, softmax over candidates.  Trained on
  5-s windows against a non-overlapping window of the same trial's attended
  envelope.

Evaluation
----------
Identical to ``baselines/linear/linear_baseline.py``: windows (1,2,5,10,20,30,60)
s, hop = window, ``MARGIN`` samples discarded on both sides, ``competing`` and
``same_talker`` rules, whole-trial competing accuracy.  The same-talker
negative windows are *the very same ones* the linear baseline drew: its
``default_rng(0)`` stream is replayed over all folds in its loop order (see
:func:`window_plan`), so every per-fold job reproduces them.

    python baselines/deep/run.py --model vlaai --dataset kuleuven --fold 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from coupleflow import paths  # noqa: E402
from coupleflow.data import load_dataset, make_folds, split_fold  # noqa: E402
from coupleflow.preprocessing.run_config import split_kw, stream  # noqa: E402

CACHE_ROOT = paths.DATA
OUT_ROOT = paths.BASELINES / "results"
SR = 64
MARGIN = 100
WINDOWS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
N_FOLDS_REQ = 5


# =============================================================================
# Evaluation plan (replays linear_baseline.py's window loop and rng)
# =============================================================================

def window_plan(data: dict, n_folds: int) -> list[list[dict]]:
    """Test trials and window positions per fold, exactly as linear_baseline.py.

    Returns:
        ``plan[fold]`` = list of ``{"trial": ti, "wins": {W: [(c0, q0 | None), ...]}}``
        where ``c0`` is the start of the scored window and ``q0`` the start of the
        same-talker negative (``None`` when the linear baseline had no candidate).
    """
    trials = data["trials"]
    rng = np.random.default_rng(0)
    plan = []
    for fold in range(n_folds):
        # Same call as linear_baseline.py (default val settings); the test set and
        # the per-subject skip rule do not depend on the val settings.
        sp = split_fold(data, fold, N_FOLDS_REQ, MARGIN, seed=42)
        by_sub: dict[str, dict[str, list]] = {}
        for split in ("train", "val", "test"):
            for s in sp[split]:
                by_sub.setdefault(trials[s.trial]["subject"],
                                  {"train": [], "val": [], "test": []})[split].append(s)
        fp = []
        for d in by_sub.values():
            if not d["test"] or not (d["train"] or d["val"]):
                continue
            for s in d["test"]:
                n = int(trials[s.trial]["eeg"].shape[1])
                wins = {}
                for W in WINDOWS:
                    win = int(W * SR)
                    L = win + 2 * MARGIN
                    qs = np.arange(0, n - L + 1, max(1, win // 2))
                    lst = []
                    for st in range(0, n - L + 1, win):
                        ok = qs[(qs + L <= st) | (qs >= st + L)]
                        q = int(rng.choice(ok)) + MARGIN if len(ok) else None
                        lst.append((st + MARGIN, q))
                    wins[W] = lst
                fp.append({"trial": s.trial, "wins": wins})
        plan.append(fp)
    return plan


def zc(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)


def rcorr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(zc(a) * zc(b)))


def empty_hits() -> dict:
    return {"windows": {f"{W:g}s": {"competing": [0, 0], "same_talker": [0, 0]}
                        for W in WINDOWS}, "trial_competing": [0, 0]}


def score_recon(trials: list[dict], fplan: list[dict], preds: dict[int, np.ndarray]) -> dict:
    """Hit counts for a reconstruction model (prediction per test trial)."""
    h = empty_hits()
    for item in fplan:
        t = trials[item["trial"]]
        p = preds[item["trial"]]
        a = t["env_att"].numpy().astype(np.float64)
        u = t["env_unatt"].numpy().astype(np.float64) if t["has_unattended"] else None
        if u is not None:
            h["trial_competing"][0] += int(rcorr(p, a) > rcorr(p, u))
            h["trial_competing"][1] += 1
        for W, lst in item["wins"].items():
            win = int(W * SR)
            hw = h["windows"][f"{W:g}s"]
            for c0, q0 in lst:
                c = slice(c0, c0 + win)
                ra = rcorr(p[c], a[c])
                if u is not None:
                    hw["competing"][0] += int(ra > rcorr(p[c], u[c]))
                    hw["competing"][1] += 1
                if q0 is not None:
                    hw["same_talker"][0] += int(ra > rcorr(p[c], a[q0:q0 + win]))
                    hw["same_talker"][1] += 1
    return h


# =============================================================================
# Match/mismatch dilation model
# =============================================================================

class DilationModel(nn.Module):
    """Accou et al. (2021) dilation model / ICASSP-2023 challenge baseline."""

    def __init__(self, n_ch: int = 64, spatial: int = 8, filters: int = 16,
                 layers: int = 3, kernel: int = 3):
        super().__init__()
        self.spatial = nn.Conv1d(n_ch, spatial, 1)
        self.eeg_convs = nn.ModuleList()
        self.env_convs = nn.ModuleList()
        ci_e, ci_s = spatial, 1
        for i in range(layers):
            self.eeg_convs.append(nn.Conv1d(ci_e, filters, kernel, dilation=kernel ** i))
            self.env_convs.append(nn.Conv1d(ci_s, filters, kernel, dilation=kernel ** i))
            ci_e = ci_s = filters
        self.readout = nn.Linear(filters * filters, 1)

    def embed_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, F, T') unit-norm over time."""
        x = self.spatial(eeg)
        for c in self.eeg_convs:
            x = F.relu(c(x))
        return F.normalize(x, dim=-1)

    def embed_env(self, env: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, F, T') unit-norm over time."""
        x = env[:, None]
        for c in self.env_convs:
            x = F.relu(c(x))
        return F.normalize(x, dim=-1)

    def score(self, e: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Embedded EEG (B,F,T') x embedded env (B,F,T') -> (B,) similarity score."""
        cos = torch.einsum("bft,bgt->bfg", e, s)
        return self.readout(cos.flatten(1))[:, 0]

    def forward(self, eeg: torch.Tensor, envs: torch.Tensor) -> torch.Tensor:
        """eeg (B,C,T), envs (B,K,T) -> logits (B,K)."""
        e = self.embed_eeg(eeg)
        B, K, T = envs.shape
        s = self.embed_env(envs.reshape(B * K, T)).reshape(B, K, *e.shape[1:])
        return torch.stack([self.score(e, s[:, k]) for k in range(K)], 1)


# =============================================================================
# Window sampling
# =============================================================================

def seg_windows(segments, trials, L: int, hop: int, lead: int = 0, lag: int = 0,
                need_neg: bool = False) -> list[tuple[int, int]]:
    """(trial, e0) starts of windows of length L inside segments.

    ``lead`` samples before e0 and ``lag`` after e0+L must also lie inside the
    segment (DECAF context / EEG delay).
    """
    out = []
    for s in segments:
        n = int(trials[s.trial]["eeg"].shape[1])
        if need_neg and n < 3 * L:
            continue
        lo, hi = s.start + lead, s.end - L - lag
        out.extend((s.trial, e0) for e0 in range(lo, hi + 1, max(1, hop)))
    return out


def neg_start(rng: np.random.Generator, n: int, st: int, L: int) -> int:
    """Random start of a length-L window of [0, n) not overlapping [st, st+L)."""
    while True:
        q = int(rng.integers(0, n - L + 1))
        if q + L <= st or q >= st + L:
            return q


def pearson_t(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Row-wise Pearson r over the last axis; eps inside the sqrt."""
    pc, tc = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
    den = (pc.pow(2).sum(-1) * tc.pow(2).sum(-1)).clamp_min(1e-16).sqrt()
    return (pc * tc).sum(-1) / den


# =============================================================================
# Model-specific pieces
# =============================================================================

class Recipe:
    """Model-specific batch construction, loss, validation and test prediction."""

    def __init__(self, name: str, trials: list[dict], device: str):
        self.name, self.trials, self.device = name, trials, device
        n_ch = int(trials[0]["eeg"].shape[0])      # 64, or 208 for MEG-MASC
        if name == "vlaai":
            from baselines.deep.vlaai.models_upstream import VLAAI
            self.model = VLAAI(nb_blocks=4, input_channels=n_ch, output_dim=1,
                               context_filter=n_ch)   # skip adds the input: widths must match (64 = paper)
            self.win, self.lead, self.delay = 5 * SR, 0, 0
            self.train_hop = int(round(0.2 * self.win))
            self.lr, self.patience, self.max_epochs, self.min_delta = 1e-3, 5, 100, 1e-4
        elif name == "decaf":
            from baselines.deep.decaf.models_upstream import (
                DECAF_MODEL_KWARGS, WARMUP_EPOCHS, PearsonL1Loss, build_decaf)
            self.model = build_decaf(prediction_seconds=5.0, in_channel=n_ch)
            self.win, self.lead, self.delay = 5 * SR, int(3.5 * SR), int(0.5 * SR)
            self.train_hop = int(round(0.2 * (self.win + self.lead)))
            self.lr, self.patience, self.max_epochs, self.min_delta = 3e-4, 3, 10, 0.0
            self.decaf_loss = PearsonL1Loss()
            self.d_model, self.warmup_epochs = DECAF_MODEL_KWARGS["d_model"], WARMUP_EPOCHS
        elif name == "mm":
            self.model = DilationModel(n_ch=n_ch)
            self.win, self.lead, self.delay = 5 * SR, 0, 0
            self.train_hop = int(round(0.2 * self.win))
            self.lr, self.patience, self.max_epochs, self.min_delta = 1e-3, 5, 100, 1e-4
        else:
            raise ValueError(name)
        self.model.to(device)

    # ---- batches -----------------------------------------------------------
    def _eeg(self, ti: int, a: int, b: int) -> torch.Tensor:
        """EEG (C, b-a) float, zero-padded outside the trial."""
        x = self.trials[ti]["eeg"]
        n = x.shape[1]
        out = torch.zeros(x.shape[0], b - a)
        lo, hi = max(a, 0), min(b, n)
        if hi > lo:
            out[:, lo - a:hi - a] = x[:, lo:hi].float()
        return out

    @staticmethod
    def _zs(x: torch.Tensor) -> torch.Tensor:          # (..., C, T) per channel
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _mm(x: torch.Tensor) -> torch.Tensor:          # (..., T) min-max
        mn, mx = x.min(-1, keepdim=True).values, x.max(-1, keepdim=True).values
        return (x - mn) / (mx - mn).clamp_min(1e-8)

    def batch_loss(self, items: list[tuple[int, int]], rng: np.random.Generator) -> torch.Tensor:
        dev, w = self.device, self.win
        if self.name == "vlaai":
            eeg = torch.stack([self._eeg(ti, e0, e0 + w) for ti, e0 in items]).to(dev)
            env = torch.stack([self.trials[ti]["env_att"][e0:e0 + w] for ti, e0 in items]).to(dev)
            pred = self.model(eeg.transpose(1, 2))[..., 0]
            return -pearson_t(pred, env).mean()
        if self.name == "decaf":
            d, c = self.delay, self.lead
            eeg = torch.stack([self._eeg(ti, e0 + d, e0 + d + w) for ti, e0 in items])
            ctx = torch.stack([self._eeg(ti, e0 - c + d, e0 + d) for ti, e0 in items])
            env = torch.stack([self.trials[ti]["env_att"][e0:e0 + w] for ti, e0 in items])
            envc = torch.stack([self.trials[ti]["env_att"][e0 - c:e0] for ti, e0 in items])
            eeg, ctx = self._zs(eeg).transpose(1, 2).to(dev), self._zs(ctx).transpose(1, 2).to(dev)
            env, envc = self._mm(env)[..., None].to(dev), self._mm(envc)[..., None].to(dev)
            pred = self.model(eeg, ctx, envelope_context_gt=envc,
                              teacher_forcing_ratio=1.0)["envelope"]
            return self.decaf_loss(pred, env)
        # mm: attended vs same-trial mismatch, random order
        eeg, envs, ys = [], [], []
        for ti, e0 in items:
            t = self.trials[ti]
            n = t["env_att"].shape[0]
            q = neg_start(rng, n, e0, w)
            pair = [t["env_att"][e0:e0 + w], t["env_att"][q:q + w]]
            y = int(rng.random() < 0.5)
            if y:
                pair = pair[::-1]
            eeg.append(self._eeg(ti, e0, e0 + w))
            envs.append(torch.stack(pair))
            ys.append(y)
        logits = self.model(torch.stack(eeg).to(dev), self._zs(torch.stack(envs)).to(dev))
        return F.cross_entropy(logits, torch.tensor(ys, device=dev))

    # ---- validation ----------------------------------------------------------
    @torch.no_grad()
    def val_metric(self, items: list[tuple[int, int]], negs: list[int | None]) -> float:
        """Mean Pearson r (recon models) or 2-way accuracy (mm) on val windows."""
        self.model.eval()
        vals = []
        for i in range(0, len(items), 128):
            chunk = items[i:i + 128]
            if self.name == "mm":
                w = self.win
                eeg = torch.stack([self._eeg(ti, e0, e0 + w) for ti, e0 in chunk])
                envs = []
                for (ti, e0), q in zip(chunk, negs[i:i + 128]):
                    t = self.trials[ti]
                    other = (t["env_unatt"][e0:e0 + w] if q is None else t["env_att"][q:q + w])
                    envs.append(torch.stack([t["env_att"][e0:e0 + w], other]))
                lg = self.model(eeg.to(self.device), self._zs(torch.stack(envs)).to(self.device))
                vals.extend((lg[:, 0] > lg[:, 1]).float().tolist())
            else:
                env = torch.stack([self.trials[ti]["env_att"][e0:e0 + self.win]
                                   for ti, e0 in chunk]).to(self.device)
                pred = self.predict_windows(chunk)
                vals.extend(pearson_t(pred, env).tolist())
        self.model.train()
        return float(np.nanmean(vals))

    def predict_windows(self, items: list[tuple[int, int]], eeg_branch: bool = False
                        ) -> torch.Tensor:
        """Envelope predictions (B, win) for envelope windows starting at e0."""
        w, d, c = self.win, self.delay, self.lead
        if self.name == "vlaai":
            eeg = torch.stack([self._eeg(ti, e0, e0 + w) for ti, e0 in items]).to(self.device)
            return self.model(eeg.transpose(1, 2))[..., 0]
        eeg = self._zs(torch.stack([self._eeg(ti, e0 + d, e0 + d + w) for ti, e0 in items]))
        eeg = eeg.transpose(1, 2).to(self.device)
        if eeg_branch:
            return self.model.eeg_branch(eeg)["envelope"][..., 0]
        ctx = self._zs(torch.stack([self._eeg(ti, e0 - c + d, e0 + d) for ti, e0 in items]))
        return self.model(eeg, ctx.transpose(1, 2).to(self.device))["envelope"][..., 0]

    # ---- test ----------------------------------------------------------------
    @torch.no_grad()
    def predict_trial(self, ti: int, eeg_branch: bool = False) -> np.ndarray:
        """Envelope reconstruction over a whole test trial."""
        self.model.eval()
        n = int(self.trials[ti]["eeg"].shape[1])
        if self.name == "vlaai":
            # Fully convolutional: 60-s chunks with 10 s of context on each side.
            ch, pad = 60 * SR, 10 * SR
            out = np.zeros(n)
            for a in range(0, n, ch):
                b = min(a + ch, n)
                lo, hi = max(0, a - pad), min(n, b + pad)
                x = self.trials[ti]["eeg"][:, lo:hi].float()[None].transpose(1, 2)
                p = self.model(x.to(self.device))[0, :, 0].cpu().numpy()
                out[a:b] = p[a - lo:a - lo + (b - a)]
            return out
        # DECAF: tile consecutive present windows (context / EEG zero-padded at edges).
        starts = list(range(0, n, self.win))
        preds = []
        for i in range(0, len(starts), 64):
            it = [(ti, e0) for e0 in starts[i:i + 64]]
            preds.append(self.predict_windows(it, eeg_branch).cpu().numpy())
        return np.concatenate(preds, 0).reshape(-1)[:n].astype(np.float64)

    @torch.no_grad()
    def score_mm(self, fplan: list[dict]) -> dict:
        """Hit counts for the match/mismatch model."""
        self.model.eval()
        h = empty_hits()
        dev = self.device
        for item in fplan:
            t = self.trials[item["trial"]]
            eeg = t["eeg"].float()
            a, u = t["env_att"], (t["env_unatt"] if t["has_unattended"] else None)

            def sc(e_list, s_list):
                e = self.model.embed_eeg(torch.stack(e_list).to(dev))
                s = self.model.embed_env(self._zs(torch.stack(s_list)).to(dev))
                return self.model.score(e, s).cpu().numpy()
            if u is not None:
                sa, su = sc([eeg], [a])[0], sc([eeg], [u])[0]
                h["trial_competing"][0] += int(sa > su)
                h["trial_competing"][1] += 1
            for W, lst in item["wins"].items():
                win = int(W * SR)
                hw = h["windows"][f"{W:g}s"]
                for i in range(0, len(lst), 256):
                    part = lst[i:i + 256]
                    ee = [eeg[:, c0:c0 + win] for c0, _ in part]
                    sa = sc(ee, [a[c0:c0 + win] for c0, _ in part])
                    if u is not None:
                        su = sc(ee, [u[c0:c0 + win] for c0, _ in part])
                        hw["competing"][0] += int((sa > su).sum())
                        hw["competing"][1] += len(part)
                    idx = [j for j, (_, q0) in enumerate(part) if q0 is not None]
                    if idx:
                        sq = sc([ee[j] for j in idx], [a[part[j][1]:part[j][1] + win] for j in idx])
                        hw["same_talker"][0] += int((sa[idx] > sq).sum())
                        hw["same_talker"][1] += len(idx)
        return h


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vlaai", "decaf", "mm"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-minutes", type=float, default=95.0, help="training time budget")
    ap.add_argument("--steps-per-epoch", type=int, default=1500)
    ap.add_argument("--max-val", type=int, default=3000, help="cap on val windows")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--groups-name", default="unified_groups.json",
                    help="content groups (a trial-level file gives the leaky control split)")
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = Path(args.out_root) / args.model / f"{args.dataset}_fold{args.fold}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = load_dataset(str(CACHE_ROOT), args.dataset, stream_name=stream(args.dataset),
                        groups_name=args.groups_name)
    trials = data["trials"]
    n_folds = len(make_folds(data["groups"], trials, N_FOLDS_REQ, 42))
    if args.fold >= n_folds:
        print(f"fold {args.fold} >= n_folds {n_folds}; nothing to do")
        return
    fplan = window_plan(data, args.fold + 1)[args.fold]
    skw = split_kw(args.dataset)
    sp = split_fold(data, args.fold, N_FOLDS_REQ, MARGIN, **skw)
    test_trials = {s.trial for s in sp["test"]}
    assert {it["trial"] for it in fplan} <= test_trials
    print(f"{args.model} {args.dataset} fold {args.fold}/{n_folds}: train {len(sp['train'])} "
          f"val {len(sp['val'])} ({sp['val_mode']}) test {len(sp['test'])} segs, "
          f"{len(fplan)} scored test trials, load {time.time() - t_start:.0f}s", flush=True)

    R = Recipe(args.model, trials, args.device)
    need_neg = args.model == "mm"
    tr_items = seg_windows(sp["train"], trials, R.win, R.train_hop, R.lead, R.delay, need_neg)
    va_items = seg_windows(sp["val"], trials, R.win, R.win, R.lead, R.delay)
    if len(va_items) > args.max_val:
        sel = np.random.default_rng(123).choice(len(va_items), args.max_val, replace=False)
        va_items = [va_items[i] for i in sorted(sel)]
    vrng = np.random.default_rng(456)
    va_negs: list[int | None] = []
    for ti, e0 in va_items:                    # mm val: competing talker, else same-trial
        t = trials[ti]
        n = t["env_att"].shape[0]
        if t["has_unattended"]:
            va_negs.append(None)
        else:
            va_negs.append(neg_start(vrng, n, e0, R.win) if n >= 3 * R.win else -1)
    if need_neg:
        keep = [i for i, q in enumerate(va_negs) if q != -1]
        va_items, va_negs = [va_items[i] for i in keep], [va_negs[i] for i in keep]
    print(f"train windows {len(tr_items)}, val windows {len(va_items)}", flush=True)

    bs = args.batch_size
    spe = max(1, min(args.steps_per_epoch, math.ceil(len(tr_items) / bs)))
    max_epochs = args.max_epochs or R.max_epochs
    opt = torch.optim.Adam(R.model.parameters(), lr=R.lr)
    sched = None
    if args.model == "decaf":
        from baselines.deep.decaf.models_upstream import NoamSchedule
        sched = NoamSchedule(opt, d_model=R.d_model, warmup_steps=R.warmup_epochs * spe)
    ckpt = out.with_suffix(".pt")
    best, best_ep, bad, hist = -1e9, -1, 0, []
    best_val0 = R.val_metric(va_items, va_negs)
    hist.append({"epoch": 0, "val": best_val0})
    best, best_ep = best_val0, 0
    torch.save(R.model.state_dict(), ckpt)
    print(f"epoch 0 val {best_val0:.4f}", flush=True)
    t_train = time.time()
    for ep in range(1, max_epochs + 1):
        R.model.train()
        perm = rng.permutation(len(tr_items))
        losses = []
        for step in range(spe):
            idx = perm[(step * bs) % len(perm):(step * bs) % len(perm) + bs]
            if len(idx) < 2:
                idx = rng.choice(len(tr_items), bs)
            loss = R.batch_loss([tr_items[i] for i in idx], rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            losses.append(loss.item())
        v = R.val_metric(va_items, va_negs)
        hist.append({"epoch": ep, "loss": float(np.nanmean(losses)), "val": v})
        print(f"epoch {ep} loss {np.nanmean(losses):.4f} val {v:.4f} "
              f"({(time.time() - t_train) / 60:.1f} min)", flush=True)
        if v > best + R.min_delta:
            best, best_ep, bad = v, ep, 0
            torch.save(R.model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= R.patience:
                break
        if (time.time() - t_train) / 60 > args.max_minutes:
            print("time budget reached", flush=True)
            break
    R.model.load_state_dict(torch.load(ckpt, map_location=args.device))
    info = {"model": args.model, "dataset": args.dataset, "fold": args.fold,
            "n_folds": n_folds, "best_epoch": best_ep, "best_val": best, "history": hist,
            "n_train_windows": len(tr_items), "n_val_windows": len(va_items),
            "val_mode": sp["val_mode"], "train_minutes": (time.time() - t_train) / 60}
    res = {}
    if args.model == "mm":
        res["mm"] = R.score_mm(fplan)
    else:
        variants = {args.model: False} | ({"decaf_eeg": True} if args.model == "decaf" else {})
        for name, eb in variants.items():
            preds = {it["trial"]: R.predict_trial(it["trial"], eb) for it in fplan}
            res[name] = score_recon(trials, fplan, preds)
            res[name]["test_r_att"] = float(np.mean([
                rcorr(preds[i], trials[i]["env_att"].numpy().astype(np.float64)) for i in preds]))
    info["results"] = res
    out.write_text(json.dumps(info, indent=1))
    ckpt.unlink(missing_ok=True)
    print(json.dumps({k: {W: v["windows"][W] for W in ("1s", "5s", "60s")} | {
        "trial": v["trial_competing"]} for k, v in res.items()}, indent=1))
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
