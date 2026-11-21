#!/usr/bin/env python
"""E6 (rebuttal): VLAAI and dilated match-mismatch on MAESTRO, release vs corrected timing.

Training is ``baselines/deep/run.py``'s recipe unchanged (its ``Recipe`` class and training
loop: one subject-independent model per fold, trained on every training listener,
early-stopped on the fold's validation trials).  Only the data and the scoring differ:
CoupleFlow's MAESTRO folds (content-disjoint within listeners, or LOSO) and 4-way candidate
sets, scored as in ``common.py`` (5-s windows, sum-of-windows and whole-trial decisions,
across-trial EEG null).  The match-mismatch model scores each of the 4 candidates with its
own similarity head (trained 2-way, as in the paper) and picks the highest.

    python scripts/rebuttal/e6/deep_maestro.py --model vlaai --timing corrected --protocol within --fold 0
Writes /fs/scratch/PAS2301/alialavi/rebuttal/e6/deep/<model>/<timing>_<protocol>_fold<k>[_s<seed>].json.
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (MARGIN, OUT_ROOT, SETS, SR, WINDOWS, Collector, acc, cand_env,  # noqa: E402
                    load, n_folds, split, trial_acc)
from coupleflow.data import Segment, eval_windows  # noqa: E402
from coupleflow.evaluate import across_trial_perm  # noqa: E402
from baselines.deep.run import Recipe, neg_start, seg_windows  # noqa: E402


class MMCollector:
    """Per-window EEG / candidate embeddings of the match-mismatch model."""

    def __init__(self, R: Recipe):
        self.R = R
        self.rows = {(W, s): [] for W in WINDOWS for s in SETS}
        self.whole = {s: [] for s in SETS}

    @torch.no_grad()
    def add(self, data: dict, ti: int, seg_start: int, seg_end: int) -> None:
        m, dev = self.R.model, self.R.device
        t = data["trials"][ti]
        sidx = data["subjects"].index(t["subject"])
        eeg = t["eeg"].float()
        for W in WINDOWS:
            win = int(round(W * SR))
            wins = eval_windows(data, [Segment(ti, seg_start, seg_end)], win, MARGIN)
            if not wins:
                continue
            ee = torch.stack([eeg[:, w["start"] + MARGIN:w["start"] + MARGIN + win] for w in wins])
            e = m.embed_eeg(ee.to(dev)).cpu()
            for s in SETS:
                c = np.stack([cand_env(t, w["start"], win + 2 * MARGIN, MARGIN, s == "cands_matched")
                              for w in wins])                      # (B, K, win)
                B, K, T = c.shape
                se = m.embed_env(self.R._zs(torch.from_numpy(c).float().reshape(B * K, T)).to(dev))
                se = se.reshape(B, K, *se.shape[1:]).cpu()
                for b in range(B):
                    self.rows[(W, s)].append((sidx, ti, e[b], se[b]))
        e = m.embed_eeg(eeg[None, :, seg_start:seg_end].to(dev))
        for s in SETS:
            c = torch.from_numpy(cand_env(t, seg_start, seg_end - seg_start, 0,
                                          s == "cands_matched")).float()
            se = m.embed_env(self.R._zs(c).to(dev))
            self.whole[s].append(np.array([float(m.score(e, se[k:k + 1])[0])
                                           for k in range(se.shape[0])]))

    @torch.no_grad()
    def _scores(self, E: torch.Tensor, S: torch.Tensor) -> np.ndarray:
        m, dev = self.R.model, self.R.device
        out = []
        for i in range(0, len(E), 512):
            e, s = E[i:i + 512].to(dev), S[i:i + 512].to(dev)
            out.append(torch.stack([m.score(e, s[:, k]) for k in range(s.shape[1])], 1).cpu())
        return torch.cat(out).numpy()

    def summary(self, n_perm: int = 20, seed: int = 0) -> dict:
        out = {"windows": {}, "trial_sum5": {}, "trial_whole": {}}
        for (W, s), rows in self.rows.items():
            if not rows:
                continue
            subj = np.array([r[0] for r in rows])
            trial = np.array([r[1] for r in rows])
            E = torch.stack([r[2] for r in rows])
            S = torch.stack([r[3] for r in rows])
            sc = self._scores(E, S)
            rng = np.random.default_rng(seed + 17)
            nulls, tnulls = [], []
            for _ in range(n_perm):
                pm = torch.from_numpy(across_trial_perm(subj, trial, rng))
                scn = self._scores(E[pm], S)
                nulls.append(acc(scn).mean())
                if W == 5.0:
                    tnulls.append(trial_acc(scn, trial)[0])
            out["windows"].setdefault(f"{W:g}s", {})[s] = {
                "acc": float(acc(sc).mean()), "n_windows": len(rows), "null": float(np.mean(nulls))}
            if W == 5.0:
                ta, nt = trial_acc(sc, trial)
                out["trial_sum5"][s] = {"acc": ta, "n_trials": nt, "null": float(np.mean(tnulls))}
        for s, lst in self.whole.items():
            if lst:
                out["trial_whole"][s] = {"acc": float(acc(np.stack(lst)).mean()), "n_trials": len(lst)}
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vlaai", "mm"])
    ap.add_argument("--timing", choices=["corrected", "release"], required=True)
    ap.add_argument("--protocol", choices=["within", "loso"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-minutes", type=float, default=95.0)     # run.py defaults
    ap.add_argument("--steps-per-epoch", type=int, default=1500)
    ap.add_argument("--max-val", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 20 steps, 4 test trials")
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = OUT_ROOT / "deep" / args.model / (
        f"{args.timing}_{args.protocol}_fold{args.fold}{f'_s{args.seed}' if args.seed else ''}"
        f"{'_smoke' if args.smoke else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.smoke:
        print(f"{out} exists")
        return
    data = load(args.timing)
    trials = data["trials"]
    if args.fold >= n_folds(data, args.protocol):
        print("fold out of range")
        return
    sp = split(data, args.protocol, args.fold)
    print(f"{args.model} {args.timing} {args.protocol} fold {args.fold}: train {len(sp['train'])} "
          f"val {len(sp['val'])} ({sp['val_mode']}) test {len(sp['test'])}", flush=True)

    # ---- training: baselines/deep/run.py main(), unchanged ----------------------------
    R = Recipe(args.model, trials, args.device)
    need_neg = args.model == "mm"
    tr_items = seg_windows(sp["train"], trials, R.win, R.train_hop, R.lead, R.delay, need_neg)
    va_items = seg_windows(sp["val"], trials, R.win, R.win, R.lead, R.delay)
    if len(va_items) > args.max_val:
        sel = np.random.default_rng(123).choice(len(va_items), args.max_val, replace=False)
        va_items = [va_items[i] for i in sorted(sel)]
    vrng = np.random.default_rng(456)
    va_negs: list[int | None] = []
    for ti, e0 in va_items:
        t = trials[ti]
        n = t["env_att"].shape[0]
        if t["has_unattended"]:
            va_negs.append(None)
        else:
            va_negs.append(neg_start(vrng, n, e0, R.win) if n >= 3 * R.win else -1)
    if need_neg:
        keep = [i for i, q in enumerate(va_negs) if q != -1]
        va_items, va_negs = [va_items[i] for i in keep], [va_negs[i] for i in keep]
    bs = args.batch_size
    spe = max(1, min(args.steps_per_epoch, math.ceil(len(tr_items) / bs)))
    max_epochs = args.max_epochs or R.max_epochs
    if args.smoke:
        spe, max_epochs = 20, 2
    print(f"train windows {len(tr_items)}, val windows {len(va_items)}, {spe} steps/epoch", flush=True)
    opt = torch.optim.Adam(R.model.parameters(), lr=R.lr)
    ckpt = out.with_suffix(".pt")
    best = R.val_metric(va_items, va_negs)
    best_ep, bad, hist = 0, 0, [{"epoch": 0, "val": best}]
    torch.save(R.model.state_dict(), ckpt)
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
    R.model.eval()

    # ---- 4-way scoring on the fold's test trials ----------------------------------------
    test = sp["test"][:4] if args.smoke else sp["test"]
    if args.model == "mm":
        col = MMCollector(R)
        for s in test:
            col.add(data, s.trial, s.start, s.end)
    else:
        col = Collector()
        for s in test:
            p = R.predict_trial(s.trial)
            col.add_recon(data, s.trial, np.asarray(p), s.start, s.end)
    res = {"model": args.model, "timing": args.timing, "protocol": args.protocol,
           "fold": args.fold, "best_epoch": best_ep, "best_val": best, "history": hist,
           "n_train_windows": len(tr_items), "n_val_windows": len(va_items),
           "val_mode": sp["val_mode"], "test_subject": sp.get("test_subject"),
           "train_minutes": (time.time() - t_train) / 60, **col.summary()}
    out.write_text(json.dumps(res, indent=1))
    ckpt.unlink(missing_ok=True)
    print(json.dumps({k: res[k] for k in ("windows", "trial_sum5", "trial_whole")}, indent=1)[:3000])
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
