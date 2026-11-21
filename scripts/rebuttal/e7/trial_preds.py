#!/usr/bin/env python
"""Per-trial CoupleFlow decisions on MAESTRO (rebuttal E7, gaze/head analysis).

The stored results give per-trial window accuracy and summed window margins, but not the
per-trial 4-way decision (which needs every candidate's summed logits).  This script reloads
each run's best.pt and its stored config.yaml, rebuilds the test windows exactly as
coupleflow.evaluate.evaluate_set does (same candidate set, window length and seed), and writes
one row per test trial:

    tag, fold, subject, trial, n_windows, correct (summed-logit 4-way decision),
    trial_margin (summed attended logit - max summed other), win_acc (fraction of correct
    5-s windows)

It checks the pooled per-trial accuracy against the stored trial_acc_coupling.

    python scripts/rebuttal/e7/trial_preds.py --tags maestro_within_lag0 maestro_within_lag0_s1 \
        maestro_within_lag0_s2 --out /fs/scratch/PAS2301/alialavi/rebuttal/e7/trial_preds.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from coupleflow import paths  # noqa: E402
from coupleflow.data import build_eval_batch, eval_windows, load_dataset, split_fold  # noqa: E402
from coupleflow.evaluate import encode, score  # noqa: E402
from coupleflow.trainer import build_model, feature_banks, streams_for  # noqa: E402

SR = 64


def run_dir_results(d: str) -> dict:
    for f in ("results.json", "results_val.json"):
        if os.path.exists(f"{d}/{f}"):
            return json.load(open(f"{d}/{f}"))["results"]
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--cand-set", default="cands_matched")
    ap.add_argument("--win", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rows, checks = [], []
    data_cache: dict = {}
    for tag in a.tags:
        for d in sorted(glob.glob(f"{paths.RUNS}/{tag}/maestro_fold*")):
            cfg = OmegaConf.load(f"{d}/config.yaml")
            ds = str(cfg.training.dataset)
            sn, gn = streams_for(cfg, ds)
            key = (sn, gn, bool(cfg.data.drop_frontal))
            if key not in data_cache:
                data_cache.clear()
                data_cache[key] = load_dataset(str(paths.DATA), ds,
                                               drop_frontal=bool(cfg.data.drop_frontal),
                                               stream_name=sn, groups_name=gn)
            data = dict(data_cache[key])
            data["cand_idx"] = list(cfg.data.get("cand_idx") or [])
            ck = torch.load(f"{d}/best.pt", map_location="cpu", weights_only=False)
            assert data["subjects"] == ck["subjects"]
            sp = split_fold(data, int(cfg.fold), int(cfg.n_folds), int(cfg.model.margin),
                            val_mode=str(cfg.data.val_mode), val_frac=float(cfg.data.val_frac),
                            tail_min_s=float(cfg.data.get("tail_min_s", 200.0)),
                            mode=str(cfg.data.get("split_mode", "stimulus")),
                            seed=int(cfg.get("split_seed", 42)))
            bank_dims = [int(data["trials"][0][f"env_att_{bk}"].shape[0])
                         for bk in feature_banks(cfg)]
            model = build_model(cfg, len(data["subjects"]), bank_dims,
                                n_channels=int(data["trials"][0]["eeg"].shape[0]))
            model.load_state_dict(ck["model"])
            model.eval()
            win = int(round(a.win * SR))
            margin = model.margin
            wins = eval_windows(data, sp["test"], win, margin)
            batch = build_eval_batch(data, wins, win, margin, a.cand_set, seed=int(cfg.seed),
                                     extra_feats=getattr(model, "extra_feats", []),
                                     glimpse=bool(getattr(model, "glimpse", False)),
                                     banks=list(getattr(model, "banks", [])),
                                     n_glimpse=int(getattr(model, "n_glimpse", 0)))
            with torch.no_grad():
                enc = encode(model, batch, "cpu")
                real = score(model, enc["e"], enc["a"], enc["side_logits"], batch["side"])
            L = real["coupling"].numpy()                      # (N, K), attended = column 0
            trial = np.asarray(batch["trial"])
            wc = L[:, 0] > L[:, 1:].max(1)
            for ti in np.unique(trial):
                m = trial == ti
                s = L[m].sum(0)
                t = data["trials"][int(ti)]
                rows.append({"tag": tag, "fold": int(cfg.fold), "subject": t["subject"],
                             "trial": t["trial"], "n_windows": int(m.sum()),
                             "correct": int(s[0] > s[1:].max()),
                             "trial_margin": float(s[0] - s[1:].max()),
                             "win_acc": float(wc[m].mean())})
            r = run_dir_results(d).get(f"{a.cand_set}_{a.win:g}s", {})
            got = np.mean([x["correct"] for x in rows if x["tag"] == tag and x["fold"] == int(cfg.fold)])
            checks.append((tag, int(cfg.fold), r.get("trial_acc_coupling"), float(got),
                           r.get("acc_coupling"), float(wc.mean())))
            print(f"{tag} fold {cfg.fold}: trial acc stored {r.get('trial_acc_coupling')} "
                  f"recomputed {got:.4f} | window acc stored {r.get('acc_coupling')} "
                  f"recomputed {wc.mean():.4f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.DataFrame(checks, columns=["tag", "fold", "trial_stored", "trial_recomputed",
                                  "win_stored", "win_recomputed"]).to_csv(
        a.out.replace(".csv", "_check.csv"), index=False)
    print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
