"""Shared MAESTRO data, split and 4-way scoring for the E6 baselines (rebuttal analysis).

Protocol = CoupleFlow's Table 4 runs (scripts/train/maestro.sh):
* candidates: the 4 attendable talkers (``data.cand_idx=[0,1,2,3]``), attended first;
  ``cands_matched`` = quantile-matched (``coupleflow.data.candidate_features``, the very
  features CoupleFlow scores), ``cands`` = as released;
* timing: corrected = ``unified_env64.pt`` + ``unified_groups.json``; release =
  ``unified_pkg.pt`` + ``unified_groups_pkg.json``;
* folds: ``split_fold`` with ``run_config.split_kw("maestro")``, 5 content folds (within
  listeners) or ``mode="loso"`` (16 held-out listeners x held-out content), margin 100;
* windows: ``eval_windows`` (hop = window, 100-sample context discarded on both sides);
  per-trial decision = sum of the trial's 5-s window scores (as ``trial_acc_coupling``),
  plus one whole-trial decision (~30 s, the benchmark's 30-s windows);
* null: ``across_trial_perm`` (each window scored against another trial's EEG of the same
  listener), as CoupleFlow's EEG-shuffle null.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from coupleflow import paths  # noqa: E402
from coupleflow.data import (Segment, candidate_features, eval_windows, load_dataset,  # noqa: E402
                             split_fold)
from coupleflow.evaluate import across_trial_perm  # noqa: E402
from coupleflow.preprocessing.run_config import split_kw  # noqa: E402

SR = 64
MARGIN = 100
CAND_IDX = [0, 1, 2, 3]
WINDOWS = (1.0, 2.0, 5.0, 10.0, 20.0)
SETS = ("cands_matched", "cands")
TIMING = {"corrected": ("unified_env64.pt", "unified_groups.json"),
          "release": ("unified_pkg.pt", "unified_groups_pkg.json")}
OUT_ROOT = Path("/fs/scratch/PAS2301/alialavi/rebuttal/e6")


def load(timing: str) -> dict:
    stream, groups = TIMING[timing]
    data = load_dataset(str(paths.DATA), "maestro", stream_name=stream, groups_name=groups)
    data["cand_idx"] = list(CAND_IDX)
    for t in data["trials"]:
        assert int(t["att_idx"]) in CAND_IDX, t["trial"]
    return data


def split(data: dict, protocol: str, fold: int) -> dict:
    kw = split_kw("maestro")
    if protocol == "loso":
        return split_fold(data, fold, 5, MARGIN, mode="loso", **kw)
    return split_fold(data, fold, 5, MARGIN, **kw)


def n_folds(data: dict, protocol: str) -> int:
    return len(data["subjects"]) if protocol == "loso" else 5


def order(t: dict) -> list[int]:
    a = int(t["att_idx"])
    return [a] + [i for i in CAND_IDX if i != a]


def cand_env(t: dict, st: int, L: int, margin: int, matched: bool) -> np.ndarray:
    """(K, L - 2*margin) candidate envelope channel exactly as CoupleFlow scores it."""
    envs = torch.stack([t["env_cands"][i, st:st + L].float() for i in order(t)])
    f = candidate_features(envs, margin, matched=matched)[:, 0]
    return f[:, margin:L - margin].numpy().astype(np.float64)


def pearson(p: np.ndarray, c: np.ndarray) -> np.ndarray:
    """r between p (T,) and each row of c (K, T)."""
    pc = p - p.mean()
    cc = c - c.mean(-1, keepdims=True)
    return (cc @ pc) / np.sqrt(np.maximum((cc ** 2).sum(-1) * (pc ** 2).sum(), 1e-16))


def acc(scores: np.ndarray) -> np.ndarray:
    """Correct when candidate 0 strictly beats every other (ties count as wrong)."""
    return (scores[:, 0] > scores[:, 1:].max(-1)).astype(float)


class Collector:
    """Collect per-window scores of a recon model (full-trial prediction p per trial)."""

    def __init__(self):
        self.rows = {(W, s): [] for W in WINDOWS for s in SETS}   # (subj, trial, p_c, cands)
        self.whole = {s: [] for s in SETS}

    def add_recon(self, data: dict, ti: int, p: np.ndarray, seg_start: int = 0,
                  seg_end: int | None = None) -> None:
        t = data["trials"][ti]
        n = t["eeg"].shape[1]
        seg_end = n if seg_end is None else seg_end
        sidx = data["subjects"].index(t["subject"])
        seg = [Segment(ti, seg_start, seg_end)]
        for W in WINDOWS:
            win = int(round(W * SR))
            for w in eval_windows(data, seg, win, MARGIN):
                st = w["start"]
                pc = p[st + MARGIN:st + MARGIN + win].astype(np.float64)
                for s in SETS:
                    c = cand_env(t, st, win + 2 * MARGIN, MARGIN, s == "cands_matched")
                    self.rows[(W, s)].append((sidx, ti, pc, c))
        for s in SETS:
            c = cand_env(t, seg_start, seg_end - seg_start, 0, s == "cands_matched")
            self.whole[s].append(pearson(p[seg_start:seg_end].astype(np.float64), c))

    def summary(self, n_perm: int = 20, seed: int = 0) -> dict:
        out = {"windows": {}, "trial_sum5": {}, "trial_whole": {}}
        for (W, s), rows in self.rows.items():
            if not rows:
                continue
            subj = np.array([r[0] for r in rows])
            trial = np.array([r[1] for r in rows])
            sc = np.stack([pearson(r[2], r[3]) for r in rows])
            rng = np.random.default_rng(seed + 17)
            nulls, tnulls = [], []
            for _ in range(n_perm):
                pm = across_trial_perm(subj, trial, rng)
                scn = np.stack([pearson(rows[j][2], r[3]) for j, r in zip(pm, rows)])
                nulls.append(acc(scn).mean())
                if W == 5.0:
                    tnulls.append(trial_acc(scn, trial)[0])
            cell = {"acc": float(acc(sc).mean()), "n_windows": len(rows),
                    "null": float(np.mean(nulls))}
            out["windows"].setdefault(f"{W:g}s", {})[s] = cell
            if W == 5.0:
                ta, nt = trial_acc(sc, trial)
                out["trial_sum5"][s] = {"acc": ta, "n_trials": nt, "null": float(np.mean(tnulls))}
        for s, lst in self.whole.items():
            if lst:
                out["trial_whole"][s] = {"acc": float(acc(np.stack(lst)).mean()), "n_trials": len(lst)}
        return out


def trial_acc(scores: np.ndarray, trial: np.ndarray) -> tuple[float, int]:
    """Sum window scores per trial, decide once per trial."""
    tot = np.stack([scores[trial == t].sum(0) for t in np.unique(trial)])
    return float(acc(tot).mean()), len(tot)
