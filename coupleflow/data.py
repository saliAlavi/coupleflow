"""Data path for CoupleFlow: trial streams, stimulus-disjoint folds, windows.

Everything works on the continuous 64-Hz trial streams built by
``paper_review/build_env64_cache.py`` (EEG z-scored per trial and channel,
VLAAI gammatone envelopes for both talkers) and on the stimulus groups
written by ``coupleflow/preprocessing/stimulus_groups.py``.

Split protocol (within-listener, stimulus-disjoint)
---------------------------------------------------
Folds are built over *stimulus groups*, not trials: a group is every trial,
of every subject, that shares any audio content.  A test fold therefore
contains stimuli that no subject heard during training.  Every listener
appears in train and test (one pooled model per fold, with a per-listener
correction).

Inner validation must also be content-disjoint from training:

* ``groups`` -- hold out a fraction of the training groups (datasets with
  many groups: DTU, NJU).
* ``tail`` -- hold out the tail of every long training trial (KU Leuven has
  only four story pairs per subject; its 2-min repeat trials replay the
  *first* two minutes of the 6-min trials, so the tail of a 6-min trial is
  heard nowhere else except in its ear-swapped twin, whose tail is held out
  as well).

Windows
-------
Every item carries ``margin`` samples of context on both sides so that the
EEG encoder can look at EEG *after* (or, for the control, *before*) the
sound without padding.  Candidate features are the envelope and its
half-wave-rectified derivative (speech onsets), z-scored over the central
window.  Candidates can be *matched*: quantile-matched so that every
candidate has exactly the same set of values and no amplitude statistic
(loudness, skew, sparsity, ...) can identify the attended one.
"""

from __future__ import annotations

import collections
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from coupleflow.envelope_common import load_streams

# Channels most exposed to ocular activity (standard 64-ch order, data/common.py):
# Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F7 F8.
FRONTAL_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 16]
SR = 64


@dataclass
class Segment:
    """A contiguous piece ``[start, end)`` of one trial assigned to a split."""

    trial: int
    start: int
    end: int


# =============================================================================
# Loading and folds
# =============================================================================

def load_dataset(cache_root: str, dataset: str, drop_frontal: bool = False,
                 stream_name: str | None = None,
                 groups_name: str = "stimulus_groups.json") -> dict:
    """Load trial streams + stimulus groups.

    Returns:
        ``{"trials": [...], "subjects": [sorted ids], "groups": [gid per trial]}``.
        Each trial dict holds ``eeg (C,T)``, ``env_att (T,)``, ``env_unatt (T,)``,
        ``subject``, ``trial``, ``attended_ear``, ``channel_mask``.
    """
    bundle = load_streams(cache_root, dataset, stream_name=stream_name)
    trials = bundle["trials"]
    gpath = Path(cache_root) / dataset / groups_name
    if not gpath.exists():
        raise FileNotFoundError(
            f"{gpath} missing -- run: python coupleflow/preprocessing/stimulus_groups.py --dataset {dataset}")
    gmap = json.loads(gpath.read_text())["groups"]
    groups = [gmap[f"{t['subject']}|{t['trial']}"] for t in trials]
    if drop_frontal:
        for t in trials:
            t["eeg"] = t["eeg"].clone()
            t["eeg"][FRONTAL_IDX] = 0.0
            t["channel_mask"] = t["channel_mask"].clone()
            t["channel_mask"][FRONTAL_IDX] = False
    subjects = sorted({t["subject"] for t in trials})
    return {"trials": trials, "subjects": subjects, "groups": groups}


def make_folds(groups: list[int], trials: list[dict], n_folds: int,
               seed: int = 42) -> list[list[int]]:
    """Deal stimulus groups into ``n_folds`` folds, balancing trial counts.

    Returns:
        ``folds[f]`` = list of *group ids* in test fold ``f``.
    """
    size = collections.Counter(groups)
    gids = sorted(size)
    random.Random(seed).shuffle(gids)
    n_folds = min(n_folds, len(gids))
    # Greedy: largest group first into the currently lightest fold.
    gids.sort(key=lambda g: -size[g])
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    load = [0] * n_folds
    for g in gids:
        f = int(np.argmin(load))
        folds[f].append(g)
        load[f] += size[g]
    return folds


def split_fold(data: dict, fold: int, n_folds: int, margin: int,
               val_mode: str = "auto", val_frac: float = 0.15,
               tail_min_s: float = 200.0, seed: int = 42, mode: str = "stimulus") -> dict:
    """Build train / val / test segments for one fold.

    Returns:
        ``{"train": [Segment], "val": [Segment], "test": [Segment],
        "test_groups": [...], "n_folds": int, "val_mode": str}``.
    """
    trials, groups = data["trials"], data["groups"]
    if mode == "loso":
        return _split_loso(data, fold, n_folds, margin, val_frac, seed, tail_min_s)
    folds = make_folds(groups, trials, n_folds, seed)
    test_g = set(folds[fold])
    train_g = sorted({g for g in groups if g not in test_g})
    if val_mode == "auto":
        val_mode = "groups" if len(train_g) >= 10 else "tail"

    val_g: set[int] = set()
    if val_mode == "groups":
        rng = random.Random(seed + 1000 + fold)
        pool = list(train_g)
        rng.shuffle(pool)
        size = collections.Counter(groups)
        target = val_frac * sum(size[g] for g in train_g)
        acc = 0
        for g in pool:
            if acc >= target:
                break
            val_g.add(g)
            acc += size[g]

    out = {"train": [], "val": [], "test": []}
    for ti, (t, g) in enumerate(zip(trials, groups)):
        n = t["eeg"].shape[1]
        if g in test_g:
            out["test"].append(Segment(ti, 0, n))
        elif g in val_g:
            out["val"].append(Segment(ti, 0, n))
        elif val_mode == "tail" and n / SR >= tail_min_s:
            cut = int(n * (1 - val_frac))
            # A gap of one margin keeps windows on either side from sharing samples.
            out["train"].append(Segment(ti, 0, cut - margin))
            out["val"].append(Segment(ti, cut + margin, n))
        else:
            out["train"].append(Segment(ti, 0, n))
    out.update(test_groups=sorted(test_g), n_folds=len(folds), val_mode=val_mode)
    return out


def _split_loso(data: dict, fold: int, n_content: int, margin: int, val_frac: float,
                seed: int, tail_min_s: float = 200.0) -> dict:
    """Leave-one-listener-out with held-out content (as in MERIT on MAESTRO).

    ``fold`` indexes the held-out listener. The content groups are dealt into
    ``n_content`` folds (``make_folds``); content fold ``fold % n_content`` is held
    out -- or, if the listener heard none of it (listeners of SparrKULee hear
    different stories), the next content fold that contains the listener's trials.
    Test = held-out listener x held-out content. Train = other listeners x other
    content. Nothing of the test listener and none of the test audio is seen in
    training. Validation = a ``val_frac`` share of the training content groups when
    at least 10 remain, otherwise the last ``val_frac`` of each training trial of at
    least ``tail_min_s`` seconds, separated by one margin (as in ``split_fold``).
    """
    trials, groups = data["trials"], data["groups"]
    subjects = data["subjects"]
    test_s = subjects[fold]
    cfolds = make_folds(groups, trials, n_content, seed)
    own = {g for t, g in zip(trials, groups) if t["subject"] == test_s}
    ci = fold % len(cfolds)
    for k in range(len(cfolds)):
        if own & set(cfolds[(fold + k) % len(cfolds)]):
            ci = (fold + k) % len(cfolds)
            break
    test_g = set(cfolds[ci])
    train_g = sorted({g for g in groups if g not in test_g})
    val_mode = "groups" if len(train_g) >= 10 else "tail"
    val_g: set = set()
    if val_mode == "groups":
        rng = random.Random(seed + 1000 + fold)
        pool = list(train_g)
        rng.shuffle(pool)
        val_g = set(pool[: max(1, int(round(val_frac * len(pool))))])
    out = {"train": [], "val": [], "test": []}
    for ti, (t, g) in enumerate(zip(trials, groups)):
        n = t["eeg"].shape[1]
        if t["subject"] == test_s:
            if g in test_g:
                out["test"].append(Segment(ti, 0, n))
        elif g in test_g:
            continue
        elif g in val_g:
            out["val"].append(Segment(ti, 0, n))
        elif val_mode == "tail" and n / SR >= tail_min_s:
            cut = int(n * (1 - val_frac))
            out["train"].append(Segment(ti, 0, cut - margin))
            out["val"].append(Segment(ti, cut + margin, n))
        else:
            out["train"].append(Segment(ti, 0, n))
    out.update(test_groups=sorted(test_g), n_folds=len(subjects), val_mode=f"loso-{val_mode}",
               test_subject=test_s)
    return out


# =============================================================================
# Candidate features
# =============================================================================

def _z(x: torch.Tensor, ref: slice) -> torch.Tensor:
    """z-score ``x`` (..., T) using stats of ``x[..., ref]``."""
    c = x[..., ref]
    return (x - c.mean(-1, keepdim=True)) / c.std(-1, keepdim=True).clamp_min(1e-6)


def quantile_match(cands: torch.Tensor) -> torch.Tensor:
    """Give every candidate the same multiset of values, keeping its time order.

    Args:
        cands: ``(K, T)``.

    Returns:
        ``(K, T)``: candidate k's values replaced, rank for rank, by the mean
        over candidates of the sorted values.
    """
    srt, _ = cands.sort(dim=-1)
    ref = srt.mean(0)                                   # (T,)
    ranks = cands.argsort(dim=-1).argsort(dim=-1)       # (K, T)
    return ref[ranks]


def _smooth_z(x: torch.Tensor, ref: slice, sigma: float = 2.0) -> torch.Tensor:
    """Gaussian-smooth sparse event streams (K, L), then z-score on ``ref``."""
    r = int(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype) / sigma) ** 2)
    y = torch.nn.functional.conv1d(x[:, None], (k / k.sum())[None, None], padding=r)[:, 0]
    c = y[..., ref]
    return (y - c.mean(-1, keepdim=True)) / c.std(-1, keepdim=True).clamp_min(1e-6)


def glimpse_channels(envs: torch.Tensor, central: slice) -> list[torch.Tensor]:
    """Glimpsed / masked parts of each candidate (Brodbeck et al., 2020).

    For candidate k, ``glimpsed_k = env_k * [env_k >= max_{j != k} env_j]`` and
    ``masked_k = env_k * [env_k < max_{j != k} env_j]``, on per-candidate
    z-scored-then-shifted-positive envelopes.  Cortex tracks an ignored talker
    mainly where it is glimpsed; an attended talker is also tracked where it is
    masked -- so the masked channel carries most of the attentional contrast.
    """
    x = envs - envs[..., central].min(-1, keepdim=True).values      # >= 0 on central
    K = x.shape[0]
    other = torch.stack([torch.cat([x[:k], x[k + 1:]]).max(0).values for k in range(K)])
    g = x * (x >= other)
    m = x * (x < other)
    return [_z(g, central), _z(m, central)]


def n_audio_channels(n_extra: int, glimpse: bool, bank_dims: list[int]) -> int:
    """Audio-encoder input width implied by the feature options.

    ``bank_dims``: channel count of each feature bank; glimpse masks apply per
    channel of the *first* bank (spectrotemporal) or, with no bank, broadband.
    """
    n = 2 + n_extra + sum(bank_dims)
    if glimpse:
        n += 2 * bank_dims[0] if bank_dims else 2
    return n


def _banks(t: dict, key: str, banks: list[str], a: int, b: int) -> torch.Tensor | None:
    """Concatenated feature banks (C, b-a) of stream ``key``; first bank first."""
    if not banks:
        return None
    return torch.cat([t[f"{key}_{bk}"][:, a:b].float() for bk in banks], 0)


def candidate_features(envs: torch.Tensor, margin: int, matched: bool,
                       extras: torch.Tensor | None = None,
                       glimpse: bool = False, sb: torch.Tensor | None = None,
                       n_glimpse: int = 0) -> torch.Tensor:
    """Envelope + onset channels for K candidates with context margins.

    Args:
        envs: ``(K, T + 2*margin)`` raw envelopes.
        margin: context samples on each side.
        matched: quantile-match candidates over the full extended window.

    Returns:
        ``(K, 2, T + 2*margin)``.
    """
    central = slice(margin, envs.shape[-1] - margin)
    x = _z(envs, central)
    if matched:
        x = quantile_match(x)
        x = _z(x, central)
    onset = torch.relu(torch.diff(x, dim=-1, prepend=x[..., :1]))
    if matched:
        onset = quantile_match(onset)
    onset = _z(onset, central)
    chans = [x, onset]
    if sb is not None:                          # (K, C, L) feature-bank channels
        sbz = torch.stack([_z(sb[:, b], central) for b in range(sb.shape[1])], 1)
        chans += [sbz[:, b] for b in range(sbz.shape[1])]
        if glimpse:                             # spectrotemporal glimpses, first bank
            for b in range(n_glimpse):
                g, m = glimpse_channels(sbz[:, b], central)
                chans += [g, m]
    elif glimpse:
        gm = glimpse_channels(_z(envs, central), central)
        chans += [gm[0], gm[1]]
    if extras is not None:                      # (K, E, L) event streams
        chans += [_smooth_z(extras[:, e], central) for e in range(extras.shape[1])]
    return torch.stack(chans, dim=1)


# =============================================================================
# Datasets
# =============================================================================

def _extras(t: dict, key: str, feats: list[str], a: int, b: int) -> torch.Tensor | None:
    """(E, b-a) extra event streams of stream ``key`` ("env_att"/"env_unatt")."""
    if not feats:
        return None
    return torch.stack([t[f"{key}_{f}"][a:b].float() for f in feats])


def cand_sides(t: dict) -> torch.Tensor:
    """Side (0 left, 1 right, -1 unknown) of every entry of ``env_cands``.

    Uses ``t["cand_sides"]`` when present; for six-speaker MAESTRO trials the fixed
    layout is speakers 1-2 left hemisphere, 3-4 right, 5-6 rear distractors.
    """
    if "cand_sides" in t:
        return torch.as_tensor(t["cand_sides"])
    n = t["env_cands"].shape[0]
    if n == 6:
        return torch.tensor([0, 0, 1, 1, -1, -1])
    return torch.full((n,), -1)


def _cand_stack(t: dict, idx: list[int], starts: list[int], L: int,
                banks: list[str]) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Envelopes ``(K, L)`` and feature banks ``(K, C, L)`` of candidates ``idx``
    (entries of ``env_cands``), each cut at its own start sample."""
    envs = torch.stack([t["env_cands"][i, s:s + L].float() for i, s in zip(idx, starts)])
    sbs = None
    if banks:
        sbs = torch.stack([torch.cat([t[f"env_cands_{bk}"][i, :, s:s + L].float()
                                      for bk in banks], 0) for i, s in zip(idx, starts)])
    return envs, sbs


class CropDataset(Dataset):
    """Random training crops, re-drawn each epoch via :meth:`resample`.

    Each item: EEG with margins, K=2 candidates (attended + competing talker)
    in random order, label = index of the attended candidate, candidate sides.
    """

    def __init__(self, data: dict, segments: list[Segment], win: int, margin: int,
                 matched: bool, density: float = 1.0, seed: int = 0,
                 extra_feats: list[str] | None = None, chan_drop: float = 0.0,
                 eeg_noise: float = 0.0, glimpse: bool = False,
                 banks: list[str] | None = None, n_glimpse: int = 0) -> None:
        self.glimpse, self.banks, self.n_glimpse = glimpse, list(banks or []), n_glimpse
        self.extra_feats = list(extra_feats or [])
        self.cands = list(data.get("cand_idx") or [])   # K-way candidates from env_cands
        self.chan_drop, self.eeg_noise = chan_drop, eeg_noise
        self.trials = data["trials"]
        self.sidx = {s: i for i, s in enumerate(data["subjects"])}
        # Single-talker trials also need room for a non-overlapping negative.
        self.segments = [s for s in segments if s.end - s.start >= win + 2 * margin
                         and (data["trials"][s.trial].get("has_unattended", True)
                              or data["trials"][s.trial]["eeg"].shape[1] >= 3 * (win + 2 * margin))]
        self.win, self.margin, self.matched = win, margin, matched
        self.density = density
        self.rng = np.random.default_rng(seed)
        self.items: list[tuple[int, int]] = []
        self.resample()

    def resample(self) -> None:
        """Draw a fresh set of crops, ~``density`` x non-overlapping coverage."""
        items = []
        L = self.win + 2 * self.margin
        for si, s in enumerate(self.segments):
            n = max(1, math.ceil(self.density * (s.end - s.start) / self.win))
            starts = self.rng.integers(s.start, s.end - L + 1, size=n)
            items.extend((si, int(st)) for st in starts)
        self.items = items

    def subject_of(self, idx: int) -> int:
        return self.sidx[self.trials[self.segments[self.items[idx][0]].trial]["subject"]]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        si, st = self.items[idx]
        t = self.trials[self.segments[si].trial]
        L = self.win + 2 * self.margin
        eeg = t["eeg"][:, st:st + L]
        cands = self.cands if (self.cands and "env_cands" in t) else None
        if cands is not None:
            # K-way matching over the listed candidates (e.g. MAESTRO's four attendable
            # talkers); random order, label = position of the attended talker.
            assert not self.extra_feats, "cand_extra is not supported with env_cands"
            perm = [cands[i] for i in self.rng.permutation(len(cands))]
            envs, sbs = _cand_stack(t, perm, [st] * len(perm), L, self.banks)
            y = perm.index(int(t["att_idx"]))
            sides = cand_sides(t)[perm]
            att_side = int(sides[y])
            eeg = eeg.float()
            if self.chan_drop > 0:
                keep = torch.from_numpy(self.rng.random(eeg.shape[0]) >= self.chan_drop).float()
                eeg = eeg * keep[:, None]
            if self.eeg_noise > 0:
                eeg = eeg + self.eeg_noise * torch.from_numpy(
                    self.rng.standard_normal(eeg.shape).astype("float32"))
            return {
                "eeg": eeg,
                "cand": candidate_features(envs, self.margin, self.matched, None,
                                           glimpse=self.glimpse, sb=sbs,
                                           n_glimpse=self.n_glimpse),
                "y": torch.tensor(y), "side": sides, "att_side": torch.tensor(att_side),
                "subj": torch.tensor(self.sidx[t["subject"]]), "env_att": envs[y].float(),
            }
        if t.get("has_unattended", True):
            envs = torch.stack([t["env_att"][st:st + L], t["env_unatt"][st:st + L]])
            sbs = (torch.stack([_banks(t, "env_att", self.banks, st, st + L),
                                _banks(t, "env_unatt", self.banks, st, st + L)])
                   if self.banks else None)
            ex = None
            if self.extra_feats:
                ex = torch.stack([_extras(t, "env_att", self.extra_feats, st, st + L),
                                  _extras(t, "env_unatt", self.extra_feats, st, st + L)])
            att_side = 1 if t["attended_ear"] == "R" else 0
            sides = torch.tensor([att_side, 1 - att_side])
        else:
            # Single-talker data (pretraining): the negative is the same story at
            # another, non-overlapping time -- the match/mismatch task.
            n = t["env_att"].shape[0]
            while True:
                ns = int(self.rng.integers(0, n - L + 1))
                if ns + L <= st or ns >= st + L:
                    break
            envs = torch.stack([t["env_att"][st:st + L], t["env_att"][ns:ns + L]])
            sbs = (torch.stack([_banks(t, "env_att", self.banks, st, st + L),
                                _banks(t, "env_att", self.banks, ns, ns + L)])
                   if self.banks else None)
            ex = None
            if self.extra_feats:
                ex = torch.stack([_extras(t, "env_att", self.extra_feats, st, st + L),
                                  _extras(t, "env_att", self.extra_feats, ns, ns + L)])
            att_side = -1
            sides = torch.tensor([-1, -1])
        if self.rng.random() < 0.5:
            envs, sides, y = envs.flip(0), sides.flip(0), 1
            ex = ex.flip(0) if ex is not None else None
            sbs = sbs.flip(0) if sbs is not None else None
        else:
            y = 0
        eeg = eeg.float()
        if self.chan_drop > 0:          # augmentation: drop whole channels
            keep = torch.from_numpy(self.rng.random(eeg.shape[0]) >= self.chan_drop).float()
            eeg = eeg * keep[:, None]
        if self.eeg_noise > 0:
            eeg = eeg + self.eeg_noise * torch.from_numpy(
                self.rng.standard_normal(eeg.shape).astype("float32"))
        return {
            "eeg": eeg,
            "cand": candidate_features(envs.float(), self.margin, self.matched, ex,
                                       glimpse=self.glimpse, sb=sbs,
                                       n_glimpse=self.n_glimpse),
            "y": torch.tensor(y),
            "side": sides,
            "att_side": torch.tensor(att_side),
            "subj": torch.tensor(self.sidx[t["subject"]]),
            "env_att": envs[y].float(),
        }


class SubjectBatchSampler(Sampler):
    """Batches drawn from a single listener, so InfoNCE cannot match on identity."""

    def __init__(self, ds: CropDataset, batch_size: int, seed: int = 0) -> None:
        self.ds, self.bs = ds, batch_size
        self.rng = random.Random(seed)

    def _batches(self) -> list[list[int]]:
        by_s = collections.defaultdict(list)
        for i in range(len(self.ds)):
            by_s[self.ds.subject_of(i)].append(i)
        batches = []
        for idx in by_s.values():
            self.rng.shuffle(idx)
            batches.extend(idx[k:k + self.bs] for k in range(0, len(idx), self.bs)
                           if len(idx[k:k + self.bs]) >= 4)
        self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return sum(math.ceil(len(v) / self.bs) for v in
                   collections.Counter(self.ds.subject_of(i)
                                       for i in range(len(self.ds))).values())


def eval_windows(data: dict, segments: list[Segment], win: int, margin: int) -> list[dict]:
    """Non-overlapping evaluation windows (hop = window) over ``segments``.

    Returns:
        List of ``{"trial", "start"}`` where ``start`` is the index of the
        first *context* sample; the central window is ``[start+margin,
        start+margin+win)``.
    """
    out = []
    L = win + 2 * margin
    for s in segments:
        for st in range(s.start, s.end - L + 1, win):
            out.append({"trial": s.trial, "start": st})
    return out


def build_eval_batch(data: dict, wins: list[dict], win: int, margin: int,
                     cand_set: str, seed: int = 0,
                     eeg_override: dict[int, torch.Tensor] | None = None,
                     extra_feats: list[str] | None = None, glimpse: bool = False,
                     banks: list[str] | None = None, n_glimpse: int = 0) -> dict:
    """Materialise evaluation windows for one candidate set.

    Candidate sets (K = 2, attended always stored at index 0 here; scoring is
    permutation-equivariant, so accuracy = fraction of windows where
    candidate 0 wins, with ties broken against the model):

    * ``competing`` -- attended vs. competing talker, as released.
    * ``matched`` -- the same pair after quantile matching.
    * ``same_talker`` -- attended vs. the attended talker's own speech from a
      random non-overlapping window of the same trial, matched.  Neither
      identity nor position can separate them.

    Returns:
        Batched tensors plus ``trial`` / ``subject`` index arrays.
    """
    rng = np.random.default_rng(seed)
    trials = data["trials"]
    sidx = {s: i for i, s in enumerate(data["subjects"])}
    L = win + 2 * margin
    eeg, cand, side, subj, tri, envs_att, envs_un = [], [], [], [], [], [], []
    keep = []
    for w in wins:
        t = trials[w["trial"]]
        st = w["start"]
        n = t["eeg"].shape[1]
        src = eeg_override[w["trial"]] if eeg_override is not None else t["eeg"]
        att = t["env_att"][st:st + L]
        cidx = list(data.get("cand_idx") or [])
        if cand_set in ("cands", "cands_matched") or (
                cand_set.startswith("same_talker") and cand_set != "same_talker"):
            # K-way sets. cands: attended + the other listed candidates, as released
            # (cands_matched: quantile-matched). same_talkerK: attended + K-1 windows
            # of the attended talker's own speech from the same trial, mutually
            # non-overlapping and not overlapping [st, st+L), matched.
            if cand_set.startswith("same_talker"):
                K = int(cand_set[len("same_talker"):])
                pool = [q for q in range(0, n - L + 1, max(1, win // 2))
                        if q + L <= st or q >= st + L]
                rng.shuffle(pool)
                negs: list[int] = []
                for q in pool:
                    if all(q + L <= r or q >= r + L for r in negs):
                        negs.append(int(q))
                    if len(negs) == K - 1:
                        break
                if len(negs) < K - 1:
                    continue
                starts = [st] + negs
                if "env_cands" in t:
                    envs, sbs = _cand_stack(t, [int(t["att_idx"])] * K, starts, L, banks or [])
                else:
                    envs = torch.stack([t["env_att"][q:q + L] for q in starts]).float()
                    sbs = (torch.stack([_banks(t, "env_att", banks, q, q + L) for q in starts])
                           if banks else None)
                sides = torch.full((K,), -1)
                matched = True
            else:
                a = int(t["att_idx"])
                order = [a] + [i for i in cidx if i != a]
                envs, sbs = _cand_stack(t, order, [st] * len(order), L, banks or [])
                sides = cand_sides(t)[order]
                matched = cand_set == "cands_matched"
            assert not extra_feats, "cand_extra is not supported with K-way sets"
            cand.append(candidate_features(envs, margin, matched=matched, extras=None,
                                           glimpse=glimpse, sb=sbs, n_glimpse=n_glimpse))
            eeg.append(src[:, st:st + L].float())
            side.append(sides)
            subj.append(sidx[t["subject"]])
            tri.append(w["trial"])
            envs_att.append(att[margin:margin + win].float())
            envs_un.append(envs[1, margin:margin + win].float())
            keep.append(w)
            continue
        if cand_set == "same_talker":
            # Negative: same talker, different time, no overlap with [st, st+L).
            ok = [s for s in range(0, n - L + 1, max(1, win // 2))
                  if s + L <= st or s >= st + L]
            if not ok:
                continue
            ns = int(rng.choice(ok))
            other = t["env_att"][ns:ns + L]
            ex_o = _extras(t, "env_att", extra_feats or [], ns, ns + L)
            sb_o = _banks(t, "env_att", banks or [], ns, ns + L)
            s_att = -1
            sides = torch.tensor([-1, -1])
        else:
            other = t["env_unatt"][st:st + L]
            ex_o = _extras(t, "env_unatt", extra_feats or [], st, st + L)
            sb_o = _banks(t, "env_unatt", banks or [], st, st + L)
            s_att = 1 if t["attended_ear"] == "R" else 0
            sides = torch.tensor([s_att, 1 - s_att])
        envs = torch.stack([att, other]).float()
        ex = None
        if extra_feats:
            ex = torch.stack([_extras(t, "env_att", extra_feats, st, st + L), ex_o])
        sbs = (torch.stack([_banks(t, "env_att", banks, st, st + L), sb_o]) if banks else None)
        cand.append(candidate_features(envs, margin, matched=cand_set != "competing", extras=ex,
                                       glimpse=glimpse, sb=sbs, n_glimpse=n_glimpse))
        eeg.append(src[:, st:st + L].float())
        side.append(sides)
        subj.append(sidx[t["subject"]])
        tri.append(w["trial"])
        envs_att.append(att[margin:margin + win].float())
        envs_un.append(t["env_unatt"][st + margin:st + margin + win].float())
        keep.append(w)
    if not eeg:          # e.g. same_talker with windows too long for a disjoint negative
        return None
    return {
        "eeg": torch.stack(eeg), "cand": torch.stack(cand), "side": torch.stack(side),
        "subj": torch.tensor(subj), "trial": np.array(tri),
        "env_att": torch.stack(envs_att), "env_unatt": torch.stack(envs_un),
        "wins": keep,
    }
