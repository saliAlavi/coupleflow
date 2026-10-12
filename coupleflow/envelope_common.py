"""SHARED data pipeline for the envelope-regression baselines.

Used by :mod:`paper_review.vlaai`, :mod:`paper_review.decaf` and
:mod:`paper_review.ssm2mel`.  All three papers solve the *same* task —
regress a single-channel speech envelope from 64-channel EEG and score it
with Pearson ``r`` — so they share one data path and differ only in the
window geometry and normalisation each paper prescribes.

Pipeline
--------
1. Load our standard 128-Hz / 5-s ``.pt`` cache for a dataset.  The cache
   writes ``sample_<seg_idx:06d}.pt`` with a **global** monotonically
   increasing ``seg_idx`` (``data/cache.py``), so sorting every file across
   ``train/``, ``val/`` and ``test/`` by that integer recovers the original
   recording order.  Segments were cut with ``overlap_s: 0.0``, so
   concatenating a trial's segments in index order rebuilds the continuous
   trial (verified: index runs are contiguous and no trial straddles two
   cache split directories for kuleuven / dtu / nju).
2. EEG 128 Hz -> 64 Hz (``resample_poly``), z-scored per (trial, channel).
3. Envelope target from the cached 16-kHz audio using the *VLAAI /
   Biesmans et al. (2017)* recipe: 28-filter ERB-spaced gammatone
   filterbank (50 Hz - 5 kHz), full-wave rectification, power-law
   compression with exponent 0.6, mean across subbands, downsample to
   64 Hz.  Computed for the attended **and** unattended stream so the same
   run yields reconstruction ``r`` *and* correlation-based AAD accuracy.
4. Strict trial-disjoint split: whole (subject, trial) pairs are held out
   for val/test, 10 % / 10 % per subject, all subjects present in every
   split.  Same construction as
   :func:`paper_review.darnet_listennet.data.build_splits` with
   ``protocol="strict"``, except the per-subject shuffle key is derived
   from SHA-256 instead of the salted builtin ``hash()`` so splits are
   bit-reproducible across processes.

Step 1-3 are expensive (gammatone filtering of ~90 h of audio per
dataset), so they are materialised once by ``build_env64_cache.py`` into
``<cache_root>/<dataset>/paper_review_env64.pt`` and memory-mapped
thereafter.
"""

from __future__ import annotations

import collections
import hashlib
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from coupleflow import paths  # noqa: E402

SR_CACHE_EEG = 128       # Hz — EEG rate of the shared .pt cache
SR_AUDIO = 16_000        # Hz — audio rate of the shared .pt cache
SR = 64                  # Hz — rate every paper works at
SEG_EEG = 640            # samples — 5 s at 128 Hz
SEG_AUDIO = 80_000       # samples — 5 s at 16 kHz

DEFAULT_CACHE_ROOT = paths.DATA
STREAM_FILENAME = "paper_review_env64.pt"


# =============================================================================
# Gammatone envelope (VLAAI / Biesmans et al. 2017 target definition)
# =============================================================================

_EAR_Q = 9.26449
_MIN_BW = 24.7


def erb_space(f_low: float = 50.0, f_high: float = 5000.0,
              n_filters: int = 28) -> np.ndarray:
    """ERB-spaced centre frequencies, ascending.

    Slaney (1993) ``ERBSpace`` from the Auditory Toolbox — the spacing used
    by the MATLAB gammatone filterbank behind VLAAI's envelope target.

    Args:
        f_low: Lowest centre frequency (Hz).
        f_high: Highest centre frequency (Hz).
        n_filters: Number of filters.

    Returns:
        ``(n_filters,)`` array of centre frequencies in Hz, ascending.
    """
    i = np.arange(1, n_filters + 1, dtype=np.float64)
    cf = -(_EAR_Q * _MIN_BW) + np.exp(
        i * (-np.log(f_high + _EAR_Q * _MIN_BW)
             + np.log(f_low + _EAR_Q * _MIN_BW)) / n_filters
    ) * (f_high + _EAR_Q * _MIN_BW)
    return np.sort(cf)


def gammatone_filterbank(
    sr: int = SR_AUDIO,
    n_filters: int = 28,
    f_low: float = 50.0,
    f_high: float = 5000.0,
    order: int = 4,
    n_taps: int | None = None,
) -> np.ndarray:
    """Build the impulse responses of an ERB-spaced gammatone filterbank.

    Uses the analytic 4th-order gammatone

    .. math:: g(t) = t^{n-1} e^{-2\\pi b t} \\cos(2\\pi f_c t)

    with :math:`b = 1.019\\,\\mathrm{ERB}(f_c)`, each filter normalised to
    unit magnitude response at its own centre frequency.  Realised as an
    FIR so the whole bank can be applied with one batched overlap-add
    convolution instead of 28 sequential IIR passes.

    Args:
        sr: Audio sample rate (Hz).
        n_filters: Number of filters.
        f_low: Lowest centre frequency (Hz).
        f_high: Highest centre frequency (Hz).
        order: Gammatone order.
        n_taps: FIR length. Defaults to 128 ms, past which the lowest-CF
            impulse response has decayed below 1e-4 of its peak.

    Returns:
        ``(n_filters, n_taps)`` float64 array of impulse responses.
    """
    if n_taps is None:
        n_taps = int(round(0.128 * sr))
    cfs = erb_space(f_low, f_high, n_filters)
    t = np.arange(n_taps, dtype=np.float64) / sr
    irs = np.empty((n_filters, n_taps), dtype=np.float64)
    for k, cf in enumerate(cfs):
        erb = _MIN_BW * (4.37e-3 * cf + 1.0)
        b = 1.019 * erb
        ir = t ** (order - 1) * np.exp(-2.0 * np.pi * b * t) * np.cos(
            2.0 * np.pi * cf * t)
        # Normalise to unit gain at the centre frequency.
        gain = np.abs(np.sum(ir * np.exp(-2j * np.pi * cf * t)))
        irs[k] = ir / max(gain, 1e-30)
    return irs


def gammatone_envelope(
    audio: np.ndarray,
    sr: int = SR_AUDIO,
    target_sr: int = SR,
    n_filters: int = 28,
    f_low: float = 50.0,
    f_high: float = 5000.0,
    power: float = 0.6,
    irs: np.ndarray | None = None,
    block_s: float = 60.0,
) -> np.ndarray:
    """Extract the broadband speech envelope exactly as VLAAI defines it.

    Gammatone filterbank -> ``|.|`` -> ``(.)**0.6`` -> mean over subbands ->
    anti-aliased downsample to ``target_sr``.  No further band-pass is
    applied: the paper's target is the raw compressed broadband envelope
    (Accou et al. 2023, "Speech envelope").

    Filtering runs in ``block_s`` blocks, each primed with the preceding
    ``n_taps - 1`` input samples, so the result is bit-identical to filtering
    the whole signal at once while keeping the ``(n_filters, block)`` subband
    buffer bounded.

    Args:
        audio: ``(T,)`` waveform at ``sr`` Hz.
        sr: Input sample rate (Hz).
        target_sr: Output sample rate (Hz). ``sr`` must be a multiple.
        n_filters: Gammatone filters.
        f_low: Lowest centre frequency (Hz).
        f_high: Highest centre frequency (Hz).
        power: Power-law compression exponent.
        irs: Pre-built filterbank from :func:`gammatone_filterbank`.
        block_s: Block length for the streaming convolution (s).

    Returns:
        ``(T * target_sr / sr,)`` float32 envelope.
    """
    from scipy.signal import oaconvolve, resample_poly

    if sr % target_sr:
        raise ValueError(f"sr={sr} is not a multiple of target_sr={target_sr}")
    if irs is None:
        irs = gammatone_filterbank(sr, n_filters, f_low, f_high)
    x = np.asarray(audio, dtype=np.float64)
    n = x.shape[0]
    n_taps = irs.shape[1]
    prime = n_taps - 1
    block = max(int(round(block_s * sr)), n_taps)

    env = np.empty(n, dtype=np.float64)
    for s in range(0, n, block):
        e = min(s + block, n)
        lo = max(0, s - prime)
        seg = x[lo:e]
        if s - prime < 0:                          # zero-prime the first block
            seg = np.concatenate([np.zeros(prime - s, dtype=np.float64), seg])
        # mode="full" then slice keeps the filters causal (mode="same" would
        # advance the response by half the FIR length).
        sub = oaconvolve(seg[None, :], irs, mode="full", axes=-1)
        sub = sub[:, prime:prime + (e - s)]
        env[s:e] = (np.abs(sub) ** power).mean(axis=0)
        del sub
    return resample_poly(env, 1, sr // target_sr).astype(np.float32)


# =============================================================================
# Trial-stream cache (step 1-3)
# =============================================================================

def _seg_index(path: Path) -> int:
    """Global segment index encoded in ``sample_<idx>.pt``."""
    m = re.search(r"_(\d+)\.pt$", path.name)
    if m is None:
        raise ValueError(f"Cannot parse segment index from {path}")
    return int(m.group(1))


def collect_cache_paths(cache_root: str | Path, dataset: str,
                        cache_name: str = "cache") -> list[Path]:
    """Every cached segment of ``dataset``, sorted into recording order.

    Files live in ``train/``, ``val/`` and ``test/`` subdirectories but carry
    a single global counter in their name, so sorting the pooled list by that
    counter restores the order in which segments were cut from the raw
    recordings.

    ``cache_name`` selects which cache directory under ``<root>/<dataset>/``
    to read.  It must be given explicitly to compare baselines against a
    NeuroToken run: the default ``cache`` is a separate, older build, so
    leaving it implicit silently compares two different preprocessings.
    """
    root = Path(cache_root) / dataset / cache_name
    paths: list[Path] = []
    for sp in ("train", "val", "test"):
        paths.extend((root / sp).glob("*.pt"))
    if not paths:
        raise RuntimeError(f"No cached samples for {dataset} under {root}")
    return sorted(paths, key=_seg_index)


def _resample_eeg(eeg: np.ndarray) -> np.ndarray:
    """128 Hz -> 64 Hz along the last axis."""
    from scipy.signal import resample_poly
    return resample_poly(eeg, 1, SR_CACHE_EEG // SR, axis=-1).astype(np.float32)


def _build_one_trial(args: tuple) -> dict:
    """Worker: rebuild one continuous trial from its cached segments."""
    paths, n_filters, f_low, f_high, power = args
    eeg_parts, att_parts, unatt_parts = [], [], []
    meta = None
    for p in paths:
        s = torch.load(p, map_location="cpu", weights_only=False)
        e = s["eeg"].float().numpy()
        if e.shape[1] < SEG_EEG:
            e = np.pad(e, ((0, 0), (0, SEG_EEG - e.shape[1])))
        eeg_parts.append(e[:, :SEG_EEG])
        att = s["attended_audio"].float().numpy()
        unatt = s["unattended_audio"].float().numpy()
        att_parts.append(np.pad(att, (0, max(0, SEG_AUDIO - att.shape[0])))[:SEG_AUDIO])
        unatt_parts.append(
            np.pad(unatt, (0, max(0, SEG_AUDIO - unatt.shape[0])))[:SEG_AUDIO])
        if meta is None:
            meta = {
                "subject": str(s["subject_id"]),
                "trial": str(s.get("trial_id", "")),
                "attended_ear": str(s.get("attended_ear", "")),
                "attention_label": int(s.get("attention_label", 0)),
                "channel_mask": s["channel_mask"].clone(),
            }

    eeg = _resample_eeg(np.concatenate(eeg_parts, axis=1))     # (64, T)
    # z-score per (trial, channel): the cache normalises each 5-s segment
    # independently, so renormalising the concatenated trial removes the
    # piecewise scale steps that stitching would otherwise introduce.
    mu = eeg.mean(axis=1, keepdims=True)
    sd = eeg.std(axis=1, keepdims=True)
    sd[sd < 1e-6] = 1.0
    eeg = (eeg - mu) / sd

    irs = gammatone_filterbank(SR_AUDIO, n_filters, f_low, f_high)
    kw = dict(n_filters=n_filters, f_low=f_low, f_high=f_high, power=power, irs=irs)
    env_att = gammatone_envelope(np.concatenate(att_parts), **kw)
    unatt_cat = np.concatenate(unatt_parts)
    has_unatt = bool(np.abs(unatt_cat).sum() > 1e-6)
    env_unatt = (gammatone_envelope(unatt_cat, **kw) if has_unatt
                 else np.zeros_like(env_att))

    t = min(eeg.shape[1], env_att.shape[0], env_unatt.shape[0])
    return {
        **meta,
        "eeg": torch.from_numpy(np.ascontiguousarray(eeg[:, :t])),
        "env_att": torch.from_numpy(env_att[:t]),
        "env_unatt": torch.from_numpy(env_unatt[:t]),
        "has_unattended": has_unatt,
        "n_segments": len(paths),
    }


def build_streams(
    cache_root: str | Path,
    dataset: str,
    out_path: str | Path | None = None,
    n_filters: int = 28,
    f_low: float = 50.0,
    f_high: float = 5000.0,
    power: float = 0.6,
    workers: int = 8,
    limit_trials: int | None = None,
    cache_name: str = "cache",
) -> dict:
    """Rebuild continuous 64-Hz trials for ``dataset`` and save them.

    Args:
        cache_root: Parent of ``<dataset>/cache/``.
        dataset: Dataset name (``kuleuven`` / ``dtu`` / ``nju``).
        out_path: Destination ``.pt``. Defaults to
            ``<cache_root>/<dataset>/paper_review_env64.pt``.
        n_filters: Gammatone filters for the envelope target.
        f_low: Lowest gammatone centre frequency (Hz).
        f_high: Highest gammatone centre frequency (Hz).
        power: Power-law compression exponent.
        workers: Process-pool size.
        limit_trials: Debug — only build the first N trials.

    Returns:
        The stream bundle that was written.
    """
    from concurrent.futures import ProcessPoolExecutor

    cache_root = Path(cache_root)
    paths = collect_cache_paths(cache_root, dataset, cache_name)
    meta = _load_trial_meta(paths)

    by_trial: dict[tuple[str, str], list[Path]] = collections.OrderedDict()
    for p, key in zip(paths, meta):
        by_trial.setdefault(key, []).append(p)
    keys = list(by_trial)
    if limit_trials is not None:
        keys = keys[:limit_trials]

    jobs = [(by_trial[k], n_filters, f_low, f_high, power) for k in keys]
    trials: list[dict] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, tr in enumerate(pool.map(_build_one_trial, jobs, chunksize=1)):
                trials.append(tr)
                if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                    print(f"  [{dataset}] {i + 1}/{len(jobs)} trials", flush=True)
    else:
        for i, job in enumerate(jobs):
            trials.append(_build_one_trial(job))
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f"  [{dataset}] {i + 1}/{len(jobs)} trials", flush=True)

    bundle = {
        "dataset": dataset,
        "sr": SR,
        "envelope": {
            "kind": "gammatone_broadband",
            "n_filters": n_filters,
            "f_low": f_low,
            "f_high": f_high,
            "power": power,
            "target_sr": SR,
        },
        "trials": trials,
    }
    if out_path is None:
        out_path = cache_root / dataset / STREAM_FILENAME
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"[{dataset}] wrote {len(trials)} trials -> {out_path}", flush=True)
    return bundle


def _load_trial_meta(paths: list[Path]) -> list[tuple[str, str]]:
    """``(subject_id, trial_id)`` per path, reusing the shared sidecar."""
    import pickle
    cache_dir = paths[0].parent.parent
    sidecar = cache_dir / "paper_review_meta.pkl"
    if sidecar.exists():
        try:
            data = pickle.loads(sidecar.read_bytes())
            lut = dict(zip((str(p) for p in data["key"]), data["meta"]))
            if all(str(p) in lut for p in paths):
                return [tuple(lut[str(p)]) for p in paths]
        except Exception:
            pass
    from concurrent.futures import ProcessPoolExecutor
    import os
    with ProcessPoolExecutor(max_workers=max(1, min(16, os.cpu_count() or 4))) as pool:
        return list(pool.map(_meta_one, [str(p) for p in paths], chunksize=64))


def _meta_one(p: str) -> tuple[str, str]:
    s = torch.load(p, map_location="cpu", weights_only=False)
    return str(s["subject_id"]), str(s.get("trial_id", ""))


_STREAM_CACHE: dict[str, dict] = {}


def load_streams(cache_root: str | Path, dataset: str,
                 subjects: list[str] | None = None,
                 stream_name: str | None = None) -> dict:
    """Load the pre-built 64-Hz trial streams for ``dataset``.

    Args:
        cache_root: Parent of ``<dataset>/``.
        dataset: Dataset name.
        subjects: Optional subject whitelist.

    Returns:
        The stream bundle, filtered to ``subjects`` when given.
    """
    path = Path(cache_root) / dataset / (stream_name or STREAM_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run:\n"
            f"    python paper_review/build_env64_cache.py --dataset {dataset}"
        )
    key = str(path)
    if key not in _STREAM_CACHE:
        _STREAM_CACHE[key] = torch.load(path, map_location="cpu", weights_only=False)
    bundle = _STREAM_CACHE[key]
    if subjects:
        keep = set(subjects)
        bundle = {**bundle,
                  "trials": [t for t in bundle["trials"] if t["subject"] in keep]}
        if not bundle["trials"]:
            raise RuntimeError(f"No trials for subjects {sorted(keep)} in {dataset}")
    return bundle


# =============================================================================
# Strict trial-disjoint split
# =============================================================================

def _shuffle_key(seed: int, subject: str) -> int:
    """Deterministic 32-bit shuffle key (builtin ``hash`` is salted)."""
    h = hashlib.sha256(f"{seed}:{subject}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def strict_trial_split(
    trials: list[dict],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Hold out whole trials per subject: 10 % val, 10 % test, rest train.

    Every subject appears in all three splits (subject-dependent model,
    stimulus-disjoint evaluation) which is exactly the protocol all three
    papers use — VLAAI's "80/10/10 per subject", DECAF's "unseen-stimulus
    test set from the same subjects", SSM2Mel's ``g_con=True``.  No window
    ever spans a split boundary because splitting happens before windowing.

    Args:
        trials: Trial dicts with ``subject`` / ``trial`` keys.
        val_frac: Fraction of each subject's trials used for validation.
        test_frac: Fraction of each subject's trials used for test.
        seed: Split seed.

    Returns:
        ``(train_idx, val_idx, test_idx)`` index lists into ``trials``.
    """
    by_subj: dict[str, list[int]] = collections.defaultdict(list)
    for i, t in enumerate(trials):
        by_subj[t["subject"]].append(i)

    train, val, test = [], [], []
    for sid in sorted(by_subj):
        ids = sorted(by_subj[sid], key=lambda i: trials[i]["trial"])
        random.Random(_shuffle_key(seed, sid)).shuffle(ids)
        n = len(ids)
        n_val = max(1, int(round(val_frac * n)))
        n_test = max(1, int(round(test_frac * n)))
        val.extend(ids[:n_val])
        test.extend(ids[n_val:n_val + n_test])
        train.extend(ids[n_val + n_test:])
    return sorted(train), sorted(val), sorted(test)


# =============================================================================
# Windowed dataset
# =============================================================================

class EnvelopeWindows(Dataset):
    """Sliding windows over continuous trials for envelope regression.

    Each item pairs a ``window_s`` window of EEG with the *simultaneous*
    attended-envelope window, optionally preceded by a ``context_s`` window
    (DECAF's "past") and optionally with the EEG advanced by ``delay_s`` to
    absorb the cortical response lag.

    Index convention: ``e0`` is the envelope start sample.  EEG is read from
    ``e0 + d`` where ``d = delay_s * sr``; context spans ``[e0 - c, e0)``.

    Args:
        trials: Trial dicts from :func:`load_streams`.
        indices: Which trials to use.
        window_s: Prediction window length (s).
        hop_s: Step between window starts (s). Defaults to ``window_s``
            (non-overlapping).
        context_s: Length of the preceding context window (s), 0 to disable.
        delay_s: EEG lead over the envelope (s).
        eeg_norm: ``"none"`` or ``"zscore"`` (per channel, per window).
        env_norm: ``"none"``, ``"minmax"`` or ``"zscore"`` (per window).
        subject_index: Mapping subject id -> integer for subject embeddings.
        min_env_std: Windows in which either stream's envelope is constant to
            within this tolerance are dropped.  Pearson ``r`` is undefined on a
            constant target, and such a window contributes neither a usable
            training gradient nor a meaningful metric.  NJU has ~2 % of them
            (both talkers silent); kuleuven and dtu have none.  The count of
            dropped windows is reported in the split ``info``.
    """

    def __init__(
        self,
        trials: list[dict],
        indices: list[int],
        window_s: float,
        hop_s: float | None = None,
        context_s: float = 0.0,
        delay_s: float = 0.0,
        eeg_norm: str = "none",
        env_norm: str = "none",
        subject_index: dict[str, int] | None = None,
        sr: int = SR,
        min_env_std: float = 1e-8,
    ):
        self.trials = trials
        self.sr = sr
        self.window = int(round(window_s * sr))
        self.hop = int(round((hop_s if hop_s is not None else window_s) * sr))
        self.context = int(round(context_s * sr))
        self.delay = int(round(delay_s * sr))
        self.eeg_norm = eeg_norm
        self.env_norm = env_norm
        if subject_index is None:
            subject_index = {s: i for i, s in enumerate(
                sorted({t["subject"] for t in trials}))}
        self.subject_index = subject_index

        self.items: list[tuple[int, int]] = []
        self.n_dropped_degenerate = 0
        for ti in indices:
            tr = self.trials[ti]
            t_len = int(tr["eeg"].shape[1])
            lo = self.context
            hi = t_len - self.window - self.delay      # inclusive upper bound
            if hi < lo:
                continue
            for e0 in range(lo, hi + 1, max(1, self.hop)):
                a = tr["env_att"][e0:e0 + self.window]
                u = tr["env_unatt"][e0:e0 + self.window]
                if (float(a.std()) < min_env_std
                        or float(u.std()) < min_env_std):
                    self.n_dropped_degenerate += 1
                    continue
                self.items.append((ti, e0))

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _norm_eeg(x: torch.Tensor, how: str) -> torch.Tensor:
        if how == "zscore":                            # x: (T, C)
            mu = x.mean(dim=0, keepdim=True)
            sd = x.std(dim=0, keepdim=True).clamp_min(1e-8)
            return (x - mu) / sd
        return x

    @staticmethod
    def _norm_env(x: torch.Tensor, how: str) -> torch.Tensor:
        if how == "minmax":                            # x: (T, 1)
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn).clamp_min(1e-8)
        if how == "zscore":
            return (x - x.mean()) / x.std().clamp_min(1e-8)
        return x

    def __getitem__(self, idx: int) -> dict:
        ti, e0 = self.items[idx]
        tr = self.trials[ti]
        w, c, d = self.window, self.context, self.delay

        eeg = tr["eeg"][:, e0 + d:e0 + d + w].transpose(0, 1).contiguous()  # (T,C)
        att = tr["env_att"][e0:e0 + w].unsqueeze(-1)                        # (T,1)
        unatt = tr["env_unatt"][e0:e0 + w].unsqueeze(-1)

        out = {
            "eeg": self._norm_eeg(eeg, self.eeg_norm),
            "env": self._norm_env(att, self.env_norm),
            "env_unatt": self._norm_env(unatt, self.env_norm),
            "subject": tr["subject"],
            "subject_idx": self.subject_index.get(tr["subject"], 0),
            "trial": tr["trial"],
            "trial_idx": ti,
        }
        if c > 0:
            out["eeg_context"] = self._norm_eeg(
                tr["eeg"][:, e0 - c + d:e0 + d].transpose(0, 1).contiguous(),
                self.eeg_norm)
            out["env_context"] = self._norm_env(
                tr["env_att"][e0 - c:e0].unsqueeze(-1), self.env_norm)
        return out


def collate(batch: list[dict]) -> dict:
    """Stack tensor fields, keep string/int fields as lists."""
    out: dict = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        elif k == "subject_idx" or k == "trial_idx":
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = [b[k] for b in batch]
    return out


def make_loaders(
    train: Dataset, val: Dataset, test: Dataset,
    batch_size: int = 64, num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders with the shared collate; train shuffles and drops the tail."""
    def _mk(ds: Dataset, shuffle: bool, bs: int) -> DataLoader:
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
            pin_memory=True, drop_last=shuffle, collate_fn=collate,
            persistent_workers=num_workers > 0,
        )
    return _mk(train, True, batch_size), _mk(val, False, batch_size), \
        _mk(test, False, batch_size)


def build_splits(
    dataset: str,
    window_s: float,
    train_overlap: float = 0.8,
    context_s: float = 0.0,
    delay_s: float = 0.0,
    eeg_norm: str = "none",
    env_norm: str = "none",
    seed: int = 42,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    subjects: list[str] | None = None,
    stream_name: str | None = None,
) -> tuple[EnvelopeWindows, EnvelopeWindows, EnvelopeWindows, dict]:
    """Strict trial-disjoint train/val/test windows for one dataset.

    Train windows use ``train_overlap`` (the papers' 80 %); val and test use
    non-overlapping windows so no test sample shares a sample with another.

    Args:
        dataset: ``kuleuven`` / ``dtu`` / ``nju``.
        window_s: Prediction window length (s).
        train_overlap: Overlap ratio for training windows.
        context_s: Preceding-context length (s), DECAF only.
        delay_s: EEG lead over the envelope (s).
        eeg_norm: Per-window EEG normalisation.
        env_norm: Per-window envelope normalisation.
        seed: Split seed.
        cache_root: Parent of ``<dataset>/``.
        subjects: Optional subject whitelist.

    Returns:
        ``(train, val, test, info)``.
    """
    bundle = load_streams(cache_root, dataset, subjects=subjects,
                          stream_name=stream_name)
    trials = bundle["trials"]
    tr_idx, va_idx, te_idx = strict_trial_split(trials, seed=seed)
    subject_index = {s: i for i, s in enumerate(
        sorted({t["subject"] for t in trials}))}

    common = dict(context_s=context_s, delay_s=delay_s, eeg_norm=eeg_norm,
                  env_norm=env_norm, subject_index=subject_index,
                  sr=int(bundle["sr"]))
    train = EnvelopeWindows(trials, tr_idx, window_s,
                            hop_s=window_s * (1.0 - train_overlap), **common)
    val = EnvelopeWindows(trials, va_idx, window_s, hop_s=window_s, **common)
    test = EnvelopeWindows(trials, te_idx, window_s, hop_s=window_s, **common)

    info = {
        "protocol": "strict_trial_disjoint",
        "dataset": dataset,
        "sr": int(bundle["sr"]),
        "window_s": window_s,
        "context_s": context_s,
        "delay_s": delay_s,
        "train_overlap": train_overlap,
        "seed": seed,
        "n_subjects": len(subject_index),
        "subjects": sorted(subject_index),
        "n_trials": len(trials),
        "n_train_trials": len(tr_idx),
        "n_val_trials": len(va_idx),
        "n_test_trials": len(te_idx),
        "n_train_windows": len(train),
        "n_val_windows": len(val),
        "n_test_windows": len(test),
        "n_dropped_degenerate_windows": {
            "train": train.n_dropped_degenerate,
            "val": val.n_dropped_degenerate,
            "test": test.n_dropped_degenerate,
        },
        "envelope": bundle["envelope"],
        "has_unattended": bool(
            all(t["has_unattended"] for t in trials)),
    }
    return train, val, test, info


# =============================================================================
# Metrics
# =============================================================================

def pearson(pred: torch.Tensor, target: torch.Tensor,
            dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation along ``dim``.

    Mirrors VLAAI's ``pearson_tf`` (and the PyTorch port shipped with
    SSM2Mel) so the training loss and the reported metric are identical to
    the papers'.

    The guard is applied *inside* the square root rather than added to it.
    ``sqrt`` has an infinite derivative at 0, so a single window whose
    prediction or target is constant — a silent stretch of stimulus, say —
    turns the whole gradient into NaN and poisons every subsequent step. The
    clamp keeps the derivative finite and, like TensorFlow's
    ``divide_no_nan`` in VLAAI's own ``pearson_tf``, returns 0 for a
    degenerate window instead of NaN.

    Args:
        pred: Predictions, e.g. ``(B, T, 1)``.
        target: Targets with the same shape.
        dim: Time axis.
        eps: Denominator guard.

    Returns:
        Correlations with ``dim`` reduced, e.g. ``(B, 1)``.
    """
    pm = pred.mean(dim=dim, keepdim=True)
    tm = target.mean(dim=dim, keepdim=True)
    pc, tc = pred - pm, target - tm
    num = (pc * tc).sum(dim=dim)
    den = (pc.pow(2).sum(dim=dim)
           * tc.pow(2).sum(dim=dim)).clamp_min(eps * eps).sqrt()
    return num / den


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``-r``, VLAAI's loss."""
    return -pearson(pred, target)


def summarize(
    r_att: np.ndarray,
    r_unatt: np.ndarray,
    subjects: list[str],
    trials: list[str],
) -> dict:
    """Reconstruction and AAD metrics from per-window correlations.

    Args:
        r_att: ``(N,)`` correlation of each prediction with the attended
            envelope.
        r_unatt: ``(N,)`` correlation with the unattended envelope.
        subjects: Subject id per window.
        trials: Trial id per window.

    Returns:
        Dict with the paper metric (``recon_r_subject_mean``, the mean over
        per-subject means, as DECAF reports) plus pooled/median variants and
        segment- and trial-level AAD accuracy.
    """
    r_att = np.asarray(r_att, dtype=np.float64)
    r_unatt = np.asarray(r_unatt, dtype=np.float64)
    subj = np.asarray(subjects)

    per_subject = {s: float(r_att[subj == s].mean()) for s in np.unique(subj)}
    seg_correct = (r_att > r_unatt)

    # Trial-level AAD: average each trial's correlations before deciding.
    trial_keys = [f"{s}||{t}" for s, t in zip(subjects, trials)]
    tk = np.asarray(trial_keys)
    trial_acc_by_subject: dict[str, list[float]] = collections.defaultdict(list)
    for key in np.unique(tk):
        m = tk == key
        hit = float(r_att[m].mean() > r_unatt[m].mean())
        trial_acc_by_subject[str(subj[m][0])].append(hit)
    per_subject_trial_acc = {
        s: float(np.mean(v)) for s, v in trial_acc_by_subject.items()}
    per_subject_seg_acc = {
        s: float(seg_correct[subj == s].mean()) for s in np.unique(subj)}

    return {
        "recon_r_subject_mean": float(np.mean(list(per_subject.values()))),
        "recon_r_subject_std": float(np.std(list(per_subject.values()))),
        "recon_r_window_mean": float(r_att.mean()),
        "recon_r_window_median": float(np.median(r_att)),
        "recon_r_unattended_window_mean": float(r_unatt.mean()),
        "recon_r_gap_window_mean": float((r_att - r_unatt).mean()),
        "aad_segment_acc": float(seg_correct.mean()),
        "aad_segment_acc_subject_mean": float(
            np.mean(list(per_subject_seg_acc.values()))),
        "aad_trial_acc_subject_mean": float(
            np.mean(list(per_subject_trial_acc.values()))),
        "n_windows": int(r_att.size),
        "n_trials": int(np.unique(tk).size),
        "per_subject_recon_r": per_subject,
        "per_subject_aad_segment_acc": per_subject_seg_acc,
        "per_subject_aad_trial_acc": per_subject_trial_acc,
    }


__all__ = [
    "SR", "SR_AUDIO", "SR_CACHE_EEG", "DEFAULT_CACHE_ROOT",
    "EnvelopeWindows", "build_splits", "build_streams", "collate",
    "collect_cache_paths", "erb_space", "gammatone_envelope",
    "gammatone_filterbank", "load_streams", "make_loaders", "pearson",
    "pearson_loss", "strict_trial_split", "summarize",
]
