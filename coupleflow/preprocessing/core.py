"""One preprocessing pipeline for every AAD / speech-tracking dataset.

Readers (``coupleflow/preprocessing/readers/<name>.py``) only *load* raw data; every
signal operation lives here so that all datasets are processed identically:

EEG (per trial, or per continuous recording before cutting trials)
    1. bad channels: std outside [0.2, 5] x median std -> excluded from the
       reference, zeroed, ``channel_mask`` False
    2. robust average reference (mean of good channels)
    3. resample native -> 128 Hz (polyphase, anti-aliased)
    4. notch at the dataset's line frequency (+ harmonic below Nyquist)
    5. band-pass 0.5-32 Hz, 4th-order Butterworth, zero phase
    6. map to the canonical 64-channel layout (by name, or by nearest
       electrode position for high-density caps)
    7. resample 128 -> 64 Hz, z-score per trial and channel

MEG gets the same steps except (1)-(2) (no reference for magnetometers /
gradiometers; bad sensors are still zeroed) and (6) (sensors are reduced to
64 virtual channels by PCA fitted on the training-independent whole
recording, which is label-free).

Audio (per stimulus, 16 kHz mono)
    28-filter causal gammatone bank (50 Hz-5 kHz, ERB spacing), |.|^0.6,
    64 Hz:  ``env`` = mean of 28 bands (the VLAAI envelope used for KU Leuven /
    DTU), ``sb`` = 8 contiguous band groups, ``sb28`` = all 28 bands.

Bundle format matches ``paper_review_env64*.pt`` so CoupleFlow can load it.
"""

from __future__ import annotations

import collections
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.preprocessing.channel_mapping import map_to_standard  # noqa: E402
from coupleflow.preprocessing.channels import STANDARD_64_CHANNELS  # noqa: E402
from coupleflow.envelope_common import gammatone_filterbank  # noqa: E402

CACHE_ROOT = paths.DATA
BUNDLE = "unified_env64.pt"
GROUPS = "unified_groups.json"
FS_MID = 128
FS_OUT = 64
FS_AUDIO = 16000
BAND = (0.5, 32.0)
PREPROC = {"bad_channel_ratio": [0.2, 5.0], "reference": "robust_average",
           "fs_mid": FS_MID, "notch": "line (50/60) + harmonic", "bandpass": list(BAND),
           "butter_order": 4, "fs_out": FS_OUT, "znorm": "per trial x channel",
           "envelope": "gammatone 28 x ERB 50-5000 Hz, |.|^0.6, causal FIR, 64 Hz",
           "banks": {"sb": 8, "sb28": 28}}


@dataclass
class RawTrial:
    """What a reader hands over: raw signals plus labels, nothing processed.

    Attributes:
        subject: listener id.
        trial: unique trial id (unique across the dataset, not only subject).
        eeg: ``(C, T)`` raw sensor data at ``fs`` (any unit).
        fs: native sampling rate.
        ch_names: channel labels (for name mapping).
        att_audio: attended (or only) stimulus waveform, ``audio_fs`` Hz,
            starting at the same instant as ``eeg``.
        unatt_audio: competing stimulus, or ``None`` for single-talker data.
        audio_fs: stimulus sampling rate.
        attended_ear: ``'L'``/``'R'`` (``''`` if not spatial).
        line_freq: mains frequency of the recording site.
        att_name / unatt_name: stimulus identifiers (for bookkeeping only;
            stimulus groups are recomputed from the envelopes).
        ch_pos: optional ``(C, 3)`` electrode positions (for position mapping).
        preprocessed: True if ``eeg`` already went through this pipeline's
            steps 1-5 on the continuous recording (then only 6-7 run here).
        modality: ``'eeg'`` or ``'meg'``.
        good: optional ``(C,)`` good-channel mask computed on the continuous
            recording.
    """

    subject: str
    trial: str
    eeg: np.ndarray
    fs: float
    ch_names: list[str]
    att_audio: np.ndarray
    unatt_audio: np.ndarray | None
    audio_fs: int
    attended_ear: str = ""
    line_freq: float = 50.0
    att_name: str = ""
    unatt_name: str = ""
    ch_pos: np.ndarray | None = None
    preprocessed: bool = False
    modality: str = "eeg"
    good: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- EEG

def bad_channels(x: np.ndarray, lo: float = 0.2, hi: float = 5.0) -> np.ndarray:
    """Good-channel mask ``(C,)`` from per-channel std (``x``: ``(C, T)``)."""
    sd = x.std(axis=1)
    med = float(np.median(sd))
    if med <= 0:
        return np.ones(len(sd), bool)
    good = (sd >= lo * med) & (sd <= hi * med)
    if good.sum() < max(8, int(0.5 * len(sd))):
        return np.ones(len(sd), bool)
    return good


def _resample(x: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
    from fractions import Fraction
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    if fr == 1:
        return x.astype(np.float32)
    return resample_poly(x, fr.numerator, fr.denominator, axis=-1).astype(np.float32)


def filter_continuous(x: np.ndarray, fs: float, line_freq: float, modality: str = "eeg",
                      good: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Steps 1-5 on ``(C, T)`` raw data. Returns ``(x_128Hz, good)``."""
    x = np.asarray(x, np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    if good is None:
        good = bad_channels(x)
    if modality == "eeg":
        x = x - x[good].mean(axis=0, keepdims=True)
    x[~good] = 0.0
    x = _resample(x, fs, FS_MID).astype(np.float64)
    for f0 in (line_freq, 2 * line_freq):
        if f0 < FS_MID / 2 - 1:
            b, a = iirnotch(f0, 30.0, fs=FS_MID)
            x = filtfilt(b, a, x, axis=1)
    sos = butter(4, list(BAND), btype="bandpass", fs=FS_MID, output="sos")
    x = sosfiltfilt(sos, x, axis=1)
    x[~good] = 0.0
    return x.astype(np.float32), good


_POS_CACHE: dict = {}


def _std_positions() -> np.ndarray:
    """``(64, 3)`` unit-sphere positions of the canonical layout."""
    if "std" not in _POS_CACHE:
        import mne
        mont = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
        alias = {"TP9": "P9", "TP10": "P10"}   # BioSemi P9/P10 fill these slots
        p = np.stack([mont[alias.get(c, c)] for c in STANDARD_64_CHANNELS])
        _POS_CACHE["std"] = p / np.linalg.norm(p, axis=1, keepdims=True)
    return _POS_CACHE["std"]


def map_by_position(x: np.ndarray, pos: np.ndarray, good: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Nearest-electrode (Hungarian, unique) mapping of a dense cap to 64 slots."""
    from scipy.optimize import linear_sum_assignment
    p = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    cost = 1.0 - _std_positions() @ p.T                      # (64, C)
    cost[:, ~good] += 10.0
    rows, cols = linear_sum_assignment(cost)
    out = np.zeros((64, x.shape[1]), np.float32)
    mask = np.zeros(64, bool)
    for r, c in zip(rows, cols):
        if good[c] and cost[r, c] < 0.05:                     # ~18 deg max displacement
            out[r] = x[c]
            mask[r] = True
    return out, mask, [int(c) for c in cols]


def meg_pca(x: np.ndarray, good: np.ndarray, n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Project good MEG sensors onto their top-``n`` principal components."""
    g = x[good]
    g = g / (g.std(axis=1, keepdims=True) + 1e-12)
    u, s, _ = np.linalg.svd(g @ g.T / g.shape[1])
    return u[:, :n].T.astype(np.float32), g


def finish_eeg(x128: np.ndarray, good: np.ndarray, ch_names: list[str],
               ch_pos: np.ndarray | None, modality: str = "eeg",
               proj: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Steps 6-7: canonical 64 channels, 64 Hz, z-scored. Returns ``(eeg, mask)``."""
    if modality == "meg" and proj is None:
        # Native sensor space: every sensor keeps its own row (same helmet for every
        # listener, so row k is the same sensor for everyone); bad sensors are zero.
        # Scale to unit variance first, as the PCA path does: MEG is in tesla (~1e-13),
        # below the flat-channel guard of the z-score below, and float16 storage.
        sd = x128.std(axis=1, keepdims=True)
        y = np.where(good[:, None], x128 / np.maximum(sd, 1e-30), 0.0)
        mask = good.copy()
    elif modality == "meg":
        g = x128[good]
        g = g / (g.std(axis=1, keepdims=True) + 1e-12)
        y = proj @ g
        mask = np.ones(64, bool)
    elif ch_pos is not None:
        y, mask, _ = map_by_position(x128, ch_pos, good)
    else:
        t, m = map_to_standard(torch.from_numpy(x128 * good[:, None]), ch_names)
        y, mask = t.numpy(), m.numpy().copy()
        # a named channel that was bad stays zero; drop it from the mask
        mask &= np.abs(y).sum(axis=1) > 0
    y = _resample(y, FS_MID, FS_OUT)
    mu = y.mean(axis=1, keepdims=True)
    sd = y.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    y = (y - mu) / sd
    y[~mask] = 0.0
    return torch.from_numpy(np.ascontiguousarray(y)), torch.from_numpy(mask)


# -------------------------------------------------------------------------- audio

_IRS: dict = {}


def audio_features(x: np.ndarray, fs: int) -> dict[str, np.ndarray]:
    """``env (T,)``, ``sb (8,T)``, ``sb28 (28,T)`` at 64 Hz from a waveform."""
    from scipy.signal import oaconvolve
    x = to_16k(x, fs)
    if "irs" not in _IRS:
        _IRS["irs"] = gammatone_filterbank(FS_AUDIO, 28, 50.0, 5000.0)
    irs = _IRS["irs"]
    prime = irs.shape[1] - 1
    dec = FS_AUDIO // FS_OUT
    ctx = FS_AUDIO                     # 1 s of context each side for the resampler
    block = 30 * FS_AUDIO
    n = len(x)
    out = []
    for s in range(0, n, block):
        e = min(s + block, n)
        a, b = max(0, s - ctx), min(n, e + ctx)
        lo = a - prime
        seg = x[max(0, lo):b]
        if lo < 0:
            seg = np.concatenate([np.zeros(-lo), seg])
        sub = oaconvolve(seg[None, :], irs, mode="full", axes=-1)[:, prime:prime + (b - a)]
        r = resample_poly(np.abs(sub) ** 0.6, 1, dec, axis=1)
        i0 = (s - a) // dec
        out.append(r[:, i0:i0 + (e - s + dec - 1) // dec])
    sb28 = np.concatenate(out, axis=1).astype(np.float32)
    sb28 = np.maximum(sb28, 0.0)
    groups = np.array_split(np.arange(28), 8)
    sb = np.stack([sb28[g].mean(0) for g in groups]).astype(np.float32)
    return {"env": sb28.mean(0).astype(np.float32), "sb": sb, "sb28": sb28}


def to_16k(x: np.ndarray, fs: int) -> np.ndarray:
    """Mono float waveform at 16 kHz."""
    x = np.asarray(x, np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1) if x.shape[1] < x.shape[0] else x.mean(axis=0)
    if int(fs) != FS_AUDIO:
        from fractions import Fraction
        fr = Fraction(FS_AUDIO, int(round(fs))).limit_denominator(2000)
        x = resample_poly(x, fr.numerator, fr.denominator)
    return x


# ------------------------------------------------------------------------- bundle

_FEAT: dict = {}


def _feat_cached(x: np.ndarray, fs: int, name: str) -> dict[str, np.ndarray]:
    """``audio_features`` memoised on (stimulus name, length) within a worker."""
    if not name:
        return audio_features(x, fs)
    key = (name, len(x), int(fs))
    if key not in _FEAT:
        if len(_FEAT) > 64:
            _FEAT.clear()
        _FEAT[key] = audio_features(x, fs)
    return _FEAT[key]

def process_trial(rt: RawTrial, proj: np.ndarray | None = None) -> dict:
    """Run the pipeline on one reader trial and return a bundle trial dict."""
    if rt.preprocessed:
        x128, good = rt.eeg, (rt.good if rt.good is not None else np.ones(len(rt.eeg), bool))
    else:
        x128, good = filter_continuous(rt.eeg, rt.fs, rt.line_freq, rt.modality, rt.good)
    eeg, mask = finish_eeg(x128, good, rt.ch_names, rt.ch_pos, rt.modality, proj)
    fa = _feat_cached(rt.att_audio, rt.audio_fs, rt.att_name)
    has_u = rt.unatt_audio is not None
    fu = (_feat_cached(rt.unatt_audio, rt.audio_fs, rt.unatt_name) if has_u
          else {k: np.zeros_like(v) for k, v in fa.items()})
    T = min(eeg.shape[1], fa["env"].shape[-1], fu["env"].shape[-1])
    ear = rt.attended_ear.upper()[:1] if rt.attended_ear else ""
    out = {"subject": rt.subject, "trial": rt.trial, "attended_ear": ear,
           "attention_label": {"L": 0, "R": 1}.get(ear, -1),
           "channel_mask": mask, "eeg": eeg[:, :T].half(),
           "env_att": torch.from_numpy(fa["env"][:T].copy()),
           "env_unatt": torch.from_numpy(fu["env"][:T].copy()),
           "has_unattended": has_u, "n_bad": int((~good).sum()),
           "att_name": rt.att_name, "unatt_name": rt.unatt_name, "modality": rt.modality}
    for b in ("sb", "sb28"):
        out[f"env_att_{b}"] = torch.from_numpy(fa[b][:, :T].copy()).half()
        out[f"env_unatt_{b}"] = torch.from_numpy(fu[b][:, :T].copy()).half()
    out.update(rt.extra)
    return out


def save_bundle(dataset: str, trials: list[dict], meta: dict | None = None,
                name: str = BUNDLE) -> Path:
    """Write ``<CACHE_ROOT>/<dataset>/<name>`` (float16 EEG) and print a summary."""
    ids = collections.Counter((t["subject"], t["trial"]) for t in trials)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise ValueError(f"duplicate (subject, trial) ids: {dup[:5]}")
    few = [t["trial"] for t in trials if int(t["channel_mask"].sum()) < 16]
    if few:
        print(f"WARNING [{dataset}] {len(few)} trials have < 16 mapped channels "
              f"(channel-name / position mapping problem?): {few[:5]}", flush=True)
    d = CACHE_ROOT / dataset
    d.mkdir(parents=True, exist_ok=True)
    bundle = {"dataset": dataset, "sr": FS_OUT, "preproc": PREPROC,
              "envelope": {"kind": "gammatone_broadband", "n_filters": 28, "f_low": 50.0,
                           "f_high": 5000.0, "power": 0.6, "target_sr": FS_OUT},
              "meta": meta or {}, "trials": trials}
    torch.save(bundle, d / name)
    hours = sum(t["eeg"].shape[1] for t in trials) / FS_OUT / 3600
    print(f"[{dataset}] wrote {d / name}: {len(trials)} trials, "
          f"{len({t['subject'] for t in trials})} subjects, {hours:.2f} h, "
          f"mean channels {np.mean([int(t['channel_mask'].sum()) for t in trials]):.1f}",
          flush=True)
    return d / name


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1))
