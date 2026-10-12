"""Broderick et al. 2018 "Cocktail Party" (O'Sullivan et al. 2015 paradigm).

33 subjects, 128-ch BioSemi (A1..D32) + 2 mastoids (dropped), EEG given at
128 Hz, unfiltered and unreferenced, time-locked to stimulus onset.  30 runs
of ~60 s; run k plays segment k of "Twenty Thousand Leagues Under the Sea"
(left ear) and "Journey to the Centre of the Earth" (right ear).  Subjects
1-17 attend 20000 / left, subjects 18-33 attend Journey / right.  Six
subjects miss one run (29 runs).  Dublin, 50 Hz mains.

ENVELOPE-ONLY DATASET.  The release ships no audio, only one 128-Hz
broadband envelope per run and book (``Stimuli/Envelopes``).  The original
recordings could not be recovered: the 20000 text is the 1870s "A Floating
Reef" translation, which no LibriVox / Gutenberg recording uses, and the
Journey reader is none of the LibriVox / Gutenberg male readings of that
text (checked by envelope cross-correlation and ASR timing).  So the audio
side cannot go through ``core.audio_features``.  Instead this module builds
the bundle itself: the EEG goes through the unchanged core pipeline
(``core.filter_continuous`` + ``core.finish_eeg``), and the provided
envelopes are resampled 128 -> 64 Hz (polyphase) as ``env_att`` /
``env_unatt``.  There are no ``sb`` / ``sb28`` subband targets, and the
bundle meta says ``envelope_source: provided``.  NOTE: the envelope differs
from the 28-band gammatone envelope of the other datasets.

Build (do NOT use ``coupleflow/preprocessing/build.py``; ``read`` refuses on purpose)::

    python -m coupleflow.preprocessing.readers.broderick_cp --workers 16            # unified_env64.pt
    python -m coupleflow.preprocessing.readers.broderick_cp --native --workers 16   # unified_native.pt

``unified_native.pt`` keeps all 128 BioSemi channels: steps 1-5 via
``core.filter_continuous``, then the same 128 -> 64 Hz polyphase resample
and per-trial x channel z-score as ``core.finish_eeg`` but without the
64-channel mapping.  ``channel_mask`` is the good-channel mask (bad channels
zeroed), and the bundle stores ``ch_names`` (A1..D32) and ``ch_pos``.
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from coupleflow import paths  # noqa: E402
from coupleflow.preprocessing import core  # noqa: E402
from coupleflow.preprocessing.core import RawTrial  # noqa: E402

RAW = paths.raw("broderick_cp")
DATASET = "broderick_cp"
LINE = 50.0
FS = 128.0
BOOKS = ("20000", "Journey")          # left ear, right ear
META = {"envelope_source": "provided",
        "envelope_note": "release envelopes (128 Hz) resampled to 64 Hz; no audio, no sb/sb28",
        "attend": "S1-17: 20000/left, S18-33: Journey/right", "native_fs": FS}


def subjects() -> list[str]:
    return [f"S{k}" for k in range(1, 34)]


def _positions() -> tuple[list[str], np.ndarray]:
    import mne
    pos = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
    names = [f"{b}{i}" for b in "ABCD" for i in range(1, 33)]
    return names, np.stack([pos[n] for n in names])


def _envelope(book: str, run: int) -> np.ndarray:
    m = scipy.io.loadmat(str(RAW / "Stimuli" / "Envelopes" / book / f"{book}_{run}_env.mat"))
    assert int(m["fsEnv"].squeeze()) == FS
    return np.asarray(m["envelope"], np.float64)[:, 0]


def read_raw(subject: str) -> list[RawTrial]:
    """Raw trials; ``att_audio`` / ``unatt_audio`` hold the 128-Hz provided envelopes."""
    k = int(subject[1:])
    att, unatt = (BOOKS[0], BOOKS[1]) if k <= 17 else (BOOKS[1], BOOKS[0])
    ear = "L" if k <= 17 else "R"
    names, pos = _positions()
    d = RAW / "EEG" / f"Subject{k}"
    runs = sorted(int(re.search(r"_Run(\d+)\.mat$", p.name).group(1))
                  for p in d.glob(f"Subject{k}_Run*.mat"))
    out = []
    for r in runs:
        m = scipy.io.loadmat(str(d / f"Subject{k}_Run{r}.mat"))
        assert float(m["fs"].squeeze()) == FS
        eeg = np.asarray(m["eegData"], np.float64)            # mastoids dropped
        if eeg.shape[0] != 128:                               # stored (T, 128) except
            eeg = eeg.T                                       # S6 runs 9-30: (128, T)
        ea, eu = _envelope(att, r), _envelope(unatt, r)
        out.append(RawTrial(subject=subject, trial=f"{subject}_run{r}", eeg=eeg, fs=FS,
                            ch_names=names, ch_pos=pos, att_audio=ea, unatt_audio=eu,
                            audio_fs=int(FS), attended_ear=ear, line_freq=LINE,
                            att_name=f"{att}_{r}", unatt_name=f"{unatt}_{r}",
                            extra={"run": r, "envelope_source": "provided"}))
    return out


def read(subject: str) -> list[RawTrial]:
    raise RuntimeError("broderick_cp has no audio (envelopes only); build with "
                       "`python -m coupleflow.preprocessing.readers.broderick_cp`, not build.py")


def _native_eeg(x128: np.ndarray, good: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """``core.finish_eeg`` step 7 without the 64-channel mapping (all 128 channels)."""
    y = core._resample(x128, core.FS_MID, core.FS_OUT)
    mu = y.mean(axis=1, keepdims=True)
    sd = y.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    y = (y - mu) / sd
    y[~good] = 0.0
    return torch.from_numpy(np.ascontiguousarray(y)), torch.from_numpy(good.copy())


def process(rt: RawTrial, native: bool = False) -> dict:
    """EEG through the core pipeline; provided envelopes 128 -> 64 Hz."""
    x128, good = core.filter_continuous(rt.eeg, rt.fs, rt.line_freq, rt.modality)
    if native:
        eeg, mask = _native_eeg(x128, good)
    else:
        eeg, mask = core.finish_eeg(x128, good, rt.ch_names, rt.ch_pos, rt.modality)
    ea = resample_poly(rt.att_audio, core.FS_OUT, rt.audio_fs).astype(np.float32)
    eu = resample_poly(rt.unatt_audio, core.FS_OUT, rt.audio_fs).astype(np.float32)
    T = min(eeg.shape[1], len(ea), len(eu))
    return {"subject": rt.subject, "trial": rt.trial, "attended_ear": rt.attended_ear,
            "attention_label": {"L": 0, "R": 1}[rt.attended_ear],
            "channel_mask": mask, "eeg": eeg[:, :T].half(),
            "env_att": torch.from_numpy(ea[:T].copy()),
            "env_unatt": torch.from_numpy(eu[:T].copy()),
            "has_unattended": True, "n_bad": int((~good).sum()),
            "att_name": rt.att_name, "unatt_name": rt.unatt_name, "modality": rt.modality,
            **rt.extra}


def _one(args: tuple[str, bool]) -> list[dict]:
    subject, native = args
    torch.set_num_threads(2)
    out = [process(rt, native) for rt in read_raw(subject)]
    print(f"[{DATASET}] {subject}: {len(out)} trials", flush=True)
    return out


def build(native: bool = False, workers: int = 8, limit: int | None = None) -> Path:
    subs = subjects()[:limit]
    trials: list[dict] = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, [(s, native) for s in subs]):
            trials += r
    meta = dict(META)
    name = core.BUNDLE
    if native:
        names, pos = _positions()
        meta.update({"layout": "biosemi128 native (no 64-ch mapping)", "ch_names": names,
                     "ch_pos": pos.tolist()})
        name = "unified_native.pt"
    if limit is not None:
        name = name.replace(".pt", ".smoke.pt")
    return core.save_bundle(DATASET, trials, meta, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build(a.native, a.workers, a.limit)
