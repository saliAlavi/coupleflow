"""KU Leuven ultra-high-density AAD (Mundanad Narayanan, Zink & Bertrand 2021; Zenodo 4518754).

30 subjects (S0-S29), 255-electrode 10-5 cap (253 EEG channels in the CURRY files),
SynAmps RT at 1 kHz.  Four ~6.5-min recordings per subject, one per trial, named
``S<x>_AAD_<P>`` with ``P`` in {1L, 1R, 2L, 2R} (L/R = attended ear).  Stimuli
(``stimuli/part{1,2}_track{1,2}_dry.wav``, 44.1 kHz) per the Zenodo table:

    P   left ear           right ear
    1*  part1_track2_dry   part1_track1_dry
    2*  part2_track1_dry   part2_track2_dry

The recording carries a trigger pulse every second while the stimulus plays; the first
rising edge marks stimulus onset (``sample_script.m`` cuts ``locs(1):locs(end)``).

NOTE: the stimuli are two parts of two of the same "radioboeken" Dutch stories as the
KU Leuven AAD dataset (Das et al. 2016) -- do not mix the two datasets across
train / test without handling stimulus overlap.  Each story part is heard twice per
subject (once attended, once ignored), so there are only two stimulus pairs.

Channel labels are 10-5 names (case differs, e.g. ``FP1``, ``AFP3``) and are resolved
case-insensitively in mne's ``standard_1005`` montage; ``ch_pos`` from that montage
drives the core's nearest-position mapping onto the 64 canonical slots.

High-density ablation bundle (all good native channels, no 64-slot mapping)::

    python -m coupleflow.preprocessing.readers.uhd255 --native --workers 8
    # -> $COUPLEFLOW_DATA/uhd255/unified_native.pt

It reuses ``core.filter_continuous`` (via ``read``) and the same 128 -> 64 Hz resample
and per-trial x channel z-score as ``core.finish_eeg``; bad channels are zeroed and
``channel_mask`` is the native good-channel mask; ``ch_names`` / ``ch_pos`` are stored
in ``meta``.
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

from coupleflow.preprocessing.core import RawTrial, filter_continuous
from coupleflow import paths  # noqa: E402

ROOT = paths.raw("uhd255")
RAW = ROOT / "x"
LINE = 50.0
TABLE = {"1": ("part1_track2_dry", "part1_track1_dry"),     # (left ear, right ear)
         "2": ("part2_track1_dry", "part2_track2_dry")}
META = {"source": "KU Leuven UHD 255-ch AAD (Zenodo 4518754)", "fs_native": 1000,
        "stimuli": "dry wavs; same radioboeken stories as KU Leuven AAD",
        "mapping": "standard_1005 positions -> 64 slots (nearest, Hungarian)"}


def subjects() -> list[str]:
    return [f"S{k}" for k in range(30)]


def _extract(subject: str) -> Path:
    d = RAW / subject
    if len(list(d.glob("*.dat"))) < 4:
        src = ROOT / f"{subject}.tar.gz"
        alt = RAW / "dl" / f"{subject}.tar.gz"             # parallel-download helper
        if (RAW / "dl" / f"{subject}.tar.gz.done").exists():
            src = alt
        with tarfile.open(src) as t:
            t.extractall(RAW, filter="data")
    return d


_MONT: dict = {}


def positions(names: list[str]) -> np.ndarray:
    """``(C, 3)`` standard_1005 positions for (case-insensitive) 10-5 labels."""
    if "m" not in _MONT:
        import mne
        m = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
        _MONT["m"] = {k.lower(): v for k, v in m.items()}
    return np.stack([_MONT["m"][n.lower()] for n in names])


def _audio(name: str) -> tuple[np.ndarray, int]:
    x, fs = sf.read(str(RAW / "stimuli" / f"{name}.wav"))
    return (x.mean(1) if x.ndim > 1 else x), fs


def read(subject: str) -> list[RawTrial]:
    import mne
    d = _extract(subject)
    out = []
    for f in sorted(d.glob(f"{subject}_AAD_*.dat")):
        p = re.search(r"_AAD_(\d)([LR])$", f.stem)
        part, ear = p.group(1), p.group(2)
        raw = mne.io.read_raw_curry(str(f), preload=True, verbose="error")
        fs = float(raw.info["sfreq"])
        # stimulus onset = first rising edge of the trigger channel (``findpeaks`` on
        # loadCurryData's markers in sample_script.m); mne's annotations sit on the
        # falling edge ~170 ms later
        trig = raw.get_data(picks=["Trigger"])[0]
        t0 = int(np.flatnonzero(np.diff(trig) > 0)[0] + 1) / fs
        raw.pick("eeg")
        names = list(raw.ch_names)
        x128, good = filter_continuous(raw.get_data(), fs, LINE)
        del raw
        left, right = TABLE[part]
        att, unatt = (left, right) if ear == "L" else (right, left)
        xa, fa = _audio(att)
        xu, fu = _audio(unatt)
        assert fa == fu
        s = int(round(t0 * 128))
        e = min(x128.shape[1], s + int(np.ceil(len(xa) / fa * 128)))
        out.append(RawTrial(subject=subject, trial=f"{subject}_{part}{ear}", eeg=x128[:, s:e],
                            fs=128.0, ch_names=names, att_audio=xa, unatt_audio=xu,
                            audio_fs=fa, attended_ear=ear, line_freq=LINE, att_name=att,
                            unatt_name=unatt, ch_pos=positions(names), preprocessed=True,
                            good=good))
    return out


# ------------------------------------------------------------- native-channel bundle

def _native_one(subject: str) -> list[dict]:
    import torch

    from coupleflow.preprocessing.core import FS_MID, FS_OUT, _resample, process_trial
    torch.set_num_threads(2)
    out = []
    try:
        raws = read(subject)
    except Exception:
        import traceback
        traceback.print_exc()
        return out
    for rt in raws:
        tr = process_trial(rt)                     # audio features + 64-slot eeg
        y = _resample(rt.eeg * rt.good[:, None], FS_MID, FS_OUT)
        mu = y.mean(axis=1, keepdims=True)
        sd = y.std(axis=1, keepdims=True)
        sd[sd < 1e-8] = 1.0
        y = (y - mu) / sd
        y[~rt.good] = 0.0
        T = tr["env_att"].shape[0]
        tr["eeg"] = torch.from_numpy(np.ascontiguousarray(y[:, :T])).half()
        tr["channel_mask"] = torch.from_numpy(rt.good.copy())
        tr["ch_names"] = rt.ch_names
        out.append(tr)
    print(f"[uhd255 native] {subject}: {len(out)} trials", flush=True)
    return out


def build_native(workers: int = 8, limit: int | None = None) -> Path:
    """Write ``unified_native.pt`` (all good native channels, 64 Hz, z-scored)."""
    from concurrent.futures import ProcessPoolExecutor

    from coupleflow.preprocessing.core import save_bundle
    subs = subjects()[:limit]
    trials: list[dict] = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_native_one, subs):
            trials += r
    names = trials[0]["ch_names"]
    assert all(t["ch_names"] == names for t in trials)
    meta = dict(META, mapping="none (native channels)", ch_names=names,
                ch_pos=positions(names).tolist())
    name = "unified_native.pt" if limit is None else "unified_native.smoke.pt"
    return save_bundle("uhd255", trials, meta, name)



if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.native:
        build_native(a.workers, a.limit)
