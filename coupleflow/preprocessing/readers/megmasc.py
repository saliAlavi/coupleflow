"""MEG-MASC (Gwilliams et al. 2023): 27 English speakers, 208-ch KIT MEG, 1 kHz, BIDS.

Each subject heard four synthesised stories (task-0..3: lw1, cable_spool_fort,
easy_money, the_black_willow) in one or two sessions; each story is split into
~3-min wav files.  ``events.tsv`` marks every played file with a
``kind == 'sound'`` event whose onset is the first audio sample, so one played
wav = one trial, the audio starting at that onset.  The wav files contain the
story interleaved with short random word lists (not separable at trial level;
kept).

Continuous runs are filtered with ``core.filter_continuous(modality='meg')``
on the 208 axial gradiometers (reference / misc channels dropped); bad sensors
are the union over the subject's runs, and ``project`` fits ``core.meg_pca``
on all of the subject's filtered segments (label-free).  NYU: 60 Hz mains.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from coupleflow.preprocessing.core import RawTrial, filter_continuous, meg_pca, FS_MID
from coupleflow import paths  # noqa: E402

RAW = paths.raw("megmasc")
LINE = 60.0
META = {"modality": "meg", "sensors": "208 KIT axial gradiometers -> 64 PCA components",
        "source": "MEG-MASC, osf.io/ag3kj"}


def _runs(subject: str) -> list[tuple[str, str, Path]]:
    """``(session, task, con_path)`` for every complete story run of ``subject``."""
    out = []
    for con in sorted((RAW / subject).glob("ses-*/meg/*_task-*_meg.con")):
        m = re.search(r"ses-(\d+)_task-(\d+)_meg", con.name)
        ses, task = m.group(1), m.group(2)
        ev = con.with_name(con.name.replace("_meg.con", "_events.tsv"))
        if task in "0123" and ev.exists():
            out.append((ses, task, con))
    return out


def subjects() -> list[str]:
    """Subjects with at least one downloaded story run (stimuli must be present)."""
    import os
    if os.environ.get("MEGMASC_SUBJECTS"):             # subset, e.g. "sub-04+sub-06"
        return re.split(r"[,+: ]+", os.environ["MEGMASC_SUBJECTS"].strip())
    subs = sorted(p.name for p in RAW.glob("sub-*") if p.is_dir())
    return [s for s in subs if _runs(s)]


def _sounds(ev_path: Path) -> list[tuple[float, str]]:
    ev = pd.read_csv(ev_path, sep="\t")
    out = []
    for onset, tt in zip(ev["onset"], ev["trial_type"]):
        d = ast.literal_eval(tt)
        if d.get("kind") == "sound":
            out.append((float(onset), str(d["sound"])))
    return out


def _wav(rel: str) -> tuple[np.ndarray, int, str]:
    """Load ``stimuli/audio/<story>_<k>.0.wav`` (names in events carry a '.0')."""
    d, name = rel.rsplit("/", 1)
    name = re.sub(r"_(\d+)\.0\.wav$", r"_\1.wav", name).lower()   # The_Black_Willow_0.0.wav
    p = RAW / d / name
    x, fs = sf.read(str(p))
    return (x.mean(1) if x.ndim > 1 else x), int(fs), p.stem.lower()


def _smooth_env(x: np.ndarray, fs: float) -> np.ndarray:
    """|x| resampled to 1 kHz and smoothed over 20 ms (for onset alignment)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import resample_poly
    e = np.abs(x) if int(fs) == 1000 else resample_poly(np.abs(x), 1000, int(fs))
    return uniform_filter1d(np.abs(e), 20)


def _mic_lag(misc: np.ndarray, onset: float, a: np.ndarray, afs: int,
             search: int = 500, dur: int = 60000) -> tuple[int, float]:
    """Delay (ms) of the wav in the recorded audio channel relative to ``onset``.

    ``misc``: ``(M, T)`` 1-kHz misc channels (one is an analog copy of the
    stimulus).  Returns ``(lag_ms, r)`` of the best channel; r ~ 0.99 when found.
    """
    ae = _smooth_env(a, afs)[:dur]
    ae = (ae - ae.mean()) / (ae.std() + 1e-12)
    i0 = int(round(onset * 1000)) - search
    best = (0, 0.0)
    if i0 < 0 or i0 + len(ae) + 2 * search > misc.shape[1]:
        return best
    for ch in misc:
        seg = ch[i0:i0 + len(ae) + 2 * search]
        if seg.std() <= 0:
            continue
        me = _smooth_env(seg - np.median(seg), 1000)
        c = np.correlate(me - me.mean(), ae, mode="valid") / len(ae)
        k = int(np.argmax(c))
        w = me[k:k + len(ae)]
        r = float(np.mean(ae * (w - w.mean()) / (w.std() + 1e-12)))
        if r > best[1]:
            best = (k - search, r)
    return best


def read(subject: str) -> list[RawTrial]:
    import mne
    mne.set_log_level("ERROR")
    out: list[RawTrial] = []
    for ses, task, con in _runs(subject):
        raw = mne.io.read_raw_kit(str(con), preload=True, verbose=False)
        misc = raw.copy().pick("misc").get_data()
        raw.pick("mag", exclude=[])                  # 208 axial gradiometers (typed 'mag')
        fs = float(raw.info["sfreq"])
        x128, good = filter_continuous(raw.get_data(), fs, LINE, modality="meg")
        names = list(raw.ch_names)
        del raw
        n128 = x128.shape[1]
        snd = []
        for onset, rel in _sounds(con.with_name(con.name.replace("_meg.con", "_events.tsv"))):
            a, afs, stem = _wav(rel)
            # the events onset precedes the sound by a small fixed delay (~40 ms): measure
            # it per trial on the recorded audio channel; run median where not found
            lag, r = _mic_lag(misc, onset, a, afs)
            snd.append([onset, a, afs, stem, lag, r])
        found = [s[4] for s in snd if s[5] > 0.8]
        fallback = int(np.median(found)) if found else 0
        del misc
        for onset, a, afs, stem, lag, r in snd:
            if r <= 0.8:
                lag = fallback
            i0 = int(round((onset + lag / 1000.0) * FS_MID))
            i1 = min(n128, i0 + int(np.ceil(len(a) / afs * FS_MID)))
            if i1 - i0 < 10 * FS_MID:
                continue
            a = a[: int((i1 - i0) / FS_MID * afs)]
            out.append(RawTrial(subject=subject, trial=f"{subject}_ses{ses}_task{task}_{stem}",
                                eeg=x128[:, i0:i1].copy(), fs=float(FS_MID), ch_names=names,
                                att_audio=a, unatt_audio=None, audio_fs=afs, line_freq=LINE,
                                att_name=stem, preprocessed=True, modality="meg", good=good,
                                extra={"session": ses, "task": task,
                                       "mic_lag_ms": int(lag), "mic_r": round(r, 3)}))
    # bad sensors: union over the subject's runs, so one projection fits all trials
    if out:
        good = np.logical_and.reduce([t.good for t in out])
        for t in out:
            t.eeg[~good] = 0.0
            t.good = good
    return out


def project(subject: str, trials: list[RawTrial]) -> np.ndarray | None:
    """``(64, n_good)`` PCA projection fitted on all of the subject's segments.

    With ``MEGMASC_NATIVE=1`` no projection is used: all 208 sensors are kept in the
    same order for every listener (``unified_native.pt``), which a model with a
    208-channel input reads directly; the per-listener PCA is not comparable across
    listeners, so it cannot be used for leave-one-listener-out evaluation.
    """
    import os
    if os.environ.get("MEGMASC_NATIVE"):
        return None
    x = np.concatenate([t.eeg for t in trials], axis=1)
    proj, _ = meg_pca(x, trials[0].good)
    return proj
