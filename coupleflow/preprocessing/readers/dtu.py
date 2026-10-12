"""DTU two-talker AAD (Fuglsang et al. 2017): 18 subjects, 64-ch BioSemi, 512 Hz.

``eeg/S<k>.mat`` holds one continuous recording per subject (64 EEG + EXG1-8 +
Status).  Events alternate ``[start trigger, 191 end marker, ...]``; the
official ``preproc_data.m`` splits the recording at the start trigger and
aligns the stimulus from its first sample, so the audio starts at the trigger.

The continuous recording (EEG channels only) goes through
``core.filter_continuous`` first; trials are cut afterwards.  Only two-talker
trials (``n_speakers == 2``) are kept, and only trials whose end marker is 191
(S8 trial 46 stops after 19.6 s with marker 192 and is dropped).  ``attend_mf`` (1 male, 2 female) picks
the attended wav among ``wavfile_male`` / ``wavfile_female``; ``attend_lr``
(1 left, 2 right) gives the attended ear.  Stimuli are the clean (dry) story
wavs at 44.1 kHz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

from coupleflow.preprocessing.core import RawTrial, filter_continuous
from coupleflow import paths  # noqa: E402

RAW = paths.raw("dtu")
LINE = 50.0
EXTRA = {f"EXG{i}" for i in range(1, 9)} | {"Status"}
META = {"source": str(RAW), "trial_span": "start trigger -> 191 end marker, audio at trigger",
        "kept": "n_speakers == 2"}


def subjects() -> list[str]:
    return [f"S{k}" for k in range(1, 19)]


def _wav(name: str) -> tuple[np.ndarray, int]:
    x, fs = sf.read(str(RAW / "audio" / name))
    return (x.mean(1) if x.ndim > 1 else x), fs


def _cellstr(c: np.ndarray) -> str:
    return str(c.flat[0]) if getattr(c, "size", 0) > 0 else ""


def read(subject: str) -> list[RawTrial]:
    d = scipy.io.loadmat(str(RAW / "eeg" / f"{subject}.mat"))["data"][0, 0]
    fs = float(d["fsample"][0, 0]["eeg"][0, 0])
    chan = d["dim"][0, 0]["chan"][0, 0]["eeg"][0, 0]
    names_all = [str(chan[i, 0].flat[0]) for i in range(chan.shape[0])]
    keep = [i for i, c in enumerate(names_all) if c not in EXTRA]
    names = [names_all[i] for i in keep]
    x = np.asarray(d["eeg"][0, 0], np.float64)[:, keep].T              # (64, T)
    ev = d["event"][0, 0]["eeg"][0, 0]
    samples = ev["sample"].ravel().astype(int)
    values = np.array([int(np.asarray(v).ravel()[0]) for v in ev["value"].ravel()])

    meta = scipy.io.loadmat(str(RAW / "metadata" / f"{subject}_metadata.mat"))
    attend_mf = meta["attend_mf"].ravel()
    attend_lr = meta["attend_lr"].ravel()
    n_spk = meta["n_speakers"].ravel()
    trig = meta["trigger"].ravel()
    wm, wf = meta["wavfile_male"], meta["wavfile_female"]

    starts, ends = samples[0::2], samples[1::2]
    ends_ok = values[1::2] == 191   # S8 trial 46 ends with 192 after 19.6 s (aborted): skipped
    n = min(len(starts), len(ends), len(attend_mf))
    assert np.array_equal(values[0::2][:n], trig[:n]), f"{subject}: triggers != expinfo"

    x128, good = filter_continuous(x, fs, LINE)
    del x
    out = []
    for i in range(n):
        if int(n_spk[i]) != 2:
            continue
        if not ends_ok[i]:
            print(f"[dtu] {subject} trial {i + 1}: end marker {values[2 * i + 1]} != 191, skipped")
            continue
        male, female = _cellstr(wm[i, 0]), _cellstr(wf[i, 0])
        if not male or not female:
            continue
        att, unatt = (male, female) if int(attend_mf[i]) == 1 else (female, male)
        xa, fa = _wav(att)
        xu, fu = _wav(unatt)
        assert fa == fu
        s0 = int(round((starts[i] - 1) * 128 / fs))    # MATLAB 1-based -> 0-based
        s1 = int(round((ends[i] - 1) * 128 / fs))
        ear = "L" if int(attend_lr[i]) == 1 else "R"
        out.append(RawTrial(subject=subject, trial=f"{subject}_trial{i + 1}",
                            eeg=x128[:, s0:s1].copy(), fs=128.0, ch_names=names,
                            att_audio=xa, unatt_audio=xu, audio_fs=fa, attended_ear=ear,
                            line_freq=LINE, att_name=att, unatt_name=unatt,
                            preprocessed=True, good=good))
    return out
