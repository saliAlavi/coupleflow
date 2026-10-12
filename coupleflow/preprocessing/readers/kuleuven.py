"""KU Leuven AAD (Das et al. 2016): 16 subjects, 64-ch BioSemi, 20 trials each.

Raw ``S<k>.mat`` holds per-trial EEG at 128 Hz (already downsampled by the
authors, no filtering or re-referencing).  The stimulus of each trial starts
with the EEG.  Envelopes use the *dry* (unprocessed) story recordings, as in
the dataset's reference pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

from coupleflow.preprocessing.core import RawTrial
from coupleflow import paths  # noqa: E402

RAW = paths.raw("kuleuven")
LINE = 50.0


def subjects() -> list[str]:
    return [f"S{k}" for k in range(1, 17)]


def _wav(name: str) -> tuple[np.ndarray, int]:
    name = name.replace("_hrtf", "_dry")
    x, fs = sf.read(str(RAW / "stimuli" / name))
    return (x.mean(1) if x.ndim > 1 else x), fs


def read(subject: str) -> list[RawTrial]:
    mat = scipy.io.loadmat(str(RAW / f"{subject}.mat"))
    cell = mat["trials"]
    out = []
    for i in range(cell.shape[1]):
        tr = cell[0, i]
        eeg = np.asarray(tr["RawData"][0, 0]["EegData"][0, 0], np.float64).T   # (64, T)
        fs = float(tr["FileHeader"][0, 0]["SampleRate"][0, 0].squeeze())
        chans = tr["FileHeader"][0, 0]["Channels"][0, 0]
        names = [str(np.asarray(c["Label"]).squeeze()) for c in chans.ravel()]
        assert len(names) == eeg.shape[0], (len(names), eeg.shape)
        ear = str(tr["attended_ear"][0, 0][0]).upper()
        att_track = int(tr["attended_track"][0, 0][0, 0])
        st = tr["stimuli"][0, 0]
        s1, s2 = str(st[0, 0][0]), str(st[1, 0][0])
        t1, t2 = (s1, s2) if "track1" in s1 else (s2, s1)
        att, unatt = (t1, t2) if att_track == 1 else (t2, t1)
        xa, fa = _wav(att)
        xu, fu = _wav(unatt)
        assert fa == fu
        out.append(RawTrial(subject=subject, trial=f"{subject}_trial{i + 1}", eeg=eeg, fs=fs,
                            ch_names=names, att_audio=xa, unatt_audio=xu, audio_fs=fa,
                            attended_ear=ear, line_freq=LINE,
                            att_name=att.replace("_hrtf", "_dry"),
                            unatt_name=unatt.replace("_hrtf", "_dry")))
    return out
