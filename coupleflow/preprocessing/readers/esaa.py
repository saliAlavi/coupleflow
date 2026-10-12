"""ESAA Mandarin EEG-speech AAD database (SCUT; Zenodo 7795585): 17 subjects, 32 trials.

Layout (after extraction into ``<RAW>/x``):

* ``S<k>/*.mat`` -- continuous BrainProducts recording at 1 kHz, one variable per
  channel (64 EEG + ECG), ``Markers`` with ``trail<K>D`` (direction cue),
  ``trail<K>S`` (stimulus start) and ``trail<K>E`` (stimulus end).  ``K`` is the
  *audio* trial number: S1-S8 heard Trail1..32 in order, S9-S17 in a shuffled order
  (``scut_suf_order`` in the authors' code), and the marker names follow the audio.
* ``clean/Trail<K>.wav`` -- unspatialised two-talker stimuli, 44.1 kHz stereo;
  channel 0 is the talker rendered on the LEFT, channel 1 the one on the RIGHT
  (verified against ``hrtf/Trail<K>.wav``).  The S-E interval equals the wav length.
* Attended direction per audio trial: ``scut_label[K-1][0]`` (0 = left) for S1-S8,
  inverted for S9-S17 (the authors' ``_reverse_label``).  Verified empirically: the
  attended envelope out-tracks the competing one in both subject groups.

Audio onset: the stimulus reaches the ear ~0.3 s after ``trail<K>S``
(``ONSET_DELAY_S``; envelope tracking peaks at +0.2-0.4 s on S2/S7 with the marker
as onset, and offset_scan on the built bundle then peaks at 0 s).

Trials the authors exclude for poor comprehension answers (``scut_remove_trail``)
are kept and flagged with ``poor_answer=True`` in the bundle trial dict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

from coupleflow.preprocessing.core import RawTrial, filter_continuous
from coupleflow import paths  # noqa: E402

ROOT = paths.raw("esaa")
RAW = ROOT / "x"
LINE = 50.0
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7',
            'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6',
            'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4',
            'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7',
            'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'Fpz', 'CPz', 'FCz']
# [direction (0 = left), speaker] per audio trial 1..32 (authors' db/SCUT.py)
SCUT_LABEL = [[0, 0], [1, 0], [0, 0], [1, 0], [0, 1], [1, 1], [0, 1], [1, 1],
              [1, 0], [0, 0], [1, 0], [0, 0], [1, 1], [0, 1], [1, 1], [0, 1],
              [0, 0], [1, 0], [0, 0], [1, 0], [0, 1], [1, 1], [0, 1], [1, 1],
              [1, 0], [0, 0], [1, 0], [0, 0], [1, 1], [0, 1], [1, 1], [0, 1]]
POOR = {1: [1, 2, 3, 9, 14, 24, 31], 2: [9, 29], 3: [1, 9, 19], 4: [31], 5: [14],
        6: [7, 9, 10, 13, 15, 29], 7: [9, 25], 8: [25, 26, 27, 28, 29, 30, 31, 32], 9: [15],
        10: [12, 15, 16], 11: [12, 17, 30], 13: [27, 29], 14: [9, 14, 26, 27], 16: [1]}
# chronological order of audio trials for S9-S17 (only used to map POOR, which the
# authors index chronologically)
SUF = [24, 8, 32, 31, 7, 15, 29, 6, 12, 26, 9, 1, 5, 28, 20, 11, 30, 21, 2, 4, 10, 13, 17,
       25, 14, 22, 23, 19, 16, 27, 18, 3]
# audio reaches the ear this long after the ``trail<K>S`` marker (estimated from the
# lag of peak envelope tracking; see module docstring)
ONSET_DELAY_S = 0.3
META = {"source": "ESAA (Zenodo 7795585)", "audio": "clean/ (unspatialised), ch0=left talker",
        "language": "Mandarin", "fs_native": 1000}


def subjects() -> list[str]:
    return [f"S{k}" for k in range(1, 18)]


def _extract(subject: str) -> Path:
    d = RAW / subject
    if not any(d.glob("*.mat")):
        import zipfile
        d.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ROOT / f"{subject}.zip") as z:      # S8.zip has no folder
            for n in z.namelist():
                if n.endswith(".mat") and not n.startswith("__MACOSX"):
                    with z.open(n) as src, open(d / Path(n).name, "wb") as dst:
                        while chunk := src.read(1 << 24):
                            dst.write(chunk)
    return d


def _markers(m: dict) -> dict[str, int]:
    M = m["Markers"]
    out = {}
    for i in range(M.shape[1]):
        desc = np.asarray(M[0, i]["Description"]).ravel()
        if desc.size:
            out[str(desc[0])] = int(np.asarray(M[0, i]["Position"]).ravel()[0]) - 1   # 1-based
    return out


def _wav(k: int) -> tuple[np.ndarray, int]:
    x, fs = sf.read(str(RAW / "clean" / f"Trail{k}.wav"))
    return x, fs


def read(subject: str) -> list[RawTrial]:
    sid = int(subject[1:])
    files = sorted(_extract(subject).glob("*.mat"))
    chrono = list(range(1, 33)) if sid <= 8 else SUF
    poor = {chrono[i - 1] for i in POOR.get(sid, [])}
    out, seen = [], set()
    for f in files:
        m = scipy.io.loadmat(str(f), variable_names=CHANNELS + ["Markers", "SampleRate"])
        fs = float(np.asarray(m["SampleRate"]).squeeze())
        eeg = np.stack([np.asarray(m[c], np.float64).ravel() for c in CHANNELS])   # (64, T)
        mk = _markers(m)
        del m
        x128, good = filter_continuous(eeg, fs, LINE)
        n = x128.shape[1]
        del eeg
        for k in range(1, 33):
            if f"trail{k}S" not in mk or k in seen:
                continue
            wav, afs = _wav(k)
            s = int(round((mk[f"trail{k}S"] / fs + ONSET_DELAY_S) * 128))
            dur = len(wav) / afs
            e = min(n, s + int(np.ceil(dur * 128)))
            if e - s < 10 * 128:
                continue
            seen.add(k)
            d = SCUT_LABEL[k - 1][0]
            if sid > 8:
                d = 1 - d
            ear = "L" if d == 0 else "R"
            a, u = (0, 1) if ear == "L" else (1, 0)
            out.append(RawTrial(subject=subject, trial=f"{subject}_trail{k}", eeg=x128[:, s:e],
                                fs=128.0, ch_names=CHANNELS, att_audio=wav[:, a],
                                unatt_audio=wav[:, u], audio_fs=afs, attended_ear=ear,
                                line_freq=LINE, att_name=f"Trail{k}_ch{a}",
                                unatt_name=f"Trail{k}_ch{u}", preprocessed=True, good=good,
                                extra={"poor_answer": k in poor}))
    return out
