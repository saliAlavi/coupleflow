"""SparrKULee (Accou et al. 2024; ICASSP 2024 challenge train set): single talker.

Raw BIDS release under ``paths.raw('sparrkulee')``: 85 subjects
(sub-001..085), 8-10 ``task-listeningActive`` runs of ~15 min each, 64-ch
BioSemi + EXG1-8 + Status, stored as ``.bdf.gz`` at 1024 Hz.  Each run's
``*_events.tsv`` names the stimulus (``stimuli/stimuli/eeg/<name>.npz.gz``,
48 kHz), its trigger track (``t_<name>.npz.gz``) and the SNR.

Alignment (SparrKULee docs / brain_pipe ``AlignPeriodicBlockTriggers``): the
sound card plays a 1 Hz pulse train on a separate channel that is recorded
in the BioSemi Status channel.  Rising edges of the stimulus trigger track
(first pulse at audio sample 0) are matched one-to-one to Status edges and a
line ``t_eeg = a * t_audio + b`` is fitted.  ``b`` is the EEG time of audio
sample 0; ``a`` (~1 - 4e-5, ~30 ms over 15 min) is the clock drift, which is
removed by re-timing the *audio* onto the EEG clock (linear interpolation at
48 kHz), so the EEG itself is never resampled here.  The sub-sample remainder
of ``b`` after cutting at a 128 Hz EEG sample is folded into the same audio
re-timing.

The whole run (EEG channels only) goes through ``core.filter_continuous``;
the stimulus span is cut afterwards.  Excluded: stories that overlap the KU
Leuven AAD stimuli (``audiobook_1*`` incl. ``_shifted`` / ``_artefact``, and
``audiobook_4``; see ``tools/build_sparrkulee_streams.py``), and the two
speech-in-noise runs (``snr < 100``: ``audiobook_7_5`` at -4 dB and
``audiobook_7_6`` at 2.5 dB), so every kept trial is clean single-talker
speech.  Runs whose Status channel carries no usable pulse train are skipped
(sub-017 ses-shortstories01 run-03: Status is 0 for the first 643 s); runs
whose EEG stops before the stimulus ends are kept up to the end of the EEG
(e.g. sub-048 ses-varyingStories05 run-04, 628 s of a 1154 s story).  ``unatt_audio`` is None.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from coupleflow.preprocessing.core import RawTrial, filter_continuous
from coupleflow import paths  # noqa: E402

RAW = paths.raw("sparrkulee")
STIM = RAW / "stimuli" / "stimuli"
LINE = 50.0
EXCLUDE = re.compile(r"^audiobook_(1|4)(_|$)")
META = {"source": str(RAW), "task": "listeningActive", "fs_native": 1024,
        "alignment": "1 Hz Status pulses vs stimulus trigger track; linear fit, audio re-timed "
                     "onto the EEG clock",
        "excluded": "audiobook_1*, audiobook_4 (KU Leuven overlap); snr < 100 runs"}


def subjects() -> list[str]:
    return sorted(p.name for p in RAW.glob("sub-*") if p.is_dir())


def _npz(path: Path) -> tuple[np.ndarray, int]:
    with gzip.open(path) as f:
        d = np.load(io.BytesIO(f.read()))
        return np.asarray(d["audio"]).ravel(), int(d["fs"])


def _rising(b: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(np.r_[0, b.astype(np.int8)]) > 0)


def _fit(ts: np.ndarray, te: np.ndarray, onset: float | None) -> tuple[float, float, int, float]:
    """Match audio pulse times ``ts`` to EEG pulse times ``te``; return ``(a, b, n, maxres)``."""
    anchors = ([int(np.argmin(np.abs(te - onset)))] if onset is not None else []) \
        + list(range(min(len(te), 60)))
    best = (0, 1.0, 0.0, np.inf)
    for k in anchors:
        a, b = 1.0, te[k] - ts[0]
        for _ in range(3):
            pred = a * ts + b
            j = np.clip(np.searchsorted(te, pred), 1, len(te) - 1)
            j = np.where(np.abs(te[j - 1] - pred) < np.abs(te[j] - pred), j - 1, j)
            ok = np.abs(te[j] - pred) < 0.02
            if ok.sum() < 3:
                break
            a, b = np.polyfit(ts[ok], te[j[ok]], 1)
        res = np.abs(a * ts[ok] + b - te[j[ok]]).max() if ok.sum() >= 3 else np.inf
        if ok.sum() > best[0]:
            best = (int(ok.sum()), float(a), float(b), float(res))
        if onset is not None and best[0] >= 0.95 * len(ts):
            break
    n, a, b, res = best
    return a, b, n, res


def _read_bdf(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, float]:
    import mne
    tmpdir = os.environ.get("TMPDIR", tempfile.gettempdir())
    with tempfile.NamedTemporaryFile(suffix=".bdf", dir=tmpdir) as tmp:
        with gzip.open(path) as f:
            while chunk := f.read(1 << 24):
                tmp.write(chunk)
        tmp.flush()
        raw = mne.io.read_raw_bdf(tmp.name, preload=True, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        names = [c for c in raw.ch_names if not c.startswith("EXG") and c != "Status"]
        eeg = raw.get_data(picks=names)
        status = raw.get_data(picks=["Status"])[0].astype(np.int64) & 0xFFFF
    return eeg, names, status, fs


def read(subject: str) -> list[RawTrial]:
    out = []
    for ev_path in sorted((RAW / subject).glob("ses-*/eeg/*task-listeningActive*_events.tsv")):
        with open(ev_path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if not rows:
            continue
        row = {k.strip(): (v or "").strip() for k, v in rows[0].items()}
        name = Path(row["stim_file"]).name.replace(".npz.gz", "")
        snr = float(row.get("snr") or 100)
        if EXCLUDE.match(name) or snr < 100:
            continue
        bdf = Path(str(ev_path).replace("_events.tsv", "_eeg.bdf.gz"))
        trig_path = STIM / row["trigger_file"]
        if not bdf.exists() or not trig_path.exists():
            print(f"[sparrkulee] {subject} {ev_path.name}: missing bdf/trigger, skipped")
            continue
        run = re.search(r"(ses-[^_]+).*(run-\d+)", ev_path.name)
        tid = f"{subject}_{run.group(1)}_{run.group(2)}"

        eeg, names, status, fs = _read_bdf(bdf)
        vals, cnt = np.unique(status, return_counts=True)
        te = _rising(status != vals[np.argmax(cnt)]) / fs
        trig, fs_t = _npz(trig_path)
        ts = _rising(trig > 0.5 * trig.max()) / fs_t
        onset = float(row["onset"]) if row.get("onset") not in (None, "", "n/a") else None
        a, b, n, res = _fit(ts, te, onset)
        # pulses that fall inside the EEG recording (some recordings stop early)
        pred = a * ts + b
        n_exp = int(((pred >= 0) & (pred <= len(status) / fs - 0.01)).sum())
        if n < 0.9 * n_exp or n_exp < 60 or res > 0.005 or abs(a - 1) > 1e-3:
            print(f"[sparrkulee] {tid}: trigger fit failed (matched {n}/{n_exp}, "
                  f"res {res * 1e3:.1f} ms, a={a:.6f}), skipped", flush=True)
            continue

        x128, good = filter_continuous(eeg, fs, LINE)
        del eeg
        audio, fs_a = _npz(STIM / row["stim_file"])
        s0 = int(np.ceil(b * 128))
        delta = b - s0 / 128.0                    # audio onset rel. to EEG sample s0 (<= 0)
        n_out = int(np.floor((delta + a * len(audio) / fs_a) * fs_a))
        src = (np.arange(n_out) / fs_a - delta) / a * fs_a
        audio = np.interp(src, np.arange(len(audio)), audio, left=0.0, right=0.0)
        s1 = min(x128.shape[1], s0 + int(np.ceil(n_out / fs_a * 128)))
        out.append(RawTrial(subject=subject, trial=tid, eeg=x128[:, s0:s1].copy(), fs=128.0,
                            ch_names=names, att_audio=audio.astype(np.float32),
                            unatt_audio=None, audio_fs=fs_a, attended_ear="", line_freq=LINE,
                            att_name=name, preprocessed=True, good=good,
                            extra={"drift": float(a - 1), "trig_res_ms": float(res * 1e3),
                                   "eeg_onset_s": float(b)}))
        print(f"[sparrkulee] {tid} {name}: a-1={a - 1:.2e} b={b:.3f}s res={res * 1e3:.2f}ms "
              f"n={n}/{len(ts)} T={(s1 - s0) / 128:.0f}s", flush=True)
    return out
