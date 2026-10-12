"""MAESTRO (OSU): 16 subjects, 32-ch ANT Neuro at 500 Hz, 6 loudspeakers, Columbus (60 Hz).

Six-talker dataset -- build it with this module, not ``build.py``::

    python -m coupleflow.preprocessing.readers.maestro [--workers 16]

then content groups and checks as for the other datasets::

    DATASETS=maestro bash scripts/data/stimulus_groups.sh   # then scripts/data/checks.sh

Release (``paths.raw('maestro')``): per subject 5
familiarisation + 100 evaluation trials of ~30 s.  Only ``eval_*`` trials go
into the bundle.  Each trial plays 3 stereo devices = 6 talkers (speaker k on
device ceil(k/2), L/R); speakers 1-4 are attendable, 5-6 are distractors.
The audio is identical for every subject (one mono FLAC per speaker).

Alignment (``media/timing/subject=S/trial=T.json``): EEG sample ``i`` is at
``eeg.first_sample_unix + i / 500`` (``t_sec`` is contiguous at 500 Hz and
starts at ``t0_internal_sec``).  The release also gives one software
``playback_start_unix`` per device; they differ by 80-250 ms, and both the old
upstream loader (fixed raw-id map, anchored at the earliest device) and the
authors' ``windows.build_trial`` (list order = Device-1,2,3, each device sliced
from its own start) trust them.  They are NOT acoustic onsets:

* Tobii scene-mic check (label-free; GCC-PHAT of every speaker FLAC against the
  glasses' microphone, 320 trials): the three devices start within ~12 ms of
  each other (median |residual|, same as the L/R multipath jitter of one
  device), and the reported inter-device offsets have slope 0.02 against the
  acoustic ones (r = 0.04) under either device mapping.  Per-device slicing
  (``align=order`` / ``id``) therefore adds 80-250 ms of jitter.
* The sound starts ~1 s AFTER the software timestamps: mic onset mapped to
  PC time (recording t -> glasses clock by exact gaze2d matches with the live
  gaze stream -> PC clock by the session-median gaze receive offset) is
  ``min(playback_start) + 1.09 s`` (within-subject IQR ~50 ms, the same for
  min/max/mean start, i.e. per-trial jitter is small).  The EEG agrees: the
  backward-model offset scan on ``align=common`` peaks at +0.78 s in every
  subject (0.69-1.03 s, 16/16), r 0.012 -> 0.075.  The mic chain cannot give
  the latency on the EEG clock (the EEG and gaze first-sample receive stamps
  have unknown relative bias), so the ONE global constant ``LATENCY`` = 0.78125 s
  (50 samples at 64 Hz) is taken from the EEG offset scan; no per-trial or
  per-subject correction is applied.

So every speaker is placed at ``min(playback_start) + LATENCY`` and the trial
is cut to the window where EEG and audio are live:
``[max(onset, eeg_start), min(onset + 30 s, eeg_end))`` (~29.4 s).

EEG: the whole trial recording goes through ``core.filter_continuous``
(robust average reference, 128 Hz, 60 Hz notch, 0.5-32 Hz) and the 128-Hz
signal is cut to the live window before ``core.finish_eeg`` (canonical 64
channels, 64 Hz, per-trial z-score).  M1/M2 are not in the canonical layout
and are dropped by the mapping (they still enter the average reference when
good).  Good channels = not listed in ``metadata/bad_channels.csv`` AND not
flat AND not clipped at the ADC rail (+-0.0839) in >= 1 % of samples
(mastoids saturate in some trials the csv misses, which would poison the
reference).

Audio: ``core.audio_features`` on every speaker's aligned 16 kHz waveform.
Per trial: ``env_att`` = attended speaker, ``env_unatt`` = the other talker
on the attended speaker's device (1<->2, 3<->4), plus ``env_cands`` (6, T)
etc. for all six speakers and ``att_idx`` (0-based).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from coupleflow import paths  # noqa: E402
from coupleflow.preprocessing import core  # noqa: E402

RAW = paths.raw("maestro")
DATASET = "maestro"
LINE = 60.0
FS = 500.0
ADC_CLIP = 0.0839
RAW_TO_LOGICAL = {6: 1, 5: 2, 3: 3}          # old upstream loader's device map (diagnostic)
LATENCY = 50 / 64                            # s, global playback latency (EEG offset scan)
PARTNER = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5}
META = {"source": str(RAW), "line_freq": LINE, "trials": "eval_001..100 only",
        "alignment": "all speakers at min(playback_start_unix) + latency (acoustic onsets are "
                     "simultaneous per scene mic; software stamps lead the sound by ~1 s)",
        "env_unatt": "other talker on the attended speaker's device",
        "env_cands": "speakers 1..6 (5-6 distractor only); att_idx 0-based",
        "good_channels": "not in bad_channels.csv, not flat, <1% samples at ADC clip"}


def subjects() -> list[str]:
    return [f"S{k:02d}" for k in range(1, 17)]


def read(subject: str) -> list:
    raise NotImplementedError("6 talkers; run `python -m coupleflow.preprocessing.readers.maestro`")


def _meta() -> tuple[list[str], dict[str, set[str]], pd.DataFrame, dict]:
    ch = json.loads((RAW / "metadata/eeg_channels.json").read_text())["channels"]
    bc = pd.read_csv(RAW / "metadata/bad_channels.csv").fillna("")
    bad = {r.subject_id: {c for c in str(r.bad_channels).split(";") if c} for r in bc.itertuples()}
    trials = pd.read_csv(RAW / "metadata/trials.csv").set_index("trial_id")
    manifest = json.loads((RAW / "metadata/audio_manifest.json").read_text())
    return ch, bad, trials, manifest


def device_starts(timing: dict, align: str, latency: float = LATENCY) -> dict[int, float]:
    """Unix time at which logical device 1/2/3 starts its FLAC (EEG clock).

    ``common`` (default): all devices at ``min(playback_start) + latency``.  Diagnostics
    (disproven by the scene mic, see module doc): ``order`` = device i is the i-th entry of
    ``audio.devices`` (authors' ``windows.build_trial``), ``id`` = fixed raw-id map
    ``RAW_TO_LOGICAL`` (old upstream loader); each device at its own start + ``latency``.
    """
    devs = timing["audio"]["devices"]
    ps = [float(d["playback_start_unix"]) for d in devs]
    if align == "order":
        return {i + 1: p + latency for i, p in enumerate(ps)}
    if align == "id":
        return {RAW_TO_LOGICAL[d["device_id_raw"]]: p + latency for d, p in zip(devs, ps)}
    if align == "common":
        return {i + 1: min(ps) + latency for i in range(len(ps))}
    raise ValueError(align)


def process_trial(subject: str, tid: str, ch: list[str], bad: set[str], row: pd.Series,
                  files: dict, align: str = "common", latency: float = LATENCY) -> dict:
    """Align, preprocess and featurise one (subject, eval trial)."""
    timing = json.loads((RAW / f"media/timing/subject={subject}/trial={tid}.json").read_text())
    df = pd.read_parquet(RAW / f"data/eeg/subject={subject}/trial={tid}.parquet")
    x = df[[f"ch_{c}" for c in ch]].to_numpy(np.float64).T                # (32, T) @ 500 Hz
    assert np.all(np.diff(df["sample_idx"].to_numpy()) == 1), (subject, tid)
    eeg_t0 = timing["eeg"]["first_sample_unix"] + (
        float(df["t_sec"].iloc[0]) - timing["eeg"]["t0_internal_sec"])
    eeg_end = eeg_t0 + x.shape[1] / FS

    dev_t0 = device_starts(timing, align, latency)
    assert sorted(dev_t0) == [1, 2, 3], (subject, tid, dev_t0)
    wavs, dur = {}, []
    for k in range(1, 7):
        w, sr = sf.read(str(RAW / "media/audio" / tid / files[str(k)]), dtype="float64")
        assert sr == core.FS_AUDIO and w.ndim == 1, (tid, k, sr, w.shape)
        wavs[k] = w
        dur.append(dev_t0[(k + 1) // 2] + len(w) / sr)
    start = max(max(dev_t0.values()), eeg_t0)
    end = min(min(dur), eeg_end)

    good = np.array([c not in bad for c in ch])
    good &= x.std(axis=1) > 1e-9
    good &= (np.abs(x) >= ADC_CLIP - 1e-4).mean(axis=1) < 0.01
    x128, good = core.filter_continuous(x, FS, LINE, "eeg", good)
    i0 = int(round((start - eeg_t0) * core.FS_MID))
    n128 = int(np.floor((end - start) * core.FS_MID))
    eeg, mask = core.finish_eeg(x128[:, i0:i0 + n128], good, ch, None)

    n16 = int(np.floor((end - start) * core.FS_AUDIO))
    feats = []
    for k in range(1, 7):
        a0 = int(round((start - dev_t0[(k + 1) // 2]) * core.FS_AUDIO))
        seg = wavs[k][a0:a0 + n16]
        assert a0 >= 0 and len(seg) == n16, (subject, tid, k, a0, len(seg), n16)
        feats.append(core.audio_features(seg, core.FS_AUDIO))
    T = min(eeg.shape[1], *(f["env"].shape[-1] for f in feats))
    cands = np.stack([f["env"][:T] for f in feats]).astype(np.float32)
    cands_sb = np.stack([f["sb"][:, :T] for f in feats])
    cands_sb28 = np.stack([f["sb28"][:, :T] for f in feats])

    att = int(row["attended_speaker"])
    un = PARTNER[att]
    ear = "L" if str(row["hemisphere_attended"]).upper().startswith("L") else "R"
    spk_ids = [files[str(k)].split("spkid")[-1].split(".")[0] for k in range(1, 7)]
    out = {"subject": subject, "trial": f"{subject}_{tid}", "attended_ear": ear,
           "attention_label": {"L": 0, "R": 1}[ear], "channel_mask": mask,
           "eeg": eeg[:, :T].half(),
           "env_att": torch.from_numpy(cands[att - 1].copy()),
           "env_unatt": torch.from_numpy(cands[un - 1].copy()),
           "has_unattended": True, "n_bad": int((~good).sum()),
           "att_name": files[str(att)], "unatt_name": files[str(un)], "modality": "eeg",
           "env_cands": torch.from_numpy(cands),
           "env_cands_sb": torch.from_numpy(cands_sb).half(),
           "env_cands_sb28": torch.from_numpy(cands_sb28).half(),
           "att_idx": att - 1, "unatt_idx": un - 1, "speaker_ids": spk_ids,
           "stimulus": tid, "snr_db": float(row["snr_db"]), "hemisphere": ear,
           "inout": str(row["inout_attended"]),
           "align": {"start_unix": start, "dur_s": end - start,
                     "dev_offset_s": [dev_t0[d] - start for d in (1, 2, 3)]}}
    for b in ("sb", "sb28"):
        out[f"env_att_{b}"] = torch.from_numpy(cands_sb[att - 1] if b == "sb"
                                               else cands_sb28[att - 1]).half()
        out[f"env_unatt_{b}"] = torch.from_numpy(cands_sb[un - 1] if b == "sb"
                                                 else cands_sb28[un - 1]).half()
    return out


def process_subject(args: tuple) -> list[dict]:
    """All eval trials of one subject."""
    subject, align, latency, limit = args
    torch.set_num_threads(2)
    ch, bad, trials, manifest = _meta()
    tids = [t for t in trials.index if t.startswith("eval_")][:limit]
    out = [process_trial(subject, t, ch, bad.get(subject, set()), trials.loc[t], manifest[t],
                         align, latency) for t in tids]
    print(f"[{DATASET}] {subject}: {len(out)} trials, mean ch "
          f"{np.mean([int(t['channel_mask'].sum()) for t in out]):.1f}", flush=True)
    return out


def build(workers: int = 16, subs: list[str] | None = None,
          align: str = "common", latency: float = LATENCY, name: str = core.BUNDLE,
          limit: int | None = None) -> Path:
    """Build ``<CACHE_ROOT>/maestro/<name>``."""
    subs = subs or subjects()
    trials: list[dict] = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(process_subject, [(s, align, latency, limit) for s in subs]):
            trials += r
    meta = {**META, "align": align, "latency_s": latency}
    return core.save_bundle(DATASET, trials, meta, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N eval trials per subject")
    ap.add_argument("--align", default="common", choices=["common", "order", "id"])
    ap.add_argument("--latency", type=float, default=LATENCY, help="s (diagnostics only)")
    ap.add_argument("--name", default=core.BUNDLE)
    a = ap.parse_args()
    build(a.workers, a.subjects, a.align, a.latency, a.name, a.limit)
