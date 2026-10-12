"""Fuglsang et al. 2020 (J Neurosci), "Effects of sensorineural hearing loss on
cortical synchronization to competing speech" -- Zenodo 3618205, BIDS
``ds-eeg-snhl``: 44 Danish listeners (22 NH, 22 HI), 64-ch BioSemi at 512 Hz
(+ EXG1-2 EOG, EXG3-8 in-ear; dropped), line 50 Hz.

Envelope-only dataset -- build it with this module, not ``build.py``::

    python -m coupleflow.preprocessing.readers.fuglsang2020 [--workers 8]

then content groups and checks as for the other datasets::

    DATASETS=fuglsang2020 bash scripts/data/stimulus_groups.sh   # then scripts/data/checks.sh

Task ``selectiveattention``: 48 trials (~50 s) per subject, 16 single-talker
and 32 two-talker (male + female talker at +-90 deg).  Only two-talker trials
are kept.  ``events.tsv`` rows per trial:

* ``targetonset`` -- attended talker starts here (``attend_left_right`` = side).
* ``maskeronset`` -- competing talker starts 3-5 s *later*.
* ``trialend``    -- end of the trial.

EEG: the whole recording goes through ``core.filter_continuous`` (bad
channels, robust average reference, notch, 0.5-32 Hz), trials are cut
[targetonset, trialend) and finished with ``core.finish_eeg`` -- identical to
every other dataset.

Audio: the public release has NO wavs (``stimuli/`` is empty; available on
request from the authors).  Only the authors' broadband envelopes are shipped
in ``derivatives/stimuli/<sub>/{target,masker}/*.mat`` (``dat.feat/fs/t``):
24 gammatone bands 100-4000 Hz, |.|^0.3, averaged, 512 Hz.  ``t`` is relative
to target onset, so the target starts at 0 and the masker is zero-padded by
``t[0]`` (its delay).  HI listeners heard CamEQ-amplified audio; we use the
unamplified ``*woa`` envelopes (``stim_file_without_cambridge_eq``) so every
listener shares one envelope per stimulus.  Envelopes are put on 64 Hz with
``core._resample``.  Consequences:

* ``env_*_sb`` / ``env_*_sb28`` are ABSENT (no waveform to compute banks from).
* envelope definition differs from ``core.audio_features`` (24 bands, ^0.3,
  100-4000 Hz); z-score per window before any cross-dataset loss.

Per-trial extras: ``group`` (NH/HI from ``participants.tsv``), ``attend_sex``,
``masker_delay_s``.  Stimulus files are numbered by trial *per subject*, so
content groups must come from ``coupleflow/preprocessing/stimulus_groups.py``.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np
import scipy.io
import torch

from coupleflow.preprocessing.channels import STANDARD_64_CHANNELS
from coupleflow.preprocessing.core import FS_MID, FS_OUT, _resample, filter_continuous, finish_eeg, save_bundle
from coupleflow import paths  # noqa: E402

DATASET = "fuglsang2020"
RAW = paths.raw("fuglsang2020")
ENV = RAW / "derivatives" / "stimuli"
LINE = 50.0
TASK = "selectiveattention"
META = {"source": "Zenodo 3618205 ds-eeg-snhl", "envelope_only": True,
        "envelope_source": "provided",
        "envelope": "24 gammatone 100-4000 Hz ^0.3 broadband",
        "envelope_detail": "derivatives *woa (no CamEQ) envelopes, 512 -> 64 Hz core._resample",
        "banks": "absent (no stimulus waveforms released)",
        "trials": "two-talker only, targetonset -> trialend",
        "group": "participants.tsv hearing_status (NH/HI)"}


def subjects() -> list[str]:
    return sorted(p.name for p in RAW.glob("sub-*")
                  if (p / "eeg" / f"{p.name}_task-{TASK}_eeg.bdf").exists())


def read(subject: str) -> list:
    raise NotImplementedError(
        "fuglsang2020 is envelope-only; run `python -m coupleflow.preprocessing.readers.fuglsang2020`")


def _groups() -> dict[str, str]:
    """``{sub-XXX: 'NH'|'HI'}`` from ``participants.tsv`` (``hearing_status`` nh/hi).

    Without the raw release, reads ``<data>/fuglsang2020/hearing_status.tsv``, the two columns
    of ``participants.tsv`` this needs (shipped with the repository; CC BY 4.0, Fuglsang et al.).
    """
    src = RAW / "participants.tsv"
    if not src.exists():
        src = paths.DATA / "fuglsang2020" / "hearing_status.tsv"
    with open(src) as f:
        return {r["participant_id"]: r["hearing_status"].strip().upper()
                for r in csv.DictReader(f, delimiter="\t")}


def _envelope(rel: str) -> tuple[np.ndarray, float, float]:
    """Shipped envelope for ``stim_file`` ``rel`` -> ``(feat, fs, t0)``."""
    d = scipy.io.loadmat(str(ENV / rel.replace(".wav", ".mat")), squeeze_me=True)["dat"]
    t = np.atleast_1d(d["t"].item())
    return np.asarray(d["feat"].item(), np.float64).ravel(), float(d["fs"].item()), float(t[0])


def process_subject(subject: str) -> list[dict]:
    """Shared EEG pipeline on one subject; shipped envelopes resampled to 64 Hz."""
    torch.set_num_threads(2)
    d = RAW / subject / "eeg"
    raw = mne.io.read_raw_bdf(str(d / f"{subject}_task-{TASK}_eeg.bdf"), verbose=False)
    names = [c for c in raw.ch_names if c in STANDARD_64_CHANNELS or c in ("P9", "P10")]
    x128, good = filter_continuous(raw.get_data(picks=names), float(raw.info["sfreq"]), LINE)
    del raw
    with open(d / f"{subject}_task-{TASK}_events.tsv") as f:
        ev = [r for r in csv.DictReader(f, delimiter="\t")
              if r["trigger_type"] in ("targetonset", "maskeronset", "trialend")]
    group = _groups()[subject]
    out = []
    for i, r in enumerate(ev):
        if r["trigger_type"] != "targetonset" or r["single_talker_two_talker"] != "twotalker":
            continue
        m, end = ev[i + 1], ev[i + 2]
        assert m["trigger_type"] == "maskeronset" and end["trigger_type"] == "trialend", (
            subject, r["onset"])
        att_rel = r["stim_file_without_cambridge_eq"]
        un_rel = m["stim_file_without_cambridge_eq"]
        ea, fa, ta0 = _envelope(att_rel)
        eu, fu, tu0 = _envelope(un_rel)
        assert fa == fu and abs(ta0) < 1e-9, (subject, att_rel, ta0)
        delay_ev = float(m["onset"]) - float(r["onset"])
        assert abs(tu0 - delay_ev) < 0.01, (subject, un_rel, tu0, delay_ev)
        eu = np.concatenate([np.zeros(int(round(tu0 * fu))), eu])
        eu = np.pad(eu, (0, max(0, len(ea) - len(eu))))[: len(ea)]
        ea64, eu64 = _resample(ea, fa, FS_OUT), _resample(eu, fu, FS_OUT)
        a = int(round(float(r["onset"]) * FS_MID))
        b = min(int(round(float(end["onset"]) * FS_MID)), x128.shape[1])
        eeg, mask = finish_eeg(x128[:, a:b], good, names, None)
        T = min(eeg.shape[1], len(ea64), len(eu64))
        ear = r["attend_left_right"].replace("attend", "")[:1].upper()
        assert ear in ("L", "R"), r["attend_left_right"]
        tid = Path(att_rel).stem.replace("woa", "")
        out.append({"subject": subject, "trial": f"{subject}_{tid}",
                    "attended_ear": ear, "attention_label": {"L": 0, "R": 1}[ear],
                    "channel_mask": mask, "eeg": eeg[:, :T].half(),
                    "env_att": torch.from_numpy(ea64[:T].copy()),
                    "env_unatt": torch.from_numpy(eu64[:T].copy()),
                    "has_unattended": True, "n_bad": int((~good).sum()),
                    "att_name": att_rel, "unatt_name": un_rel, "modality": "eeg",
                    "group": group,
                    "attend_sex": r["attend_male_female"].replace("attend", ""),
                    "masker_delay_s": tu0})
    print(f"[{DATASET}] {subject} ({group}): {len(out)} trials", flush=True)
    return out


def build_envelope_only(workers: int = 8, limit: int | None = None) -> Path:
    """Build ``<CACHE_ROOT>/fuglsang2020/unified_env64.pt`` (no sb / sb28 banks)."""
    trials: list[dict] = []
    subs = subjects()[:limit]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(process_subject, subs):
            trials += r
    name = "unified_env64.pt" if limit is None else "unified_env64.smoke.pt"
    return save_bundle(DATASET, trials, META, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build_envelope_only(a.workers, a.limit)
