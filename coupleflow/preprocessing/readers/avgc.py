"""KU Leuven AV-GC-AAD (Rotaru et al. 2024): 13 public subjects, 64-ch BioSemi, 128 Hz.

Envelope-only dataset -- build it with this module, not ``build.py``::

    python -m coupleflow.preprocessing.readers.avgc [--workers 8]

then content groups and checks as for the other datasets::

    DATASETS=avgc bash scripts/data/stimulus_groups.sh   # then scripts/data/checks.sh

Release format (``2024-AV-GC-AAD-sub<NN>_preprocessed.mat``): per trial one
10-min recording ``(T, 68)`` = 64 EEG + 4 EOG (EXG3-6) at 128 Hz.  The authors
already down-sampled (8192 -> 256 -> 128 Hz) and band-passed it 1-40 Hz
(Chebyshev II, zero phase), but did NOT re-reference it.

EEG: each recording is one continuous block, so it goes through
``core.filter_continuous`` (bad channels, robust average reference, notch,
0.5-32 Hz) and ``core.finish_eeg`` exactly like every other dataset; the EOG
channels are dropped first.  Subject 16 ships 63 EEG channels (P9 removed by
the authors, header labels match the columns); P9 is zero with mask False.  The authors' 1-40 Hz pre-filter means the
effective pass band is 1-32 Hz here (0.5-32 Hz elsewhere).

Audio: the wavs are withheld (copyright).  Only the authors' broadband
envelopes are released: gammatone subbands, |.|^0.6, each band-passed with the
same 1-40 Hz filter as the EEG, summed, at 128 Hz.  They are put on the 64 Hz
time base with ``core._resample`` (the same polyphase resampler the pipeline
uses) and stored as ``env_att`` / ``env_unatt``.  Consequences:

* ``env_*_sb`` / ``env_*_sb28`` are ABSENT from every trial (no waveform to
  compute subbands from).  Consumers that need banks must skip this dataset.
* the envelope definition differs slightly from ``core.audio_features``
  (sum vs mean of bands, 1-40 Hz band-pass vs none, their gammatone design).
  Scale and DC are therefore not comparable across datasets; z-score per
  window before any cross-dataset loss.
* ``att_name`` / ``unatt_name`` are the (withheld) wav file names.

Attended stream: the talkers swap insert phones at ``switch_times`` (300 s),
but ``attendedEnvelopes`` follows the same talker end to end, and the
alpha-lateralisation check (``paper_review/avgc_spatial_label_check.py``,
59.4 % constant vs 49.3 % flipped) shows the attended EAR label must be held
at ``first_attended_side`` for the whole recording.  Each 10-min recording is
kept as ONE trial (not split at the swap) so a stimulus-grouped split can
never separate its two halves; ``switch_idx`` (64 Hz sample of the swap) and
``condition`` are stored for analyses that need them.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io
import torch

from coupleflow import paths
from coupleflow.preprocessing.core import (FS_OUT, _resample, filter_continuous, finish_eeg,
                                save_bundle)

RAW = paths.raw("avgc")
LINE = 50.0
SUBJECTS = (1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
META = {"source": str(RAW), "envelope_only": True,
        "envelope": "released broadband gammatone envelope (|.|^0.6, 1-40 Hz band-passed, "
                    "summed over bands), 128 -> 64 Hz with core._resample",
        "banks": "absent (no stimulus waveforms released)",
        "attended_ear": "first_attended_side, constant through the 300 s swap",
        "eeg_note": "release pre-filtered 1-40 Hz by the authors; effective band 1-32 Hz"}


def subjects() -> list[str]:
    return [f"S{k}" for k in SUBJECTS]


def read(subject: str) -> list:
    raise NotImplementedError("avgc is envelope-only; run `python -m coupleflow.preprocessing.readers.avgc`")


def process_subject(subject: str) -> list[dict]:
    """Load one subject and run the shared EEG pipeline; envelopes resampled to 64 Hz."""
    torch.set_num_threads(2)
    k = int(subject[1:])
    m = scipy.io.loadmat(str(RAW / f"2024-AV-GC-AAD-sub{k:02d}_preprocessed.mat"),
                         squeeze_me=True, struct_as_record=False)
    fs = float(m["fs"])
    data = np.atleast_1d(m["data"])
    cond = np.atleast_1d(m["conditionID"])
    rand = np.atleast_1d(m["randomization"])
    md = np.atleast_1d(m["metadata"])
    stim = m["stimulus"]
    env_a = np.atleast_1d(stim.attendedEnvelopes)
    env_u = np.atleast_1d(stim.unattendedEnvelopes)
    out = []
    for i in range(len(data)):
        labels = [str(c.Label) for c in np.atleast_1d(md[i].FileHeader.Channels)]
        eeg_idx = [j for j, c in enumerate(labels) if not c.startswith("EXG")]
        # S16 ships 63 EEG channels (P9 removed by the authors); labels match columns
        assert len(labels) == np.asarray(data[i]).shape[1], (subject, i, len(labels))
        names = [labels[j] for j in eeg_idx]
        raw = np.asarray(data[i], np.float64)[:, eeg_idx].T                # (64, T)
        x128, good = filter_continuous(raw, fs, LINE)
        eeg, mask = finish_eeg(x128, good, names, None)
        ea = _resample(np.asarray(env_a[i], np.float64).ravel(), fs, FS_OUT)
        eu = _resample(np.asarray(env_u[i], np.float64).ravel(), fs, FS_OUT)
        T = min(eeg.shape[1], len(ea), len(eu))
        r = rand[i]
        ear = str(r.first_attended_side).upper()[:1]
        assert ear in ("L", "R"), r.first_attended_side
        att, unatt = (r.left, r.right) if ear == "L" else (r.right, r.left)
        sw = np.atleast_1d(np.asarray(r.switch_times, float)).ravel()
        c = str(cond[i])
        out.append({"subject": subject, "trial": f"{subject}_{c}", "attended_ear": ear,
                    "attention_label": {"L": 0, "R": 1}[ear], "channel_mask": mask,
                    "eeg": eeg[:, :T].half(),
                    "env_att": torch.from_numpy(ea[:T].copy()),
                    "env_unatt": torch.from_numpy(eu[:T].copy()),
                    "has_unattended": True, "n_bad": int((~good).sum()),
                    "att_name": str(att), "unatt_name": str(unatt), "modality": "eeg",
                    "condition": c.rstrip("12"),
                    "switch_idx": int(round(sw[0] * FS_OUT)) if sw.size else -1})
    print(f"[avgc] {subject}: {len(out)} trials", flush=True)
    return out


def build_envelope_only(workers: int = 8) -> Path:
    """Build ``<CACHE_ROOT>/avgc/unified_env64.pt`` (no sb / sb28 banks)."""
    trials: list[dict] = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(process_subject, subjects()):
            trials += r
    return save_bundle("avgc", trials, META)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    build_envelope_only(ap.parse_args().workers)
