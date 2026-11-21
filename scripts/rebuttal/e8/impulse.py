#!/usr/bin/env python
"""Temporal spread of the zero-phase preprocessing chains (E8, rebuttal).

EEG chain (coupleflow/preprocessing/core.py, steps 3-5 and 7): resample native -> 128 Hz
(resample_poly), notch filtfilt at the line frequency (+ harmonic below Nyquist), 0.5-32 Hz
4th-order Butterworth sosfiltfilt, resample 128 -> 64 Hz.  Run on a unit impulse placed on the
64-Hz output grid, and on a causal TRF-like response (P1/N1/P2 bumps at 50/100/200 ms), using
core.filter_continuous and core._resample themselves.

Audio envelope chain (core.audio_features): causal gammatone FIR (no pre-response), |.|^0.6,
resample_poly 16 kHz -> 64 Hz (linear-phase anti-alias FIR, the only acausal step).

Reports, per native rate, the fraction of output energy at tau <= -125/-250/-500/-1000 ms (EEG
moved earlier, i.e. into a lag-control window that ends 125 ms before the sound) and the
symmetric 99%-energy half-width.

    python scripts/rebuttal/e8/impulse.py [--out /fs/scratch/.../impulse.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from coupleflow.preprocessing import core  # noqa: E402

RATES = {"kuleuven": (128.0, 50.0), "dtu": (512.0, 50.0), "sparrkulee": (8192.0, 50.0),
         "megmasc": (1000.0, 60.0), "maestro": (500.0, 60.0)}
CUTS_MS = (125, 250, 500, 1000)


def stats(h: np.ndarray, k0: float, fs: float) -> dict:
    """``h``: (n_offsets, T) responses; ``k0``: (n_offsets,) impulse times in output samples."""
    h = np.atleast_2d(h).astype(np.float64)
    k0 = np.atleast_1d(np.asarray(k0, np.float64))
    tau = (np.arange(h.shape[1])[None, :] - k0[:, None]) / fs
    h, tau = h.ravel(), tau.ravel()
    e = h ** 2
    tot = e.sum()
    out = {f"frac_energy_tau_le_-{c}ms": float(e[tau <= -c / 1000 + 1e-9].sum() / tot) for c in CUTS_MS}
    out.update({f"frac_energy_tau_ge_+{c}ms": float(e[tau >= c / 1000 - 1e-9].sum() / tot) for c in CUTS_MS})
    peak = np.abs(h).max()
    out.update({f"max_abs_tau_le_-{c}ms_rel_peak": float(np.abs(h[tau <= -c / 1000 + 1e-9]).max() / peak)
                for c in CUTS_MS})
    order = np.argsort(np.abs(tau))
    cum = np.cumsum(e[order]) / tot
    out["halfwidth99_ms"] = float(1000 * np.abs(tau[order][np.searchsorted(cum, 0.99)]))
    return out


def eeg_chain(x: np.ndarray, fs: float, line: float) -> np.ndarray:
    x2 = np.stack([x, np.zeros_like(x)])
    y128, _ = core.filter_continuous(x2, fs, line, modality="meg", good=np.array([True, True]))
    return core._resample(y128, core.FS_MID, core.FS_OUT)[0].astype(np.float64)


def trf_kernel(fs: float) -> np.ndarray:
    t = np.arange(int(0.6 * fs)) / fs
    g = lambda m, s: np.exp(-0.5 * ((t - m) / s) ** 2)  # noqa: E731
    return 1.0 * g(0.05, 0.015) - 1.5 * g(0.10, 0.025) + 1.0 * g(0.20, 0.04)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/fs/scratch/PAS2301/alialavi/rebuttal/e8/impulse.json")
    a = ap.parse_args()
    res: dict = {"eeg": {}, "audio": {}}
    T, t0 = 60.0, 30.0
    for ds, (fs, line) in RATES.items():
        n = int(T * fs)
        i0 = int(round(t0 * fs))
        k0 = int(round(t0 * core.FS_OUT))
        step = fs / core.FS_OUT                     # native samples per output sample
        offs = sorted({int(round(j * step / 8)) for j in range(8)})   # sub-output-sample positions
        hs, ks = [], []
        for o in offs:
            x = np.zeros(n)
            x[i0 + o] = 1.0
            hs.append(eeg_chain(x, fs, line))
            ks.append(k0 + o / step)
        r = {"native_fs": fs, "impulse_offsets": len(offs), "impulse": stats(np.stack(hs), ks, core.FS_OUT)}
        # causal TRF-like response starting at tau = 0: any energy before 0 is the chain's doing
        x = np.zeros(n)
        k = trf_kernel(fs)
        x[i0:i0 + len(k)] = k
        ht = eeg_chain(x, fs, line)
        # reference: the same response sampled at 64 Hz without filtering
        tau = (np.arange(len(ht)) - k0) / core.FS_OUT
        r["trf"] = stats(ht, k0, core.FS_OUT)
        r["trf"]["frac_energy_tau_lt_0"] = float((ht[tau < 0] ** 2).sum() / (ht ** 2).sum())
        res["eeg"][ds] = r
    # audio envelope: |.|^0.6 output resampled 16 kHz -> 64 Hz (core.audio_features)
    # resample_poly(x, 1, dec) filters with firwin(2*10*dec+1, 1/dec, window=("kaiser", 5.0))
    # (scipy default), centred: env[t] mixes rectified audio from t-156 ms to t+156 ms.
    from scipy.signal import firwin
    fsa, dec = core.FS_AUDIO, core.FS_AUDIO // core.FS_OUT
    h = firwin(2 * 10 * dec + 1, 1.0 / dec, window=("kaiser", 5.0))
    tau = (np.arange(len(h)) - 10 * dec) / fsa                  # seconds; >0 = audio after t
    e = h ** 2
    res["audio"] = {"anti_alias_fir_halfwidth_ms": 1000 * 10 * dec / fsa,
                    "frac_energy_future_any": float(e[tau > 0].sum() / e.sum()),
                    "frac_energy_future_ge_62.5ms": float(e[tau >= 0.0625].sum() / e.sum()),
                    "frac_energy_future_ge_125ms": float(e[tau >= 0.125].sum() / e.sum()),
                    "max_abs_future_ge_125ms_rel_peak": float(np.abs(h[tau >= 0.125]).max() / np.abs(h).max()),
                    "note": "gammatone bank is causal; this FIR is the only acausal audio step"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
