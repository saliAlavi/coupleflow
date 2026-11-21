#!/usr/bin/env python
"""Per-trial gaze and head-orientation features of the MAESTRO release (rebuttal E7).

Inputs (release, ``paths.raw('maestro')``):
  data/gaze/subject=S/trial=T.parquet   Tobii Glasses 3, ~50 Hz, recording-relative ``t``;
                                         ``gaze3d_{x,y,z}`` = gaze point in the glasses' (head)
                                         frame, mm.  The eye tracker is uncalibrated (README).
  data/imu/subject=S/trial=T.parquet    same clock, ~120 Hz; gyroscope ``gx,gy,gz`` (deg/s);
                                         gravity is on +y, so ``gy`` is the yaw rate.
  metadata/trials.csv                    attended speaker (1-4) per trial id.

Per trial, over the audio-live interval (gaze clock [ONSET, ONSET + 30 s]; the sound starts
~1.09 s after the Tobii recording start, coupleflow/preprocessing/readers/maestro.py):
  eye_az, eye_el    mean eye-in-head azimuth / elevation (deg) of the gaze3d vector
  head_yaw          mean head yaw (deg) relative to the trial's first sample: cumulative
                    integral of gy minus the listener's median gy (gyro bias; integrating the raw
                    rate drifts by ~100 deg per trial)
  world_az          mean of eye_az + S * head_yaw sample-wise, S = +-1 chosen to minimise the
                    pooled within-trial variance (the sign convention between the two devices
                    is not documented; eye and head movements compensate while fixating)
  valid_frac        fraction of gaze samples with a gaze vector
Per sample (gaze rate) the world azimuth is kept for the gaze-on-target fraction (analysis.py).

    python scripts/rebuttal/e7/gaze_features.py --out /fs/scratch/PAS2301/alialavi/rebuttal/e7
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from coupleflow import paths  # noqa: E402

ONSET, DUR = 1.09, 30.0


def load_trial(raw: Path, s: str, t: str):
    g = raw / "data" / "gaze" / f"subject={s}" / f"trial={t}.parquet"
    m = raw / "data" / "imu" / f"subject={s}" / f"trial={t}.parquet"
    if not (g.exists() and m.exists()):
        return None
    g = pd.read_parquet(g, columns=["t", "gaze3d_x", "gaze3d_y", "gaze3d_z"])
    m = pd.read_parquet(m, columns=["t", "gy"])
    g = g[(g.t >= ONSET) & (g.t < ONSET + DUR)]
    m = m[(m.t >= ONSET) & (m.t < ONSET + DUR)]
    if len(g) < 100 or len(m) < 100:
        return None
    az = np.degrees(np.arctan2(g.gaze3d_x.values, g.gaze3d_z.values))
    el = np.degrees(np.arctan2(g.gaze3d_y.values, np.hypot(g.gaze3d_x.values, g.gaze3d_z.values)))
    return {"tg": g.t.values, "az": az, "el": el, "ti": m.t.values, "gy": m.gy.values}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = paths.raw("maestro")
    trials = pd.read_csv(raw / "metadata" / "trials.csv").set_index("trial_id")
    ev = [t for t in trials.index if t.startswith("eval_")]
    subs = [f"S{i:02d}" for i in range(1, 17)]
    rec = {}
    for s in subs:
        for t in ev:
            x = load_trial(raw, s, t)
            if x is not None:
                rec[(s, t)] = x
        print(s, sum(k[0] == s for k in rec), flush=True)
    # per-listener gyro bias, then head yaw relative to the trial start, on the gaze clock
    for s in subs:
        ks = [k for k in rec if k[0] == s]
        bias = float(np.nanmedian(np.concatenate([rec[k]["gy"] for k in ks])))
        for k in ks:
            r = rec[k]
            dt = np.diff(r["ti"], prepend=r["ti"][0])
            rate = np.nan_to_num(r["gy"] - bias, nan=0.0)          # IMU dropouts: no rotation
            yaw = np.cumsum(rate * dt)
            r["imu_nan"] = float(np.isnan(r["gy"]).mean())
            r["yaw"] = np.interp(r["tg"], r["ti"], yaw)
            r["bias"] = bias
    # sign between eye azimuth and head yaw: minimise pooled within-trial variance
    var = {}
    for S in (+1, -1):
        v = [np.nanvar(r["az"] + S * r["yaw"]) for r in rec.values()]
        var[S] = float(np.nanmean(v))
    S = min(var, key=var.get)
    rows = []
    for (s, t), r in rec.items():
        r["world"] = r["az"] + S * r["yaw"]
        ok = np.isfinite(r["az"])
        rows.append({"subject": s, "trial_id": t, "attended": int(trials.loc[t, "attended_speaker"]),
                     "eye_az": float(np.nanmean(r["az"])), "eye_el": float(np.nanmean(r["el"])),
                     "eye_az_sd": float(np.nanstd(r["az"])),
                     "head_yaw": float(np.mean(r["yaw"])), "world_az": float(np.nanmean(r["world"])),
                     "valid_frac": float(ok.mean()), "gyro_bias": r["bias"],
                     "imu_nan_frac": r["imu_nan"]})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "gaze_features.csv", index=False)
    with open(out / "gaze_samples.pkl", "wb") as f:
        pickle.dump({k: {"world": r["world"].astype(np.float32), "az": r["az"].astype(np.float32),
                         "yaw": r["yaw"].astype(np.float32)} for k, r in rec.items()}, f)
    (out / "gaze_meta.txt").write_text(
        f"trials with gaze+imu: {len(rec)} of {len(subs) * len(ev)}\n"
        f"sign S (world = eye_az + S*head_yaw): {S}; pooled within-trial variance {var}\n"
        f"audio-live interval on the gaze clock: [{ONSET}, {ONSET + DUR}) s\n")
    print((out / "gaze_meta.txt").read_text())


if __name__ == "__main__":
    main()
