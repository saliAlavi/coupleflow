#!/usr/bin/env python
"""Rebuttal E7: does gaze or head orientation explain CoupleFlow's MAESTRO decisions?

Inputs (in --dir): gaze_features.csv + gaze_samples.pkl (gaze_features.py) and
trial_preds.csv (trial_preds.py: per-trial 4-way decisions of maestro_within_lag0, seeds 0-2,
latency-corrected timing, matched candidates, 5-s windows summed per trial).

1. Gaze/head alone.  Per listener, stratified 5-fold CV (fixed seed) of a standardised
   multinomial logistic regression on per-trial features -> attended speaker (4-way, chance 25%)
   and attended hemisphere (chance 50%).  Feature sets: eye (eye_az, eye_el, eye_az_sd),
   head (head_yaw), world (world_az), all.
2. Per-trial gaze measures, each with an attended-talker score (on_att) and a picked talker:
   eye_samples / world_samples: per listener, leave-one-trial-out centroids of the per-trial mean
   eye-in-head (or eye + head) azimuth for each attended speaker; every gaze sample is assigned
   to the nearest centroid; on_att = fraction assigned to the attended speaker, pick = speaker
   with the largest fraction.  classifier_eye / classifier_all: the CV posterior of the
   attended speaker and the CV prediction of step 1.  No trial's own label enters its measure.
3. CoupleFlow vs each measure.  Per-trial correctness (3 seeds) by within-listener terciles,
   Spearman of the seed-mean trial margin, a logistic regression correct ~ z(on_att) +
   C(subject) + C(seed) with trial-clustered s.e., and CoupleFlow accuracy on trials whose gaze
   pick is not the attended talker.

    python scripts/rebuttal/e7/analysis.py --dir /fs/scratch/PAS2301/alialavi/rebuttal/e7
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATS = {"eye": ["eye_az", "eye_el", "eye_az_sd"], "head": ["head_yaw"], "world": ["world_az"],
         "all": ["eye_az", "eye_el", "eye_az_sd", "head_yaw", "world_az"]}


def gaze_only(df: pd.DataFrame) -> dict:
    out = {}
    for name, cols in FEATS.items():
        acc4, acc2, per = [], [], {}
        for s, g in df.groupby("subject"):
            g = g.dropna(subset=cols)
            X = g[cols].values
            y4 = g["attended"].values
            y2 = (y4 >= 3).astype(int)
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            p4 = cross_val_predict(clf, X, y4, cv=cv)
            p2 = cross_val_predict(clf, X, y2, cv=cv)
            acc4 += list(p4 == y4)
            acc2 += list(p2 == y2)
            per[s] = float(np.mean(p4 == y4))
        v = np.array(list(per.values()))
        out[name] = {"speaker4": float(np.mean(acc4)), "hemisphere": float(np.mean(acc2)),
                     "speaker4_per_listener_min_median_max":
                         [float(v.min()), float(np.median(v)), float(v.max())],
                     "n_listeners_speaker4_ge_50pct": int((v >= 0.5).sum()), "n_trials": len(acc4)}
    return out


def gaze_on_target(df: pd.DataFrame, samples: dict, col: str, key: str) -> pd.DataFrame:
    """LOTO per-listener centroids of per-trial mean ``col``; per-sample nearest centroid."""
    rows = []
    for s, g in df.groupby("subject"):
        g = g.set_index("trial_id")
        for t in g.index:
            other = g.drop(index=t)
            cent = other.groupby("attended")[col].mean()             # LOTO centroids
            w = samples[(s, t)][key]
            w = w[np.isfinite(w)]
            if len(w) == 0 or len(cent) < 4 or not np.isfinite(cent.values).all():
                continue
            k = cent.index.values[np.abs(w[:, None] - cent.values[None, :]).argmin(1)]
            frac = {int(c): float(np.mean(k == c)) for c in cent.index}
            att = int(g.loc[t, "attended"])
            rows.append({"subject": s, "trial_id": t, "on_att": frac[att],
                         "pick": max(frac, key=frac.get), "attended": att})
    return pd.DataFrame(rows)


def classifier_posterior(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-listener 5-fold CV posterior of the attended speaker from per-trial features."""
    rows = []
    for s, g in df.groupby("subject"):
        g = g.dropna(subset=cols)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        P = cross_val_predict(clf, g[cols].values, g["attended"].values, cv=cv,
                              method="predict_proba")
        cls = np.sort(g["attended"].unique())
        for (_, r), p in zip(g.iterrows(), P):
            rows.append({"subject": s, "trial_id": r.trial_id, "attended": int(r.attended),
                         "on_att": float(p[list(cls).index(r.attended)]),
                         "pick": int(cls[p.argmax()])})
    return pd.DataFrame(rows)


def relate(pr: pd.DataFrame, got: pd.DataFrame) -> dict:
    """CoupleFlow per-trial correctness vs a per-trial gaze measure (on_att, pick)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    res = {"measure_mean_on_att": float(got.on_att.mean()),
           "gaze_pick_acc": float((got.pick == got.attended).mean()), "n_trials": int(len(got))}
    m = pr.merge(got, on=["subject", "trial_id"], how="inner")
    m["terc"] = m.groupby("subject")["on_att"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    res["acc_by_tercile_within_listener"] = {
        int(k): {"acc": float(v.correct.mean()), "on_att_mean": float(v.on_att.mean()),
                 "n_rows": int(len(v))} for k, v in m.groupby("terc")}
    pt = m.groupby(["subject", "trial_id"]).agg(margin=("trial_margin", "mean"),
                                                 correct=("correct", "mean"),
                                                 g=("on_att", "first")).reset_index()
    r1 = spearmanr(pt.g, pt.margin)
    res["spearman_vs_trial_margin"] = {"rho": float(r1.statistic), "p": float(r1.pvalue),
                                       "n": int(len(pt))}
    m["gz"] = (m.on_att - m.on_att.mean()) / m.on_att.std()
    m["cluster"] = pd.factorize(m.subject + "_" + m.trial_id)[0]
    fit = smf.glm("correct ~ gz + C(subject) + C(seed)", data=m,
                  family=sm.families.Binomial()).fit(cov_type="cluster",
                                                     cov_kwds={"groups": m["cluster"]})
    lo, hi = np.exp(fit.conf_int().loc["gz"].values)
    res["logit_correct_on_measure"] = {"odds_ratio_per_sd": float(np.exp(fit.params["gz"])),
                                       "ci95": [float(lo), float(hi)], "p": float(fit.pvalues["gz"])}
    off, on = m[m.pick != m.attended], m[m.pick == m.attended]
    res["coupleflow_acc_gaze_not_on_attended"] = {
        "acc": float(off.correct.mean()), "n_trials": int(off[["subject", "trial_id"]]
                                                          .drop_duplicates().shape[0])}
    res["coupleflow_acc_gaze_on_attended"] = {
        "acc": float(on.correct.mean()), "n_trials": int(on[["subject", "trial_id"]]
                                                         .drop_duplicates().shape[0])}
    po = off.groupby("subject").correct.agg(["mean", "size"])
    po = po[po["size"] >= 30]   # 3 seeds x >= 10 trials
    res["acc_gaze_not_on_attended_per_listener_min_max"] = (
        [float(po["mean"].min()), float(po["mean"].max()), int(len(po))] if len(po) else None)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    feats = pd.read_csv(d / "gaze_features.csv")
    samples = pickle.load(open(d / "gaze_samples.pkl", "rb"))
    res = {"n_trials_with_gaze": int(len(feats))}
    res["gaze_only_decoding"] = gaze_only(feats)
    pr = pd.read_csv(d / "trial_preds.csv")
    pr["trial_id"] = pr["trial"].str.split("_", n=1).str[1]
    pr["seed"] = pr["tag"].map(lambda x: int(x[-1]) if x.endswith(("_s1", "_s2")) else 0)
    res["coupleflow_trial_acc_all"] = float(pr.correct.mean())
    res["coupleflow_trial_acc_per_seed"] = {int(k): float(v) for k, v in
                                            pr.groupby("seed").correct.mean().items()}
    measures = {"eye_samples": gaze_on_target(feats, samples, "eye_az", "az"),
                "world_samples": gaze_on_target(feats, samples, "world_az", "world"),
                "classifier_all": classifier_posterior(feats, FEATS["all"]),
                "classifier_eye": classifier_posterior(feats, FEATS["eye"])}
    res["relation"] = {k: relate(pr, g) for k, g in measures.items()}
    for k, g in measures.items():
        g.to_csv(d / f"gaze_measure_{k}.csv", index=False)
    (d / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
