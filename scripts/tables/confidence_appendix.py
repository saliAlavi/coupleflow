#!/usr/bin/env python
"""Appendix: calibrated evidence, sequential decisions, overlap and training-window length.

Inputs (all produced beforehand, nothing is trained here):

* ``<run>/conf_margins.npz`` of every CoupleFlow run (``dump_margins.py``): per-window
  margins ``s(attended) - s(competing)`` on the val and test splits, 1/2/5-s windows,
  1-s windows with hop 0.5 / 0.25 s.
* ``<cache>/<ds>/linear_margins.npz`` (``baselines/linear/linear_margins.py``).
* ``<run>/results.json`` of the training-window arms (tags ``twin_<arm>_s<seed>``).

Protocol: for every fold (and seed) one positive scale ``a`` is fitted on the fold's
validation margins (logistic loss, no bias: the task is symmetric), giving calibrated
log-odds ``l = a * margin`` on its test windows.  Everything reported is on test windows,
pooled over folds (window-weighted), mean over 3 seeds.  Thresholds of the sequential rule
are fixed values, never chosen on test data.

Outputs: ``paper/tables/conf_*.tex``, ``tables/numbers_conf.tex``,
``figures/conf_*.pdf`` and ``paper/conf_results.json``.

    python scripts/tables/confidence_appendix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.preprocessing.core import CACHE_ROOT  # noqa: E402

RUNS = paths.RUNS
PAPER = Path("paper")
TABLES, FIGS = PAPER / "tables", Path("figures")
TWO = ["kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp"]
ONE = ["sparrkulee", "megmasc"]
NAMES = {"kuleuven": "KU Leuven", "dtu": "DTU", "fuglsang2020": "Fuglsang'20", "esaa": "ESAA",
         "uhd255": "KUL-255ch", "broderick_cp": "Broderick", "sparrkulee": "SparrKULee",
         "megmasc": "MEG-MASC"}
SEED_TAGS = ["final", "final_s1", "final_s2"]
ARMS = [("w1", "1 s"), ("w2", "2 s"), ("w5", "5 s (paper)"), ("w10", "10 s"),
        ("mix", "mixed 1--10 s")]
EVAL_W = [1, 2, 5, 10, 20, 30, 60]
KS = [1, 2, 3, 5, 8, 10, 15, 20, 30]                  # fixed-length decisions (s)
A_TAB = [1.0, 2.0]                                    # thresholds in the tables
A_CURVE = np.round(np.arange(0.25, 5.01, 0.25), 2)   # thresholds on the curves
# Categorical order (validated palette: blue, orange, aqua, yellow, magenta) + markers.
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MRK = ["o", "s", "^", "D", "v", "P"]
A_DET = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]                # thresholds of the detailed analysis
LEN_BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10 ** 6)]
rng = np.random.default_rng(0)
MISSING: list[str] = []


# =============================================================================
# Calibration and decision rules
# =============================================================================

def fit_scale(m: np.ndarray) -> float:
    """ML scale a > 0 with P(candidate 0 attended) = sigmoid(a * margin)."""
    return minimize_scalar(lambda a: np.mean(np.logaddexp(0.0, -a * m)),
                           bounds=(1e-4, 1e3), method="bounded").x


def by_trial(l: np.ndarray, tr: np.ndarray, st: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per trial: (log-odds, start in samples), time-ordered."""
    out = []
    for t in np.unique(tr):
        m = tr == t
        o = np.argsort(st[m])
        out.append((l[m][o], st[m][o]))
    return out


def fixed(trials, T: float, w: float = 1.0) -> list[bool]:
    """Non-overlapping T-s segments; each sums the log-odds of the windows inside it."""
    c = []
    for l, st in trials:
        t0 = (st - st[0]) / 64.0
        for s0 in np.arange(0, t0[-1] + w - T + 1e-9, T):
            m = (t0 >= s0 - 1e-9) & (t0 + w <= s0 + T + 1e-9)
            if m.any():
                c.append(l[m].sum() > 0)
    return c


def sprt(trials, A: float, w: float = 1.0, h: float = 1.0, scale: float = 1.0,
         adjacent: bool = False):
    """Sequential rule: add windows until |sum| >= A, decide, restart; leftover forced.

    Time of a decision that used n windows = (n - 1) h + w seconds.  With overlapping
    windows (h < w) the next decision starts at the first window that shares no data
    with the previous decision; restarting at the adjacent window would re-use the
    window that crossed the threshold and inflate accuracy.
    """
    c, n = [], []
    skip = 0 if adjacent else int(round(w / h)) - 1
    for l, _ in trials:
        s, k, i = 0.0, 0, 0
        while i < len(l):
            s += l[i] * scale
            k += 1
            i += 1
            if abs(s) >= A:
                c.append(s > 0)
                n.append((k - 1) * h + w)
                s, k = 0.0, 0
                i += skip
        if k:
            c.append(s > 0 if s != 0 else rng.random() < 0.5)
            n.append((k - 1) * h + w)
    return float(np.mean(c)), float(np.mean(n)), len(c), float(np.percentile(n, 90))


def fit_kappa(curve: dict[float, float]) -> float:
    """Least squares in probit space: Phi^-1(acc) = kappa sqrt(T)."""
    T = np.array([k for k, a in curve.items() if 0.5 < a < 0.995])
    z = norm.ppf([curve[k] for k in T])
    return float((np.sqrt(T) * z).sum() / T.sum())


def theory_sprt(kappa: float, acc: np.ndarray) -> np.ndarray:
    """Wald: mean time (s) of the symmetric SPRT reaching accuracy acc (Prop. conf-sprt)."""
    a = 1 - acc
    return (1 - 2 * a) * np.log((1 - a) / a) / (2 * kappa ** 2)


def ece(l: np.ndarray) -> float:
    p = 1 / (1 + np.exp(-l))
    conf, c = np.maximum(p, 1 - p), l > 0
    b = np.minimum(((conf - 0.5) / 0.05).astype(int), 9)
    return float(sum(abs(conf[b == i].mean() - c[b == i].mean()) * (b == i).mean()
                     for i in range(10) if (b == i).any()))


def nll_bits(l: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, -l)) / np.log(2))


# =============================================================================
# Loading
# =============================================================================

def folds_of(tag: str, ds: str) -> list[Path]:
    return sorted((RUNS / tag).glob(f"{ds}_fold*"), key=lambda p: int(p.name.rsplit("fold", 1)[1]))


def load_cf(tag: str, ds: str, key: str):
    """Calibrated test log-odds, trial ids and starts pooled over folds (scale per fold)."""
    L, TR, ST, RAW = [], [], [], []
    for f, rd in enumerate(folds_of(tag, ds)):
        p = rd / "conf_margins.npz"
        if not p.exists():
            MISSING.append(str(p))
            return None
        z = np.load(p)
        if f"test_{key}__coupling" not in z.files:
            MISSING.append(f"{p}:{key}")
            return None
        a = fit_scale(z[f"val_{key}__coupling"])
        m = z[f"test_{key}__coupling"]
        L.append(a * m)
        RAW.append(m)
        TR.append(z[f"test_{key}__trial"] + 100000 * f)
        ST.append(z[f"test_{key}__start"])
    if not L:
        MISSING.append(f"{tag}/{ds}")
        return None
    return {"l": np.concatenate(L), "raw": np.concatenate(RAW), "tr": np.concatenate(TR),
            "st": np.concatenate(ST)}


def load_cf_ens(ds: str, key: str):
    """3-seed ensemble: average calibrated log-odds, re-fit one scale on validation."""
    L, TR, ST = [], [], []
    for f in range(len(folds_of(SEED_TAGS[0], ds))):
        zs = [np.load(folds_of(t, ds)[f] / "conf_margins.npz") for t in SEED_TAGS]
        st0 = zs[0][f"test_{key}__start"]
        assert all((z[f"test_{key}__start"] == st0).all() for z in zs)
        sc = [fit_scale(z[f"val_{key}__coupling"]) for z in zs]
        v = np.mean([a * z[f"val_{key}__coupling"] for a, z in zip(sc, zs)], 0)
        t = np.mean([a * z[f"test_{key}__coupling"] for a, z in zip(sc, zs)], 0)
        L.append(fit_scale(v) * t)
        TR.append(zs[0][f"test_{key}__trial"] + 100000 * f)
        ST.append(st0)
    return {"l": np.concatenate(L), "tr": np.concatenate(TR), "st": np.concatenate(ST)}


def load_linear(ds: str, key: str):
    p = CACHE_ROOT / ds / "linear_margins.npz"
    if not p.exists():
        MISSING.append(str(p))
        return None
    z = np.load(p)
    L, TR, ST = [], [], []
    has_val = f"val_{key}__coupling" in z.files
    for f in np.unique(z[f"test_{key}__fold"]):
        tm = z[f"test_{key}__fold"] == f
        vm = (z[f"val_{key}__coupling"][z[f"val_{key}__fold"] == f] if has_val else [])
        if len(vm) == 0:
            # No validation split for this dataset (e.g. Fuglsang'20: one trial per listener
            # and condition): fit the scale on the other folds' test windows instead, which
            # never uses the fold being scored.
            vm = z[f"test_{key}__coupling"][~tm]
        L.append(fit_scale(vm) * z[f"test_{key}__coupling"][tm])
        TR.append(z[f"test_{key}__trial"][tm])
        ST.append(z[f"test_{key}__start"][tm])
    return {"l": np.concatenate(L), "tr": np.concatenate(TR), "st": np.concatenate(ST)}


def seq_block(d: dict, w: float = 1.0, h: float = 1.0, scale: float = 1.0) -> dict:
    """Fixed-length curve, kappa, sequential curve and table points for one score."""
    trials = by_trial(d["l"], d["tr"], d["st"])
    fc = {k: float(np.mean(fixed(trials, k, w))) for k in KS}
    kap = fit_kappa(fc)
    curve = [sprt(trials, A, w, h, scale) for A in A_CURVE]
    tab = {}
    for A in A_TAB:
        acc, t, n, t90 = sprt(trials, A, w, h, scale)
        fa = float(np.interp(t, KS, list(fc.values())))
        tab[A] = {"acc": acc, "time": t, "t90": t90, "n": n, "fixed": fa, "gain": acc - fa,
                  "theory_time": float(theory_sprt(kap, np.array([acc]))[0])}
    return {"fixed": fc, "kappa": kap, "curve": curve, "tab": tab}


def pooled_acc(tag: str, ds: str, cell: str) -> float | None:
    num = den = 0.0
    for rd in folds_of(tag, ds):
        p = rd / "results.json"
        if not p.exists():
            p = rd / "results_val.json"            # re-evaluated runs (MODE=val)
        if not p.exists():
            MISSING.append(str(rd / "results.json"))
            return None
        r = json.loads(p.read_text())["results"].get(cell, {})
        if not r.get("n_windows"):
            continue
        num += r["acc_coupling"] * r["n_windows"]
        den += r["n_windows"]
    return num / den if den else None


def arm_tag(arm: str, seed: int) -> str:
    return SEED_TAGS[seed] if arm == "w5" else f"twin_{arm}_s{seed}"


def m_sd(v: list) -> tuple[float | None, float | None]:
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None)


# =============================================================================
# Analyses
# =============================================================================

def analyse() -> dict:
    res: dict = {"calib": {}, "seq": {}, "overlap": {}, "trainlen": {}}
    for ds in TWO + ONE:
        # ---- calibration, ensemble, Gaussian-LLR diagnostics ----------------
        c = {}
        seeds = [load_cf(t, ds, "5s") for t in SEED_TAGS]
        seeds1 = [load_cf(t, ds, "1s") for t in SEED_TAGS]
        if any(s is None for s in seeds + seeds1):
            continue
        c["acc5"] = float(np.mean([(s["l"] > 0).mean() for s in seeds]))
        c["ece_raw"] = float(np.mean([ece(s["raw"]) for s in seeds]))
        c["ece_cal"] = float(np.mean([ece(s["l"]) for s in seeds]))
        c["nll_raw"] = float(np.mean([nll_bits(s["raw"]) for s in seeds]))
        c["nll_cal"] = float(np.mean([nll_bits(s["l"]) for s in seeds]))
        e5, e1 = load_cf_ens(ds, "5s"), load_cf_ens(ds, "1s")
        c["acc5_ens"] = float((e5["l"] > 0).mean())
        c["acc1"] = float(np.mean([(s["l"] > 0).mean() for s in seeds1]))
        c["acc1_ens"] = float((e1["l"] > 0).mean())
        # Exact Gaussian LLRs have var = 2 mean (Prop. conf-gauss); lag-1 autocorrelation
        # of consecutive 1-s log-odds tests the independence assumption.
        ratio, ac = [], []
        for s in seeds1:
            ratio.append(s["l"].var() / (2 * s["l"].mean()))
            x = [(l[:-1], l[1:]) for l, _ in by_trial(s["l"], s["tr"], s["st"]) if len(l) > 2]
            a0 = np.concatenate([u for u, _ in x]) - s["l"].mean()
            a1 = np.concatenate([v for _, v in x]) - s["l"].mean()
            ac.append(float((a0 * a1).mean() / s["l"].var()))
        c["var_ratio"], c["lag1"] = float(np.mean(ratio)), float(np.mean(ac))
        if ds in TWO:
            vote, sm = [], []
            for s in seeds:
                v_, s_ = [], []
                for l, _ in by_trial(s["l"], s["tr"], s["st"]):
                    fr = (l > 0).mean()
                    v_.append(fr > 0.5 if fr != 0.5 else 0.5)
                    s_.append(l.sum() > 0)
                vote.append(np.mean(v_))
                sm.append(np.mean(s_))
            c["trial_vote"], c["trial_sum"] = float(np.mean(vote)), float(np.mean(sm))
        res["calib"][ds] = c
        print("calib", ds, flush=True)

        if ds not in TWO:
            continue
        # ---- sequential vs fixed: CoupleFlow (per seed, ensemble) and linear -------
        blocks = [seq_block(s) for s in seeds1]
        sq = {"cf": average_blocks(blocks), "cf_ens": seq_block(e1)}
        lin = load_linear(ds, "1s")
        if lin is not None:
            sq["linear"] = seq_block(lin)
        res["seq"][ds] = sq

        # ---- overlapping 1-s windows (seed 0) ---------------------------------------
        ov = {}
        for h in (1.0, 0.5, 0.25):
            key = "1s" if h == 1.0 else f"1s_h{h:g}"
            d = load_cf("final", ds, key)
            if d is None:
                continue
            trials = by_trial(d["l"], d["tr"], d["st"])
            fc = {k: float(np.mean(fixed(trials, k))) for k in (1, 2, 5, 10)}
            row = {"fixed": fc}
            # Summed overlapping log-odds: raw (over-confident by w/h) and corrected.
            seg = [(l, st) for l, st in trials]
            sums = []
            for l, st in seg:
                t0 = (st - st[0]) / 64.0
                for s0 in np.arange(0, t0[-1] + 1 - 5 + 1e-9, 5):
                    m = (t0 >= s0 - 1e-9) & (t0 + 1 <= s0 + 5 + 1e-9)
                    if m.any():
                        sums.append(l[m].sum())
            sums = np.array(sums)
            row["ece5_raw"], row["ece5_corr"] = ece(sums), ece(sums * h)
            for name, sc in (("raw", 1.0), ("corr", h)):
                row[name] = {A: sprt(trials, A, 1.0, h, sc) for A in A_TAB}
            row["curve"] = [sprt(trials, A, 1.0, h, h) for A in A_CURVE]
            row["adjacent"] = sprt(trials, 2.0, 1.0, h, h, adjacent=True)
            x = [l for l, _ in trials if len(l) > 3]
            mu, var = np.concatenate(x).mean(), np.concatenate(x).var()
            row["lagcorr"] = [float(np.mean(np.concatenate([(l[:-j] - mu) * (l[j:] - mu)
                                                            for l in x])) / var)
                              for j in (1, 2, 3)]
            ov[h] = row
        if 1.0 in ov:       # sequential accuracy of hop 1 at each hop's mean time (A = 2)
            base = sorted((c[1], c[0]) for c in ov[1.0]["curve"])
            for h, row in ov.items():
                t = row["corr"][2.0][1]
                row["hop1_at_t"] = float(np.interp(t, [b[0] for b in base], [b[1] for b in base]))
        res["overlap"][ds] = ov
        print("seq/overlap", ds, flush=True)

    # ---- training-window length -------------------------------------------------------
    for ds in TWO + ONE:
        task = "same_talker" if ds in ONE else "competing"
        tl = {}
        for arm, _ in ARMS:
            accs = {W: m_sd([pooled_acc(arm_tag(arm, s), ds, f"{task}_{W}s") for s in range(3)])
                    for W in EVAL_W}
            row = {"acc": accs}
            if ds in TWO:
                bl = []
                for s in range(3):
                    d = load_cf(arm_tag(arm, s), ds, "1s")
                    if d is not None:
                        bl.append(seq_block(d))
                if bl:
                    row["seq"] = average_blocks(bl)
            tl[arm] = row
        res["trainlen"][ds] = tl
        print("trainlen", ds, flush=True)
    return res


def average_blocks(bl: list[dict]) -> dict:
    """Mean over seeds of fixed curve, kappa, sequential curve and table points."""
    out = {"fixed": {k: float(np.mean([b["fixed"][k] for b in bl])) for k in KS},
           "kappa": float(np.mean([b["kappa"] for b in bl])),
           "curve": [tuple(float(np.mean([b["curve"][i][j] for b in bl])) for j in range(4))
                     for i in range(len(A_CURVE))],
           "tab": {A: {k: float(np.mean([b["tab"][A][k] for b in bl]))
                       for k in bl[0]["tab"][A]} for A in A_TAB},
           "n_seeds": len(bl)}
    return out


# =============================================================================
# Tables, figures, macros
# =============================================================================

def pct(x: float | None, d: int = 1) -> str:
    return "--" if x is None else f"{100 * x:.{d}f}"


def write_tables(res: dict) -> None:
    rows = []
    for ds, c in res["calib"].items():
        tv = (f"{pct(c['trial_vote'])} & {pct(c['trial_sum'])}" if "trial_vote" in c
              else "-- & --")
        rows.append(f"{NAMES[ds]} & {c['ece_raw']:.3f} & {c['ece_cal']:.3f} & "
                    f"{c['nll_raw']:.2f} & {c['nll_cal']:.2f} & {pct(c['acc5'])} & "
                    f"{pct(c['acc5_ens'])} & {tv} & {c['var_ratio']:.2f} & {c['lag1']:.2f} \\\\")
    (TABLES / "conf_calib.tex").write_text(
        "\\begin{tabular}{lcccccccccc}\\toprule\n"
        "& \\multicolumn{2}{c}{ECE (5\\,s)} & \\multicolumn{2}{c}{NLL, bits (5\\,s)} & "
        "\\multicolumn{2}{c}{acc.\\ 5\\,s} & \\multicolumn{2}{c}{trial} & "
        "\\multicolumn{2}{c}{1-s log-odds} \\\\\n"
        "Dataset & raw & cal. & raw & cal. & single & ens. & vote & sum & "
        "$\\frac{\\mathrm{var}}{2\\,\\mathrm{mean}}$ & lag-1 $r$ \\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

    rows = []
    for ds, sq in res["seq"].items():
        cells = []
        for who in ("cf", "linear"):
            if who not in sq:
                cells += ["--"] * 4
                continue
            b = sq[who]
            cells.append(f"{b['kappa']:.3f}")
            for A in A_TAB:
                t = b["tab"][A]
                cells.append(f"{pct(t['acc'])} / {t['time']:.1f} ({t['t90']:.0f}) / {pct(t['fixed'])}")
            cells.append(f"{100 * sq[who]['tab'][2.0]['gain']:+.1f}")
        rows.append(f"{NAMES[ds]} & " + " & ".join(cells) + " \\\\")
    hdr = ("\\begin{tabular}{l" + "c" * 8 + "}\\toprule\n"
           "& \\multicolumn{4}{c}{CoupleFlow (mean of 3 seeds)} & "
           "\\multicolumn{4}{c}{Linear backward model} \\\\\n"
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
           "Dataset & $\\kappa$ & $A{=}1$ & $A{=}2$ & gain & $\\kappa$ & $A{=}1$ & $A{=}2$ & gain "
           "\\\\\\midrule\n")
    (TABLES / "conf_seq.tex").write_text(hdr + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

    rows = []
    for ds, ov in res["overlap"].items():
        cells = []
        for h in (1.0, 0.5, 0.25):
            r = ov.get(h)
            if r is None:
                cells += ["--"] * 3
                continue
            a, t, _, _ = r["corr"][2.0]
            cells += [pct(r["fixed"][5]), f"{r['ece5_raw']:.2f}/{r['ece5_corr']:.2f}",
                      f"{pct(a)} / {t:.1f}" + ("" if h == 1.0 else
                                                f" ({100 * (a - r['hop1_at_t']):+.1f})")]
        rows.append(f"{NAMES[ds]} & " + " & ".join(cells) + " \\\\")
    (TABLES / "conf_overlap.tex").write_text(
        "\\begin{tabular}{l" + "ccc" * 3 + "}\\toprule\n"
        + "".join(f"& \\multicolumn{{3}}{{c}}{{hop {h:g}\\,s}} " for h in (1, 0.5, 0.25))
        + "\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
        "Dataset" + " & fixed 5\\,s & ECE raw/corr. & seq.\\ $A{=}2$" * 3 + " \\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

    rows = []
    for ds, tl in res["trainlen"].items():
        for i, (arm, lab) in enumerate(ARMS):
            r = tl.get(arm, {})
            acc = r.get("acc", {})
            cells = []
            for W in (1, 5, 30):
                m, sd = acc.get(W, (None, None))
                cells.append(pct(m) + (f"{{\\scriptsize$\\pm${100 * sd:.1f}}}" if sd else ""))
            sq = r.get("seq")
            cells.append(f"{sq['kappa']:.3f}" if sq else "--")
            cells.append(f"{pct(sq['tab'][2.0]['acc'])} / {sq['tab'][2.0]['time']:.1f}"
                         if sq else "--")
            name = NAMES[ds] if i == 0 else ""
            rows.append(f"{name} & {lab} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    (TABLES / "conf_trainlen.tex").write_text(
        "\\begin{tabular}{llccccc}\\toprule\n"
        "Dataset & training window & 1\\,s & 5\\,s & 30\\,s & $\\kappa$ (1-s sums) & "
        "seq.\\ $A{=}2$: acc.\\ / time \\\\\\midrule\n"
        + "\n".join(rows[:-1]) + "\n\\bottomrule\\end{tabular}\n")


def fig_seq(res: dict) -> None:
    dss = [d for d in TWO if d in res["seq"]]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True, sharey=True)
    for ax, ds in zip(axes.flat, dss):
        sq = res["seq"][ds]
        series = [("cf", "fixed", "CoupleFlow, fixed"), ("cf", "seq", "CoupleFlow, sequential"),
                  ("linear", "fixed", "Linear, fixed"), ("linear", "seq", "Linear, sequential")]
        for i, (who, kind, lab) in enumerate(series):
            if who not in sq:
                continue
            b = sq[who]
            if kind == "fixed":
                x, y = KS, [100 * b["fixed"][k] for k in KS]
            else:
                x, y = [c[1] for c in b["curve"]], [100 * c[0] for c in b["curve"]]
            ax.plot(x, y, color=COL[i], marker=MRK[i], ms=3.5, lw=1.5, label=lab)
        acc = np.linspace(0.55, 0.97, 60)
        ax.plot(theory_sprt(sq["cf"]["kappa"], acc), 100 * acc, color="#8a8a8a", ls=":", lw=1.2,
                label="Wald prediction (CoupleFlow $\\kappa$)")
        ax.set_xscale("log")
        ax.set_xlim(0.9, 40)
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_ylim(50, 100)
        ax.set_title(NAMES[ds], fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("mean decision time (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)", fontsize=8)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGS / "conf_seq.pdf")
    plt.close(fig)


def fig_trainlen(res: dict) -> None:
    dss = [d for d in TWO + ONE if d in res["trainlen"]]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9), sharex=True)
    for ax, ds in zip(axes.flat, dss):
        for i, (arm, lab) in enumerate(ARMS):
            acc = res["trainlen"][ds].get(arm, {}).get("acc", {})
            pts = [(W, 100 * acc[W][0]) for W in EVAL_W if acc.get(W, (None,))[0] is not None]
            if pts:
                ax.plot(*zip(*pts), color=COL[i], marker=MRK[i], ms=3.5, lw=1.5, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(EVAL_W)
        ax.set_xticklabels([str(w) for w in EVAL_W])
        ax.set_title(NAMES[ds], fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=6.5)
    for ax in axes[-1]:
        ax.set_xlabel("test window (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)", fontsize=8)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, [f"trained on {x}" for x in lab], loc="upper center", ncol=5, fontsize=7,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "conf_trainlen.pdf")
    plt.close(fig)


def fig_overlap(res: dict) -> None:
    dss = [d for d in TWO if res["overlap"].get(d)]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True, sharey=True)
    for ax, ds in zip(axes.flat, dss):
        ov = res["overlap"][ds]
        if 1.0 in ov:
            fc = ov[1.0]["fixed"]
            ax.plot(list(fc), [100 * v for v in fc.values()], color="#8a8a8a", ls=":", lw=1.2,
                    marker=".", label="fixed length (any hop)")
        for i, h in enumerate((1.0, 0.5, 0.25)):
            if h not in ov:
                continue
            cv = ov[h]["curve"]
            ax.plot([c[1] for c in cv], [100 * c[0] for c in cv], color=COL[i], marker=MRK[i],
                    ms=3, lw=1.5, label=f"sequential, hop {h:g} s")
        ax.set_xscale("log")
        ax.set_xlim(0.9, 40)
        ax.set_ylim(50, 100)
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_title(NAMES[ds], fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("mean decision time (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)", fontsize=8)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "conf_overlap.pdf")
    plt.close(fig)


def signed(x: float) -> str:
    """+1.2 / -1.2 / 0.0 (no signed zero after rounding)."""
    return "0.0" if round(x, 1) == 0 else f"{x:+.1f}"


def write_numbers(res: dict) -> None:
    """Macros for the appendix text (never hand-typed)."""
    m = {}
    cal = res["calib"]
    two = [d for d in TWO if d in cal]
    if two:
        m["confEceRawMin"] = f"{min(cal[d]['ece_raw'] for d in two):.2f}"
        m["confEceRawMax"] = f"{max(cal[d]['ece_raw'] for d in two):.2f}"
        m["confEceCalMax"] = f"{max(cal[d]['ece_cal'] for d in cal):.3f}"
        g = [100 * (cal[d]["acc5_ens"] - cal[d]["acc5"]) for d in two]
        m["confEnsMin"], m["confEnsMax"] = f"{min(g):.1f}", f"{max(g):.1f}"
        g = [100 * (cal[d]["trial_sum"] - cal[d]["trial_vote"]) for d in two]
        m["confSumMin"], m["confSumMax"] = f"{min(g):.1f}", f"{max(g):.1f}"
        m["confLagMin"] = f"{min(cal[d]['lag1'] for d in two):.2f}"
        m["confLagMax"] = f"{max(cal[d]['lag1'] for d in two):.2f}"
        m["confVarMin"] = f"{min(cal[d]['var_ratio'] for d in two):.2f}"
        m["confVarMax"] = f"{max(cal[d]['var_ratio'] for d in two):.2f}"
    sq = res["seq"]
    if sq:
        for A, nm in ((1.0, "One"), (2.0, "Two")):
            g = [100 * sq[d]["cf"]["tab"][A]["gain"] for d in sq]
            m[f"confGain{nm}Min"], m[f"confGain{nm}Max"] = f"{min(g):.1f}", f"{max(g):.1f}"
            gl = [100 * sq[d]["linear"]["tab"][A]["gain"] for d in sq if "linear" in sq[d]]
            if gl:
                m[f"confLinGain{nm}Min"] = f"{min(gl):.1f}"
                m[f"confLinGain{nm}Max"] = f"{max(gl):.1f}"
        rt = [sq[d]["cf"]["tab"][2.0]["time"] / sq[d]["cf"]["tab"][2.0]["theory_time"] for d in sq]
        m["confTheoryRatioMin"], m["confTheoryRatioMax"] = f"{min(rt):.2f}", f"{max(rt):.2f}"
    ov = res["overlap"]
    if ov and all(0.25 in ov[d] for d in ov):
        m["confOvCorOne"], m["confOvCorTwo"], m["confOvCorThree"] = (
            f"{x:.2f}" for x in ov["kuleuven"][0.25]["lagcorr"])
        m["confOvFixedMax"] = f"{max(abs(100 * (ov[d][h]['fixed'][5] - ov[d][1.0]['fixed'][5])) for d in ov for h in (0.5, 0.25)):.1f}"
        m["confOvEceRawMax"] = f"{max(ov[d][h]['ece5_raw'] for d in ov for h in (0.5, 0.25)):.2f}"
        m["confOvEceCorrMax"] = f"{max(ov[d][h]['ece5_corr'] for d in ov for h in (0.5, 0.25)):.2f}"
        g = [100 * (ov[d][h]["corr"][2.0][0] - ov[d][h]["hop1_at_t"]) for d in ov for h in (0.5, 0.25)]
        m["confOvSeqMin"], m["confOvSeqMax"] = f"{min(g):.1f}", f"{max(g):.1f}"
        g = [100 * (ov[d][h]["adjacent"][0] - ov[d][h]["corr"][2.0][0]) for d in ov for h in (0.5, 0.25)]
        m["confOvAdjMin"], m["confOvAdjMax"] = f"{min(g):.1f}", f"{max(g):.1f}"
    tl = res["trainlen"]

    def acc(ds, arm, W):
        return tl.get(ds, {}).get(arm, {}).get("acc", {}).get(W, (None, None))[0]

    for arm, nm in (("w1", "One"), ("w2", "Two"), ("w10", "Ten"), ("mix", "Mix")):
        for W, wn in ((1, "OneS"), (5, "FiveS"), (30, "ThirtyS")):
            d = [100 * (acc(ds, arm, W) - acc(ds, "w5", W)) for ds in tl
                 if acc(ds, arm, W) is not None and acc(ds, "w5", W) is not None]
            if d:
                m[f"confTl{nm}{wn}Min"], m[f"confTl{nm}{wn}Max"] = signed(min(d)), signed(max(d))
                m[f"confTl{nm}{wn}N"] = str(len(d))
        k = [tl[ds][arm]["seq"]["kappa"] / tl[ds]["w5"]["seq"]["kappa"] for ds in tl
             if "seq" in tl[ds].get(arm, {}) and "seq" in tl[ds].get("w5", {})]
        t = [tl[ds][arm]["seq"]["tab"][2.0]["time"] / tl[ds]["w5"]["seq"]["tab"][2.0]["time"]
             for ds in tl if "seq" in tl[ds].get(arm, {}) and "seq" in tl[ds].get("w5", {})]
        if k:
            m[f"confTl{nm}KapMin"], m[f"confTl{nm}KapMax"] = f"{min(k):.2f}", f"{max(k):.2f}"
            m[f"confTl{nm}TimeMin"], m[f"confTl{nm}TimeMax"] = f"{min(t):.2f}", f"{max(t):.2f}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (TABLES / "numbers_conf.tex").write_text("% generated by scripts/tables/confidence_appendix.py\n"
                                             + "\n".join(lines) + "\n")


# =============================================================================
# Detailed sequential analysis: chosen lengths, accuracy by length, thresholds
# =============================================================================

def sprt_decisions(trials, A: float) -> dict:
    """Every decision of the sequential rule on 1-s log-odds (non-overlapping windows).

    Returns arrays: correct, length (s), forced (trial ended before |S| reached A),
    |S| at the decision.  By Prop. conf-add the running sum is the posterior log-odds, so
    a decision taken at |S| has posterior accuracy sigmoid(|S|) whatever its length.
    """
    c, n, f, s_abs = [], [], [], []
    for l, _ in trials:
        s, k = 0.0, 0
        for x in l:
            s += x
            k += 1
            if abs(s) >= A:
                c.append(s > 0); n.append(k); f.append(False); s_abs.append(abs(s))
                s, k = 0.0, 0
        if k:
            c.append(s > 0 if s != 0 else rng.random() < 0.5)
            n.append(k); f.append(True); s_abs.append(abs(s))
    return {"c": np.array(c, float), "n": np.array(n, float), "f": np.array(f),
            "s": np.array(s_abs)}


def analyse_detail(res: dict) -> dict:
    """Pooled over folds and the 3 seeds (each seed's decisions counted once)."""
    out = {}
    for ds in TWO:
        seeds = [load_cf(t, ds, "1s") for t in SEED_TAGS]
        if any(d is None for d in seeds) or ds not in res["seq"]:
            continue
        fc = res["seq"][ds]["cf"]["fixed"]
        per_a = {}
        for A in A_DET:
            D = [sprt_decisions(by_trial(d["l"], d["tr"], d["st"]), A) for d in seeds]
            c, n, f, sa = (np.concatenate([d[k] for d in D]) for k in ("c", "n", "f", "s"))
            nf = ~f
            row = {"acc": c.mean(), "acc_decided": c[nf].mean(), "nominal": 1 / (1 + np.exp(-A)),
                   "posterior": float(np.mean(1 / (1 + np.exp(-sa[nf])))),
                   "mean": n.mean(), "median": float(np.median(n)),
                   "p10": float(np.percentile(n, 10)), "p90": float(np.percentile(n, 90)),
                   "forced": f.mean(), "acc_forced": (c[f].mean() if f.any() else None),
                   "per_min": 60.0 / n.mean(), "n_dec": int(len(c) / len(D))}
            row["fixed_same"] = float(np.interp(row["mean"], KS, [fc[k] for k in KS]))
            row["gain"] = row["acc"] - row["fixed_same"]
            # Distribution of chosen lengths (fraction of decisions per second, 1..60, >60).
            h = np.bincount(np.minimum(n.astype(int), 61), minlength=62)[1:]
            row["hist"] = (h / h.sum()).tolist()
            bins = []
            for lo, hi in LEN_BINS:
                m = nf & (n >= lo) & (n <= hi)
                bins.append({"lo": lo, "hi": hi, "frac": float(m.mean()),
                             "acc": (float(c[m].mean()) if m.sum() >= 20 else None),
                             "posterior": (float(np.mean(1 / (1 + np.exp(-sa[m]))))
                                           if m.sum() >= 20 else None)})
            row["bins"] = bins
            per_a[A] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in row.items()}
        out[ds] = per_a
        print("detail", ds, flush=True)
    return out


def write_detail_table(det: dict) -> None:
    rows = []
    for ds, per_a in det.items():
        for i, A in enumerate(A_DET):
            r = per_a[A]
            name = NAMES[ds] if i == 0 else ""
            rows.append(
                f"{name} & {A:g} & {pct(r['nominal'])} & {pct(r['acc'])} & {pct(r['acc_decided'])} & "
                f"{r['mean']:.1f} & {r['median']:.0f} & {r['p10']:.0f}--{r['p90']:.0f} & "
                f"{pct(r['forced'])} & {pct(r['acc_forced'])} & {pct(r['fixed_same'])} & "
                f"{100 * r['gain']:+.1f} \\\\")
        rows.append("\\midrule")
    (TABLES / "conf_seq_detail.tex").write_text(
        "\\begin{tabular}{lccccccccccc}\\toprule\n"
        "& & \\multicolumn{3}{c}{accuracy (\\%)} & \\multicolumn{3}{c}{chosen length (s)} & "
        "\\multicolumn{2}{c}{forced at trial end} & \\multicolumn{2}{c}{vs.\\ fixed length} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\n"
        "Dataset & $A$ & $\\Lambda(A)$ & all & decided & mean & median & 10--90\\% & "
        "share (\\%) & acc.\\ (\\%) & fixed (\\%) & gain \\\\\\midrule\n"
        + "\n".join(rows[:-1]) + "\n\\bottomrule\\end{tabular}\n")


def fig_threshold(det: dict) -> None:
    """Effect of the threshold: accuracy, chosen length and forced share against A."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for i, (ds, per_a) in enumerate(det.items()):
        kw = dict(color=COL[i], marker=MRK[i], ms=3.5, lw=1.5, label=NAMES[ds])
        axes[0].plot(A_DET, [100 * per_a[A]["acc_decided"] for A in A_DET], **kw)
        axes[1].plot(A_DET, [per_a[A]["mean"] for A in A_DET], **kw)
        axes[2].plot(A_DET, [100 * per_a[A]["forced"] for A in A_DET], **kw)
    a = np.linspace(0.4, 4.1, 50)
    axes[0].plot(a, 100 / (1 + np.exp(-a)), color="#3a3a3a", ls="--", lw=1.0,
                 label="$\\Lambda(A)$ (guarantee)")
    axes[0].set_ylabel("accuracy of decisions (%)", fontsize=8)
    axes[1].set_ylabel("mean chosen length (s)", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_yticks([2, 5, 10, 20, 50])
    axes[1].set_yticklabels(["2", "5", "10", "20", "50"])
    axes[2].set_ylabel("forced at trial end (%)", fontsize=8)
    for ax, t in zip(axes, ("(a) accuracy (threshold reached)", "(b) chosen length",
                            "(c) undecided at trial end")):
        ax.set_xlabel("threshold $A$ (log-odds)", fontsize=8)
        ax.set_title(t, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=7, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIGS / "conf_threshold.pdf")
    plt.close(fig)


def fig_lengths(det: dict) -> None:
    """Cumulative distribution of the chosen decision length, per threshold."""
    shown = [0.5, 1.0, 2.0, 3.0]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True, sharey=True)
    for ax, (ds, per_a) in zip(axes.flat, det.items()):
        for i, A in enumerate(shown):
            cdf = np.cumsum(per_a[A]["hist"][:60])
            ax.step(np.arange(1, 61), 100 * cdf, where="post", color=COL[i], lw=1.5,
                    label=f"$A={A:g}$")
            ax.plot(per_a[A]["median"], 50, marker=MRK[i], color=COL[i], ms=4)
        ax.set_xscale("log")
        ax.set_xlim(1, 60)
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
        ax.set_title(NAMES[ds], fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("chosen decision length (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("decisions made by then (%)", fontsize=8)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "conf_lengths.pdf")
    plt.close(fig)


def fig_acc_by_length(det: dict) -> None:
    """Accuracy of the decisions that stopped at each length (threshold reached)."""
    shown = [1.0, 2.0, 3.0]
    labels = [f"{b[0]}" if b[0] == b[1] else (f"{b[0]}-{b[1]}" if b[1] < 10 ** 5 else f">{b[0] - 1}")
              for b in LEN_BINS]
    x = np.arange(len(LEN_BINS))
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True, sharey=True)
    for ax, (ds, per_a) in zip(axes.flat, det.items()):
        for i, A in enumerate(shown):
            b = per_a[A]["bins"]
            y = [100 * v["acc"] if v["acc"] is not None else np.nan for v in b]
            ax.plot(x, y, color=COL[i], marker=MRK[i], ms=3.5, lw=1.5, label=f"$A={A:g}$")
            ax.axhline(100 / (1 + np.exp(-A)), color=COL[i], ls=":", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylim(55, 100)
        ax.set_title(NAMES[ds], fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("chosen decision length (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)", fontsize=8)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab + [], loc="upper center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "conf_acc_by_length.pdf")
    plt.close(fig)


def write_detail_numbers(det: dict) -> None:
    m = {}
    if not det:
        return
    for A, nm in ((1.0, "One"), (2.0, "Two")):
        r = [det[d][A] for d in det]
        m[f"confDet{nm}MedMin"] = f"{min(x['median'] for x in r):.0f}"
        m[f"confDet{nm}MedMax"] = f"{max(x['median'] for x in r):.0f}"
        m[f"confDet{nm}PNineMin"] = f"{min(x['p90'] for x in r):.0f}"
        m[f"confDet{nm}PNineMax"] = f"{max(x['p90'] for x in r):.0f}"
        m[f"confDet{nm}AccMin"] = f"{100 * min(x['acc_decided'] for x in r):.1f}"
        m[f"confDet{nm}AccMax"] = f"{100 * max(x['acc_decided'] for x in r):.1f}"
        m[f"confDet{nm}Nominal"] = f"{100 / (1 + np.exp(-A)):.1f}"
        m[f"confDet{nm}ForcedMax"] = f"{100 * max(x['forced'] for x in r):.1f}"
        m[f"confDet{nm}OneSec"] = f"{100 * min(x['hist'][0] for x in r):.0f}"
        m[f"confDet{nm}OneSecMax"] = f"{100 * max(x['hist'][0] for x in r):.0f}"
    fa = [det[d][A]["acc_forced"] for d in det for A in (1.0, 2.0) if det[d][A]["acc_forced"]]
    m["confDetForcedAccMin"], m["confDetForcedAccMax"] = (f"{100 * min(fa):.1f}", f"{100 * max(fa):.1f}")
    # decided accuracy minus sigma(A), all datasets and thresholds
    d = [100 * (det[x][A]["acc_decided"] - det[x][A]["nominal"]) for x in det for A in A_DET]
    m["confDetAboveMin"], m["confDetAboveMax"] = f"{min(d):+.1f}", f"{max(d):+.1f}"
    # observed vs posterior-predicted accuracy per length bin (A = 2)
    dev = [abs(b["acc"] - b["posterior"]) for x in det for b in det[x][2.0]["bins"]
           if b["acc"] is not None]
    m["confDetPostDevMax"] = f"{100 * max(dev):.1f}"
    # shortfall of the longest decisions (>= 17 s) below sigma(A) at A = 2
    sh = [100 * (det[x][2.0]["nominal"] - b["acc"]) for x in det for b in det[x][2.0]["bins"]
          if b["acc"] is not None and b["lo"] >= 17]
    m["confDetLongDevMax"] = f"{max(sh):.1f}"
    sh = [100 * (det[x][2.0]["nominal"] - b["acc"]) for x in det for b in det[x][2.0]["bins"]
          if b["acc"] is not None and b["hi"] <= 8]
    m["confDetShortDevMax"] = f"{max(sh):.1f}"
    worst = max(det, key=lambda x: det[x][2.0]["forced"])
    m["confDetForcedWorst"] = NAMES[worst]
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    with open(TABLES / "numbers_conf.tex", "a") as fh:
        fh.write("\n".join(lines) + "\n")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    res = analyse()
    (PAPER / "conf_results.json").write_text(json.dumps(jsonable(res), indent=1))
    write_tables(res)
    write_numbers(res)
    fig_seq(res)
    fig_trainlen(res)
    fig_overlap(res)
    res["detail"] = analyse_detail(res)
    (PAPER / "conf_results.json").write_text(json.dumps(jsonable(res), indent=1))
    write_detail_table(res["detail"])
    write_detail_numbers(res["detail"])
    fig_threshold(res["detail"])
    fig_lengths(res["detail"])
    fig_acc_by_length(res["detail"])
    if MISSING:
        print(f"MISSING ({len(set(MISSING))}):", *sorted(set(MISSING))[:20], sep="\n  ")
