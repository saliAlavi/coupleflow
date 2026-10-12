#!/usr/bin/env python
"""Main-text comparison table and envelope-tracking table.

compare.tex   CoupleFlow (mean of 3 seeds) against the linear, VLAAI and dilated
              match-mismatch baselines at 1, 5 and 20 s on the content-disjoint folds, and
              against the pooled linear backward model under LOSO at 5 s (CoupleFlow LOSO:
              1 seed). Best value per column group in bold.
tracking.tex  Envelope tracking r with the attended and the competing envelope: the
              per-listener linear backward model on held-out trials (the values of App.
              Table kth) and CoupleFlow's linear envelope read-out of its neural embedding
              on 5-s test windows of the content-disjoint folds (mean of 3 seeds).
numbers_compare.tex  macros used in the text.

    python scripts/tables/compare_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.analysis.aggregate import CACHE, load, pooled  # noqa: E402
from coupleflow.preprocessing.run_config import check_name  # noqa: E402

T = Path("paper/tables")
TWO = ["kuleuven", "dtu", "fuglsang2020", "esaa", "uhd255", "broderick_cp"]
ONE = ["sparrkulee", "megmasc"]
NAMES = {"kuleuven": "KU Leuven", "dtu": "DTU", "fuglsang2020": "Fuglsang'20",
         "esaa": "ESAA", "uhd255": "KUL-255ch", "broderick_cp": "Broderick",
         "sparrkulee": "SparrKULee", "megmasc": "MEG-MASC"}
SEED_TAGS = ["final", "final_s1", "final_s2"]
WINS = ["1s", "5s", "20s"]
BL = [("linear_baseline", "Linear"), ("baseline_vlaai", "VLAAI"), ("baseline_mm", "Dil.\\ MM")]


def task(ds: str) -> str:
    return "same_talker" if ds in ONE else "competing"


def cf_mean(ds: str, cell: str, key: str = "acc_coupling") -> float | None:
    v = [pooled(load(t).get(ds, []), cell, key) for t in SEED_TAGS]
    return float(np.mean(v)) if all(x is not None for x in v) else None


def bl(fn: str, ds: str, w: str) -> float | None:
    p = CACHE / ds / f"{fn}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["windows"].get(w, {}).get(task(ds), [None])[0]


def fmt(x: float | None, best: float | None = None, nd: int = 1, scale: float = 100) -> str:
    if x is None:
        return "--"
    s = f"{scale * x:.{nd}f}"
    return f"\\textbf{{{s}}}" if best is not None and round(scale * x, nd) == round(scale * best, nd) else s


def main() -> None:
    rows, res = [], {}
    for ds in TWO + ONE:
        cells, res[ds] = [], {}
        for w in WINS:
            v = [bl(fn, ds, w) for fn, _ in BL] + [cf_mean(ds, f"{task(ds)}_{w}")]
            best = max([x for x in v if x is not None], default=None)
            cells += [fmt(x, best) for x in v]
            res[ds][w] = v
        lo_cf = pooled(load("loso_s0").get(ds, []), f"{task(ds)}_5s", "acc_coupling")
        lp = CACHE / ds / "linear_loso.json"
        lo_lin = json.loads(lp.read_text())["acc"] if lp.exists() else None
        best = max([x for x in (lo_lin, lo_cf) if x is not None], default=None)
        cells += [fmt(lo_lin, best), fmt(lo_cf, best)]
        res[ds]["loso"] = [lo_lin, lo_cf]
        rows.append(f"{NAMES[ds]} & {'comp.' if ds in TWO else 'mm'} & " + " & ".join(cells) + " \\\\")
    names = " & ".join(n for _, n in BL) + " & Ours"
    hdr = ("\\begin{tabular}{ll" + "|cccc" * len(WINS) + "|cc}\\toprule\n"
           " & & " + " & ".join(f"\\multicolumn{{4}}{{c|}}{{{w[:-1]}\\,s}}" for w in WINS)
           + " & \\multicolumn{2}{c}{LOSO, 5\\,s} \\\\\n"
           "Dataset & task & " + " & ".join([names] * len(WINS)) + " & Linear & Ours \\\\\\midrule\n")
    (T / "compare.tex").write_text(hdr + "\n".join(rows[:len(TWO)]) + "\n\\midrule\n"
                                   + "\n".join(rows[len(TWO):]) + "\n\\bottomrule\\end{tabular}\n")

    # envelope tracking
    trows, rr = [], {}
    for ds in TWO + ONE:
        ps = json.loads((CACHE / ds / check_name(ds)).read_text())["per_subject"]
        la = float(np.mean([v["r_att"] for v in ps.values()]))
        lu = float(np.mean([v["r_unatt"] for v in ps.values()])) if ds in TWO else None
        ca = cf_mean(ds, f"{task(ds)}_5s", "recon_r_att")
        cu = cf_mean(ds, "competing_5s", "recon_r_unatt") if ds in TWO else None
        rr[ds] = (la, lu, ca, cu)
        trows.append(f"{NAMES[ds]} & {fmt(la, nd=3, scale=1)} & {fmt(lu, nd=3, scale=1)} & "
                     f"{fmt(ca, nd=3, scale=1)} & {fmt(cu, nd=3, scale=1)} \\\\")
    (T / "tracking.tex").write_text(
        "\\begin{tabular}{lcccc}\\toprule\n"
        " & \\multicolumn{2}{c}{Linear (trials)} & \\multicolumn{2}{c}{Ours (5\\,s)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Dataset & $r_{\\rm att}$ & $r_{\\rm comp}$ & $r_{\\rm att}$ & $r_{\\rm comp}$ \\\\\\midrule\n"
        + "\n".join(trows[:len(TWO)]) + "\n\\midrule\n" + "\n".join(trows[len(TWO):])
        + "\n\\bottomrule\\end{tabular}\n")

    # macros
    gain1 = [res[d]["1s"][3] - max(x for x in res[d]["1s"][:3] if x is not None) for d in TWO + ONE]
    gain20 = [res[d]["20s"][3] - max(x for x in res[d]["20s"][:3] if x is not None)
              for d in TWO + ONE if res[d]["20s"][3] is not None]
    lg = [res[d]["loso"][1] - res[d]["loso"][0] for d in TWO
          if d != "broderick_cp" and None not in res[d]["loso"]]
    cfa = [rr[d][2] for d in TWO]
    g1 = dict(zip(TWO + ONE, gain1))
    grow = [d for d in TWO if d != "broderick_cp" and res[d]["20s"][3] - max(res[d]["20s"][:3]) > g1[d]]
    higher = [d for d in TWO + ONE if rr[d][2] > rr[d][0]]
    from coupleflow.analysis.aggregate import WINDOWS
    nl = {w: [cf_mean(d, f"{tk}_{w}", "null_coupling") for d in TWO + ONE
              for tk in (["competing", "same_talker"] if d in TWO else ["same_talker"])]
          for w in WINDOWS}
    short = [x for w in ("1s", "2s", "5s", "10s") for x in nl[w] if x is not None]
    long_ = [x for w in ("20s", "30s", "60s") for x in nl[w] if x is not None]
    mac = {
        "nullShortMin": f"{100 * min(short):.1f}", "nullShortMax": f"{100 * max(short):.1f}",
        "nullLongMin": f"{100 * min(long_):.1f}", "nullLongMax": f"{100 * max(long_):.1f}",
        "cmpOneMin": f"{100 * min(gain1):+.1f}", "cmpOneMax": f"{100 * max(gain1):+.1f}",
        "cmpTwentyMin": f"{100 * min(gain20):+.1f}", "cmpTwentyMax": f"{100 * max(gain20):+.1f}",
        "cmpLosoMin": f"{100 * min(lg):.1f}", "cmpLosoMax": f"{100 * max(lg):.1f}",
        "rCFattMin": f"{min(cfa):.3f}", "rCFattMax": f"{max(cfa):.3f}",
        "rCFmeg": f"{rr['megmasc'][2]:.3f}", "rCFsparr": f"{rr['sparrkulee'][2]:.3f}",
        "cmpGrowN": str(len(grow)), "rCFhigherN": str(len(higher)),
        "cmpOneMM": f"{100 * min(g1['sparrkulee'], g1['megmasc']):.1f}",
    }
    mac.update(extra_macros())
    (T / "numbers_compare.tex").write_text(
        "% generated by scripts/tables/compare_table.py\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
    print((T / "compare.tex").read_text())
    print((T / "tracking.tex").read_text())
    print(mac)
    print("grow", grow, "higher", higher)
    print("gain1", [round(100 * g, 1) for g in gain1], "gain20", [round(100 * g, 1) for g in gain20],
          "loso", [round(100 * g, 1) for g in lg])


def extra_macros() -> dict:
    """Numbers quoted in the text that come straight from runs (no hand-typing)."""
    import glob
    import os
    out = {}
    # hearing-impaired vs normal-hearing listeners, Fuglsang'20, 5 s, mean of 3 seeds
    from coupleflow.preprocessing.readers.fuglsang2020 import _groups
    grp = _groups()
    acc = {"HI": [], "NH": []}
    for t in SEED_TAGS:
        num, den = {"HI": 0.0, "NH": 0.0}, {"HI": 0, "NH": 0}
        for r in load(t).get("fuglsang2020", []):
            for sub, v in r["results"]["competing_5s"]["per_subject"].items():
                g = grp[sub]
                num[g] += v["coupling"] * v["n"]
                den[g] += v["n"]
        for g in acc:
            acc[g].append(num[g] / den[g])
    out["hiAcc"] = f"{100 * np.mean(acc['HI']):.1f}"
    out["nhAcc"] = f"{100 * np.mean(acc['NH']):.1f}"
    # training time per fold (minutes, includes test evaluation)
    mins = [r["minutes"] for t in SEED_TAGS for rs in load(t).values() for r in rs if r.get("minutes")]
    out["trainMinMin"], out["trainMinMax"] = f"{min(mins):.0f}", f"{max(mins):.0f}"
    # KU Leuven without fold 0 (fold 0 was used to choose the input features)
    ex0 = [pooled([r for r in load(t)["kuleuven"] if r["fold"] != 0], "competing_5s", "acc_coupling")
           for t in SEED_TAGS]
    out["kulExFoldZero"] = f"{100 * np.mean(ex0):.1f}"
    # ESAA with the 125-ms window (seed 0) and listeners whose tracking peaks 125 ms early
    out["esaaLateFive"] = f"{100 * pooled(load('unified')['esaa'], 'competing_5s', 'acc_coupling'):.1f}"
    sc = json.loads((paths.RESULTS / "offset_scan" / "esaa_unified_env64.pt.json").read_text())
    off, r = np.array(sc["offsets_s"]), np.array(sc["r"])
    out["esaaEarlyN"] = str(int(np.sum(np.isclose(off[r.argmax(1)], -0.125))))
    out["esaaN"] = str(r.shape[0])
    # MAESTRO: 5-s EEG-shuffle nulls of every run in Table maestro (matched, 4-way)
    nul = []
    for tag in ["maestro_within_lag0", "maestro_within_lag0_s1", "maestro_within_lag0_s2",
                "maestro_loso_lag0", "maestro_loso_lag0_s1", "maestro_loso_lag0_s2",
                "maestro_pkg_within", "maestro_pkg_within_s1", "maestro_pkg_within_s2",
                "maestro_pkg_loso", "maestro_pkg_loso_s1", "maestro_pkg_loso_s2"]:
        num = den = 0.0
        for d in glob.glob(f"{paths.RUNS}/{tag}/maestro_fold*"):
            f = f"{d}/results.json" if os.path.exists(f"{d}/results.json") else f"{d}/results_val.json"
            c = json.load(open(f))["results"].get("cands_matched_5s", {})
            if c.get("null_coupling") is not None:
                num += c["null_coupling"] * c["n_windows"]
                den += c["n_windows"]
        if den:
            nul.append(num / den)
    out["maMainNullMin"], out["maMainNullMax"] = f"{100 * min(nul):.1f}", f"{100 * max(nul):.1f}"
    # flow table (tables.json from make_tables.py): audio-prior accuracy, unconditional gaps
    fl = json.loads((T / "tables.json").read_text())["flow"]
    pri = [v["acc_flow_prior_only"] for v in fl.values()]
    out["pmiPriorMin"], out["pmiPriorMax"] = f"{100 * min(pri):.1f}", f"{100 * max(pri):.1f}"
    du = {d: v["R_u_un"] - v["R_u_att"] for d, v in fl.items()}
    out["dRuncMaxOther"] = f"{max(abs(x) for d, x in du.items() if d != 'kuleuven'):.3f}"
    out["dRuncKUL"] = f"{du['kuleuven']:+.3f}"
    lb = json.loads((CACHE / "sparrkulee" / "linear_baseline.json").read_text())
    out["linInnerCV"] = str(lb["n_listener_folds_inner_cv"])
    return out


if __name__ == "__main__":
    main()
