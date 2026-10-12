#!/usr/bin/env python
"""Fit the evidence-accumulation law  Acc(T) = Phi(kappa * sqrt(T))  per dataset.

Derivation (paper, Prop. 2): with a decoder output correlated rho_a with the
attended and rho_u with the competing envelope, over a window holding N_eff =
f*T effectively independent samples, Fisher-z statistics give
z_a - z_u ~ N(Delta_zeta, 2 / (N_eff - 3)), so
    P(correct) = Phi(Delta_zeta * sqrt((f T - 3) / 2)) ~= Phi(kappa sqrt(T)),
    kappa = Delta_zeta * sqrt(f / 2).
One parameter per dataset and task; the fit uses every window length, and the
leave-one-window-out error tests whether the sqrt(T) law (rather than a free
curve) explains accuracy.

    python scripts/figures/snr_law.py --tag unified --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.analysis.aggregate import WINDOWS, load, pooled  # noqa: E402

SECS = np.array([float(w[:-1]) for w in WINDOWS])


def fit_kappa(T: np.ndarray, acc: np.ndarray, n: np.ndarray) -> float:
    """Binomial maximum-likelihood kappa."""
    acc = np.clip(acc, 1e-4, 1 - 1e-4)

    def nll(k: float) -> float:
        p = np.clip(norm.cdf(k * np.sqrt(T)), 1e-6, 1 - 1e-6)
        return -float(np.sum(n * (acc * np.log(p) + (1 - acc) * np.log(1 - p))))
    return float(minimize_scalar(nll, bounds=(1e-4, 5.0), method="bounded").x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="unified")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    runs = load(args.tag)
    out, rows = {}, []
    for ds in sorted(runs):
        for task, pre in (("comp", "competing"), ("same", "same_talker")):
            T, A, N = [], [], []
            for s, w in zip(SECS, WINDOWS):
                a = pooled(runs[ds], f"{pre}_{w}", "acc_coupling")
                n = sum(r["results"].get(f"{pre}_{w}", {}).get("n_windows", 0) for r in runs[ds])
                if a is not None and n > 0:
                    T.append(s), A.append(a), N.append(n)
            if len(T) < 3:
                continue
            T, A, N = map(np.array, (T, A, N))
            k = fit_kappa(T, A, N)
            pred = norm.cdf(k * np.sqrt(T))
            loo = []
            for i in range(len(T)):
                m = np.arange(len(T)) != i
                loo.append(abs(norm.cdf(fit_kappa(T[m], A[m], N[m]) * np.sqrt(T[i])) - A[i]))
            # windows needed for 95 % under the law
            t95 = (norm.ppf(0.95) / k) ** 2
            out[f"{ds}/{task}"] = {"kappa": k, "T": T.tolist(), "acc": A.tolist(),
                                   "pred": pred.tolist(), "mae": float(np.mean(np.abs(pred - A))),
                                   "loo_mae": float(np.mean(loo)), "T95_s": t95}
            rows.append((ds, task, k, float(np.mean(np.abs(pred - A))), float(np.mean(loo)), t95))
    print("| dataset | task | kappa (1/sqrt(s)) | MAE | LOO-MAE | T for 95% (s) |")
    print("|---|---|---|---|---|---|")
    for ds, task, k, mae, loo, t95 in rows:
        print(f"| {ds} | {task} | {k:.3f} | {100 * mae:.1f} | {100 * loo:.1f} | {t95:.1f} |")
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"snr_law_{args.tag}.json").write_text(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task, title in ((axes[0], "comp", "attended vs competing talker"),
                            (axes[1], "same", "match vs mismatch (same talker)")):
        for key, v in out.items():
            ds, t = key.split("/")
            if t != task:
                continue
            tt = np.linspace(0.5, 65, 200)
            line, = ax.plot(tt, 100 * norm.cdf(v["kappa"] * np.sqrt(tt)), lw=1)
            ax.plot(v["T"], 100 * np.array(v["acc"]), "o", color=line.get_color(), ms=4,
                    label=f"{ds} (κ={v['kappa']:.2f})")
        ax.set_xscale("log")
        ax.set_xlabel("decision window T (s)")
        ax.set_title(title)
        ax.axhline(95, color="gray", ls=":", lw=0.8)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("accuracy (%)")
    fig.tight_layout()
    fig.savefig(od / f"snr_law_{args.tag}.pdf")
    print(f"wrote {od / f'snr_law_{args.tag}.pdf'}")


if __name__ == "__main__":
    main()
