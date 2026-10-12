#!/usr/bin/env python
"""Figure: accumulation-law kappa of the linear decoder vs the Fisher-z tracking margin."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

r = json.loads(Path("paper/tables/kappa_theory.json").read_text())
a = r["absolute"]
fig, ax = plt.subplots(figsize=(3.3, 2.6))
xs = np.linspace(0, 0.3, 50)
ax.plot(xs, a["slope"] * xs, color="0.6", lw=1, zorder=1,
        label=rf"$\kappa=\sqrt{{f/2}}\,\Delta\zeta$, $f={a['f_eff']:.1f}$/s")
style = {"comp": ("o", "tab:blue", "EEG, competing"), "same": ("s", "tab:orange", "EEG, match-mismatch")}
seen = set()
for lab, v in a["points"].items():
    ds, t = lab.split("/")
    if ds == "megmasc":
        ax.scatter(v["dz"], v["kappa"], marker="*", s=90, color="tab:red", zorder=3,
                   label="MEG, match-mismatch")
        continue
    m, c, name = style[t]
    ax.scatter(v["dz"], v["kappa"], marker=m, s=22, color=c, zorder=2,
               label=None if t in seen else name)
    seen.add(t)
ax.set_xlabel(r"$\Delta\zeta$ (per-sample tracking margin)")
ax.set_ylabel(r"$\kappa$ (s$^{-1/2}$), linear decoder")
ax.set_title(f"r = {a['r']:.3f}, mean error {100 * a['mape']:.1f}%", fontsize=8)
ax.legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
fig.savefig("figures/kappa_theory.pdf")
print("wrote figures/kappa_theory.pdf")
