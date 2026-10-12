#!/usr/bin/env python
"""Bar chart of the accumulation-law kappa per dataset (64-ch EEG, native high-density, MEG).

    python scripts/figures/kappa_fig.py   # after snr_law.py --tag unified / unified_native
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

F = Path("figures")
u = json.loads((F / "snr_law_final.json").read_text())
NAMES = {"kuleuven": "KU Leuven", "dtu": "DTU", "fuglsang2020": "Fuglsang'20",
         "esaa": "ESAA", "uhd255": "KUL-255", "broderick_cp": "Broderick",
         "sparrkulee": "SparrKULee", "megmasc": "MEG-MASC"}
order = ["broderick_cp", "uhd255", "esaa", "kuleuven", "dtu", "fuglsang2020", "sparrkulee",
         "megmasc"]
fig, ax = plt.subplots(figsize=(6.5, 2.6))
x = np.arange(len(order))
for i, ds in enumerate(order):
    k = u.get(f"{ds}/same", {}).get("kappa")
    c = u.get(f"{ds}/comp", {}).get("kappa")
    if c is not None:
        ax.bar(i - 0.2, c, 0.4, color="tab:blue", label="comp., 64 ch" if i == 0 else None)
    ax.bar(i + (0.2 if c is not None else 0), k, 0.4,
           color="tab:red" if ds == "megmasc" else "tab:orange",
           label="mm, 64 ch" if i == 0 else None)
ax.axhline(norm.ppf(0.95) / np.sqrt(5), color="gray", ls=":", lw=1)
ax.text(len(order) - 1.5, norm.ppf(0.95) / np.sqrt(5) + 0.02, "95% at 5 s", ha="right",
        fontsize=7, color="gray")
ax.set_xticks(x, [NAMES[d] for d in order], rotation=20, fontsize=8)
ax.set_ylabel(r"$\kappa$ (s$^{-1/2}$)")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(F / "kappa_bars.pdf")
print("wrote", F / "kappa_bars.pdf")
