#!/usr/bin/env python
"""Figure 1 signal thumbnails (real MAESTRO data) for the TikZ diagram.

The diagram itself is `paper/fig_arch.tex` (TikZ, typeset with the
paper's fonts); this script only draws the two text-free signal thumbnails it embeds.

    python scripts/figures/fig_architecture.py
    # -> figures/arch_eeg.pdf, figures/arch_env.pdf
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = np.load(paths.REPO / "figures" / "data" / "fig1_data.npz")
OUT = Path("figures")
NEURAL, AUDIO, AUDIO_LIGHT = "#2a78d6", "#eb6834", "#f5c1a8"
SIZE = (0.95, 0.62)                                        # inches, matches fig_arch.tex
CTX = 100                                                  # context samples on each side


def _axes() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


def eeg_thumb() -> None:
    fig, ax = _axes()
    sig = D["eeg"][[0, 1, 2, 3, 4, 5, 7], CTX:-CTX]
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-6)
    t = np.arange(sig.shape[1])
    for i, s in enumerate(sig):
        ax.plot(t, -3.2 * i + 0.9 * np.clip(s, -3, 3), color=NEURAL, lw=0.35)
    ax.set_xlim(0, t[-1])
    ax.margins(y=0.02)
    fig.savefig(OUT / "arch_eeg.pdf", transparent=True)


def env_thumb() -> None:
    fig, ax = _axes()
    att = int(D["att"])
    env = D["env"][:, CTX:-CTX]
    t = np.arange(env.shape[1])
    for k, e in enumerate(env):
        e = (e - e.min()) / (e.max() - e.min() + 1e-9)
        base = -1.15 * k
        ax.fill_between(t, base, base + e, color=AUDIO if k == att else AUDIO_LIGHT, lw=0)
    ax.set_xlim(0, t[-1])
    ax.margins(y=0.02)
    fig.savefig(OUT / "arch_env.pdf", transparent=True)


if __name__ == "__main__":
    eeg_thumb()
    env_thumb()
    print("wrote", OUT / "arch_eeg.pdf", OUT / "arch_env.pdf")
