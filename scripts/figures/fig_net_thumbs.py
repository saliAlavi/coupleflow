#!/usr/bin/env python
"""Real-data thumbnails for the network-details figures (App. D).

From the KU Leuven fold-0 checkpoint (seed 0) and one test window of that fold:

* ``net_vs_topo.pdf``   -- the 8 learned virtual-sensor filters (``spatial.weight``).
* ``net_lc_topo.pdf``   -- one listener's rank-2 correction: read-out patterns V and
  write-back patterns U.
* ``net_feat_*.pdf``    -- the candidate features of one 5-s window: broadband envelope
  (before / after quantile matching), onset, 8-band sub-band envelopes, glimpsed and
  masked channels, value distributions before / after matching.

    python scripts/figures/fig_net_thumbs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.preprocessing.channels import STANDARD_64_CHANNELS  # noqa: E402
from coupleflow.data import (_z, glimpse_channels, load_dataset,  # noqa: E402
                                 quantile_match, split_fold)

RUN = paths.RUNS / "final" / "kuleuven_fold0"
FIGS = Path("figures")
NEURAL, ACOUSTIC, INK, MUTED = "#2A78D6", "#EB6834", "#2b2b2b", "#7A7974"
# Diverging blue <-> red with a neutral grey midpoint (validated palette).
DIV = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b3261e"])
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
SR, MARGIN, WIN = 64, 100, 320


def info64() -> mne.Info:
    info = mne.create_info(list(STANDARD_64_CHANNELS), 64.0, "eeg")
    mont = mne.channels.make_standard_montage("standard_1005")
    low = {n.lower(): n for n in mont.ch_names}
    ren = {c: low[c.lower()] for c in info.ch_names if c.lower() in low}
    info.rename_channels(ren)
    info.set_montage(mont, on_missing="ignore")
    return info


def topo_row(maps: np.ndarray, titles: list[str], path: Path, width: float) -> None:
    info = info64()
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(width, width / n * 1.12))
    for ax, m, t in zip(np.atleast_1d(axes), maps, titles):
        v = np.abs(m).max()
        mne.viz.plot_topomap(m, info, axes=ax, show=False, cmap=DIV, vlim=(-v, v),
                             contours=0, sensors=False, outlines="head")
        ax.set_title(t, fontsize=7, color=INK, pad=1)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.86, wspace=0.08)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def spatial_maps(data: dict, train_segs: list) -> None:
    """Virtual sensors as activation patterns A = C W^T (W C W^T)^-1 (Haufe et al. 2014),
    C the covariance of the listener-corrected training EEG; filters W are hard to read."""
    ck = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    sd, subjects = ck["model"], ck["subjects"]
    W = sd["spatial.weight"][:, :, 0].numpy()                      # (8, 64)
    U, V = sd["U"].numpy(), sd["V"].numpy()                         # (L, 64, 2)
    C, n = np.zeros((64, 64)), 0
    for sg in train_segs:
        t = data["trials"][sg.trial]
        x = t["eeg"][:, sg.start:sg.end].float().numpy()
        li = subjects.index(t["subject"])
        x = x + U[li] @ (V[li].T @ x)                               # listener correction
        C += x @ x.T
        n += x.shape[1]
    C /= n
    A = C @ W.T @ np.linalg.inv(W @ C @ W.T)                        # (64, 8)
    topo_row(A.T, [f"sensor {i + 1}" for i in range(8)], FIGS / "net_vs_topo.pdf", 6.4)
    li = int(np.argmax(np.linalg.norm(U, axis=(1, 2))))             # most-corrected listener
    maps = [V[li, :, 0], V[li, :, 1], U[li, :, 0], U[li, :, 1]]
    topo_row(np.array(maps), ["$V_\\ell$ col. 1", "$V_\\ell$ col. 2",
                              "$U_\\ell$ col. 1", "$U_\\ell$ col. 2"],
             FIGS / "net_lc_topo.pdf", 3.4)


def load():
    data = load_dataset(str(paths.DATA), "kuleuven",
                        stream_name="unified_env64.pt", groups_name="unified_groups.json")
    return data, split_fold(data, 0, 5, MARGIN, val_mode="auto", seed=42)


def window(data: dict, sp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw attended / competing envelope and 8-band sub-bands of one KU Leuven test window."""
    seg = sp["test"][0]
    t = data["trials"][seg.trial]
    a = seg.start + 40 * SR                                         # 40 s into the segment
    L = WIN + 2 * MARGIN
    envs = torch.stack([t["env_att"][a:a + L], t["env_unatt"][a:a + L]]).float()
    sb = torch.stack([t["env_att_sb"][:, a:a + L], t["env_unatt_sb"][:, a:a + L]]).float()
    return envs, sb


def trace(ax, y, col, lw=0.8, **kw):
    ax.plot(np.arange(len(y)) / SR, y, color=col, lw=lw, **kw)


def clean(ax, label=None):
    ax.set_yticks([])
    ax.set_xlim(0, WIN / SR)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if label:
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7, color=INK)


def feature_panels(data: dict, sp: dict) -> None:
    envs, sb = window(data, sp)
    c = slice(MARGIN, MARGIN + WIN)
    x = _z(envs, c)
    xm = quantile_match(x)          # identical value multisets over the whole window
    xm_z = _z(xm, c)
    onset = _z(torch.relu(torch.diff(xm_z, dim=-1, prepend=xm_z[..., :1])), c)
    sbz = torch.stack([_z(sb[:, b], c) for b in range(sb.shape[1])], 1)   # (2, 8, L)
    g, m = glimpse_channels(sbz[:, 3], c)                                   # one band

    # --- time panels ------------------------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(3.3, 3.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.5, 1.1]})
    trace(axes[0], x[0, c], ACOUSTIC, label="talker 1")
    trace(axes[0], x[1, c], MUTED, lw=0.7, label="talker 2")
    clean(axes[0], "envelope\n(z-scored)")
    trace(axes[1], xm_z[0, c], ACOUSTIC)
    trace(axes[1], xm_z[1, c], MUTED, lw=0.7)
    clean(axes[1], "quantile-\nmatched")
    trace(axes[2], onset[0, c], ACOUSTIC)
    clean(axes[2], "onset")
    axes[3].imshow(sbz[0, :, c].numpy(), aspect="auto", origin="lower", cmap="Oranges",
                   extent=(0, WIN / SR, 0.5, 8.5), vmin=-1.5, vmax=3)
    clean(axes[3], "8 sub-band\nenvelopes")
    trace(axes[4], g[0, c], ACOUSTIC, label="glimpsed")
    trace(axes[4], m[0, c], "#1BAF7A", lw=0.7, label="masked")
    clean(axes[4], "glimpsed /\nmasked (band 4)")
    axes[4].set_xlabel("time (s)", fontsize=7)
    for ax in (axes[0], axes[4]):
        lg = ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower right",
                       bbox_to_anchor=(1.0, 0.98), borderaxespad=0, handlelength=1.6)
        for h in lg.get_lines():
            h.set_linewidth(1.6)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.93, bottom=0.12, hspace=0.45)
    fig.savefig(FIGS / "net_feat_time.pdf", transparent=True)
    plt.close(fig)

    # --- value distributions before / after matching ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(1.9, 0.95), sharey=True)
    bins = np.linspace(-2.2, 3.5, 24)
    for ax, y, tt in ((axes[0], x, "before"), (axes[1], xm, "after")):
        ax.hist(y[0].numpy(), bins, color=ACOUSTIC, alpha=0.75, label="talker 1")
        ax.hist(y[1].numpy(), bins, histtype="step", color=INK, lw=0.8, label="talker 2")
        ax.set_title(tt, fontsize=7, color=INK, pad=1)
        ax.set_yticks([])
        ax.set_xticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.05, wspace=0.12)
    fig.savefig(FIGS / "net_feat_hist.pdf", transparent=True)
    plt.close(fig)


def embeddings(data: dict, sp: dict) -> None:
    """The trained model's EEG and candidate embeddings on the same test window, the
    per-dimension correlations and the learned weights (coupling-score figure)."""
    from omegaconf import OmegaConf

    from coupleflow.data import build_eval_batch
    from coupleflow.evaluate import encode
    from coupleflow.model import time_centred_corr
    from coupleflow.trainer import build_model, feature_banks
    cfg = OmegaConf.load(RUN / "config.yaml")
    ck = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    t0 = data["trials"][0]
    bd = [int(t0[f"env_att_{b}"].shape[0]) for b in feature_banks(cfg)]
    model = build_model(cfg, len(data["subjects"]), bd)
    model.load_state_dict(ck["model"])
    model.eval()
    seg = sp["test"][0]
    wins = [{"trial": seg.trial, "start": seg.start + 40 * SR}]
    b = build_eval_batch(data, wins, WIN, MARGIN, "competing", glimpse=model.glimpse,
                         banks=model.banks, n_glimpse=model.n_glimpse)
    with torch.no_grad():
        enc = encode(model, b, "cpu")
        e, a = enc["e"][0], enc["a"][0]                               # (16,T), (2,16,T)
        rho = time_centred_corr(enc["e"], enc["a"])[0].numpy()        # (2, 16)
    w = model.w.detach().numpy()
    wa = np.abs(w).max()

    def strip(z, path, cmap):
        z = (z - z.mean(-1, keepdims=True)) / (z.std(-1, keepdims=True) + 1e-8)
        fig, ax = plt.subplots(figsize=(2.3, 0.62))
        ax.imshow(z, aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(path, transparent=True)
        plt.close(fig)
    strip(e.numpy(), FIGS / "net_emb_e.pdf", DIV)
    strip(a[0].numpy(), FIGS / "net_emb_a1.pdf", DIV)
    strip(a[1].numpy(), FIGS / "net_emb_a2.pdf", DIV)

    fig, axes = plt.subplots(2, 1, figsize=(2.3, 1.35), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    d = np.arange(16)
    axes[0].bar(d - 0.2, rho[0], 0.4, color=ACOUSTIC, label="$\\hat a_1$ (attended)")
    axes[0].bar(d + 0.2, rho[1], 0.4, color=MUTED, label="$\\hat a_2$")
    axes[0].axhline(0, color=INK, lw=0.4)
    axes[0].set_ylabel("$\\rho_{k,d}$", fontsize=7)
    axes[0].legend(frameon=False, fontsize=5.5, ncol=2, loc="lower right",
                   bbox_to_anchor=(1, 0.98), borderaxespad=0)
    axes[1].bar(d, w, 0.6, color="#1BAF7A")
    axes[1].axhline(0, color=INK, lw=0.4)
    axes[1].set_ylim(-1.1 * wa, 1.1 * wa)
    axes[1].set_ylabel("$w_d$", fontsize=7)
    axes[1].set_xticks([0, 5, 10, 15])
    axes[1].set_xticklabels(["1", "6", "11", "16"])
    axes[1].set_xlabel("embedding dimension $d$", fontsize=6.5)
    for ax in axes:
        ax.tick_params(labelsize=5.5, length=2)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.26, hspace=0.25)
    fig.savefig(FIGS / "net_emb_corr.pdf", transparent=True)
    plt.close(fig)
    s = model.log_tau.exp().item() * (rho * w).sum(-1)
    sc = s - s.mean()
    tau = 1 / model.log_tau.exp().item()
    Path("paper/tables/net_values.tex").write_text(
        "% generated by scripts/figures/fig_net_thumbs.py (KU Leuven fold 0, one test window)\n"
        f"\\newcommand{{\\netSOne}}{{{sc[0]:+.2f}}}\n"
        f"\\newcommand{{\\netSTwo}}{{{sc[1]:+.2f}}}\n"
        f"\\newcommand{{\\netTau}}{{{tau:.3f}}}\n")
    print("scores", s, "centred", sc, "tau", tau)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    data, sp = load()
    spatial_maps(data, sp["train"])
    feature_panels(data, sp)
    embeddings(data, sp)
    print("wrote", *sorted(p.name for p in FIGS.glob("net_*.pdf")))
