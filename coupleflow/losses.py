"""Training objective for CoupleFlow.

``L = CE_K + w_side * CE_side + w_con * InfoNCE + w_null * L_null
      + w_col * L_VICReg + w_adv * L_adv + w_flow * L_flow + w_env * L_env``

* ``CE_K`` -- cross-entropy over candidates, label smoothing 0.1.
* ``CE_side`` -- side classifier (no audio) against the attended side.
* ``InfoNCE`` -- two-way, EEG window i vs. attended speech of window j; the
  batch comes from one listener, so identity cannot solve it, and a constant
  encoder leaves it at log B.
* ``L_null`` -- real EEG must give the attended candidate a higher coupling
  score than another window's EEG, and than a constant (score 0) EEG, by a
  margin ``mu``.
* ``L_VICReg`` -- temporal spread of every embedding dimension above
  ``gamma``, and decorrelated dimensions.
* ``L_adv`` -- audio-only adversary; the GRL inside the model flips its
  gradient into the audio encoder.
* ``L_flow`` -- conditional flow matching on the attended envelope with
  conditioning dropout (density model, not a discriminator).
* ``L_env`` -- ``1 - r`` of the linear envelope read-out (for reporting).

Confidence-aware variants (``robust``, default ``none`` = the objective above):
``gce`` (generalised CE, q = ``robust_q``), ``focal`` (gamma = ``robust_q``),
``trim`` (after ``robust_warmup`` epochs, drop the ``trim_frac`` highest-loss
windows of each batch from CE / InfoNCE / null hinge), ``confw`` (after warm-up,
weight those terms by the detached probability of the labelled candidate,
normalised to mean 1).  ``trim``/``confw`` target lapses of attention, i.e.
windows whose instructed label is wrong.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from coupleflow.model import CoupleFlow, time_centred_corr


def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(-1, keepdim=True)
    yc = y - y.mean(-1, keepdim=True)
    den = (xc.pow(2).sum(-1) * yc.pow(2).sum(-1)).clamp_min(1e-8).sqrt()
    return (xc * yc).sum(-1) / den


def compute_losses(model: CoupleFlow, batch: dict, out: dict, w: dict) -> dict:
    """Return a dict of weighted loss terms plus ``total``."""
    y = batch["y"]
    B = y.shape[0]
    L = {}
    mode = str(w.get("robust", "none"))
    ce = F.cross_entropy(out["logits"], y, label_smoothing=0.1, reduction="none")    # (B,)
    sw = torch.ones_like(ce)                                        # per-window weights
    warm = int(w.get("_epoch", 0)) >= int(w.get("robust_warmup", 5))
    p_y = F.softmax(out["logits"].detach(), -1)[torch.arange(B), y]
    if mode == "gce":
        q = float(w.get("robust_q", 0.7))
        py = F.softmax(out["logits"], -1)[torch.arange(B), y].clamp_min(1e-6)
        L["task"] = ((1 - py.pow(q)) / q).mean()
    elif mode == "focal":
        L["task"] = ((1 - p_y).pow(float(w.get("robust_q", 2.0))) * ce).mean()
    else:
        if mode == "trim" and warm and B > 3:
            k = int(round(float(w.get("trim_frac", 0.2)) * B))
            if k > 0:
                sw[ce.detach().topk(k).indices] = 0.0
        elif mode == "confw" and warm:
            sw = p_y / p_y.mean().clamp_min(1e-6)
        L["task"] = (sw * ce).sum() / sw.sum().clamp_min(1e-6)

    if model.use_side and w.get("side", 0) > 0 and bool((batch["att_side"] >= 0).any()):
        L["side"] = w["side"] * F.cross_entropy(out["side_logits"], batch["att_side"],
                                                ignore_index=-1)

    e, a = out["e"], out["a"]
    a_pos = a[torch.arange(B), y]                                   # (B,D,T)
    if w.get("con", 0) > 0 and B > 1:
        # S[i, j] = s(e_i, a_j^+)
        S = model.log_tau.exp() * (
            time_centred_corr(e, a_pos.unsqueeze(0).expand(B, *a_pos.shape)) * model.w
        ).sum(-1)                                                   # (B,B)
        tgt = torch.arange(B, device=S.device)
        nce = 0.5 * (F.cross_entropy(S, tgt, reduction="none")
                     + F.cross_entropy(S.T, tgt, reduction="none"))
        L["con"] = w["con"] * (sw * nce).sum() / sw.sum().clamp_min(1e-6)

    if w.get("null_hinge", 0) > 0 and B > 1:
        mu = w.get("null_margin", 0.5)
        s_pos = out["s"][torch.arange(B), y]
        s_rho = model.coupling(e.roll(1, 0), a_pos.unsqueeze(1)).squeeze(1)
        hinge = F.softplus(s_rho - s_pos + mu) + F.softplus(0.0 - s_pos + mu)
        L["null"] = w["null_hinge"] * (sw * hinge).sum() / sw.sum().clamp_min(1e-6)

    if w.get("col", 0) > 0:
        ec = e - e.mean(-1, keepdim=True)                           # (B,D,T)
        std = ec.pow(2).mean(-1).clamp_min(1e-8).sqrt()            # (B,D)
        var_term = F.relu(w.get("col_gamma", 0.5) - std).mean()
        flat = ec.permute(0, 2, 1).reshape(-1, ec.shape[1])         # (BT,D)
        C = flat.T @ flat / max(flat.shape[0] - 1, 1)
        off = C - torch.diag(torch.diag(C))
        cov_term = off.pow(2).sum() / ec.shape[1]
        L["col"] = w["col"] * (var_term + 0.04 * cov_term)

    if w.get("adv", 0) > 0:
        L["adv"] = w["adv"] * F.cross_entropy(model.adversary_logits(a), y)

    if model.use_flow and w.get("flow", 0) > 0:
        x1 = model.central_env(batch["cand"])[torch.arange(B), y].unsqueeze(1)
        L["flow"] = w["flow"] * model.flow.loss(x1, e)

    if w.get("env", 0) > 0:
        M = model.margin
        tgt = batch["env_att"][:, M:batch["env_att"].shape[-1] - M]
        L["env"] = w["env"] * (1 - _pearson(out["env_hat"], tgt)).mean()

    L["total"] = sum(v for k, v in L.items())
    return L
