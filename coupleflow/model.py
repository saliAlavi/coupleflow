"""CoupleFlow: shortcut-proof EEG-speech coupling with a PMI flow head.

Components (see ``docs`` in the proposal):

* **EEG coupling encoder** -- per-listener rank-r channel correction, 1x1
  conv to virtual sensors, five dilated convs (dilations 1..16, RF 63
  samples = 0.98 s) with GroupNorm and *no* activation on the last layer.
  Computed without padding over an EEG window with context margins, then
  sliced so the embedding at audio time ``t`` sees EEG from
  ``t + 8 .. t + 70`` (``lag_mode="after"``, +125..+1109 ms: where a cortical
  response to the sound must be) or the mirror ``t - 70 .. t - 8``
  (``"before"``: the control, where none can be).
* **Audio encoder** shared by all candidates (envelope + onset channels),
  causal in ``"after"`` mode (anti-causal in ``"before"``) so the lag
  direction is set jointly by both encoders.
* **Time-centred coupling score** (Eq. 1 of MERIT): per-dimension Pearson
  correlation over time, combined by a bias-free weight vector, scaled by a
  learned temperature and centred over candidates.  A constant EEG embedding
  gives every candidate exactly the same score: the audio shortcut is
  impossible, not merely discouraged.
* **Side head** (no audio): predicts the attended side from EEG statistics;
  mapped onto candidates by their known side.  Its credit is measured
  separately (it can read gaze).
* **Flow head**: conditional flow matching on the attended envelope with
  conditioning dropout, so one network gives both ``p(a | EEG)`` and
  ``p(a)``.  The AAD score is the pointwise mutual information estimate
  ``R_u(a) - R_c(a)`` (unconditional minus conditional velocity residual),
  with Monte-Carlo draws shared across candidates and across the two
  passes.  If the flow ignores the EEG both residuals are equal and the
  score is zero for every candidate.
* **Audio-only adversary** behind gradient reversal.
* **Envelope head**: linear read-out of the EEG embedding, for the
  reconstruction metric.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LAG_OFFSET = 8           # samples: +125 ms at 64 Hz
EEG_RF = 63              # 1 + 2 * (1+2+4+8+16)
AUDIO_RF = 31            # 1 + 2 * (1+2+4+8)


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return -ctx.alpha * g, None


class DilatedStack(nn.Module):
    """Unpadded dilated conv stack; output length = input - (RF - 1).

    Args:
        c_in: input channels.
        c_hid: hidden width.
        c_out: output width.
        dilations: one layer per dilation, kernel 3.
    """

    def __init__(self, c_in: int, c_hid: int, c_out: int,
                 dilations: tuple[int, ...], dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        c = c_in
        for i, d in enumerate(dilations):
            last = i == len(dilations) - 1
            co = c_out if last else c_hid
            layers.append(nn.Conv1d(c, co, 3, dilation=d))
            layers.append(nn.GroupNorm(4, co))
            if not last:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            c = co
        self.net = nn.Sequential(*layers)
        self.rf = 1 + 2 * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def time_centred_corr(e: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Per-dimension Pearson correlation over time.

    Args:
        e: ``(B, D, T)`` EEG embedding.
        a: ``(B, K, D, T)`` candidate embeddings.

    Returns:
        ``(B, K, D)``.  Exactly zero wherever ``e`` is constant in time.
    """
    ec = e - e.mean(-1, keepdim=True)
    ac = a - a.mean(-1, keepdim=True)
    # Clamp inside the norm: an eps after sqrt NaNs on constant windows.
    ec = ec / ec.pow(2).sum(-1, keepdim=True).clamp_min(1e-8).sqrt()
    ac = ac / ac.pow(2).sum(-1, keepdim=True).clamp_min(1e-8).sqrt()
    return (ec.unsqueeze(1) * ac).sum(-1)


class FlowHead(nn.Module):
    """Conditional flow matching on the (z-scored) envelope, with cond dropout."""

    def __init__(self, d_cond: int, d_model: int = 64, p_uncond: float = 0.2,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16)) -> None:
        super().__init__()
        self.p_uncond = p_uncond
        self.d_t = 16
        self.inp = nn.Conv1d(1 + d_cond + 1 + self.d_t, d_model, 1)
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, dilation=d, padding=d),
                nn.GroupNorm(8, d_model), nn.GELU(),
                nn.Conv1d(d_model, d_model, 1)))
        self.out = nn.Conv1d(d_model, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _t_emb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_t // 2
        f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
        ang = t[:, None] * f[None] * 1000.0
        return torch.cat([ang.sin(), ang.cos()], -1)          # (N, d_t)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                 has_cond: torch.Tensor) -> torch.Tensor:
        """``x_t`` (N,1,T), ``t`` (N,), ``cond`` (N,D,T), ``has_cond`` (N,) -> (N,1,T)."""
        N, _, T = x_t.shape
        flag = has_cond.float().view(N, 1, 1).expand(N, 1, T)
        cond = cond * flag
        te = self._t_emb(t).unsqueeze(-1).expand(N, self.d_t, T)
        h = self.inp(torch.cat([x_t, cond, flag, te], 1))
        for layer in self.layers:
            h = h + layer(h)
        return self.out(h)

    def loss(self, x1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Flow-matching MSE on the attended envelope ``x1`` (B,1,T)."""
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device)
        x0 = torch.randn_like(x1)
        xt = (1 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
        has = torch.rand(B, device=x1.device) >= self.p_uncond
        v = self.velocity(xt, t, cond, has)
        return F.mse_loss(v, x1 - x0)

    @torch.no_grad()
    def residuals(self, cands: torch.Tensor, cond: torch.Tensor, n_mc: int = 8,
                  seed: int = 0, t_bins: int = 5) -> dict:
        """Per-candidate conditional and unconditional velocity residuals.

        Monte-Carlo draws ``(t, x0)`` are shared across candidates and across
        the conditional / unconditional passes, so their noise cancels in
        every difference we report.

        Args:
            cands: ``(B, K, T)`` z-scored candidate envelopes.
            cond: ``(B, D, T)`` EEG conditioning.

        Returns:
            ``R_c``, ``R_u``: ``(B, K)`` mean squared residuals;
            ``R_c_t``, ``R_u_t``: ``(B, K, t_bins)`` by flow-time bin;
            ``spec_c``, ``spec_u``: ``(B, K, F)`` residual power spectra.
        """
        B, K, T = cands.shape
        g = torch.Generator(device=cands.device).manual_seed(seed)
        Rc = cands.new_zeros(B, K)
        Ru = cands.new_zeros(B, K)
        Rct = cands.new_zeros(B, K, t_bins)
        Rut = cands.new_zeros(B, K, t_bins)
        cnt = cands.new_zeros(B, 1, t_bins)
        nf = T // 2 + 1
        Sc = cands.new_zeros(B, K, nf)
        Su = cands.new_zeros(B, K, nf)
        x1 = cands.reshape(B * K, 1, T)
        cond_k = cond.unsqueeze(1).expand(B, K, *cond.shape[1:]).reshape(B * K, -1, T)
        for i in range(n_mc):
            # Stratified flow time so every bin is visited.
            u = torch.rand(B, device=cands.device, generator=g)
            t = ((i + u) / n_mc).clamp(1e-3, 1 - 1e-3)
            x0 = torch.randn(B, 1, T, device=cands.device, generator=g)
            tk = t.repeat_interleave(K)
            x0k = x0.repeat_interleave(K, 0)
            xt = (1 - tk.view(-1, 1, 1)) * x0k + tk.view(-1, 1, 1) * x1
            tgt = x1 - x0k
            both_x = torch.cat([xt, xt])
            both_t = torch.cat([tk, tk])
            both_c = torch.cat([cond_k, cond_k])
            has = torch.cat([torch.ones(B * K, dtype=torch.bool, device=cands.device),
                             torch.zeros(B * K, dtype=torch.bool, device=cands.device)])
            v = self.velocity(both_x, both_t, both_c, has)
            r = v - torch.cat([tgt, tgt])                        # (2BK,1,T)
            rc, ru = r[:B * K, 0], r[B * K:, 0]
            mc = rc.pow(2).mean(-1).view(B, K)
            mu = ru.pow(2).mean(-1).view(B, K)
            Rc += mc
            Ru += mu
            b = (t * t_bins).long().clamp(max=t_bins - 1)        # (B,)
            oh = F.one_hot(b, t_bins).float().unsqueeze(1)       # (B,1,bins)
            Rct += mc.unsqueeze(-1) * oh
            Rut += mu.unsqueeze(-1) * oh
            cnt += oh
            Sc += torch.fft.rfft(rc, dim=-1).abs().pow(2).view(B, K, nf) / T
            Su += torch.fft.rfft(ru, dim=-1).abs().pow(2).view(B, K, nf) / T
        return {"R_c": Rc / n_mc, "R_u": Ru / n_mc,
                "R_c_t": Rct / cnt.clamp_min(1), "R_u_t": Rut / cnt.clamp_min(1),
                "t_count": cnt, "spec_c": Sc / n_mc, "spec_u": Su / n_mc}


class CoupleFlow(nn.Module):
    """Full model. See module docstring.

    Args:
        n_channels: EEG channels (64, zero-padded where missing).
        n_subjects: listeners, for the per-listener correction.
        margin: context samples each side of the central window.
        lag_mode: ``"after"`` (EEG after sound) or ``"before"`` (control).
    """

    def __init__(self, n_channels: int = 64, n_subjects: int = 1, margin: int = 100,
                 lag_mode: str = "after", d: int = 16, hidden: int = 32,
                 n_virtual: int = 8, rank: int = 2, dropout: float = 0.1,
                 use_side: bool = True, use_flow: bool = True,
                 flow_d_model: int = 64, p_uncond: float = 0.2,
                 score_mode: str = "independent",
                 eeg_dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
                 extra_feats: tuple[str, ...] = (), glimpse: bool = False,
                 banks: tuple[str, ...] = (), bank_dims: tuple[int, ...] = (),
                 cand_side: bool = False, score_head: str = "corr", n_sub: int = 5,
                 lag_offset: int = LAG_OFFSET) -> None:
        super().__init__()
        assert lag_mode in ("after", "before")
        self.eeg_rf = 1 + 2 * sum(eeg_dilations)
        self.lag_offset = lag_offset
        assert margin >= self.eeg_rf + lag_offset and margin >= AUDIO_RF
        self.margin, self.lag_mode, self.d = margin, lag_mode, d
        self.use_side, self.use_flow = use_side, use_flow
        # Per-listener rank-r correction x -> x + U_s V_s^T x (U starts at 0).
        self.U = nn.Parameter(torch.zeros(n_subjects, n_channels, rank))
        self.V = nn.Parameter(torch.randn(n_subjects, n_channels, rank) * 0.01)
        self.spatial = nn.Conv1d(n_channels, n_virtual, 1)
        self.eeg_enc = DilatedStack(n_virtual, hidden, d, tuple(eeg_dilations), dropout)
        self.extra_feats = list(extra_feats)      # extra candidate channels (data side)
        from coupleflow.data import n_audio_channels
        self.glimpse, self.banks, self.bank_dims = glimpse, list(banks), list(bank_dims)
        self.n_glimpse = self.bank_dims[0] if (glimpse and self.bank_dims) else 0
        n_aud = n_audio_channels(len(self.extra_feats), glimpse, self.bank_dims)
        # Candidate side as a constant input channel (+1 left, -1 right, 0 unknown):
        # lets the coupling use hemispheric lateralisation of time-locked tracking.
        self.cand_side = cand_side
        n_aud += 1 if cand_side else 0
        # "subwin": correlations on the full window and on ``n_sub`` sub-windows,
        # combined per candidate by a small MLP (zero features -> zero score).
        assert score_head in ("corr", "subwin")
        self.score_head, self.n_sub = score_head, n_sub
        self.aud_enc = DilatedStack(n_aud, hidden, d, (1, 2, 4, 8), dropout)
        # "joint": two-talker hypothesis scoring.  The EEG is compared with the
        # prediction A(a_k) + mean_{j != k} U(a_j): attended and ignored speech
        # get separate pathways (cortex tracks both, differently).
        assert score_mode in ("independent", "joint")
        self.score_mode = score_mode
        if score_mode == "joint":
            self.unatt_enc = DilatedStack(n_aud, hidden, d, (1, 2, 4, 8), dropout)
        self.w = nn.Parameter(torch.full((d,), 1.0 / d))
        if score_head == "subwin":
            self.head = nn.Sequential(nn.Linear((n_sub + 1) * d, 64), nn.GELU(),
                                      nn.Linear(64, 1))
        self.log_tau = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.env_head = nn.Conv1d(d, 1, 1)
        if use_side:
            self.side_spatial = nn.Conv1d(n_channels, 16, 1)
            self.side_enc = DilatedStack(16, hidden, d, (1, 2, 4, 8), dropout)
            self.side_fc = nn.Linear(2 * d, 2)
            self.side_gain = nn.Parameter(torch.tensor(-2.0))
        self.adv = nn.Sequential(nn.Linear(2 * d, 32), nn.GELU(), nn.Linear(32, 1))
        if use_flow:
            self.flow = FlowHead(d, flow_d_model, p_uncond)

    # ------------------------------------------------------------------ encoders
    def correct(self, eeg: torch.Tensor, subj: torch.Tensor) -> torch.Tensor:
        U, V = self.U[subj], self.V[subj]                          # (B,C,r)
        return eeg + torch.einsum("bcr,bdr,bdt->bct", U, V, eeg)

    def encode_eeg(self, eeg: torch.Tensor, subj: torch.Tensor) -> torch.Tensor:
        """``eeg`` (B,C,T+2M) -> ``e`` (B,D,T) aligned to audio time."""
        T = eeg.shape[-1] - 2 * self.margin
        y = self.eeg_enc(self.spatial(self.correct(eeg, subj)))   # (B,D,T+2M-62)
        # y[j] sees eeg[j .. j+62]; audio time t sits at index t+M.
        o = self.margin + self.lag_offset if self.lag_mode == "after" \
            else self.margin - self.lag_offset - (self.eeg_rf - 1)
        return y[..., o:o + T]

    def encode_audio(self, cand: torch.Tensor, side: torch.Tensor | None = None) -> torch.Tensor:
        """``cand`` (B,K,C,T+2M) -> ``a`` (B,K,D,T)."""
        if self.cand_side:
            sv = torch.zeros(cand.shape[:2], device=cand.device, dtype=cand.dtype)
            if side is not None:
                sv = torch.where(side == 0, -1.0, torch.where(side == 1, 1.0, 0.0)).to(cand.dtype)
            cand = torch.cat([cand, sv[:, :, None, None].expand(*cand.shape[:2], 1,
                                                                  cand.shape[-1])], 2)
        B, K, C, L = cand.shape
        T = L - 2 * self.margin
        # y[j] sees audio[j .. j+30].  "after": causal, audio[t-30..t] -> j=t+M-30.
        o = self.margin - (AUDIO_RF - 1) if self.lag_mode == "after" else self.margin
        y = self.aud_enc(cand.reshape(B * K, C, L))[..., o:o + T].reshape(B, K, self.d, T)
        if self.score_mode == "independent":
            return y
        u = self.unatt_enc(cand.reshape(B * K, C, L))[..., o:o + T].reshape(B, K, self.d, T)
        others = (u.sum(1, keepdim=True) - u) / max(K - 1, 1)   # mean over j != k
        return y + others

    def side_logits(self, eeg: torch.Tensor, subj: torch.Tensor) -> torch.Tensor:
        """No-audio side classifier -> ``(B, 2)`` logits over {L, R}."""
        M = self.margin
        x = self.correct(eeg, subj)[..., M - 15:eeg.shape[-1] - M + 15]
        h = self.side_enc(self.side_spatial(x))                    # (B,D,T)
        feat = torch.cat([h.mean(-1), h.std(-1).clamp_min(1e-4).log()], -1)
        return self.side_fc(feat)

    # ------------------------------------------------------------------ scores
    def coupling(self, e: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Uncentred coupling score ``s(e, a_k)``: ``(B, K)``.  Zero for constant ``e``."""
        c = time_centred_corr(e, a)                                   # (..., K, D)
        if self.score_head == "corr":
            return self.log_tau.exp() * (c * self.w).sum(-1)
        B, D, T0 = e.shape
        K, n = a.shape[1], self.n_sub
        t = T0 // n
        e2 = e[..., : n * t].reshape(B, D, n, t).permute(0, 2, 1, 3).reshape(B * n, D, t)
        a2 = (a[..., : n * t].reshape(B, K, D, n, t).permute(0, 3, 1, 2, 4)
              .reshape(B * n, K, D, t))
        cs = time_centred_corr(e2, a2).reshape(B, n, K, D).permute(0, 2, 1, 3).reshape(B, K, n * D)
        feat = torch.cat([c, cs], -1)
        zero = self.head(torch.zeros_like(feat[..., :1, :]))
        return self.log_tau.exp() * 0.1 * (self.head(feat) - zero).squeeze(-1)

    @staticmethod
    def centre(s: torch.Tensor) -> torch.Tensor:
        return s - s.mean(-1, keepdim=True)

    def side_scores(self, side_logits: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """Map side logits onto candidates; unknown side (-1) contributes 0."""
        lp = side_logits.log_softmax(-1)                           # (B,2)
        known = side >= 0
        sc = torch.gather(lp, 1, side.clamp_min(0)) * known
        sc = torch.where(known.all(-1, keepdim=True), self.centre(sc),
                         torch.zeros_like(sc))
        return F.softplus(self.side_gain) * sc

    def forward(self, batch: dict) -> dict:
        """Encode once; return everything the losses and evaluator need."""
        eeg, cand, subj = batch["eeg"], batch["cand"], batch["subj"]
        e = self.encode_eeg(eeg, subj)
        a = self.encode_audio(cand, batch.get("side"))
        s = self.coupling(e, a)
        out = {"e": e, "a": a, "s": s, "coupling_logits": self.centre(s)}
        if self.use_side:
            sl = self.side_logits(eeg, subj)
            out["side_logits"] = sl
            out["side_cand"] = self.side_scores(sl, batch["side"])
            out["logits"] = out["coupling_logits"] + out["side_cand"]
        else:
            out["logits"] = out["coupling_logits"]
        out["env_hat"] = self.env_head(e).squeeze(1)
        return out

    def adversary_logits(self, a: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Audio-only candidate classifier behind a GRL: ``(B, K)``."""
        a = _GRL.apply(a, alpha)
        f = torch.cat([a.mean(-1), a.std(-1)], -1)                 # (B,K,2D)
        return self.adv(f).squeeze(-1)

    def central_env(self, cand: torch.Tensor) -> torch.Tensor:
        """z-scored envelope channel of candidates over the central window: (B,K,T)."""
        M = self.margin
        return cand[:, :, 0, M:cand.shape[-1] - M]
