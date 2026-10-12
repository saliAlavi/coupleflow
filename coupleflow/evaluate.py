"""Evaluation for CoupleFlow: accuracy, nulls, per-stream credit, flow residuals.

Every accuracy is reported next to its own shuffled baseline.

Streams and credit
------------------
The model has two routes from EEG to the decision: the coupling path (EEG
vs. candidate speech over time) and the side path (EEG -> attended side, no
audio).  ``v(T)`` is accuracy when the streams in ``T`` keep their own EEG
and the rest receive EEG from a window of a *different trial* of the same
listener (candidates and labels untouched).  ``v(empty)`` is what this model
gets from the audio alone.  Shapley credit splits ``v(all) - v(empty)``
exactly across the two streams.

Two nulls
---------
* ``across_trial`` -- EEG from another trial: removes everything the EEG
  carries (timing *and* trial identity).
* ``circular`` -- the trial's own EEG, circularly shifted by >= 2 s before
  windowing: keeps listener, trial, and slow state, breaks only the
  time-locking.  A decoder that recognises the trial passes the first null
  but fails this one.

Flow residuals (per source)
---------------------------
For each window and candidate: conditional residual ``R_c`` and
unconditional residual ``R_u``, with shared Monte-Carlo draws.
``PMI = R_u - R_c`` estimates how much the EEG explains that candidate.
We report the 2x2 table, ``PMI_att``, ``PMI_un``, the attention index,
the audio-prior margin ``R_u(un) - R_u(att)`` (the shortcut the old
AttuneFlow could use), the decision ``PMI_att - PMI_un``, and breakdowns by
flow time and by modulation frequency.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from coupleflow.data import build_eval_batch, eval_windows

SR = 64
BANDS = {"delta_0.5-4": (0.5, 4.0), "theta_4-8": (4.0, 8.0),
         "alpha_8-13": (8.0, 13.0), "beta_13-32": (13.0, 32.0)}


# =============================================================================
# Encoding
# =============================================================================

@torch.no_grad()
def encode(model, batch: dict, device: str, chunk: int = 256,
           zero_eeg: bool = False) -> dict:
    """Encode all windows once: EEG embedding, candidate embeddings, side logits."""
    model.eval()
    N = batch["eeg"].shape[0]
    es, as_, sls, envh = [], [], [], []
    for i in range(0, N, chunk):
        eeg = batch["eeg"][i:i + chunk].to(device)
        if zero_eeg:
            eeg = torch.zeros_like(eeg)
        subj = batch["subj"][i:i + chunk].to(device)
        e = model.encode_eeg(eeg, subj)
        a = model.encode_audio(batch["cand"][i:i + chunk].to(device),
                               batch["side"][i:i + chunk].to(device))
        es.append(e)
        as_.append(a)
        envh.append(model.env_head(e).squeeze(1))
        if model.use_side:
            sls.append(model.side_logits(eeg, subj))
    out = {"e": torch.cat(es), "a": torch.cat(as_), "env_hat": torch.cat(envh)}
    out["side_logits"] = torch.cat(sls) if sls else None
    return out


@torch.no_grad()
def score(model, e: torch.Tensor, a: torch.Tensor, side_logits: torch.Tensor | None,
          side: torch.Tensor, chunk: int = 512) -> dict:
    """Candidate logits per rule: ``coupling``, ``side``, ``fused`` (N,K)."""
    cs = []
    for i in range(0, e.shape[0], chunk):
        cs.append(model.centre(model.coupling(e[i:i + chunk], a[i:i + chunk])))
    coup = torch.cat(cs)
    out = {"coupling": coup}
    # The side rule only exists when candidate sides are known (not same-talker).
    if side_logits is not None and bool((side >= 0).all()):
        sc = model.side_scores(side_logits, side.to(e.device))
        out["side"] = sc
        out["fused"] = coup + sc
    else:
        out["fused"] = coup
    return out


def _oth(x: torch.Tensor, how: str = "max") -> torch.Tensor:
    """Competitors of candidate 0 (index 1..K-1): max or mean. Equals ``x[:, 1]`` at K = 2."""
    return x[:, 1:].max(1).values if how == "max" else x[:, 1:].mean(1)


def _acc(logits: torch.Tensor) -> torch.Tensor:
    """Per-window correctness (attended is candidate 0; ties count as wrong)."""
    return (logits[:, 0] > logits[:, 1:].max(-1).values).float()


# =============================================================================
# Permutations
# =============================================================================

def across_trial_perm(subj: np.ndarray, trial: np.ndarray, rng: np.random.Generator
                      ) -> np.ndarray:
    """Permutation within listener such that every window gets another trial's EEG."""
    perm = np.arange(len(subj))
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        if len(np.unique(trial[idx])) < 2:
            perm[idx] = rng.permutation(idx)
            continue
        for _ in range(50):
            p = rng.permutation(idx)
            bad = trial[p] == trial[idx]
            if not bad.any():
                break
            # Repair collisions by swapping with random non-colliding partners.
            for j in np.where(bad)[0]:
                cands = np.where((trial[p] != trial[idx[j]]) & (trial[idx] != trial[p[j]]))[0]
                if len(cands):
                    k = rng.choice(cands)
                    p[j], p[k] = p[k], p[j]
            if not (trial[p] == trial[idx]).any():
                break
        perm[idx] = p
    return perm


def shapley_two(v: dict[frozenset, float], streams: list[str]) -> dict[str, float]:
    """Exact Shapley values from the coalition table ``v``."""
    n = len(streams)
    phi = {}
    for m in streams:
        others = [s for s in streams if s != m]
        tot = 0.0
        for r in range(len(others) + 1):
            for T in itertools.combinations(others, r):
                wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                tot += wgt * (v[frozenset(T) | {m}] - v[frozenset(T)])
        phi[m] = tot
    return phi


# =============================================================================
# Main evaluation for one (window, candidate set)
# =============================================================================

def _pearson_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(-1, keepdim=True)
    yc = y - y.mean(-1, keepdim=True)
    den = (xc.pow(2).sum(-1) * yc.pow(2).sum(-1)).clamp_min(1e-8).sqrt()
    return (xc * yc).sum(-1) / den


def _trial_acc(logits: torch.Tensor, trial: np.ndarray) -> tuple[float, int]:
    """Sum window logits per trial, decide once per trial."""
    tot = {}
    for t in np.unique(trial):
        m = torch.from_numpy(trial == t).to(logits.device)
        tot[t] = logits[m].sum(0)
    L = torch.stack(list(tot.values()))
    return float(_acc(L).mean()), len(tot)


@torch.no_grad()
def evaluate_set(model, data: dict, segments, win_s: float, cand_set: str,
                 device: str, n_perm: int = 20, n_circ: int = 5, seed: int = 0,
                 full: bool = True, n_mc: int = 8) -> dict:
    """Evaluate one window length on one candidate set. See module docstring."""
    win = int(round(win_s * SR))
    margin = model.margin
    wins = eval_windows(data, segments, win, margin)
    if not wins:
        return {"n_windows": 0}
    xf = getattr(model, "extra_feats", [])
    gl = bool(getattr(model, "glimpse", False))
    bk = list(getattr(model, "banks", []))
    ng = int(getattr(model, "n_glimpse", 0))
    batch = build_eval_batch(data, wins, win, margin, cand_set, seed=seed, extra_feats=xf,
                             glimpse=gl, banks=bk, n_glimpse=ng)
    if batch is None:
        return {"n_windows": 0}
    N = batch["eeg"].shape[0]
    subj = batch["subj"].numpy()
    trial = batch["trial"]
    enc = encode(model, batch, device)
    side = batch["side"]
    real = score(model, enc["e"], enc["a"], enc["side_logits"], side)
    res: dict = {"n_windows": int(N), "n_trials": int(len(np.unique(trial))),
                 "n_subjects": int(len(np.unique(subj)))}
    rules = list(real.keys())
    for r in rules:
        res[f"acc_{r}"] = float(_acc(real[r]).mean())

    # ---- across-trial null and per-stream coalitions ----------------------
    rng = np.random.default_rng(seed + 17)
    has_side = enc["side_logits"] is not None and bool((side >= 0).all())
    streams = ["eeg_coupling"] + (["eeg_side"] if has_side else [])
    null = {r: [] for r in rules}
    coal = {frozenset(T): [] for k in range(len(streams) + 1)
            for T in itertools.combinations(streams, k)}
    flips = []
    real_dec = real["fused"].argmax(-1)
    for _ in range(n_perm if full else 3):
        p = torch.from_numpy(across_trial_perm(subj, trial, rng)).to(device)
        e_p = enc["e"][p]
        sl_p = enc["side_logits"][p] if enc["side_logits"] is not None else None
        shuf = score(model, e_p, enc["a"], sl_p, side)
        for r in rules:
            null[r].append(float(_acc(shuf[r]).mean()))
        flips.append(float((shuf["fused"].argmax(-1) != real_dec).float().mean()))
        for T in coal:
            e_use = enc["e"] if "eeg_coupling" in T else e_p
            sl_use = None
            if enc["side_logits"] is not None:
                sl_use = enc["side_logits"] if "eeg_side" in T else sl_p
            coal[T].append(float(_acc(score(model, e_use, enc["a"], sl_use, side)["fused"]).mean()))
    for r in rules:
        nm = float(np.mean(null[r]))
        res[f"null_{r}"] = nm
        res[f"contrib_{r}"] = res[f"acc_{r}"] - nm
        res[f"p_{r}"] = (sum(x >= res[f"acc_{r}"] for x in null[r]) + 1) / (len(null[r]) + 1)
    res["flip_rate"] = float(np.mean(flips))
    v = {T: float(np.mean(x)) for T, x in coal.items()}
    res["coalitions"] = {"+".join(sorted(T)) or "none": x for T, x in v.items()}
    res["shapley"] = shapley_two(v, streams)
    res["shapley_total"] = v[frozenset(streams)] - v[frozenset()]
    if not full:
        return res

    # ---- zero-EEG and collapse ------------------------------------------------
    enc0 = encode(model, batch, device, zero_eeg=True)
    zero = score(model, enc0["e"], enc["a"], enc0["side_logits"], side)
    res["acc_fused_zero_eeg"] = float(_acc(zero["fused"]).mean())
    ec = enc["e"] - enc["e"].mean(-1, keepdim=True)
    flat = ec.reshape(N, -1)
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    i = torch.from_numpy(rng.integers(0, N, 2000)).to(device)
    j = torch.from_numpy(rng.integers(0, N, 2000)).to(device)
    res["collapse"] = float((flat[i] * flat[j]).sum(-1).mean())

    # ---- circular-shift (time-locking-only) null ---------------------------
    circ = {r: [] for r in rules}
    for c in range(n_circ):
        crng = np.random.default_rng(seed + 1000 + c)
        over = {}
        for ti in np.unique(trial):
            x = data["trials"][ti]["eeg"]
            n = x.shape[1]
            lo, hi = 2 * SR, max(2 * SR + 1, n - 2 * SR)
            over[int(ti)] = torch.roll(x, int(crng.integers(lo, hi)), dims=1)
        # Only the EEG changes: keep every window and its candidates exactly as scored
        # (re-drawing candidates would re-sample same-talker negatives).
        L = win + 2 * margin
        b2 = dict(batch)
        b2["eeg"] = torch.stack([over[int(w["trial"])][:, w["start"]:w["start"] + L].float()
                                 for w in batch["wins"]])
        enc2 = encode(model, b2, device)
        sc2 = score(model, enc2["e"], enc["a"], enc2["side_logits"], side)
        for r in rules:
            circ[r].append(float(_acc(sc2[r]).mean()))
    for r in rules:
        res[f"null_circ_{r}"] = float(np.mean(circ[r]))
        res[f"contrib_circ_{r}"] = res[f"acc_{r}"] - res[f"null_circ_{r}"]

    # ---- per-subject and per-trial -------------------------------------------
    subjects = data["subjects"]
    acc_f = _acc(real["fused"]).cpu().numpy()
    acc_c = _acc(real["coupling"]).cpu().numpy()
    res["per_subject"] = {subjects[s]: {"fused": float(acc_f[subj == s].mean()),
                                        "coupling": float(acc_c[subj == s].mean()),
                                        "n": int((subj == s).sum())}
                          for s in np.unique(subj)}
    # Raw per-window margins (attended minus other), for offline ensembling.
    res["_windows"] = {"trial": trial.astype(int).tolist(), "subject": subj.astype(int).tolist(),
                       **{f"margin_{r}": (real[r][:, 0] - _oth(real[r])).cpu().tolist()
                          for r in rules}}
    # Per-trial breakdown: correlated errors within a trial show up here.
    coup_margin = real["coupling"][:, 0] - _oth(real["coupling"])
    res["per_trial"] = []
    for ti in np.unique(trial):
        m = trial == ti
        t = data["trials"][int(ti)]
        res["per_trial"].append({
            "trial": t["trial"], "subject": t["subject"],
            "attended_ear": t["attended_ear"], "n": int(m.sum()),
            "acc_coupling": float(acc_c[m].mean()), "acc_fused": float(acc_f[m].mean()),
            "sum_margin_coupling": float(coup_margin[torch.from_numpy(m).to(coup_margin.device)]
                                         .sum()),
        })
    # Per-trial decisions need the same candidates in every window of a trial, which
    # the same-talker sets (fresh negatives per window) do not have.
    if not cand_set.startswith("same_talker"):
        for r in rules:
            res[f"trial_acc_{r}"], res["n_trial_decisions"] = _trial_acc(real[r], trial)
        p = torch.from_numpy(across_trial_perm(subj, trial, rng)).to(device)
        shuf = score(model, enc["e"][p], enc["a"],
                     enc["side_logits"][p] if enc["side_logits"] is not None else None, side)
        res["trial_null_fused"], _ = _trial_acc(shuf["fused"], trial)
        res["trial_null_coupling"], _ = _trial_acc(shuf["coupling"], trial)

    # ---- envelope reconstruction ---------------------------------------------
    r_att = _pearson_rows(enc["env_hat"].cpu(), batch["env_att"])
    r_un = _pearson_rows(enc["env_hat"].cpu(), batch["env_unatt"])
    res["recon_r_att"] = float(r_att.mean())
    res["recon_r_unatt"] = float(r_un.mean())
    if cand_set == "competing":
        res["acc_env_r"] = float((r_att > r_un).float().mean())

    # ---- flow: per-source residuals --------------------------------------------
    if model.use_flow:
        res["flow"] = flow_report(model, enc, batch, device, rng, subj, trial,
                                  real["fused"], n_mc=n_mc, seed=seed)
        res["_windows"]["margin_flow_pmi"] = res["flow"].pop("_pmi_margin")
    return res


@torch.no_grad()
def flow_report(model, enc: dict, batch: dict, device: str, rng, subj, trial,
                fused: torch.Tensor, n_mc: int = 8, seed: int = 0,
                chunk: int = 128) -> dict:
    """2x2 residual table, PMI scores, nulls, breakdowns by t and frequency."""
    cands = model.central_env(batch["cand"]).to(device)            # (N,K,T)
    N, K, T = cands.shape
    p = torch.from_numpy(across_trial_perm(subj, trial, rng)).to(device)

    def run(e):
        parts = [model.flow.residuals(cands[i:i + chunk], e[i:i + chunk], n_mc, seed + i)
                 for i in range(0, N, chunk)]
        return {k: torch.cat([q[k] for q in parts]) for k in parts[0]}

    R = run(enc["e"])
    Rs = run(enc["e"][p])
    pmi = R["R_u"] - R["R_c"]                                      # (N,K)
    pmi_s = Rs["R_u"] - Rs["R_c"]
    out = {}

    def m(x):
        return float(x.mean())

    def se(x):
        return float(x.std() / math.sqrt(max(len(x), 1)))

    out["R_c_att"], out["R_c_un"] = m(R["R_c"][:, 0]), m(_oth(R["R_c"], "mean"))
    out["R_u_att"], out["R_u_un"] = m(R["R_u"][:, 0]), m(_oth(R["R_u"], "mean"))
    out["PMI_att"], out["PMI_un"] = m(pmi[:, 0]), m(_oth(pmi, "mean"))
    out["PMI_att_se"], out["PMI_un_se"] = se(pmi[:, 0]), se(_oth(pmi, "mean"))
    out["attention_index"] = (out["PMI_att"] - out["PMI_un"]) / max(
        abs(out["PMI_att"]) + abs(out["PMI_un"]), 1e-12)
    d = pmi[:, 0] - _oth(pmi)
    prior = _oth(R["R_u"], "mean") - R["R_u"][:, 0]
    out["decision_margin"], out["decision_margin_se"] = m(d), se(d)
    out["audio_prior_margin"], out["audio_prior_margin_se"] = m(prior), se(prior)
    out["old_cond_margin"] = m(_oth(R["R_c"], "mean") - R["R_c"][:, 0])
    out["shuffled_PMI_att"], out["shuffled_PMI_un"] = m(pmi_s[:, 0]), m(_oth(pmi_s, "mean"))

    out["_pmi_margin"] = (pmi[:, 0] - _oth(pmi)).cpu().tolist()
    out["acc_flow_pmi"] = float(_acc(pmi).mean())
    out["null_flow_pmi"] = float(_acc(pmi_s).mean())
    out["contrib_flow_pmi"] = out["acc_flow_pmi"] - out["null_flow_pmi"]
    out["acc_flow_cond_only"] = float(_acc(-R["R_c"]).mean())       # old AttuneFlow rule
    out["null_flow_cond_only"] = float(_acc(-Rs["R_c"]).mean())
    out["acc_flow_prior_only"] = float(_acc(-R["R_u"]).mean())      # audio alone
    # Label-free fusion: scale each score by its own spread, then add.
    zf = fused / fused.std().clamp_min(1e-8)
    zp = (pmi - pmi.mean(-1, keepdim=True))
    zp = zp / zp.std().clamp_min(1e-8)
    out["acc_fused_plus_flow"] = float(_acc(zf + zp).mean())
    if batch["side"][0, 0] >= 0:
        out["trial_acc_flow_pmi"], _ = _trial_acc(pmi, trial)
        out["trial_acc_fused_plus_flow"], _ = _trial_acc(zf + zp, trial)

    # By flow time.
    pt = R["R_u_t"] - R["R_c_t"]                                   # (N,K,bins)
    cnt = R["t_count"].squeeze(1)                                   # (N,bins)
    w = cnt / cnt.sum(0, keepdim=True).clamp_min(1)
    out["PMI_by_t_att"] = (pt[:, 0] * w).sum(0).tolist()
    out["PMI_by_t_un"] = (pt[:, 1:].mean(1) * w).sum(0).tolist()
    # By modulation frequency.
    freqs = np.fft.rfftfreq(T, d=1.0 / SR)
    ps = (R["spec_u"] - R["spec_c"]).mean(0)                        # (K,F)
    out["PMI_by_band"] = {}
    for name, (lo, hi) in BANDS.items():
        sel = torch.from_numpy((freqs >= lo) & (freqs < hi)).to(ps.device)
        out["PMI_by_band"][name] = {"att": float(ps[0, sel].sum() / T),
                                    "un": float(ps[1, sel].sum() / T)}
    # Per subject: PMI gap vs accuracy.
    subjects = np.unique(subj)
    out["per_subject"] = {int(s): {"PMI_att": m(pmi[subj == s, 0]),
                                   "PMI_un": m(pmi[subj == s, 1]),
                                   "acc_flow_pmi": float(_acc(pmi[subj == s]).mean())}
                          for s in subjects}
    return out


def evaluate_split(model, data: dict, segments, device: str, windows: list[float],
                   cand_sets: list[str], n_perm: int = 20, n_circ: int = 5,
                   n_mc: int = 8, seed: int = 0, log=print) -> dict:
    """Full evaluation grid. Flow and circular nulls only at the 5-s window."""
    out = {}
    for cs in cand_sets:
        for w in windows:
            full = abs(w - 5.0) < 1e-6
            key = f"{cs}_{w:g}s"
            r = evaluate_set(model, data, segments, w, cs, device, n_perm=n_perm,
                             n_circ=n_circ if full else 0, seed=seed, full=True,
                             n_mc=n_mc) if full else \
                evaluate_set(model, data, segments, w, cs, device, n_perm=n_perm,
                             n_circ=0, seed=seed, full=False)
            out[key] = r
            if r.get("n_windows", 0):
                log(f"  [{key}] acc fused {r['acc_fused']:.4f} null {r['null_fused']:.4f} "
                    f"contrib {r['contrib_fused']:+.4f} | coupling {r['acc_coupling']:.4f} "
                    f"(contrib {r['contrib_coupling']:+.4f}) | shapley "
                    + " ".join(f"{k}={v:+.4f}" for k, v in r["shapley"].items()))
    return out
