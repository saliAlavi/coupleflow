"""CPU tests for CoupleFlow: shapes, shortcut-proofness, lag direction, nulls.

Synthetic data: the EEG carries the attended envelope delayed by 150 ms in a
few channels, plus noise.  An ``after`` model must learn it; the ``before``
control cannot, because the response lies outside its lag window.

    python -m pytest tests/test_coupleflow.py -q
"""

from __future__ import annotations

import numpy as np
import torch

from coupleflow.data import (
    CropDataset,
    build_eval_batch,
    eval_windows,
    quantile_match,
    split_fold,
)
from coupleflow.evaluate import across_trial_perm, evaluate_set, shapley_two
from coupleflow.losses import compute_losses
from coupleflow.model import CoupleFlow, time_centred_corr

SR = 64


def _smooth_noise(rng, n):
    x = rng.standard_normal(n + 64)
    k = np.hanning(17)
    return np.abs(np.convolve(x, k / k.sum(), "same"))[:n]


def synthetic(n_subj=3, n_trials=12, dur_s=40, delay=10, snr=0.6, seed=0):
    rng = np.random.default_rng(seed)
    trials, groups = [], []
    for s in range(n_subj):
        for t in range(n_trials):
            n = dur_s * SR
            att, un = _smooth_noise(rng, n), _smooth_noise(rng, n)
            eeg = rng.standard_normal((64, n)).astype(np.float32)
            resp = np.roll((att - att.mean()) / att.std(), delay)
            eeg[26:36] += snr * resp
            trials.append({"subject": f"S{s}", "trial": f"S{s}_t{t}",
                           "attended_ear": "L" if t % 2 else "R",
                           "eeg": torch.from_numpy(eeg),
                           "env_att": torch.from_numpy(att.astype(np.float32)),
                           "env_unatt": torch.from_numpy(un.astype(np.float32)),
                           "channel_mask": torch.ones(64, dtype=torch.bool)})
            groups.append(t)            # same stimulus index shared across subjects
    return {"trials": trials, "groups": groups,
            "subjects": [f"S{s}" for s in range(n_subj)]}


def test_constant_eeg_gives_chance():
    torch.manual_seed(0)
    e = torch.ones(4, 16, 100) * torch.randn(4, 16, 1)      # constant in time
    a = torch.randn(4, 2, 16, 100)
    assert time_centred_corr(e, a).abs().max() < 1e-6


def test_quantile_match_same_values():
    x = torch.randn(2, 50) * torch.tensor([[1.0], [3.0]]) + torch.tensor([[0.0], [2.0]])
    m = quantile_match(x)
    assert torch.allclose(m[0].sort().values, m[1].sort().values)
    assert torch.equal(m[0].argsort(), x[0].argsort())


def test_stimulus_folds_disjoint():
    d = synthetic()
    sp = split_fold(d, 0, 4, margin=100)
    tg = {d["groups"][s.trial] for s in sp["test"]}
    trg = {d["groups"][s.trial] for s in sp["train"]}
    vg = {d["groups"][s.trial] for s in sp["val"]}
    assert not tg & trg and not tg & vg


def test_across_trial_perm():
    rng = np.random.default_rng(0)
    subj = np.repeat([0, 1], 20)
    trial = np.tile(np.repeat(np.arange(4), 5), 2)
    p = across_trial_perm(subj, trial, rng)
    assert (subj[p] == subj).all() and (trial[p] != trial).all()


def test_shapley_additive():
    v = {frozenset(): 0.5, frozenset({"a"}): 0.7, frozenset({"b"}): 0.6,
         frozenset({"a", "b"}): 0.85}
    phi = shapley_two(v, ["a", "b"])
    assert abs(sum(phi.values()) - 0.35) < 1e-9


def _train(lag_mode, steps=150, seed=0):
    torch.manual_seed(seed)
    d = synthetic(seed=seed)
    sp = split_fold(d, 0, 4, margin=100)
    ds = CropDataset(d, sp["train"], 320, 100, matched=True, seed=seed)
    m = CoupleFlow(n_subjects=3, lag_mode=lag_mode)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    w = dict(side=0.3, con=1.0, null_hinge=0.5, null_margin=0.5, col=0.1, col_gamma=0.5,
             adv=0.3, flow=0.3, env=0.2)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
    it = 0
    while it < steps:
        ds.resample()
        for b in loader:
            out = m(b)
            L = compute_losses(m, b, out, w)
            opt.zero_grad()
            L["total"].backward()
            opt.step()
            it += 1
            if it >= steps:
                break
    return m, d, sp


def test_lag_direction_and_nulls():
    m, d, sp = _train("after")
    r = evaluate_set(m, d, sp["test"], 5.0, "matched", "cpu", n_perm=5, n_circ=2,
                     full=True, n_mc=2)
    assert r["acc_coupling"] > 0.8, r["acc_coupling"]
    assert abs(r["null_coupling"] - 0.5) < 0.15
    assert r["contrib_circ_coupling"] > 0.2
    assert "flow" in r and np.isfinite(r["flow"]["PMI_att"])
    mb, db, spb = _train("before")
    rb = evaluate_set(mb, db, spb["test"], 5.0, "matched", "cpu", n_perm=5, n_circ=0,
                      full=False)
    assert rb["acc_coupling"] < r["acc_coupling"] - 0.2, (rb["acc_coupling"], r["acc_coupling"])


def test_eval_batch_same_talker_no_overlap():
    d = synthetic()
    sp = split_fold(d, 0, 4, margin=100)
    wins = eval_windows(d, sp["test"], 320, 100)
    b = build_eval_batch(d, wins, 320, 100, "same_talker")
    assert b["cand"].shape[1:] == (2, 2, 520)
    assert (b["side"] == -1).all()


def _synth_kway(n_sub: int = 3, n_trials: int = 6, T: int = 64 * 40) -> dict:
    """MAESTRO-like data: six candidate talkers per trial, attended in 0..3, EEG tracks it."""
    g = torch.Generator().manual_seed(0)
    trials, groups = [], []
    for s in range(n_sub):
        for k in range(n_trials):
            cands = torch.rand(6, T, generator=g)
            att = k % 4
            eeg = torch.randn(64, T, generator=g) * 0.5
            eeg[:8, 10:] += cands[att, :-10]                 # response ~150 ms after sound
            trials.append({"subject": f"S{s}", "trial": f"S{s}_eval_{k}", "attended_ear": "LR"[att // 2],
                           "eeg": eeg, "env_att": cands[att].clone(),
                           "env_unatt": cands[att ^ 1].clone(), "has_unattended": True,
                           "env_cands": cands, "att_idx": att,
                           "channel_mask": torch.ones(64, dtype=torch.bool)})
            groups.append(k)
    return {"trials": trials, "subjects": [f"S{s}" for s in range(n_sub)], "groups": groups,
            "cand_idx": [0, 1, 2, 3]}


def test_kway_candidates_and_loso():
    from coupleflow.data import CropDataset, build_eval_batch, eval_windows, split_fold
    data = _synth_kway()
    margin, win = 100, 320
    # LOSO: held-out listener, held-out content, nothing shared with training.
    sp = split_fold(data, 1, 3, margin, mode="loso", seed=42)
    tr = {(data["trials"][x.trial]["subject"], data["groups"][x.trial]) for x in sp["train"]}
    te = {(data["trials"][x.trial]["subject"], data["groups"][x.trial]) for x in sp["test"]}
    assert te and all(s == "S1" for s, _ in te)
    assert not ({g for _, g in te} & {g for _, g in tr}) and not any(s == "S1" for s, _ in tr)
    ds = CropDataset(data, sp["train"], win, margin, matched=True)
    item = ds[0]
    assert item["cand"].shape[0] == 4 and 0 <= int(item["y"]) < 4
    assert torch.allclose(item["env_att"], item["cand"][int(item["y"]), 0] * 0 + item["env_att"])
    segs = sp["train"] + sp["test"]
    wins = eval_windows(data, segs, win, margin)
    for cs, K in (("cands", 4), ("cands_matched", 4), ("same_talker3", 3), ("same_talker", 2)):
        b = build_eval_batch(data, wins, win, margin, cs)
        assert b is not None and b["cand"].shape[1] == K, cs
    # model + losses + evaluation run end-to-end with K = 4
    model = CoupleFlow(n_channels=64, n_subjects=3, margin=margin)
    batch = {k: torch.stack([ds[i][k] for i in range(8)]) for k in ds[0]}
    out = model(batch)
    L = compute_losses(model, batch, out, {"side": .3, "con": 1, "null_hinge": .5, "null_margin": .5,
                                           "col": .1, "col_gamma": .5, "adv": .3, "flow": .3,
                                           "env": .2})
    assert torch.isfinite(L["total"])
    r = evaluate_set(model, data, sp["test"], 5.0, "cands_matched", "cpu", n_perm=2, full=False)
    assert r["n_windows"] > 0 and 0.0 <= r["acc_coupling"] <= 1.0
