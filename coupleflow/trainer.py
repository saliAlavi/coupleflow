"""CoupleFlow training loop: one pooled model per stimulus-disjoint fold.

Called by ``main.py``::

    python main.py --mode train --dataset kuleuven fold=0

Each run trains on the fold's training segments, selects the checkpoint by
*validation contribution* (accuracy minus the across-trial shuffled
accuracy, on content-disjoint validation data) rather than by accuracy, then
evaluates the chosen checkpoint once on the test fold with the full battery
and writes ``results.json`` next to the checkpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from coupleflow.data import (CropDataset, Segment, SubjectBatchSampler, load_dataset,
                                 split_fold)
from coupleflow.evaluate import evaluate_set, evaluate_split
from coupleflow.losses import compute_losses
from coupleflow.model import CoupleFlow


def _seed(s: int) -> None:
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def apply_dss(data: dict, train_segs: list, n_keep: int, log=print, pooled: bool = False) -> None:
    """Per-listener DSS denoising fitted on training trials only.

    Denoising Source Separation (de Cheveigne & Parra, 2014): for each listener,
    training presentations of the *same* stimulus (KU Leuven replays each story
    up to five times) are aligned at onset; C1 = covariance of their average,
    C0 = average single-trial covariance.  The generalised eigenvectors of
    (C1, C0) rank spatial components by reproducibility across repeats, i.e. by
    stimulus-locked SNR.  Every trial of that listener (train, val and test) is
    projected onto the top ``n_keep`` components and back to channel space.
    Test trials are transformed but never used in fitting.
    """
    from scipy.linalg import eigh
    base = lambda nm: nm.replace("rep_", "")  # noqa: E731
    train_trials = {sg.trial for sg in train_segs}
    subjects = sorted({t["subject"] for t in data["trials"]})
    # pooled=True: one filter from all listeners' repeat groups (16x more data).
    units = [subjects] if pooled else [[x] for x in subjects]
    for unit in units:
        sname = "pooled" if pooled else unit[0]
        idx = [i for i, t in enumerate(data["trials"]) if t["subject"] in unit]
        groups: dict[str, list[int]] = {}
        for i in idx:
            t = data["trials"][i]
            if i in train_trials and "env_att_ling_src" in t:
                groups.setdefault(t["subject"] + "|" + base(t["env_att_ling_src"][0]) + "|" +
                                  base(t["env_unatt_ling_src"][0]), []).append(i)
        C0 = np.zeros((64, 64))
        C1 = np.zeros((64, 64))
        n_used = 0
        for g in groups.values():
            if len(g) < 2:
                continue
            L = min(data["trials"][i]["eeg"].shape[1] for i in g)
            X = np.stack([data["trials"][i]["eeg"][:64, :L].float().numpy() for i in g])
            m = X.mean(0)
            C1 += m @ m.T / L
            C0 += sum(x @ x.T for x in X) / (L * len(g))
            n_used += 1
        if n_used == 0:
            continue
        C0 += 1e-3 * np.trace(C0) / 64 * np.eye(64)
        w, V = eigh(C1, C0)
        W = V[:, ::-1][:, :n_keep]                     # (64, k) most reproducible first
        A = C0 @ W @ np.linalg.inv(W.T @ C0 @ W)       # back-projection
        P = torch.from_numpy((A @ W.T).astype(np.float32))
        for i in idx:
            t = data["trials"][i]
            x = P @ t["eeg"][:64].float()
            x = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)
            t["eeg"] = torch.cat([x, t["eeg"][64:].float()], 0) if t["eeg"].shape[0] > 64 else x
        log(f"  DSS {sname}: {n_used} repeat groups, top eigenvalue ratios "
            f"{np.round(w[::-1][:4], 3).tolist()}")


def apply_mwf(data: dict, train_segs: list, log=print, thr: float = 4.0,
              delay: int = 3) -> None:
    """Multichannel Wiener filter artifact removal (Somers et al., 2018), per listener.

    Artifact samples are marked where frontal channels (Fp1, Fpz, Fp2, AF7, AF8)
    exceed ``thr`` robust s.d. (dilated by 0.25 s).  With time-lagged channel
    stacks (+-``delay`` samples), R_yy = covariance in artifact periods and
    R_nn = covariance in clean periods, fitted on the listener's *training*
    trials only; W = R_yy^-1 (R_yy - R_nn) estimates the artifact, which is
    subtracted from every trial of that listener (train, val, test).
    """
    from scipy.ndimage import binary_dilation
    frontal = [0, 1, 2, 3, 7]
    train_trials = {sg.trial for sg in train_segs}
    lags = list(range(-delay, delay + 1))

    def stack(x: np.ndarray) -> np.ndarray:                    # (C*L, T)
        return np.concatenate([np.roll(x, lg, axis=1) for lg in lags], 0)

    for sname in sorted({t["subject"] for t in data["trials"]}):
        idx = [i for i, t in enumerate(data["trials"]) if t["subject"] == sname]
        C = 64 * len(lags)
        Ryy = np.zeros((C, C)); Rnn = np.zeros((C, C)); ny = nn = 0
        for i in idx:
            if i not in train_trials:
                continue
            x = data["trials"][i]["eeg"][:64].float().numpy()
            f = x[frontal]
            med = np.median(f, 1, keepdims=True)
            mad = np.median(np.abs(f - med), 1, keepdims=True) * 1.4826 + 1e-6
            art = (np.abs(f - med) / mad > thr).any(0)
            art = binary_dilation(art, iterations=8)
            X = stack(x)
            if art.sum() > 64:
                Ryy += X[:, art] @ X[:, art].T; ny += art.sum()
            Rnn += X[:, ~art] @ X[:, ~art].T; nn += (~art).sum()
        if ny < 64 * 20 or nn == 0:
            log(f"  MWF {sname}: too few artifact samples ({ny}) -- skipped")
            continue
        Ryy /= ny; Rnn /= nn
        Ryy += 1e-6 * np.trace(Ryy) / C * np.eye(C)
        W = np.linalg.solve(Ryy, Ryy - Rnn)                     # (C, C)
        W0 = W[:, lags.index(0) * 64:(lags.index(0) + 1) * 64]  # artifact estimate at lag 0
        for i in idx:
            t = data["trials"][i]
            x = t["eeg"][:64].float().numpy()
            d = W0.T @ stack(x)
            y = x - d
            y = (y - y.mean(1, keepdims=True)) / (y.std(1, keepdims=True) + 1e-6)
            y = torch.from_numpy(y.astype(np.float32))
            t["eeg"] = torch.cat([y, t["eeg"][64:].float()], 0) if t["eeg"].shape[0] > 64 else y
        log(f"  MWF {sname}: artifact fraction {ny / (ny + nn):.3f}")


def apply_clip(data: dict, k: float = 5.0) -> None:
    """Winsorise each channel at +-k robust s.d. (median / MAD), then re-z-score."""
    for t in data["trials"]:
        x = t["eeg"][:64].float()
        med = x.median(1, keepdim=True).values
        mad = (x - med).abs().median(1, keepdim=True).values * 1.4826 + 1e-6
        x = torch.maximum(torch.minimum(x, med + k * mad), med - k * mad)
        x = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)
        t["eeg"] = torch.cat([x, t["eeg"][64:].float()], 0) if t["eeg"].shape[0] > 64 else x


def append_eeg_bands(data: dict) -> None:
    """Append band-power envelopes (``eeg_bands``) to each trial's EEG channels."""
    for t in data["trials"]:
        if t["eeg"].shape[0] == 64 and "eeg_bands" in t:
            t["eeg"] = torch.cat([t["eeg"].float(), t["eeg_bands"].float()], 0)


def feature_banks(cfg: DictConfig) -> list[str]:
    banks = list(cfg.data.get("feat_banks", None) or [])
    if bool(cfg.data.get("subbands", False)) and "sb" not in banks:
        banks = ["sb"] + banks                   # legacy flag
    return banks


def build_model(cfg: DictConfig, n_subjects: int, bank_dims: list[int] | None = None,
                n_channels: int = 64) -> CoupleFlow:
    """``n_channels``: EEG rows per trial as stored (64 canonical, or native HD)."""
    m = cfg.model
    return CoupleFlow(
        n_channels=n_channels * (4 if bool(cfg.data.get("eeg_bands", False)) else 1),
        n_subjects=n_subjects, margin=int(m.margin), lag_mode=str(m.lag_mode),
        d=int(m.d), hidden=int(m.hidden), n_virtual=int(m.n_virtual), rank=int(m.rank),
        dropout=float(m.dropout), use_side=bool(m.use_side), use_flow=bool(m.use_flow),
        flow_d_model=int(m.flow_d_model), p_uncond=float(m.p_uncond),
        score_mode=str(m.get("score_mode", "independent")),
        eeg_dilations=tuple(int(x) for x in m.get("eeg_dilations", [1, 2, 4, 8, 16])),
        extra_feats=tuple(cfg.data.get("cand_extra", None) or []),
        glimpse=bool(cfg.data.get("glimpse", False)),
        banks=tuple(feature_banks(cfg)), bank_dims=tuple(bank_dims or []),
        cand_side=bool(m.get("cand_side", False)), score_head=str(m.get("score_head", "corr")),
        n_sub=int(m.get("n_sub", 5)), lag_offset=int(m.get("lag_offset", 8)))


def streams_for(cfg: DictConfig, ds: str) -> tuple[str | None, str]:
    """Stream / stimulus-group files for ``ds``: explicit override, else the
    verified per-dataset entry, else the defaults."""
    table = cfg.data.get("dataset_streams", {}) or {}
    ent = table.get(ds, {}) or {}
    sn = cfg.data.get("stream_name") if ds == str(cfg.training.dataset) else None
    gn = cfg.data.get("groups_name") if ds == str(cfg.training.dataset) else None
    return (sn or ent.get("stream_name"),
            str(gn or ent.get("groups_name") or "stimulus_groups.json"))


def run_dir(cfg: DictConfig) -> Path:
    tag = str(cfg.get("tag", "main"))
    d = Path(cfg.paths.out_root) / tag / f"{cfg.training.dataset}_fold{int(cfg.fold)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to(batch: dict, device: str) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def adapt_and_evaluate(cfg: DictConfig, model: CoupleFlow, data: dict, adapt_segs: list,
                       eval_segs: list, device: str, w: dict, log) -> dict:
    """Unsupervised test-time adaptation from the model's own trial-level decisions.

    No test label is used.  Each test trial is labelled by the sign of its summed
    coupling margin (the model's own trial decision); trials in the least
    confident ``1 - keep_frac`` are dropped.  The model is fine-tuned on the
    pseudo-labelled test trials (the predicted-attended stream plays the role of
    the attended envelope, and the attended side follows the prediction), then
    the full battery is re-run.  This is transductive: it is reported apart from
    the supervised result, as the deployment scenario of a device that adapts
    to a listener while in use.  Repeated for ``rounds`` rounds.
    """
    from coupleflow.evaluate import evaluate_set
    A = cfg.adapt
    margin = int(cfg.model.margin)
    win = int(round(float(cfg.data.train_window_s) * 64))
    out = {}
    for rnd in range(int(A.rounds)):
        # Pseudo-labels always come from the adaptation segments, never the eval ones.
        r5 = evaluate_set(model, data, adapt_segs, 5.0, "competing", device, n_perm=1,
                          full=True, n_circ=0, n_mc=1)
        cur = r5.pop("_windows")
        tr = np.asarray(cur["trial"])
        mg = np.asarray(cur["margin_coupling"])
        sums = {int(t): float(mg[tr == t].sum() / np.sqrt((tr == t).sum())) for t in np.unique(tr)}
        order = sorted(sums, key=lambda t: -abs(sums[t]))
        keep = set(order[: max(1, int(round(float(A.keep_frac) * len(order))))])
        pseudo = []
        for t in keep:
            src = data["trials"][t]
            q = dict(src)
            if sums[t] < 0:        # model says the other stream is attended
                q["env_att"], q["env_unatt"] = src["env_unatt"], src["env_att"]
                q["attended_ear"] = "L" if src["attended_ear"] == "R" else "R"
            pseudo.append(q)
        truth = float(np.mean([sums[t] > 0 for t in keep]))   # for reporting only
        pdata = {"trials": pseudo, "subjects": data["subjects"]}
        segs = [Segment(i, 0, q["eeg"].shape[1]) for i, q in enumerate(pseudo)]
        ds_ad = CropDataset(pdata, segs, win, margin, matched=bool(cfg.data.matched),
                            density=float(cfg.data.crop_density),
                           extra_feats=list(cfg.data.get('cand_extra', None) or []),
                           glimpse=bool(cfg.data.get('glimpse', False)),
                           banks=feature_banks(cfg), n_glimpse=model.n_glimpse, seed=int(cfg.seed) + 100 + rnd)
        opt = torch.optim.AdamW(model.parameters(), lr=float(A.lr),
                                weight_decay=float(cfg.optim.weight_decay))
        for ep in range(int(A.epochs)):
            model.train()
            ds_ad.resample()
            loader = DataLoader(ds_ad, batch_sampler=SubjectBatchSampler(
                ds_ad, int(cfg.optim.batch_size), seed=int(cfg.seed) + 1000 * rnd + ep),
                num_workers=int(cfg.num_workers), pin_memory=device == "cuda")
            for batch in loader:
                batch = _to(batch, device)
                L = compute_losses(model, batch, model(batch), w)
                opt.zero_grad(set_to_none=True)
                L["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.grad_clip))
                opt.step()
        log(f"=== adapt round {rnd}: {len(keep)}/{len(order)} test trials pseudo-labelled "
            f"(pseudo-label accuracy {truth:.3f}, not used) ===")
        res = evaluate_split(model, data, eval_segs, device,
                             [float(x) for x in cfg.eval.windows], list(cfg.eval.cand_sets),
                             n_perm=int(cfg.eval.test_perms), n_circ=int(cfg.eval.circ_perms),
                             n_mc=int(cfg.eval.flow_mc), seed=int(cfg.seed), log=log)
        for v in res.values():
            v.pop("_windows", None)
        out[f"round{rnd}"] = {"results": res, "n_pseudo": len(keep),
                              "pseudo_label_accuracy": truth}
    return out


def subject_finetune(cfg: DictConfig, model: CoupleFlow, data: dict, sp: dict, device: str,
                     w: dict, log) -> dict:
    """Per-listener fine-tuning of the pooled model; returns pooled test metrics.

    For each listener: copy the pooled model, fine-tune on that listener's
    training segments, select the epoch on that listener's validation segments
    (coupling accuracy), and evaluate that listener's test segments.
    """
    import copy
    from coupleflow.evaluate import evaluate_set
    F = cfg.subject_ft
    margin = int(cfg.model.margin)
    win = int(round(float(cfg.data.train_window_s) * 64))
    subj_of = lambda seg: data["trials"][seg.trial]["subject"]  # noqa: E731
    base_state = copy.deepcopy(model.state_dict())
    agg: dict[str, list] = {}
    per = {}
    for sname in sorted({subj_of(x) for x in sp["test"]}):
        tr = [x for x in sp["train"] if subj_of(x) == sname]
        va = [x for x in sp["val"] if subj_of(x) == sname]
        te = [x for x in sp["test"] if subj_of(x) == sname]
        if not tr or not va:
            continue
        model.load_state_dict(base_state)
        ds_s = CropDataset(data, tr, win, margin, matched=bool(cfg.data.matched),
                           density=float(cfg.data.crop_density),
                           extra_feats=list(cfg.data.get('cand_extra', None) or []),
                           glimpse=bool(cfg.data.get('glimpse', False)),
                           banks=feature_banks(cfg), n_glimpse=model.n_glimpse, seed=int(cfg.seed))
        opt = torch.optim.AdamW(model.parameters(), lr=float(F.lr),
                                weight_decay=float(cfg.optim.weight_decay))
        best, best_state = -1.0, copy.deepcopy(model.state_dict())
        v0 = evaluate_set(model, data, va, 5.0, "competing", device, n_perm=1, full=False)
        best = v0["acc_coupling"]
        for ep in range(int(F.epochs)):
            model.train()
            ds_s.resample()
            loader = DataLoader(ds_s, batch_size=int(cfg.optim.batch_size), shuffle=True,
                                num_workers=int(cfg.num_workers), drop_last=True)
            for batch in loader:
                batch = _to(batch, device)
                L = compute_losses(model, batch, model(batch), w)
                opt.zero_grad(set_to_none=True)
                L["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.grad_clip))
                opt.step()
            v = evaluate_set(model, data, va, 5.0, "competing", device, n_perm=1, full=False)
            if v["acc_coupling"] > best:
                best, best_state = v["acc_coupling"], copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        per[sname] = {"val_before": v0["acc_coupling"], "val_after": best}
        for cs in ("competing", "same_talker"):
            r = evaluate_set(model, data, te, 5.0, cs, device, n_perm=5, full=False)
            agg.setdefault(cs, []).append((r["acc_coupling"], r["null_coupling"], r["n_windows"]))
            per[sname][f"test_{cs}"] = r["acc_coupling"]
        log(f"  subject-ft {sname}: val {v0['acc_coupling']:.3f} -> {best:.3f} | test "
            f"{per[sname]['test_competing']:.3f}")
    model.load_state_dict(base_state)
    out = {"per_subject": per}
    for cs, v in agg.items():
        n = np.array([x[2] for x in v])
        out[f"{cs}_5s_acc_coupling"] = float(np.average([x[0] for x in v], weights=n))
        out[f"{cs}_5s_null_coupling"] = float(np.average([x[1] for x in v], weights=n))
    log(f"=== subject-ft: competing 5s {out.get('competing_5s_acc_coupling', float('nan')):.4f} "
        f"same-talker {out.get('same_talker_5s_acc_coupling', float('nan')):.4f} ===")
    return out


def train(cfg: DictConfig) -> None:
    """Train one fold and evaluate it. See module docstring."""
    t_start = time.time()
    _seed(int(cfg.seed))
    device = "cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu"
    out_dir = run_dir(cfg)
    if (out_dir / "results.json").exists() and cfg.get("skip_if_done", True):
        print(f"{out_dir}/results.json exists -- skipping")
        return
    log_f = open(out_dir / "train.log", "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    ds = str(cfg.training.dataset)
    sn, gn = streams_for(cfg, ds)
    data = load_dataset(cfg.paths.cache_root, ds, drop_frontal=bool(cfg.data.drop_frontal),
                        stream_name=sn, groups_name=gn)
    if bool(cfg.data.get("eeg_bands", False)):
        append_eeg_bands(data)
    if cfg.data.get("max_subjects"):
        keep = set(data["subjects"][: int(cfg.data.max_subjects)])
        idx = [i for i, t in enumerate(data["trials"]) if t["subject"] in keep]
        data = {"trials": [data["trials"][i] for i in idx],
                "groups": [data["groups"][i] for i in idx],
                "subjects": sorted(keep)}
    margin = int(cfg.model.margin)
    # K-way candidates (datasets with env_cands, e.g. MAESTRO): used by training
    # crops and the "cands"/"cands_matched"/"same_talkerK" evaluation sets.
    data["cand_idx"] = list(cfg.data.get("cand_idx") or [])
    sp = split_fold(data, int(cfg.fold), int(cfg.n_folds), margin,
                    val_mode=str(cfg.data.val_mode), val_frac=float(cfg.data.val_frac),
                    tail_min_s=float(cfg.data.get("tail_min_s", 200.0)),
                    mode=str(cfg.data.get("split_mode", "stimulus")),
                    seed=int(cfg.get("split_seed", 42)))
    # Learning-curve experiments: keep only ``train_frac`` of each listener's
    # training audio (whole segments, random order, fixed seed).
    frac = float(cfg.data.get("train_frac", 1.0))
    if frac < 1.0:
        import random as _r
        by_s: dict[str, list] = {}
        for sg in sp["train"]:
            by_s.setdefault(data["trials"][sg.trial]["subject"], []).append(sg)
        kept = []
        for sname, segs in sorted(by_s.items()):
            _r.Random(int(cfg.get("split_seed", 42)) + len(sname)).shuffle(segs)
            tot = sum(x.end - x.start for x in segs)
            acc = 0
            for x in segs:
                if acc >= frac * tot:
                    break
                kept.append(x)
                acc += x.end - x.start
        sp["train"] = kept
    if float(cfg.data.get("clip_mad", 0)) > 0:
        apply_clip(data, float(cfg.data.clip_mad))
    if bool(cfg.data.get("mwf", False)):
        apply_mwf(data, sp["train"], log, thr=float(cfg.data.get("mwf_thr", 6.0)))
    if int(cfg.data.get("dss_keep", 0)) > 0:
        apply_dss(data, sp["train"], int(cfg.data.dss_keep), log,
                  pooled=bool(cfg.data.get("dss_pooled", False)))
    # Extra training-only datasets (other corpora, other stimuli): every trial goes
    # to train; validation and test stay on the target dataset's folds.
    for ex_name in cfg.data.get("extra_train_names", None) or []:
        sn, gn = streams_for(cfg, str(ex_name))
        d2 = load_dataset(cfg.paths.cache_root, str(ex_name),
                          drop_frontal=bool(cfg.data.drop_frontal),
                          stream_name=sn, groups_name=gn)
        base = len(data["trials"])
        for t in d2["trials"]:
            t = dict(t)
            t["subject"] = f"{ex_name}:{t['subject']}"
            data["trials"].append(t)
            data["groups"].append(-1)
        for i in range(base, len(data["trials"])):
            sp["train"].append(Segment(i, 0, data["trials"][i]["eeg"].shape[1]))
        data["subjects"] = sorted({t["subject"] for t in data["trials"]})
        log(f"+ extra train dataset {ex_name}: {len(d2['trials'])} trials")
    n_tr = sum(s.end - s.start for s in sp["train"]) / 64 / 3600
    n_va = sum(s.end - s.start for s in sp["val"]) / 64 / 3600
    n_te = sum(s.end - s.start for s in sp["test"]) / 64 / 3600
    log(f"[{ds} fold {cfg.fold}/{sp['n_folds']}] val_mode={sp['val_mode']} "
        f"test groups={sp['test_groups']} | hours train {n_tr:.2f} val {n_va:.2f} "
        f"test {n_te:.2f} | device {device}")
    OmegaConf.save(cfg, out_dir / "config.yaml")

    win = int(round(float(cfg.data.train_window_s) * 64))
    if cfg.data.get("train_window_choices"):
        win = int(round(max(float(x) for x in cfg.data.train_window_choices) * 64))
    win_rng = np.random.default_rng(int(cfg.seed) + 7)
    t0_ = data["trials"][0]
    bank_dims = [int(t0_[f"env_att_{bk}"].shape[0]) for bk in feature_banks(cfg)]
    model = build_model(cfg, len(data["subjects"]), bank_dims,
                        n_channels=int(data["trials"][0]["eeg"].shape[0])
                        // (4 if bool(cfg.data.get("eeg_bands", False)) else 1)).to(device)
    train_ds = CropDataset(data, sp["train"], win, margin, matched=bool(cfg.data.matched),
                           chan_drop=float(cfg.data.get("chan_drop", 0.0)),
                           eeg_noise=float(cfg.data.get("eeg_noise", 0.0)),
                           density=float(cfg.data.crop_density),
                           extra_feats=list(cfg.data.get('cand_extra', None) or []),
                           glimpse=bool(cfg.data.get('glimpse', False)),
                           banks=feature_banks(cfg), n_glimpse=model.n_glimpse, seed=int(cfg.seed))
    if cfg.get("init_from"):
        # Warm start (e.g. from SparrKULee pretraining).  Per-listener tensors
        # (U, V) do not transfer across subject sets and are skipped.
        src = torch.load(str(cfg.init_from), map_location=device, weights_only=False)["model"]
        own = model.state_dict()
        keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        log(f"init_from {cfg.init_from}: loaded {len(keep)}/{len(own)} tensors")
    log(f"params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr),
                            weight_decay=float(cfg.optim.weight_decay))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=int(cfg.optim.plateau_patience))
    w = OmegaConf.to_container(cfg.losses, resolve=True)
    val_set = cfg.eval.get("val_set") or (
        ("cands_matched" if cfg.data.matched else "cands") if data["cand_idx"]
        else ("matched" if cfg.data.matched else "competing"))

    best, best_ep, bad = -1e9, -1, 0
    history = []
    for ep in range(int(cfg.optim.epochs)):
        model.train()
        w["_epoch"] = ep                                    # for warm-up of robust losses
        if cfg.data.get("train_window_choices"):
            # Mixed-length training: one crop length per epoch.  The dataset was built
            # with the longest length, so every segment fits every choice.
            train_ds.win = int(round(float(win_rng.choice(list(cfg.data.train_window_choices)))
                                     * 64))
        train_ds.resample()
        sampler = SubjectBatchSampler(train_ds, int(cfg.optim.batch_size), seed=int(cfg.seed) + ep)
        loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=int(cfg.num_workers),
                            pin_memory=device == "cuda", persistent_workers=False)
        agg: dict[str, float] = {}
        n = 0
        t0 = time.time()
        for bi, batch in enumerate(loader):
            batch = _to(batch, device)
            out = model(batch)
            L = compute_losses(model, batch, out, w)
            opt.zero_grad(set_to_none=True)
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.grad_clip))
            opt.step()
            for k, v in L.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
            if cfg.get("max_steps_per_epoch") and bi + 1 >= int(cfg.max_steps_per_epoch):
                break
        tr = {k: v / max(n, 1) for k, v in agg.items()}

        # Selection: validation contribution on content-disjoint data.
        vr = evaluate_set(model, data, sp["val"], float(cfg.data.val_window_s), val_set,
                          device, n_perm=int(cfg.eval.val_perms), seed=int(cfg.seed),
                          full=False)
        # Select on the coupling path by default: on KU Leuven the validation data are
        # tails of training trials, so anything that recognises the trial (the side
        # path) looks better on validation than it is on unseen stimuli.
        rule = str(cfg.eval.get("select_on", "coupling"))
        # select_metric=acc is the ablation of contribution-based selection.
        metric = "acc" if str(cfg.eval.get("select_metric", "contrib")) == "acc" else "contrib"
        crit = vr[f"{metric}_{rule}"] + float(cfg.eval.select_beta) * min(vr["shapley"].values())
        sched.step(crit)
        history.append({"epoch": ep, "train": tr, "val_acc": vr["acc_fused"],
                        "val_null": vr["null_fused"], "val_contrib": vr["contrib_fused"],
                        "val_acc_coupling": vr["acc_coupling"], "val_shapley": vr["shapley"],
                        "crit": crit})
        log(f"ep {ep:3d} | loss {tr['total']:.3f} (task {tr['task']:.3f}"
            + "".join(f" {k} {v:.3f}" for k, v in tr.items() if k not in ("total", "task"))
            + f") | val acc {vr['acc_fused']:.4f} null {vr['null_fused']:.4f} contrib "
            f"{vr['contrib_fused']:+.4f} coup {vr['acc_coupling']:.4f} | tau "
            f"{model.log_tau.exp().item():.1f} | {time.time() - t0:.0f}s")
        if crit > best:
            best, best_ep, bad = crit, ep, 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "cfg": OmegaConf.to_container(cfg, resolve=True),
                        "subjects": data["subjects"]}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= int(cfg.optim.patience):
                log(f"early stop at epoch {ep} (best {best_ep})")
                break

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    log(f"=== test (best epoch {best_ep}) ===")
    results = evaluate_split(model, data, sp["test"], device,
                             [float(x) for x in cfg.eval.windows], list(cfg.eval.cand_sets),
                             n_perm=int(cfg.eval.test_perms), n_circ=int(cfg.eval.circ_perms),
                             n_mc=int(cfg.eval.flow_mc), seed=int(cfg.seed), log=log)
    windows = {k: v.pop("_windows") for k, v in results.items() if "_windows" in v}
    np.savez_compressed(out_dir / "test_windows.npz",
                        **{f"{cell}__{k}": np.asarray(a) for cell, d in windows.items()
                           for k, a in d.items()})
    sft = None
    if cfg.get("subject_ft") and bool(cfg.subject_ft.enabled):
        sft = subject_finetune(cfg, model, data, sp, device, w, log)
    adapt_results = None
    if cfg.get("adapt") and bool(cfg.adapt.enabled):
        # Transductive: adapt on the test trials themselves.  See validator
        # (adapt.cross) for the stimulus-disjoint adapt/evaluate protocol.
        adapt_results = adapt_and_evaluate(cfg, model, data, sp["test"], sp["test"], device,
                                           w, log)
    payload = {"dataset": ds, "fold": int(cfg.fold), "n_folds": sp["n_folds"],
               "val_mode": sp["val_mode"], "test_groups": sp["test_groups"],
               "best_epoch": best_ep, "best_crit": best, "history": history,
               "hours": {"train": n_tr, "val": n_va, "test": n_te},
               "config": OmegaConf.to_container(cfg, resolve=True),
               "results": results, "adapt": adapt_results, "subject_ft": sft,
               "minutes": (time.time() - t_start) / 60}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1, default=float))
    log(f"wrote {out_dir / 'results.json'} ({payload['minutes']:.1f} min)")
