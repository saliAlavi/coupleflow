#!/usr/bin/env python
"""Rebuttal E1, MAESTRO per trial: coupling vs PMI agreement with 4-way candidates.

Per-trial decisions sum each candidate's window logits over the trial, which needs the
full (N, 4) logits; ``test_windows.npz`` stores only attended-minus-best margins.  This
re-runs the evaluation forward pass of ``coupleflow.evaluate.evaluate_set`` /
``flow_report`` (same windows, candidates, seed, MC chunking) on CPU for the
``cands_matched`` 5-s windows of ``maestro_within_lag0`` and saves the logits.  The flow
head's MC draws come from a device-specific torch generator, so CPU PMI scores differ
from the CUDA ones by MC noise; the coupling logits are deterministic.  The script
reports both against ``results_val.json`` and against the stored per-window margins.

    python scripts/rebuttal/e1/maestro_trial.py --out /fs/scratch/PAS2301/alialavi/rebuttal/e1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from coupleflow.data import build_eval_batch, eval_windows, load_dataset, split_fold  # noqa: E402
from coupleflow.evaluate import SR, _acc, _trial_acc, encode, score  # noqa: E402
from coupleflow.trainer import build_model, feature_banks, streams_for  # noqa: E402

RUNS = Path(os.environ.get("COUPLEFLOW_RUNS", "/fs/scratch/PAS2301/alialavi/runs/coupleflow"))


@torch.no_grad()
def run_fold(d: Path, data_cache: dict, cell: str = "cands_matched", win_s: float = 5.0) -> dict:
    cfg = OmegaConf.load(d / "config.yaml")
    cfg.paths.cache_root = os.environ.get("COUPLEFLOW_DATA", cfg.paths.cache_root)
    ds = str(cfg.training.dataset)
    sn, gn = streams_for(cfg, ds)
    key = (sn, gn)
    if key not in data_cache:
        data_cache[key] = load_dataset(cfg.paths.cache_root, ds,
                                       drop_frontal=bool(cfg.data.drop_frontal),
                                       stream_name=sn, groups_name=gn)
    data = data_cache[key]
    ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    assert data["subjects"] == ck["subjects"]
    data["cand_idx"] = list(cfg.data.get("cand_idx") or [])
    sp = split_fold(data, int(cfg.fold), int(cfg.n_folds), int(cfg.model.margin),
                    val_mode=str(cfg.data.val_mode), val_frac=float(cfg.data.val_frac),
                    tail_min_s=float(cfg.data.get("tail_min_s", 200.0)),
                    mode=str(cfg.data.get("split_mode", "stimulus")),
                    seed=int(cfg.get("split_seed", 42)))
    bank_dims = [int(data["trials"][0][f"env_att_{bk}"].shape[0]) for bk in feature_banks(cfg)]
    model = build_model(cfg, len(data["subjects"]), bank_dims,
                        n_channels=int(data["trials"][0]["eeg"].shape[0]))
    model.load_state_dict(ck["model"])
    model.eval()
    seed = int(cfg.seed)
    win = int(round(win_s * SR))
    wins = eval_windows(data, sp["test"], win, model.margin)
    batch = build_eval_batch(data, wins, win, model.margin, cell, seed=seed,
                             extra_feats=getattr(model, "extra_feats", []),
                             glimpse=bool(getattr(model, "glimpse", False)),
                             banks=list(getattr(model, "banks", [])),
                             n_glimpse=int(getattr(model, "n_glimpse", 0)))
    enc = encode(model, batch, "cpu")
    coup = score(model, enc["e"], enc["a"], enc["side_logits"], batch["side"])["coupling"]
    cands = model.central_env(batch["cand"])
    N, chunk, n_mc = cands.shape[0], 128, int(cfg.eval.flow_mc)
    parts = [model.flow.residuals(cands[i:i + chunk], enc["e"][i:i + chunk], n_mc, seed + i)
             for i in range(0, N, chunk)]
    Ru = torch.cat([q["R_u"] for q in parts])
    Rc = torch.cat([q["R_c"] for q in parts])
    pmi = Ru - Rc
    trial = batch["trial"]
    r = json.loads((d / ("results.json" if (d / "results.json").exists()
                         else "results_val.json")).read_text())["results"][f"{cell}_5s"]
    z = np.load(d / "test_windows.npz")
    stored_c = z[f"{cell}_5s__margin_coupling"] > 0
    stored_p = z[f"{cell}_5s__margin_flow_pmi"] > 0
    cw, pw = _acc(coup).numpy().astype(bool), _acc(pmi).numpy().astype(bool)
    return {"run": d.name, "n": int(N), "coup": coup.numpy(), "pmi": pmi.numpy(),
            "trial": np.asarray(trial),
            "check": {"acc_coupling": float(cw.mean()), "json_coupling": r["acc_coupling"],
                      "acc_pmi": float(pw.mean()), "json_pmi": r["flow"]["acc_flow_pmi"],
                      "trial_coupling": _trial_acc(coup, trial)[0],
                      "json_trial_coupling": r["trial_acc_coupling"],
                      "trial_pmi": _trial_acc(pmi, trial)[0],
                      "json_trial_pmi": r["flow"]["trial_acc_flow_pmi"],
                      "same_coupling_decisions_as_stored": float((cw == stored_c).mean()),
                      "same_pmi_correctness_as_stored_cuda": float((pw == stored_p).mean())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/fs/scratch/PAS2301/alialavi/rebuttal/e1")
    ap.add_argument("--tag", default="maestro_within_lag0")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    folds = []
    for d in sorted((RUNS / args.tag).glob("maestro_fold*")):
        f = run_fold(d, cache)
        print(d.name, json.dumps(f["check"]), flush=True)
        folds.append(f)
    np.savez_compressed(out / f"{args.tag}_logits.npz",
                        **{f"{f['run']}__{k}": f[k] for f in folds for k in ("coup", "pmi", "trial")})
    # per-trial 2x2 (window logits summed per trial, as evaluate._trial_acc)
    tc, tp = [], []
    for f in folds:
        for u in np.unique(f["trial"]):
            m = f["trial"] == u
            Lc, Lp = f["coup"][m].sum(0), f["pmi"][m].sum(0)
            tc.append(Lc[0] > Lc[1:].max())
            tp.append(Lp[0] > Lp[1:].max())
    tc, tp = np.array(tc), np.array(tp)
    sys.path.insert(0, str(Path(__file__).parent))
    from agreement import table  # noqa: E402
    res = {"checks": [f["check"] for f in folds], "trial": table(tc, tp)}
    (out / f"{args.tag}_trial.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["trial"], indent=1, default=float))


if __name__ == "__main__":
    main()
