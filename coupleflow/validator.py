"""Re-evaluate a trained CoupleFlow checkpoint on its test fold.

    python main.py --mode val --dataset kuleuven fold=0 tag=final \
        --checkpoint runs/final/kuleuven_fold0/best.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import DictConfig

from coupleflow.data import load_dataset, split_fold
from coupleflow.evaluate import evaluate_split
from coupleflow.trainer import adapt_and_evaluate, build_model, streams_for


def validate(cfg: DictConfig) -> None:
    """Load ``cfg.checkpoint`` and run the full evaluation battery on the test fold."""
    device = "cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu"
    ck = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    sn, gn = streams_for(cfg, str(cfg.training.dataset))
    data = load_dataset(cfg.paths.cache_root, str(cfg.training.dataset),
                        drop_frontal=bool(cfg.data.drop_frontal),
                        stream_name=sn, groups_name=gn)
    if bool(cfg.data.get("eeg_bands", False)):
        from coupleflow.trainer import append_eeg_bands
        append_eeg_bands(data)
    # Extra training datasets only contribute listeners (for the per-listener
    # correction); their trials are not evaluated.
    extra = []
    for ex_name in cfg.data.get("extra_train_names", None) or []:
        esn, egn = streams_for(cfg, str(ex_name))
        d2 = load_dataset(cfg.paths.cache_root, str(ex_name), stream_name=esn, groups_name=egn)
        extra += [f"{ex_name}:{x}" for x in d2["subjects"]]
    data["subjects"] = sorted(set(data["subjects"]) | set(extra))
    assert data["subjects"] == ck["subjects"], "subject list differs from training"
    data["cand_idx"] = list(cfg.data.get("cand_idx") or [])
    sp = split_fold(data, int(cfg.fold), int(cfg.n_folds), int(cfg.model.margin),
                    val_mode=str(cfg.data.val_mode), val_frac=float(cfg.data.val_frac),
                    tail_min_s=float(cfg.data.get("tail_min_s", 200.0)),
                    mode=str(cfg.data.get("split_mode", "stimulus")),
                    seed=int(cfg.get("split_seed", 42)))
    from coupleflow.trainer import feature_banks
    bank_dims = [int(data["trials"][0][f"env_att_{bk}"].shape[0]) for bk in feature_banks(cfg)]
    model = build_model(cfg, len(data["subjects"]), bank_dims,
                        n_channels=int(data["trials"][0]["eeg"].shape[0])
                        // (4 if bool(cfg.data.get("eeg_bands", False)) else 1)).to(device)
    model.load_state_dict(ck["model"])
    res = evaluate_split(model, data, sp["test"], device,
                         [float(x) for x in cfg.eval.windows], list(cfg.eval.cand_sets),
                         n_perm=int(cfg.eval.test_perms), n_circ=int(cfg.eval.circ_perms),
                         n_mc=int(cfg.eval.flow_mc), seed=int(cfg.seed))
    import numpy as np
    windows = {k: v.pop("_windows") for k, v in res.items() if "_windows" in v}
    np.savez_compressed(Path(cfg.checkpoint).with_name("test_windows.npz"),
                        **{f"{cell}__{k}": np.asarray(a) for cell, d in windows.items()
                           for k, a in d.items()})
    if cfg.get("adapt") and bool(cfg.adapt.get("cross", False)):
        cross_adapt(cfg, model, ck, data, sp, device)
        return
    out = Path(cfg.checkpoint).with_name("results_val.json")
    out.write_text(json.dumps({"dataset": str(cfg.training.dataset), "fold": int(cfg.fold),
                               "results": res}, indent=1, default=float))
    print(f"wrote {out}")


def cross_adapt(cfg: DictConfig, model, ck: dict, data: dict, sp: dict, device: str) -> None:
    """Stimulus-disjoint test-time adaptation.

    The test fold's stimulus groups are split into two halves A and B.  Starting
    from the supervised checkpoint each time, the model adapts (no labels) on A
    and is evaluated on B, then adapts on B and is evaluated on A.  No window
    it is scored on was used for adaptation, and no audio it is scored on was
    heard during adaptation.  Folds with a single stimulus group are skipped.
    """
    from omegaconf import OmegaConf
    groups = sorted({data["groups"][s.trial] for s in sp["test"]})
    out = {"groups": groups, "halves": []}
    if len(groups) < 2:
        print(f"fold {cfg.fold}: one stimulus group -- cross adaptation skipped")
    else:
        half = [set(groups[: len(groups) // 2]), set(groups[len(groups) // 2:])]
        segs = [[s for s in sp["test"] if data["groups"][s.trial] in h] for h in half]
        w = OmegaConf.to_container(cfg.losses, resolve=True)
        for a, e in ((0, 1), (1, 0)):
            model.load_state_dict(ck["model"])
            res0 = evaluate_split(model, data, segs[e], device,
                                  [float(x) for x in cfg.eval.windows], list(cfg.eval.cand_sets),
                                  n_perm=int(cfg.eval.test_perms),
                                  n_circ=int(cfg.eval.circ_perms), n_mc=int(cfg.eval.flow_mc),
                                  seed=int(cfg.seed))
            for v in res0.values():
                v.pop("_windows", None)
            ad = adapt_and_evaluate(cfg, model, data, segs[a], segs[e], device, w, print)
            out["halves"].append({"adapt_groups": sorted(half[a]), "eval_groups": sorted(half[e]),
                                  "supervised": res0, "adapted": ad})
    p = Path(cfg.checkpoint).with_name("results_crossadapt.json")
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {p}")
