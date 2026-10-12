#!/usr/bin/env python
"""CoupleFlow command-line entry point: train one fold, or re-evaluate a checkpoint.

    python main.py --mode train --dataset kuleuven fold=0 seed=0 tag=final
    python main.py --mode val --checkpoint runs/final/kuleuven_fold0/best.pt

Every ``key=value`` argument overrides ``coupleflow/config.yaml`` (OmegaConf dotlist).  In
``val`` mode the base configuration is the run's own ``config.yaml`` next to the checkpoint
(``--config`` overrides that choice), so a paper run is re-evaluated exactly as it was trained;
``results_val.json`` is written next to the checkpoint.
``scripts/slurm/train.sbatch`` adds the paper recipe; ``scripts/train/*.sh`` launch each
experiment.  Data and run locations come from ``coupleflow/paths.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from coupleflow import paths

CONFIG = Path(__file__).resolve().parent / "coupleflow" / "config.yaml"


def parse_cli_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the known flags; everything else is a dotlist override."""
    parser = argparse.ArgumentParser(description="CoupleFlow training and evaluation")
    parser.add_argument("--mode", required=True, choices=["train", "val"])
    parser.add_argument("--dataset", default=None, help="dataset name (directory under the data root)")
    parser.add_argument("--checkpoint", default=None, help="best.pt to re-evaluate (--mode val)")
    parser.add_argument("--config", default=None,
                        help="base config (default: coupleflow/config.yaml; in val mode the "
                             "config.yaml saved next to --checkpoint)")
    return parser.parse_known_args()


def main() -> None:
    args, overrides = parse_cli_args()
    base = Path(args.config) if args.config else CONFIG
    if args.config is None and args.mode == "val" and args.checkpoint:
        saved = Path(args.checkpoint).with_name("config.yaml")
        if saved.exists():
            base = saved
    cfg = OmegaConf.load(base)
    if overrides:
        # A leading "+" (Hydra's add-key marker) would otherwise create a key named "+foo".
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([o.lstrip("+") for o in overrides]))
    cfg.model_name = "coupleflow"
    cfg.mode = args.mode
    cfg.checkpoint = args.checkpoint
    if args.dataset is not None:
        cfg.training.dataset = args.dataset
    if cfg.paths.cache_root is None or base != CONFIG:
        cfg.paths.cache_root = str(paths.DATA)     # a saved config records the paper's cluster paths
    if cfg.paths.out_root is None or base != CONFIG:
        cfg.paths.out_root = str(paths.RUNS)
    OmegaConf.resolve(cfg)

    if args.mode == "train":
        from coupleflow.trainer import train
        train(cfg)
    else:
        if args.checkpoint is None:
            sys.exit("--checkpoint is required for --mode val")
        from coupleflow.validator import validate
        validate(cfg)


if __name__ == "__main__":
    main()
