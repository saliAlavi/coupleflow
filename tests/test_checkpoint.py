"""Shipped checkpoints load without data and match their configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from coupleflow.checkpoint import load_run

RUNS = Path(__file__).resolve().parents[1] / "runs"
CASES = ["final/kuleuven_fold0",          # two-talker, sub-band features + glimpses
         "final/fuglsang2020_fold0",      # envelope only
         "final/megmasc_fold0",           # MEG, 208 sensors, single talker
         "maestro_within_lag0/maestro_fold0"]


@pytest.mark.parametrize("rel", CASES)
def test_load_run(rel):
    run = RUNS / rel
    if not (run / "best.pt").exists():
        pytest.skip(f"{rel} not present")
    model, cfg, listeners = load_run(run)
    assert not model.training
    assert len(listeners) == model.U.shape[0]
    assert sum(p.numel() for p in model.parameters()) > 0
    assert torch.isfinite(torch.cat([p.flatten() for p in model.parameters()])).all()
