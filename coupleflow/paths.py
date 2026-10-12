"""Filesystem locations used by every CoupleFlow script.

All defaults live inside the repository, so tables and figures rebuild from a fresh clone.
Point the environment variables at other locations to train or preprocess
(``scripts/env/osc.sh`` sets them for the cluster the paper was run on):

==========================  ================================  ===================================
Variable                    Default                           Contents
==========================  ================================  ===================================
``COUPLEFLOW_DATA``         ``<repo>/data``                   ``<dataset>/unified_env64.pt`` bundles
                                                              and their small sidecars (content
                                                              groups, checks, baseline results)
``COUPLEFLOW_RUNS``         ``<repo>/runs``                   ``<tag>/<dataset>_fold<k>/`` runs
                                                              (``best.pt``, ``results.json``, ...)
``COUPLEFLOW_BASELINES``    ``<repo>/results/baselines``      per-fold deep-baseline outputs
                                                              (``results/``, ``results_leak/``,
                                                              ``spatial_results/``)
``COUPLEFLOW_THIRDPARTY``   ``<repo>/baselines/spatial/``     authors' DARNet / DBPNet /
                            ``thirdparty``                    ListenNet repositories
``COUPLEFLOW_RAW``          ``<repo>/data/raw``               raw dataset releases (readers only)
``COUPLEFLOW_RAW_<NAME>``   ``$COUPLEFLOW_RAW/<name>``        one raw release, e.g.
                                                              ``COUPLEFLOW_RAW_DTU``
==========================  ================================  ===================================
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


DATA = _env("COUPLEFLOW_DATA", REPO / "data")
RUNS = _env("COUPLEFLOW_RUNS", REPO / "runs")
BASELINES = _env("COUPLEFLOW_BASELINES", REPO / "results" / "baselines")
THIRDPARTY = _env("COUPLEFLOW_THIRDPARTY", REPO / "baselines" / "spatial" / "thirdparty")
RAW = _env("COUPLEFLOW_RAW", REPO / "data" / "raw")
RESULTS = REPO / "results"


def raw(name: str) -> Path:
    """Root of the raw release of dataset ``name`` (``COUPLEFLOW_RAW_<NAME>`` overrides)."""
    return _env(f"COUPLEFLOW_RAW_{name.upper()}", RAW / name)
