#!/usr/bin/env python
"""Check that scripts/train/*.sh rebuild the configuration of every paper run, key by key.

For each launch script, the sbatch commands it prints under ``DRY_RUN=1`` are parsed.  For
every command and array index, ``scripts/slurm/train.sbatch`` itself is executed by bash with
``python -u main.py`` intercepted, which yields the exact argument list main.py would get
(including train.sbatch's fixed recipe and its fold gate).  ``main.main()`` then builds the
config from those arguments (training stubbed out), and the result is compared with
``<runs>/<tag>/<dataset>_fold<k>/config.yaml`` of the paper run.  Ignored keys: ``paths.*``,
``model_name``, ``mode``, ``checkpoint``, ``run_id``, ``eval_mode``.

Two documented exceptions are accepted and reported:
* RELOCATED: runs trained under another tag and moved into the tag the tables read; their
  stored ``tag`` must equal the original tag listed below (the tag only names the run directory).
* Keys absent from older stored configs (added to config.yaml after the run): accepted only if
  the launch command pins the key explicitly and the pinned value equals the fallback the code
  uses for a missing key (``cfg.get(key, fallback)`` in coupleflow/trainer.py and data.py).
Stored runs of a launched tag that no script reproduces fail, unless listed in UNUSED_STORED
(runs no paper table reads).

    source scripts/env/osc.sh            # COUPLEFLOW_DATA / COUPLEFLOW_RUNS of the paper runs
    python scripts/train/verify_configs.py [--scripts main loso ...] [--jobs 32]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import types
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coupleflow import paths  # noqa: E402

SCRIPTS = ["main", "loso", "ablations", "maestro", "maestro_ablations", "train_windows",
           "variants", "leakage"]          # dump_margins.sh launches no training
IGNORED = {"model_name", "mode", "checkpoint", "run_id", "eval_mode"}

# Fallback the code uses when the key is missing (keys added after some runs were made).
FALLBACK = {"data.split_mode": "stimulus",     # data.get("split_mode", "stimulus")
            "data.cand_idx": None,             # data.get("cand_idx") or []
            "data.tail_min_s": 200.0,          # data.get("tail_min_s", 200.0)
            "eval.select_metric": "contrib"}   # eval.get("select_metric", "contrib")

# (tag the tables read, dataset) -> tag the run was trained under (stored config.yaml `tag`).
RELOCATED = {
    ("unified", "megmasc"): "megnat_125", ("unified_s1", "megmasc"): "megnat_125_s1",
    ("unified_s2", "megmasc"): "megnat_125_s2", ("esaa_lag0", "megmasc"): "megnat_0",
    ("esaa_lag0_s1", "megmasc"): "megnat_0_s1", ("esaa_lag0_s2", "megmasc"): "megnat_0_s2",
    ("loso_s0", "megmasc"): "loso_megnat_0", ("_pca_megmasc/loso_s0", "megmasc"): "loso_s0",
}

# Stored runs inside launched tags that no paper script reads, so no script reproduces them.
UNUSED_STORED = {
    (t, ds): "125-ms seeds 1-2 where 0 ms was chosen; not an ablation dataset"
    for t in ("unified_s1", "unified_s2") for ds in ("esaa", "uhd255", "sparrkulee")
}

# python stub exported into train.sbatch: records `python -u ARGS` (main.py), runs the rest.
STUB = ('() { if [ "$1" = -u ]; then shift; printf "%s\\0" "$@" > "$VERIFY_ARGV"; return 0; fi; '
        'command python "$@"; }')


def leaves(node, prefix: str = ""):
    if OmegaConf.is_dict(node):
        for k in node:
            yield from leaves(node[k], f"{prefix}{k}.")
    else:
        yield prefix[:-1], (OmegaConf.to_container(node) if OmegaConf.is_list(node) else node)


def parse_array(spec: str) -> list[int]:
    out = []
    for part in spec.split("%")[0].split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def launches(script: str) -> list[dict]:
    """sbatch commands printed by scripts/train/<script>.sh under DRY_RUN=1."""
    env = dict(os.environ, DRY_RUN="1")
    res = subprocess.run(["bash", str(REPO / "scripts" / "train" / f"{script}.sh")], env=env,
                         capture_output=True, text=True, check=True)
    out = []
    for line in res.stdout.splitlines():
        if not line.startswith("sbatch "):
            continue
        a = shlex.split(line)
        i = a.index("scripts/slurm/train.sbatch")
        opts = dict(x.split("=", 1) for x in a[1:i])
        exports = dict(kv.split("=", 1) for kv in opts["--export"].split(",") if kv != "ALL")
        out.append({"script": script, "line": line, "env": exports,
                    "array": parse_array(opts["--array"]), "overrides": a[i + 1:]})
    return out


def sbatch_argv(job: dict, index: int, workdir: Path) -> list[str] | None:
    """main.py argv that train.sbatch builds for one array index (None: index skipped)."""
    rec = workdir / f"argv_{uuid.uuid4().hex}"
    env = dict(os.environ, **job["env"], SLURM_ARRAY_TASK_ID=str(index),
               SLURM_SUBMIT_DIR=str(workdir), VERIFY_ARGV=str(rec))
    env["BASH_FUNC_python%%"] = STUB
    res = subprocess.run(["bash", str(REPO / "scripts" / "slurm" / "train.sbatch"),
                          *job["overrides"]], env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"train.sbatch failed for {job['line']} [{index}]:\n{res.stderr}")
    if not rec.exists():
        return None
    argv = rec.read_bytes().split(b"\0")[:-1]
    rec.unlink()
    return [x.decode() for x in argv]


def build_config(argv: list[str]):
    """Config exactly as main.py builds it for this argv (training stubbed out)."""
    import main as entry
    got = {}
    fake = types.ModuleType("coupleflow.trainer")
    fake.train = lambda cfg: got.setdefault("cfg", cfg)
    saved = sys.modules.get("coupleflow.trainer"), sys.argv
    sys.modules["coupleflow.trainer"] = fake
    sys.argv = argv
    try:
        entry.main()
    finally:
        sys.argv = saved[1]
        if saved[0] is None:
            sys.modules.pop("coupleflow.trainer", None)
        else:
            sys.modules["coupleflow.trainer"] = saved[0]
    return got["cfg"]


def keyval(cfg) -> dict:
    return {k: v for k, v in leaves(cfg) if not k.startswith("paths.") and k not in IGNORED}


def pinned_keys(overrides: list[str]) -> set[str]:
    return {o.lstrip("+").split("=", 1)[0] for o in overrides if "=" in o}


def compare(job: dict, ds: str, tag: str, rebuilt: dict, stored: dict) -> tuple[list, list]:
    """(differences, accepted notes) between a rebuilt and a stored config."""
    diffs, notes = [], []
    pins = pinned_keys(job["overrides"])
    for k in sorted(set(rebuilt) | set(stored)):
        r, s = rebuilt.get(k, "<absent>"), stored.get(k, "<absent>")
        if k == "tag" and (tag, ds) in RELOCATED:
            if s == RELOCATED[(tag, ds)]:
                notes.append(f"relocated from tag {s}")
            else:
                diffs.append((k, r, s))
            continue
        if k not in stored and k in rebuilt:
            if k in FALLBACK and k in pins and r == FALLBACK[k]:
                notes.append(f"pinned {k}={r} (absent in stored config)")
            else:
                diffs.append((k, r, s))
            continue
        if r != s or isinstance(r, bool) != isinstance(s, bool):   # False == 0 in Python
            diffs.append((k, r, s))
    return diffs, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scripts", nargs="*", default=SCRIPTS)
    ap.add_argument("--runs", default=str(paths.RUNS), help="paper runs (default $COUPLEFLOW_RUNS)")
    ap.add_argument("--jobs", type=int, default=32, help="parallel train.sbatch emulations")
    args = ap.parse_args()
    runs = Path(args.runs)

    jobs = [j for s in args.scripts for j in launches(s) if j["env"].get("MODE", "train") == "train"]
    with tempfile.TemporaryDirectory(prefix="verify_", dir=os.environ.get("TMPDIR")) as tmp:
        work = Path(tmp)
        (work / "scripts").symlink_to(REPO / "scripts")      # train.sbatch sources scripts/env
        tasks = [(j, k) for j in jobs for k in j["array"]]
        with ThreadPoolExecutor(args.jobs) as ex:
            argvs = list(ex.map(lambda t: sbatch_argv(t[0], t[1], work), tasks))

    per_tag: dict[str, dict] = defaultdict(lambda: {"script": "", "n": 0, "skipped": 0,
                                                    "diffs": defaultdict(list),
                                                    "notes": defaultdict(Counter),
                                                    "missing": []})
    produced: dict[str, set] = defaultdict(set)
    for (job, k), argv in zip(tasks, argvs):
        tag, ds = job["env"]["TAG"], job["env"]["DS"]
        t = per_tag[tag]
        t["script"] = job["script"]
        if argv is None:                        # train.sbatch's fold gate: nothing to train
            t["skipped"] += 1
            continue
        produced[tag].add(f"{ds}_fold{k}")
        stored_path = runs / tag / f"{ds}_fold{k}" / "config.yaml"
        if not stored_path.exists():
            t["missing"].append(f"{ds}_fold{k}")
            continue
        rebuilt = keyval(build_config(argv))
        stored = keyval(OmegaConf.load(stored_path))
        diffs, notes = compare(job, ds, tag, rebuilt, stored)
        t["n"] += 1
        for key, r, s in diffs:
            t["diffs"][(key, repr(r), repr(s))].append(f"{ds}_fold{k}")
        for n in notes:
            t["notes"][n][ds] += 1

    n_pass = n_fail = n_runs = 0
    print(f"paper runs: {runs}\n")
    for tag in sorted(per_tag, key=lambda x: (per_tag[x]["script"], x)):
        t = per_tag[tag]
        stored_dirs = {p.name for p in (runs / tag).glob("*_fold*")} if (runs / tag).is_dir() else set()
        extra = sorted(stored_dirs - produced[tag])
        unused = [d for d in extra if (tag, d.rsplit("_fold", 1)[0]) in UNUSED_STORED]
        extra = [d for d in extra if d not in unused]
        ok = not t["diffs"] and not t["missing"] and not extra
        n_pass += ok
        n_fail += not ok
        n_runs += t["n"]
        print(f"{'PASS' if ok else 'FAIL'}  {tag:26s} {t['script'] + '.sh':22s} "
              f"{t['n']:4d} runs compared" + (f", {t['skipped']} array indices skipped by the fold gate"
                                         if t["skipped"] else ""))
        for (key, r, s), where in sorted(t["diffs"].items()):
            print(f"      DIFF {key}: rebuilt {r} != stored {s}  [{len(where)} runs: "
                  f"{', '.join(where[:6])}{' ...' if len(where) > 6 else ''}]")
        if t["missing"]:
            print(f"      NO STORED RUN for {len(t['missing'])}: {', '.join(t['missing'][:8])}")
        if extra:
            print(f"      STORED BUT NOT LAUNCHED: {', '.join(extra)}")
        if unused:
            dss = sorted({d.rsplit('_fold', 1)[0] for d in unused})
            print(f"      not launched (unused by the paper): {len(unused)} stored runs of "
                  f"{', '.join(dss)} -- {UNUSED_STORED[(tag, dss[0])]}")
        for n, by_ds in sorted(t["notes"].items()):
            print(f"      accepted: {n}  [{sum(by_ds.values())} runs: "
                  + ", ".join(f"{d} {c}" for d, c in sorted(by_ds.items())) + "]")
    print(f"\n{n_pass} tags PASS, {n_fail} tags FAIL; {n_runs} runs compared "
          f"({len(jobs)} sbatch commands from {', '.join(s + '.sh' for s in args.scripts)})")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
