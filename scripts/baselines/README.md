# Baselines

Three launchers regenerate every baseline number in the paper's tables. They need the bundles and
content groups from `scripts/data/` (see `scripts/data/README.md`); all three can run at the same
time.

```bash
source scripts/env/osc.sh                    # or export the COUPLEFLOW_* variables yourself
bash scripts/baselines/linear.sh             # linear backward models          (CPU, ~12 node-h)
bash scripts/baselines/deep.sh               # VLAAI, DECAF, dilated MM        (GPU, ~11 GPU-h)
bash scripts/baselines/spatial.sh            # DARNet, DBPNet, ListenNet       (GPU, ~77 GPU-h)
```

Every baseline uses CoupleFlow's folds, windows and inputs: the scripts read
`coupleflow/preprocessing/run_config.py` (MEG-MASC on all 208 sensors, Fuglsang'20 validation
tails) and the `unified_groups.json` content groups. Results go to `$COUPLEFLOW_DATA/<dataset>/`
(pooled, read by the tables) and `$COUPLEFLOW_BASELINES/` (per-fold outputs of the GPU baselines),
both set by `scripts/env/osc.sh` (defaults in `coupleflow/paths.py`). Logs land in `logs/`.

| Variable | Meaning |
|---|---|
| `DRY_RUN=1` | print every command (and, for `deep.sh`, the decoded array indices) instead of submitting |
| `MODE=local` | no Slurm: run every job (and array index) in the foreground, one after another |
| `STEPS` | subset of the steps listed below |
| `DATASETS`, `MODELS` (deep), `TASKS` (spatial) | restrict the runs |
| `DEPEND=afterok:<jobid>` | add a Slurm dependency to every job |
| `COUPLEFLOW_ENV` | environment file the jobs source (default `scripts/env/osc.sh`) |

GPU jobs (`scripts/slurm/baseline_deep.sbatch`, `scripts/slurm/baseline_spatial.sbatch`) skip
tasks whose per-fold output already exists; delete a file to rerun it. The linear scripts and the
aggregations overwrite their outputs.

## `linear.sh` — ridge backward models (CPU)

Steps `baseline trialsplit loso margins` (plus the opt-in `persub`); one `scripts/slurm/cpu.sbatch`
job per dataset and step, 22 jobs in all.

| Output (`$COUPLEFLOW_DATA/<ds>/`) | Script | Datasets | Read by |
|---|---|---|---|
| `linear_baseline.json` | `baselines/linear/linear_baseline.py` (`--single` on single-talker data) | 6 two-talker sets; sparrkulee, megmasc with `--single` | `compare_table.py` (main Table 1, `tables/compare.tex`, `numbers_compare.tex`), `make_tables.py` (`tables/main.tex`), `appendix_tables.py` (`tables/app_baselines.tex`), `leak_table.py`, `scripts/figures/kappa_linear.py`, `baselines/deep/aggregate.py` |
| `linear_baseline_trialsplit.json` | same, `--groups-name unified_groups_trialsplit.json` | kuleuven, dtu | `leak_table.py` (`tables/leak.tex`, App. F.2) |
| `linear_loso.json` | `baselines/linear/linear_loso.py` | 6 two-talker sets | `compare_table.py` (Table 1, LOSO column), `loso_table.py` (`tables/loso.tex`, App. F.6) |
| `linear_margins.npz` | `baselines/linear/linear_margins.py` | 6 two-talker sets | `confidence_appendix.py` (App. G) |
| `linear_baseline_persub.json` (`STEPS=persub`) | same as `linear_baseline.json`, `--out-name` | kuleuven, esaa | no table; an earlier export, now contained in `linear_baseline.json` (`per_subject_5s_competing`) |

The two-talker sets are kuleuven, dtu, fuglsang2020, esaa, uhd255 and broderick_cp; LOSO and
margins compare the attended with the competing talker, so they do not apply to the single-talker
sets. MEG-MASC (208 sensors) needs `--mem=180G`, which the script sets. Typical run times: 5–30 min
per `linear_baseline` job (SparrKULee ~70 min, MEG-MASC ~40 min), 15 min to 2.2 h per `linear_loso`
job (Fuglsang'20 the longest), 10–35 min per `linear_margins` job.

## `deep.sh` — VLAAI, DECAF, dilated match-mismatch (GPU)

Steps `main leak aggregate`.

- **main:** `scripts/slurm/baseline_deep.sbatch`, one subject-independent model per (model,
  dataset, content fold), trained on all training listeners and early-stopped on the fold's
  validation data (`baselines/deep/run.py`). Array index = model × 40 + dataset × 5 + fold:

  | indices | model | | indices | dataset (+ fold) |
  |---|---|---|---|---|
  | 0–39 | VLAAI | | +0–4 | kuleuven (4 folds) |
  | 40–79 | DECAF (also scored as `decaf_eeg`, its EEG branch) | | +5–9 | dtu |
  | 80–119 | dilated MM | | +10–14 | esaa |
  | | | | +15–19 | uhd255 (2 folds) |
  | | | | +20–24 | fuglsang2020 (4 folds) |
  | | | | +25–29 | broderick_cp |
  | | | | +30–34 | sparrkulee |
  | | | | +35–39 | megmasc (208 sensors): 35–39, 75–79, 115–119 |

  Only existing folds are submitted (101 tasks); `MODELS=` / `DATASETS=` select a subset, e.g.
  `MODELS=vlaai DATASETS=megmasc` submits 35–39. Output:
  `$COUPLEFLOW_BASELINES/results/<model>/<ds>_fold<k>.json`.
- **leak:** VLAAI and MM on KU Leuven and DTU under the leaky trial split
  (`--groups-name unified_groups_trialsplit.json`, 5 folds each; indices 0–9 and 80–89) into
  `$COUPLEFLOW_BASELINES/results_leak/` (`OUT_SUBDIR=results_leak`). Read directly by
  `scripts/tables/leak_table.py` (`tables/leak.tex`).
- **aggregate:** after the main array (`afterany`), `python -m baselines.deep.aggregate` pools the
  folds into `$COUPLEFLOW_DATA/<ds>/baseline_{vlaai,decaf,decaf_eeg,mm}.json` and rewrites
  `baselines/deep/RESULTS.md`; a dataset with a missing fold is listed as a failure and not
  written. Read by `compare_table.py` (Table 1), `make_tables.py` (`tables/main.tex`),
  `appendix_tables.py` (`tables/app_baselines.tex`, the only table with DECAF) and `leak_table.py`.

Each task takes 1–20 min on one A100 (limit 2 h 45 min, 110 GB): ~10 GPU-hours for the main array
and ~1 for the leakage array. VLAAI and DECAF are vendored under `baselines/deep/{vlaai,decaf}/upstream`
with their licences; `baselines/deep/{vlaai,decaf}/models_upstream.py` hold the PyTorch ports.

## `spatial.sh` — DARNet, DBPNet, ListenNet (GPU)

Steps `fetch tasks aggregate`.

- **fetch:** `baselines/spatial/fetch_thirdparty.sh` clones the authors' repositories at the
  commits used in the paper into `$COUPLEFLOW_THIRDPARTY` (default `baselines/spatial/thirdparty`,
  git-ignored). They carry no licence, so they are not redistributed:

  | directory | repository | commit |
  |---|---|---|
  | `darnet/` | https://github.com/fchest/darnet | `6254e7ebd2aa2a63b64b407a1be3013088381c10` |
  | `DBPNet/` | https://github.com/fchest/DBPNet | `99a153fdbb8276f227ac6393529a24af9afc5dbc` |
  | `ListenNet/` | https://github.com/fchest/ListenNet | `286b8ac4bddc7e6baee870940457809131249737` |

  Run it where there is network access (the launcher runs it on the submitting node). Existing
  clones at the pinned commits are left alone.
- **tasks:** `scripts/slurm/baseline_spatial.sbatch`, array 0–404 = the 405 rows of
  `python baselines/spatial/run.py --list`: 3 models × 5 two-talker datasets (kuleuven, dtu, esaa,
  fuglsang2020, uhd255) × windows 1, 2, 5 s × {protocol A subject-independent and per-listener on
  each content fold, protocol B (the papers' within-trial split) once}. Output:
  `$COUPLEFLOW_BASELINES/spatial_results/<model>/<ds>_<protocol>_<w>s_fold<k>.json`. 3–52 min per
  task on one A100 (limit 8 h, 110 GB), ~77 GPU-hours in all.
- **aggregate:** after the array (`afterany`), `baselines/spatial/aggregate.py` writes
  `$COUPLEFLOW_DATA/<ds>/spatial_{darnet,dbpnet,listennet}.json` and prints Markdown tables into
  its log. Read by `scripts/tables/spatial_table.py` (`tables/spatial.tex`, Sec. 5) and
  `appendix_tables.py` (`tables/app_spatial.tex`).

## Files on the paper's cluster

The commands above reproduce every baseline file the tables read: 35 per-fold files per deep model
in `results/`, 20 in `results_leak/`, 135 per spatial model in `spatial_results/`, and the pooled
`baseline_*.json`, `spatial_*.json` and `linear_*` files of each dataset. Not recreated:
`baselines_tmp/smoke/` (smoke tests of `run.py`).
