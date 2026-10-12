# CoupleFlow

**Null-calibrated auditory attention decoding with time-locked brain–speech coupling and
flow-matching likelihood ratios** (AISTATS 2027 submission, under review).

Ali Alavi (alavibajestan.1@osu.edu) and Donald S. Williamson (williamson.413@osu.edu),
Department of Computer Science and Engineering, The Ohio State University.

When several people talk at once, the auditory cortex follows the talker a listener attends to.
Auditory attention decoding (AAD) asks whether that talker can be identified from a few seconds of
EEG or MEG, by matching the recording against the speech of each candidate talker. CoupleFlow
reaches its decision only through the time-locked coupling between a neural embedding and each
candidate's acoustic embedding, so it is at chance whenever the brain signal carries no
information. A conditional flow-matching head adds a likelihood-ratio rule (the pointwise mutual
information between speech and brain). The model is evaluated on content-disjoint splits
recovered from the audio, with null tests that separate time-locking from trial recognition,
across nine EEG and MEG datasets under one preprocessing pipeline.

This repository holds everything behind the paper: the model, preprocessing, baselines, the
launch scripts for every training run, the trained checkpoints and their results, the scripts
that regenerate every table and figure, the LaTeX source, and the project page.

- Paper: [`paper/main.pdf`](paper/main.pdf) (source in [`paper/`](paper/))
- Project page: [`docs/index.html`](docs/index.html) (GitHub Pages from `/docs`, once public)

## Main result

5-s accuracy (%) on content-disjoint folds; CoupleFlow is the mean of 3 seeds, baselines are
trained on the same folds, windows and inputs (from `paper/tables/compare.tex`; 1-s, 20-s and
LOSO columns there).

| Dataset | Task | Linear | VLAAI | Dil. MM | CoupleFlow |
|---|---|---|---|---|---|
| KU Leuven | competing | 64.9 | 63.8 | 65.9 | **72.7** |
| DTU | competing | 72.4 | 64.8 | 69.1 | **76.2** |
| Fuglsang'20 | competing | 77.3 | 68.9 | 73.0 | **79.6** |
| ESAA | competing | 74.1 | 66.4 | 70.6 | **77.9** |
| KUL-255ch | competing | 65.5 | 62.1 | 65.0 | **69.4** |
| Broderick | competing | **63.7** | 52.2 | 52.3 | 63.6 |
| SparrKULee | match–mismatch | 76.3 | 70.5 | 83.3 | **92.2** |
| MEG-MASC | match–mismatch | 92.0 | 89.1 | 96.2 | **99.0** |

## Repository layout

```
coupleflow/                 the Python package
  model.py losses.py        CoupleFlow (coupling score, listener correction, flow head) and its objective
  data.py evaluate.py       content-disjoint folds, crops, candidate sets, evaluation battery and nulls
  trainer.py validator.py   training loop (one pooled model per fold) and checkpoint re-evaluation
  config.yaml               default configuration (every run overrides it from the command line)
  paths.py                  all filesystem locations (environment variables, defaults inside the repo)
  preprocessing/            the single preprocessing path (core.py), one reader per dataset,
                            content groups (stimulus_groups.py), per-dataset run settings (run_config.py)
  analysis/aggregate.py     pools run results over folds (used by every table)
baselines/
  linear/                   ridge backward models (per listener, LOSO, per-window margins)
  deep/                     VLAAI, DECAF and dilated match-mismatch on the same folds and inputs
  spatial/                  DARNet, DBPNet, ListenNet (authors' code, fetched by fetch_thirdparty.sh)
scripts/
  train/                    one launcher per training experiment (+ verify_configs.py)
  data/                     raw releases -> bundles, content groups, data checks
  baselines/                launchers for all baselines
  tables/  figures/         one generator per table / figure (see the map below)
  results/                  lag selection, final tags, per-window margins
  checks/                   tracking check, split audit, offset scan
  paper/                    figure sync, abstract, LaTeX build
  slurm/  env/              job scripts and cluster locations
  build_paper.sh            regenerate every table and figure, then the PDF
runs/                       checkpoints and results of every run the paper uses (see below)
data/                       small per-dataset files: content groups, checks, baseline results
results/                    per-fold baseline outputs, the earlier-decoder table, offset scan
figures/                    generated figures (paper/figures holds the ones the paper includes)
paper/                      LaTeX source, generated tables/macros, figures, main.pdf
docs/                       project page
tests/                      CPU unit tests
```

## Installation

```bash
conda env create -f environment.yml && conda activate coupleflow   # Python 3.11, versions of the paper runs
pip install -e .
make test                                                          # 12 CPU tests, about 1 minute
```

`pip install -e ".[dev]"` into any Python ≥ 3.10 environment with PyTorch also works.
Training needs a CUDA GPU; everything else in this README runs on CPU.

## Reproduce the paper's tables and figures (no data needed)

Every number in the paper is a LaTeX macro written by a script; nothing is typed by hand. The runs
and baseline results these scripts read are in the repository, so from a fresh clone:

```bash
bash scripts/fetch_release_assets.sh   # App. G margins (721 MB release asset; needs repo access)
make paper                             # = bash scripts/build_paper.sh  (~7 min on CPU)
git status                             # nothing changes: outputs are byte-identical
```

`make tables` skips LaTeX. Without the release asset the build skips App. G. The App. D data
thumbnails need the KU Leuven bundle (`WITH_DATA=1`); their outputs are committed.

| Paper item | Generator | Output |
|---|---|---|
| Fig. 1 (architecture) | `paper/fig_arch.tex` (TikZ) + `scripts/figures/fig_architecture.py` | `figures/arch_{eeg,env}.pdf` |
| Table 1 (windows, LOSO), Table 2 (tracking) | `scripts/tables/compare_table.py` | `tables/compare.tex`, `tracking.tex`, `numbers_compare.tex` |
| Fig. 2 (accuracy vs. window, law fit) | `scripts/figures/snr_law.py --tag final` | `figures/snr_law_final.{pdf,json}` |
| Fig. 3 (κ per dataset) | `scripts/figures/kappa_linear.py`, `kappa_fig.py` | `figures/kappa_bars.pdf` |
| Fig. 4, App. Table (κ from tracking) | `scripts/figures/kappa_theory.py`, `kappa_theory_fig.py` | `figures/kappa_theory.pdf`, `tables/kappa_theory.tex` |
| Table 3 (flow-head rules), ablation, channels | `scripts/tables/make_tables.py` | `tables/{flow,ablation,channels}.tex`, `tables.json` |
| Table 4 (MAESTRO), MAESTRO ablation | `scripts/tables/maestro_paper.py`, `maestro_ablation.py` | `tables/maestro_main.tex`, `maestro_ablation.tex` |
| Table 5 / App. (side classifiers) | `scripts/tables/spatial_table.py`, `appendix_tables.py` | `tables/spatial.tex`, `app_spatial.tex` |
| App. C content groups | `scripts/tables/groups_table.py` | `tables/groups.tex`, `numbers_groups.tex` |
| App. D modules, network figures | `scripts/tables/arch_table.py`, `scripts/figures/fig_net_thumbs.py`, `fig_net_encoders.py` | `tables/arch.tex`, `figures/net_*.pdf`, `fig_net_encoders.tex` |
| App. E loss terms | `scripts/tables/loss_table.py` | `tables/loss_terms.tex` |
| App. F baselines at every window, variants | `scripts/tables/appendix_tables.py` | `tables/app_baselines.tex`, `app_variants.tex` |
| App. F leakage, earlier decoders | `scripts/tables/leak_table.py`, `earlier_split_table.py` | `tables/leak.tex`, `earlier_split.tex` |
| App. F LOSO | `scripts/tables/loso_table.py` | `tables/loso.tex` |
| App. G confidence and sequential decisions | `scripts/tables/confidence_appendix.py` | `tables/conf_*.tex`, `figures/conf_*.pdf`, `paper/conf_results.json` |
| All text macros | `scripts/tables/make_numbers.py` (last) | `tables/numbers.tex` |
| OpenReview abstract | `scripts/paper/make_abstract.py` | `paper/abstract.md` |

Tables 1–5 are numbered as in the main text; `tables/` and `figures/` are under `paper/`
and the repository root respectively.

## Checkpoints and results (`runs/`)

`runs/<tag>/<dataset>_fold<k>/` holds, for each of the 1,317 runs the paper uses:

| File | Contents |
|---|---|
| `best.pt` | checkpoint selected on validation (`model` state dict, `subjects`, epoch) |
| `config.yaml` | the full configuration the run was trained with |
| `results.json` | test results at every window and candidate set, nulls, per-listener and per-trial accuracy, training history |
| `train.log` | training log |
| `conf_margins.npz` | per-window margins for App. G (release asset, not in git) |

Tags: `final`, `final_s1`, `final_s2` are the paper's main runs (seeds 0–2), symlinks chosen per
dataset by validation from `unified*` (125-ms lag window) and `esaa_lag0*` (0-ms window). `loso_s0` is
leave-one-listener-out, `abl_*`/`mabl_*` are ablations, `twin_*` training-window arms,
`maestro_*` MAESTRO, `leak_trial` the trial-split leakage control, `unified_envonly`,
`unified_native`, `esaa_rank8`, `esaa_sft` variants. `scripts/train/README.md` lists which
table reads which tag.

Load a trained model (no data needed):

```python
from coupleflow.checkpoint import load_run

model, cfg, listeners = load_run("runs/final/kuleuven_fold0")   # CoupleFlow in eval mode
```

Re-evaluate a checkpoint on its test fold with the configuration it was trained with (needs the
dataset bundle, see below; writes `results_val.json` next to the checkpoint):

```bash
python main.py --mode val --checkpoint runs/final/kuleuven_fold0/best.pt
```

`scripts/results/dump_margins.py` shows the full path from a run directory to per-window scores.

## Data

The nine datasets are public but are not redistributed here (sizes and licenses).
`scripts/data/README.md` gives the source of each release, the raw layout each reader expects,
and the commands that turn them into the preprocessed bundles (`unified_env64.pt`; MEG-MASC
`unified_native.pt`), content groups and checks:

| Dataset | Listeners | Release | License |
|---|---|---|---|
| KU Leuven | 16 | Zenodo 10.5281/zenodo.4004271 | CC BY-NC-SA 4.0 |
| DTU | 18 | Zenodo 10.5281/zenodo.1199011 | CC BY-NC 4.0 |
| Fuglsang'20 | 44 | Zenodo 10.5281/zenodo.3618205 | CC BY 4.0 |
| ESAA | 17 | Zenodo 10.5281/zenodo.7795585 | CC BY 4.0 |
| KUL-255ch | 30 | Zenodo 10.5281/zenodo.4518754 | CC BY-NC 4.0 |
| Broderick | 33 | Dryad 10.5061/dryad.070jc | CC0 1.0 |
| SparrKULee | 85 | KU Leuven RDR 10.48804/K3VSND | CC BY-NC 4.0 |
| MEG-MASC | 27 | OSF 10.17605/OSF.IO/AG3KJ | CC0 1.0 |
| MAESTRO | 16 | Hugging Face aspire-osu/maestro-eeg-dataset | CC BY-NC-SA 4.0 |

`data/<dataset>/` in this repository holds only small derived files: the content groups that
define the folds (`unified_groups.json`), tracking checks, linear and deep baseline results, and
two-column extracts of release metadata (`fuglsang2020/hearing_status.tsv`,
`maestro/trial_snr.csv`, under their releases' licenses). Put the bundles next to them, or point
`COUPLEFLOW_DATA` at another directory with the same layout.

## Training

```bash
source scripts/env/osc.sh            # cluster locations; elsewhere export the COUPLEFLOW_* variables
DRY_RUN=1 bash scripts/train/main.sh # print the sbatch commands
bash scripts/train/main.sh           # main runs: 8 datasets x 5 folds x 2 lag windows x 3 seeds
python scripts/results/select_lag.py && python scripts/results/make_final.py
```

One fold by hand:

```bash
python main.py --mode train --dataset kuleuven fold=0 seed=0 tag=mine \
    data.stream_name=unified_env64.pt data.groups_name=unified_groups.json \
    "data.feat_banks=[sb,sb28]" data.glimpse=true optim.epochs=100 optim.patience=20
```

`scripts/train/README.md` lists every experiment (main, LOSO, ablations, MAESTRO, training
windows, variants, leakage, margins) with its tags, paper items, run count and GPU-hours, and
`scripts/train/verify_configs.py` checks that the launchers rebuild each paper run's
`config.yaml`. Baselines: `scripts/baselines/README.md`.

## Configuration

| Variable | Default | Contents |
|---|---|---|
| `COUPLEFLOW_DATA` | `data/` | `<dataset>/` bundles and sidecar files |
| `COUPLEFLOW_RUNS` | `runs/` | `<tag>/<dataset>_fold<k>/` runs |
| `COUPLEFLOW_BASELINES` | `results/baselines/` | per-fold baseline outputs |
| `COUPLEFLOW_THIRDPARTY` | `baselines/spatial/thirdparty/` | DARNet, DBPNet, ListenNet repositories |
| `COUPLEFLOW_RAW_<NAME>` | `data/raw/<name>/` | raw release of one dataset (readers only) |

Training options are OmegaConf keys of `coupleflow/config.yaml`, overridden as `key=value` on the
`main.py` command line.

## Citation

```bibtex
@inproceedings{alavi2027coupleflow,
  title     = {CoupleFlow: Null-Calibrated Auditory Attention Decoding With Time-Locked
               Brain--Speech Coupling and Flow-Matching Likelihood Ratios},
  author    = {Alavi, Ali and Williamson, Donald S.},
  booktitle = {Under review at AISTATS 2027},
  year      = {2027}
}
```

## License

MIT (see `LICENSE`), except third-party code: `baselines/deep/vlaai/` is GPL-3.0 (port of
exporl/vlaai), `baselines/deep/decaf/upstream/` is MIT (carankt/DECAF). DARNet, DBPNet and
ListenNet are not redistributed. Dataset-derived files in `data/` keep their releases' licenses.
