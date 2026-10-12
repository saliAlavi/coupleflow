# Training launchers

One script per experiment family. Each submits `scripts/slurm/train.sbatch` (one array task per
fold or held-out listener) with the settings of the paper runs, under the run tags the table
and figure scripts read. `common.sh` holds the per-dataset settings shared by all of them.

```bash
source scripts/env/osc.sh                       # or export COUPLEFLOW_DATA / COUPLEFLOW_RUNS
DRY_RUN=1 bash scripts/train/main.sh            # print the sbatch commands, submit nothing
bash scripts/train/main.sh                      # submit
python scripts/train/verify_configs.py          # rebuilt configs vs the paper runs' config.yaml
```

## Experiments

GPU-hours are the paper runs' own training time (`minutes` in `results.json`, from the start of
training to the written results, test evaluation included), summed over the runs a script launches.

| Experiment | Script | Tags | Paper items | Runs | GPU-h |
|---|---|---|---|---|---|
| Main runs, both lag windows, 3 seeds | `main.sh` | `unified`, `esaa_lag0` (seed 0, 8 datasets); `unified_s1/_s2` (KUL, Fuglsang'20, Broderick, DTU, MEG-MASC); `esaa_lag0_s1/_s2` (DTU, ESAA, KUL-255ch, SparrKULee, MEG-MASC) -> `final*` | Tables tab:main, tab:track, tab:flow, tab:spatial (CoupleFlow row); Figs. fig:law, fig:kappa, fig:kth; App. D (arch, net figures); App. E loss terms; App. F tab:app_baselines, windows, tab:ablation base and 0-ms rows, tab:app_variants base columns, tab:channels, tab:leak; App. G; `numbers*.tex` | 160 | 105 |
| Leave-one-listener-out | `loso.sh` | `loso_s0`; `_pca_megmasc/loso_s0` | Table tab:main (LOSO column), App. F.6 tab:loso, `numbers_loso.tex`; `\losoMEGpca` (App. C) | 297 | 256 |
| Ablations (KUL, DTU, MEG-MASC) | `ablations.sh` | `abl_{noflow,norank,nomatch,noglimpse,noadv,nohinge,nocon,selacc,noside,noaux,before}` | App. F.1 tab:ablation, `\ablRank*`, `\lagCtl*` | 154 | 67 |
| MAESTRO | `maestro.sh` | `maestro_within_lag0{,_s1,_s2}`, `maestro_loso_lag0{,_s1,_s2}`, `maestro_pkg_within{,_s1,_s2}`, `maestro_pkg_loso{,_s1,_s2}`, `maestro_within_pre50` | Table tab:maestro, `\mae*`, `\maMainNull*` | 131 | 18 |
| MAESTRO ablations | `maestro_ablations.sh` | `mabl_{noflow,norank,nomatch,noglimpse,nobanks,noadv,nohinge,nocon,selacc,noside,noaux,before}` | App. F.3 tab:mabl, `\maAbl*`, `\maBefore*`, `\maSafeMax*` | 60 | 10 |
| Training-window arms | `train_windows.sh` | `twin_{w1,w2,w10,mix}_s{0,1,2}` | App. G training length (`conf_trainlen.tex`, `conf_trainlen.pdf`, `\confTl*`) | 420 | 425 |
| Variants | `variants.sh` | `unified_envonly`, `unified_native`, `esaa_rank8`, `esaa_sft` | App. F tab:app_variants; tab:channels and `\kN*`/`\tN*` (`unified_native`) | 61 | 22 |
| Trial-split leakage | `leakage.sh` | `leak_trial` | App. F.2 tab:leak, `\leakMin/Max` | 10 | 3 |
| Per-window margins | `dump_margins.sh` | writes `conf_margins.npz` into `final*` (all datasets) and `twin_*` (two-talker datasets) | App. G (calibration, sequential decisions, overlap, training length) | 405 jobs | ~70 (est.) |
| **Total** | | | | **1293 runs** | **~905 + 70** |

Per-dataset settings (`common.sh`, identical in every experiment): Fuglsang'20 and Broderick are
envelope-only (`BANKS=none`); SparrKULee and MEG-MASC are single-talker (`SINGLE=1`); MEG-MASC
uses its 208 sensors (`STREAM=unified_native.pt`, 120 GB); Fuglsang'20 validates on the last 20%
of trials of at least 30 s (`data.tail_min_s=30 data.val_frac=0.2`). The 125-ms lag window is the
model default; the 0-ms window adds `model.lag_offset=0`. LOSO uses `train.sbatch` with
`SPLIT=loso` (array index = held-out listener).

## Order

```bash
source scripts/env/osc.sh
STAGE=seed0 bash scripts/train/main.sh          # 1. both lag windows, seed 0 (70 runs)
python scripts/results/select_lag.py            # 2. lag window per dataset, on validation
STAGE=seeds bash scripts/train/main.sh          # 3. seeds 1, 2 of the chosen window (90 runs)
python scripts/results/make_final.py            # 4. symlink tags final, final_s1, final_s2
bash scripts/train/loso.sh                      # 5. need lag_selection.json (chosen window)
bash scripts/train/train_windows.sh
bash scripts/train/ablations.sh                 #    independent of the lag choice
bash scripts/train/maestro.sh
bash scripts/train/maestro_ablations.sh
bash scripts/train/variants.sh
bash scripts/train/leakage.sh
bash scripts/train/dump_margins.sh              # 6. after 4 and train_windows.sh
# 7. tables and figures (scripts/tables, scripts/figures, scripts/paper)
```

`main.sh` without `STAGE` submits both stages using the committed
`paper/tables/lag_selection.json`; `loso.sh` and `train_windows.sh` read the same file (override
with `LAG_JSON=...`).

## Verification

`verify_configs.py` runs every script with `DRY_RUN=1`, executes `train.sbatch` for each array
index with `python -u main.py` intercepted, rebuilds the config through `main.py`, and compares it
key by key with the stored `config.yaml` of the paper run (ignoring `paths.*`, `model_name`,
`mode`, `checkpoint`, `run_id`, `eval_mode`). Current result: **61 tags PASS, 0 FAIL, 1293 runs**.
Accepted and reported exceptions:

- **Relocated runs.** The MEG-MASC runs of `unified`, `unified_s1/_s2`, `esaa_lag0`,
  `esaa_lag0_s1/_s2` were trained as `megnat_125[_s1|_s2]` / `megnat_0[_s1|_s2]` (208-sensor
  rerun) and the MEG-MASC LOSO runs of `loso_s0` as `loso_megnat_0`, then moved into the tags the
  tables read; the PCA-input LOSO runs were trained as `loso_s0` and moved to
  `_pca_megmasc/loso_s0`. Their stored `tag` is the original one (it only names the directory).
- **Pinned keys.** Runs made before a key existed lack it in `config.yaml`; the code then used its
  `cfg.get` fallback, which equals today's default. The affected tags pin it explicitly
  (`PIN_PRE_LOSO` = `data.split_mode=stimulus data.cand_idx=null`; `PIN_PRE_TAIL` =
  `data.tail_min_s=200.0 eval.select_metric=contrib`) and the check requires the pinned value to
  equal the fallback. The train/validation/test split recorded in `train.log` of all 274 such runs
  equals that of the runs that store the key.

## Notes

- Runs evaluated in a second job (`MODE=val`, `results_val.json`) after an evaluation crash that
  has since been fixed: `unified` (Broderick, DTU, ESAA), `unified_native` (Broderick),
  `maestro_within_lag0`, `maestro_loso_lag0`. A fresh run writes `results.json`; the tables read
  either.
- Stored runs no paper script reads, therefore not launched: `unified_s1/_s2` on ESAA,
  KUL-255ch and SparrKULee; `maestro_within`, `maestro_loso` (125-ms MAESTRO); `loso_megnat_125`;
  `conf_{base,confw,focal,gce,trim}_s{0,1,2}`; the emptied `megnat_*`/`loso_megnat_0`
  directories; and the pre-unified exploration tags (`main*`, `before*`, `it*`, `v_best_*`,
  `exp`, `smoke*`).
