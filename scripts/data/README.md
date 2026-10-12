# Data preparation and checks

Three launchers turn the nine public releases into the inputs that CoupleFlow, every baseline
and every table read. Run them from anywhere (they `cd` to the repository root), in this order,
waiting for each script's jobs to finish before starting the next:

```bash
source scripts/env/osc.sh                    # or export the COUPLEFLOW_* variables yourself
bash scripts/data/build_bundles.sh           # raw releases -> preprocessed bundles   (~2.5 CPU node-h)
bash scripts/data/stimulus_groups.sh         # bundles -> content groups              (minutes)
bash scripts/data/checks.sh                  # tracking checks + split audit          (~1 CPU node-h)
```

Then run the baselines (`scripts/baselines/README.md`) and CoupleFlow training (`scripts/train/`).

| Variable | Meaning |
|---|---|
| `DRY_RUN=1` | print every command instead of submitting or running it |
| `MODE=local` | no Slurm: run each job in the foreground, one after another (default `MODE=slurm`) |
| `DATASETS="kuleuven dtu"` | restrict to some datasets (default: all nine) |
| `DEPEND=afterok:<jobid>` | add a Slurm dependency to every job, e.g. to chain a script after the previous one |
| `COUPLEFLOW_ENV` | environment file the jobs source (default `scripts/env/osc.sh`, the cluster the paper ran on) |

All jobs go through `scripts/slurm/build.sbatch` (32 cores, 180 GB) or `scripts/slurm/cpu.sbatch`
(16 cores, 96 GB by default); logs land in `logs/`. Every location comes from
`coupleflow/paths.py`: bundles and their sidecar files in `$COUPLEFLOW_DATA/<dataset>/` (default
`<repo>/data`), raw releases in `$COUPLEFLOW_RAW_<NAME>` (default `$COUPLEFLOW_RAW/<name>`, i.e.
`<repo>/data/raw/<name>`). Per-dataset run settings (MEG-MASC reads `unified_native.pt`,
Fuglsang'20 validation tails) live in `coupleflow/preprocessing/run_config.py`; every check and
baseline reads them.

## What each script produces

All paths are relative to `$COUPLEFLOW_DATA/<dataset>/` unless stated.

| Script | Output | Datasets | Read by |
|---|---|---|---|
| `build_bundles.sh` | `unified_env64.pt` | all 9 | every training run, baseline and check (MEG-MASC: only the content groups and the per-listener-PCA LOSO comparison `\losoMEGpca`, App. C) |
| | `unified_native.pt` | megmasc (208 sensors) | every MEG-MASC model and baseline (`run_config.stream`) |
| | `unified_native.pt` | uhd255 (253 ch), broderick_cp (128 ch) | `unified_native` runs: `tables/channels.tex`, `tables/app_variants.tex` |
| | `unified_pkg.pt` | maestro (release timing, no 0.781-s latency) | `maestro_pkg_*` runs: `tables/maestro_main.tex`, MAESTRO timing comparison (App. C) |
| `stimulus_groups.sh` | `unified_groups.json` | all 9 | content folds of every run, baseline and check |
| | `unified_groups_v2.json` (copy) | all 9 | `scripts/tables/groups_table.py` (`tables/groups.tex`, `numbers_groups.tex`); reference of `check_fold_leak.py` |
| | `unified_groups_trialsplit.json` | kuleuven, dtu | leakage runs (linear, VLAAI, MM, CoupleFlow `leak_trial`) -> `tables/leak.tex` |
| | `unified_groups_pkg.json` (copy) | maestro | `maestro_pkg_*` runs (same folds as the main MAESTRO runs) |
| `checks.sh` | `unified_env64.check.json` | all 9 | `compare_table.py` (main Table 2, `tables/tracking.tex`), `kappa_theory.py` (`tables/kappa_theory.tex`) via `run_config.check_name` |
| | `unified_native.check.json` | megmasc | same as above (MEG-MASC entry) |
| | `unified_native.check.json` | uhd255, broderick_cp | sanity check only |
| | `paper/tables/split_audit.jsonl` (in the repo) | all 9, content folds + LOSO | record of the split audit (Sec. 2, App. C); no table script reads it |

Notes:

- **MEG-MASC** has two bundles. `unified_native.pt` (all 208 sensors, `MEGMASC_NATIVE=1`) is the
  input of every model in the paper; `unified_env64.pt` is the abandoned 64-component
  per-listener PCA, kept for the LOSO comparison quoted in App. C. Checks and baselines on the
  208-sensor bundle need `--mem=180G` (the launchers set it).
- **Content groups** (`coupleflow/preprocessing/stimulus_groups.py`): 10-s chunks matched at 8 Hz
  (r > 0.9), the overlap confirmed at 64 Hz (r > 0.95), connected components = groups. SparrKULee
  needs `--block 512` to fit in 96 GB (set by the script).
- **`unified_groups_v2.json`** is the audit's re-derivation of the groups with the final algorithm
  (offsets ranked by their support). One run of the current code produces it, so the script copies
  `unified_groups.json`. The paper's `unified_groups.json` files predate that change; they are
  byte-identical to `_v2` for eight datasets, and for Fuglsang'20 they hold the same groups with a
  different `min_verified_r` (0.966 vs 0.977) — the folds are identical.
- **Split audit:** `checks.sh` empties `paper/tables/split_audit.jsonl` before auditing all nine
  datasets (it only appends when `DATASETS` is restricted; override with `RESET_AUDIT=0|1`). The 18
  audit jobs run one after another, so the lines come in a fixed order. All lines must read
  `"PASS": true`.
- `STEPS=leak bash scripts/data/checks.sh` runs `scripts/checks/check_fold_leak.py` (folds of
  `unified_groups.json` checked against `unified_groups_v2.json`; log output only).
- `SMOKE=1 bash scripts/data/build_bundles.sh` builds one listener per dataset into `*.smoke.pt`
  (MAESTRO: two trials of S01), to check a raw layout before the full build.
- Tracking checks use `--subjects 1000` (= every listener). The checks of the paper's run used 200
  (also every listener); the uhd255 / broderick_cp native-channel checks on the cluster cover only
  4 / 8 listeners, so regenerated versions of those two sanity checks contain more listeners.

## Raw releases

Download each release from its official page (all public; versions as in App. C,
Table "Releases, versions and licenses") and point `COUPLEFLOW_RAW_<NAME>` at it, or place it at
`<repo>/data/raw/<name>`. We redistribute no data. Layouts below are what the readers in
`coupleflow/preprocessing/readers/` open.

| Dataset (`<name>`) | Source | Version |
|---|---|---|
| KU Leuven (`kuleuven`) | Zenodo, https://doi.org/10.5281/zenodo.4004271 | v2.0 (2019); v2.1 (https://doi.org/10.5281/zenodo.21415059) fixes a 70-ms offset and was not used |
| DTU (`dtu`) | Zenodo, https://doi.org/10.5281/zenodo.1199011 | v1 (2018) |
| Fuglsang'20 (`fuglsang2020`) | Zenodo, https://doi.org/10.5281/zenodo.3618205 | 2020 (BIDS 1.3.0) |
| ESAA (`esaa`) | Zenodo, https://doi.org/10.5281/zenodo.7795585 | v1.0 (2022) |
| KUL-255ch (`uhd255`) | Zenodo, https://doi.org/10.5281/zenodo.4518754 | v1.1.0 (2021) |
| Broderick (`broderick_cp`) | Dryad, https://doi.org/10.5061/dryad.070jc | v3 (2019) |
| SparrKULee (`sparrkulee`) | KU Leuven RDR, https://doi.org/10.48804/K3VSND | v3.1 (2023); the recordings distributed as the ICASSP 2024 auditory-EEG challenge training set |
| MEG-MASC (`megmasc`) | OSF, https://doi.org/10.17605/OSF.IO/AG3KJ | 2022 |
| MAESTRO (`maestro`) | Hugging Face, https://huggingface.co/datasets/aspire-osu/maestro-eeg-dataset | 2026 |

Expected layout under each `$COUPLEFLOW_RAW_<NAME>`:

- **kuleuven**: `S1.mat` ... `S16.mat`; `stimuli/*_dry.wav` (the readers map every `*_hrtf.wav`
  stimulus name to its `*_dry.wav`).
- **dtu**: `eeg/S1.mat` ... `eeg/S18.mat`; `metadata/S<k>_metadata.mat`; `audio/*.wav` (the clean
  story wavs named in the metadata, e.g. `aske_story1_trial_10.wav`).
- **fuglsang2020**: the BIDS folder `ds-eeg-snhl` itself: `participants.tsv`,
  `sub-XXX/eeg/sub-XXX_task-selectiveattention_{eeg.bdf,events.tsv}`, and the authors' envelopes in
  `derivatives/stimuli/sub-XXX/{target,masker}/*.mat` (the release has no audio).
- **esaa**: the release files `S1.zip` ... `S17.zip` at the root (each is extracted on first read to
  `x/S<k>/*.mat`; pre-extracted `x/S<k>/*.mat` also works), and the clean stimuli unpacked by hand:
  `unzip clean.zip -d x` -> `x/clean/Trail1.wav` ... `Trail32.wav`.
- **uhd255**: `S0.tar.gz` ... `S29.tar.gz` at the root (extracted on first read into
  `x/S<k>/S<k>_AAD_{1L,1R,2L,2R}.{dat,dap,rs3,ceo}`; pre-extracted `x/S<k>/` also works), and
  `unzip stimuli.zip -d x` -> `x/stimuli/part{1,2}_track{1,2}_dry.wav`. The native-channel build
  waits for the first build so that the archives are extracted only once.
- **broderick_cp**: the `Cocktail Party` folder of the release: `EEG/Subject<k>/Subject<k>_Run<r>.mat`
  and `Stimuli/Envelopes/{20000,Journey}/<book>_<r>_env.mat` (the release has no audio).
- **sparrkulee**: the BIDS tree `sub-XXX/ses-*/eeg/*task-listeningActive*_{events.tsv,eeg.bdf.gz}`
  and the stimuli unpacked to `stimuli/stimuli/eeg/<name>.npz.gz` and `t_<name>.npz.gz` (the
  `stim_file` / `trigger_file` columns of `events.tsv`, e.g. `eeg/audiobook_5_1.npz.gz`, are
  relative to `stimuli/stimuli/`). The `.bdf.gz` files are decompressed to `$TMPDIR` on the fly.
- **megmasc**: the BIDS tree `sub-XX/ses-<s>/meg/sub-XX_ses-<s>_task-<t>_{meg.con,events.tsv}` and
  `stimuli/audio/<story>_<k>.wav` (lower-case names, e.g. `cable_spool_fort_0.wav`). Listeners
  without any downloaded story run are skipped; `MEGMASC_SUBJECTS="sub-01+sub-02"` restricts the
  build to some listeners.
- **maestro**: the Hugging Face snapshot: `metadata/{eeg_channels.json,bad_channels.csv,trials.csv,
  audio_manifest.json}`, `data/eeg/subject=S<k>/trial=<id>.parquet`,
  `media/timing/subject=S<k>/trial=<id>.json` and `media/audio/<trial>/*.flac`.

## Files on the paper's cluster that these scripts do not recreate

They were made by the earlier pipeline or by one-off diagnostics and are read by no table:
`paper_review_env64*.pt`, `stimulus_groups*.json` (old stream caches and their groups),
`sparrkulee/kul_overlap.json` (story-overlap check behind the SparrKULee exclusions, now hard-coded
in the reader), `*.smoke.pt` / `*.smoke.check.json` (smoke tests), and the downloaded archives and
download logs that share the dataset folders (`*.zip`, `dl.log`, MEG-MASC BIDS sidecars).
