# Published spatial AAD classifiers under a leakage-free vs. the papers' within-trial split

DARNet (NeurIPS 2024), ListenNet (IJCAI 2025) and DBPNet (IJCAI 2024) classify the
attended **side** (L/R) from an EEG window. We use no audio. Model code is imported
unmodified from the authors' repos. They are cloned under
`$COUPLEFLOW_THIRDPARTY/{darnet,ListenNet,DBPNet}` (`fetch_thirdparty.sh`).
The repos have no licence file, so none of their code is copied here. Training
recipes follow the authors' code (see `run.py` docstring). Runner: `run.py`. SLURM:
`scripts/baseline_spatial.sbatch` (405 tasks). Aggregation: `aggregate.py`, which writes
`$COUPLEFLOW_DATA/<ds>/spatial_<model>.json`. Per-task JSON files are in
`$COUPLEFLOW_BASELINES/spatial_results/` (in this repository: `results/baselines/spatial_results/`).

**Protocols**

* **A, subject-independent.** Uses CoupleFlow's content-disjoint folds:
  `split_fold(data, fold, 5, margin=100, val_mode="auto", val_frac=0.15, seed=42)`, and
  for fuglsang2020 `tail_min_s=30, val_frac=0.2`. There is one model per fold, trained
  on all training listeners and early-stopped on the fold's val segments. It is scored on
  non-overlapping windows (hop = window, at CoupleFlow's `eval_windows` positions)
  of the held-out stimulus groups. Folds are pooled; n_folds is 4 / 5 / 5 / 4 / 2 for
  kuleuven / dtu / esaa / fuglsang2020 / uhd255.
* **A, subject-dependent.** Same folds, with one model per listener per fold, trained
  on that listener's training segments and early-stopped on its val segments. Every
  listener had val segments, so the random-10 % fallback was never used.
* **B, the papers' split.** Subject-dependent. Each trial is cut at 90 % of its samples.
  Windows from the first part (50 % overlap) are split randomly 9:1 into train / val,
  and windows from the last part are the test set. CSP is fitted on the training part.
  This is the corrected protocol reported as "DARNet2 / DBPNet2" in the DARNet paper
  (Table 4) and reused in ListenNet's Table 2. Test windows are adjacent to training
  windows of the **same trial**. KU Leuven uses only the eight 6-min trials, as in the
  papers.

Chance is 50 %. Tables give the pooled accuracy over all scored windows, then the
mean ± sd over listeners. Windows are strongly correlated within a trial, so read the
listener spread, not the window count, as the uncertainty.

## Sanity check: protocol B vs. published numbers (subject-dependent, %)

| dataset | window | DARNet paper* | DARNet ours | DBPNet paper* | DBPNet ours | ListenNet paper | ListenNet ours |
|---|---|---|---|---|---|---|---|
| KUL | 1 s | 94.8 ± 4.5 | 96.8 ± 2.8 | 94.4 ± 4.6 | 97.1 ± 3.3 | 96.9 ± 3.0 | 98.1 ± 2.2 |
| KUL | 2 s | 95.5 ± 4.9 | 96.9 ± 3.1 | 95.3 ± 4.6 | 97.7 ± 2.6 | 97.3 ± 2.6 | 97.8 ± 2.4 |
| DTU | 1 s | 80.1 ± 6.9 | 67.7 ± 6.8 | 79.8 ± 6.9 | 66.5 ± 7.3 | 86.2 ± 5.6 | 74.5 ± 6.7 |
| DTU | 2 s | 81.2 ± 6.3 | 68.0 ± 7.6 | 80.2 ± 6.8 | 67.6 ± 6.4 | 86.6 ± 4.8 | 74.4 ± 7.8 |

\* These are the corrected CSP-after-split numbers ("DARNet2 / DBPNet2", DARNet
paper Table 4, repeated in ListenNet Table 2). The originally published numbers
(CSP fitted before the split) are higher: DARNet KUL 96.2 / 97.2, DTU 87.8 / 89.9;
DBPNet KUL 95.0 / 96.5, DTU 83.9 / 86.5.

On KU Leuven the re-implementations reach the published level; they are even 1–3 pp
above it. On DTU they fall about 12 pp short, and the ranking (ListenNet above DARNet
≈ DBPNet) is kept. The likely cause is preprocessing, not the models. The unified
bundle is high-passed at 0.5 Hz, whereas the papers use 0.1 Hz. It is also z-scored per
trial and channel. Both steps remove the slow trial-specific offsets that a
within-trial split can exploit. The papers' DTU data (the authors' `DATA_preproc`) also
use a mastoid reference and joint-decorrelation EOG removal, while we use a robust
average reference.

## Protocol B: papers' within-trial split (subject-dependent), accuracy %

| dataset | model | 1 s | 2 s | 5 s |
|---|---|---|---|---|
| kuleuven | darnet | 96.8 ± 2.8 | 96.9 ± 3.1 | 98.0 ± 2.8 |
| kuleuven | listennet | 98.1 ± 2.2 | 97.8 ± 2.4 | 97.0 ± 3.5 |
| kuleuven | dbpnet | 97.1 ± 3.3 | 97.7 ± 2.6 | 97.9 ± 3.0 |
| dtu | darnet | 67.7 ± 6.8 | 68.0 ± 7.6 | 67.5 ± 8.6 |
| dtu | listennet | 74.5 ± 6.7 | 74.4 ± 7.8 | 70.8 ± 12.0 |
| dtu | dbpnet | 66.5 ± 7.3 | 67.6 ± 6.4 | 68.8 ± 8.1 |
| esaa | darnet | 85.4 ± 8.2 | 85.7 ± 8.4 | 86.0 ± 9.6 |
| esaa | listennet | 91.6 ± 5.3 | 91.2 ± 6.0 | 88.5 ± 10.4 |
| esaa | dbpnet | 87.8 ± 7.7 | 89.6 ± 7.4 | 89.1 ± 9.9 |
| fuglsang2020 | darnet | 82.7 ± 12.3 | 82.8 ± 12.3 | 82.8 ± 13.6 |
| fuglsang2020 | listennet | 88.6 ± 9.1 | 87.6 ± 10.3 | 85.4 ± 11.6 |
| fuglsang2020 | dbpnet | 83.6 ± 11.6 | 83.6 ± 13.1 | 81.8 ± 15.0 |
| uhd255 | darnet | 98.5 ± 2.1 | 98.9 ± 1.9 | 99.0 ± 1.7 |
| uhd255 | listennet | 98.6 ± 1.8 | 98.4 ± 1.8 | 98.1 ± 3.2 |
| uhd255 | dbpnet | 99.4 ± 1.0 | 99.6 ± 0.8 | 99.6 ± 0.8 |

(Every listener has the same number of test windows, so the pooled and mean values are equal.)

## Protocol A: content-disjoint, subject-dependent, accuracy % (pooled | listener mean ± sd)

| dataset | model | 1 s | 2 s | 5 s |
|---|---|---|---|---|
| kuleuven | darnet | 52.3 \| 52.3 ± 22.3 | 53.1 \| 53.1 ± 25.7 | 53.8 \| 53.8 ± 24.6 |
| kuleuven | listennet | 41.9 \| 41.9 ± 14.2 | 43.9 \| 43.9 ± 16.2 | 44.0 \| 44.0 ± 17.2 |
| kuleuven | dbpnet | 46.8 \| 46.8 ± 23.5 | 45.5 \| 45.5 ± 26.1 | 45.0 \| 45.0 ± 27.4 |
| dtu | darnet | 54.5 \| 54.5 ± 6.3 | 53.2 \| 53.2 ± 5.3 | 53.2 \| 53.3 ± 6.1 |
| dtu | listennet | 54.1 \| 54.1 ± 6.7 | 55.0 \| 55.0 ± 7.3 | 55.2 \| 55.2 ± 7.5 |
| dtu | dbpnet | 50.8 \| 50.8 ± 4.7 | 52.2 \| 52.2 ± 4.1 | 51.4 \| 51.4 ± 4.3 |
| esaa | darnet | 56.0 \| 56.1 ± 15.2 | 58.6 \| 58.6 ± 14.2 | 59.4 \| 59.4 ± 14.6 |
| esaa | listennet | 56.5 \| 56.5 ± 11.5 | 58.8 \| 58.8 ± 12.3 | 59.5 \| 59.5 ± 13.6 |
| esaa | dbpnet | 59.8 \| 59.8 ± 15.5 | 61.6 \| 61.7 ± 14.8 | 61.6 \| 61.6 ± 16.3 |
| fuglsang2020 | darnet | 67.1 \| 67.1 ± 14.1 | 68.7 \| 68.7 ± 14.3 | 68.9 \| 68.9 ± 14.8 |
| fuglsang2020 | listennet | 70.5 \| 70.5 ± 14.9 | 71.2 \| 71.2 ± 15.8 | 71.3 \| 71.3 ± 16.2 |
| fuglsang2020 | dbpnet | 68.3 \| 68.3 ± 14.3 | 68.8 \| 68.8 ± 14.6 | 68.6 \| 68.6 ± 15.2 |
| uhd255 | darnet | 41.9 \| 41.9 ± 19.5 | 40.9 \| 40.9 ± 19.0 | 40.2 \| 40.2 ± 19.4 |
| uhd255 | listennet | 47.1 \| 47.1 ± 25.4 | 46.5 \| 46.5 ± 26.6 | 45.6 \| 45.6 ± 26.4 |
| uhd255 | dbpnet | 40.8 \| 40.8 ± 18.7 | 43.3 \| 43.3 ± 19.2 | 42.3 \| 42.3 ± 20.4 |

## Protocol A: content-disjoint, subject-independent, accuracy % (pooled | listener mean ± sd)

| dataset | model | 1 s | 2 s | 5 s |
|---|---|---|---|---|
| kuleuven | darnet | 54.2 \| 54.2 ± 16.5 | 58.5 \| 58.5 ± 19.4 | 58.9 \| 58.9 ± 21.2 |
| kuleuven | listennet | 55.1 \| 55.1 ± 20.2 | 55.7 \| 55.7 ± 20.1 | 57.4 \| 57.4 ± 22.1 |
| kuleuven | dbpnet | 51.6 \| 51.6 ± 18.1 | 53.8 \| 53.8 ± 19.5 | 53.4 \| 53.4 ± 20.9 |
| dtu | darnet | 49.2 \| 49.2 ± 3.9 | 49.9 \| 49.9 ± 4.7 | 47.3 \| 47.3 ± 4.6 |
| dtu | listennet | 49.1 \| 49.1 ± 5.5 | 48.1 \| 48.1 ± 4.0 | 49.6 \| 49.6 ± 4.5 |
| dtu | dbpnet | 47.7 \| 47.7 ± 3.6 | 48.2 \| 48.2 ± 3.8 | 48.2 \| 48.2 ± 3.2 |
| esaa | darnet | 55.8 \| 55.8 ± 13.5 | 54.9 \| 55.0 ± 15.0 | 54.5 \| 54.6 ± 12.8 |
| esaa | listennet | 56.0 \| 56.0 ± 11.0 | 57.9 \| 58.0 ± 11.1 | 59.3 \| 59.4 ± 14.2 |
| esaa | dbpnet | 55.8 \| 55.8 ± 12.5 | 56.8 \| 56.9 ± 13.9 | 58.1 \| 58.2 ± 12.2 |
| fuglsang2020 | darnet | 65.7 \| 65.6 ± 12.2 | 65.3 \| 65.3 ± 12.9 | 67.8 \| 67.8 ± 14.2 |
| fuglsang2020 | listennet | 68.9 \| 68.9 ± 13.9 | 71.0 \| 71.0 ± 14.1 | 71.2 \| 71.1 ± 14.7 |
| fuglsang2020 | dbpnet | 66.8 \| 66.8 ± 12.7 | 68.4 \| 68.4 ± 13.0 | 69.7 \| 69.7 ± 12.6 |
| uhd255 | darnet | 44.6 \| 44.6 ± 17.0 | 41.8 \| 41.8 ± 17.9 | 44.4 \| 44.4 ± 20.8 |
| uhd255 | listennet | 48.4 \| 48.4 ± 17.9 | 47.1 \| 47.1 ± 17.6 | 48.8 \| 48.8 ± 20.3 |
| uhd255 | dbpnet | 43.1 \| 43.1 ± 18.6 | 41.9 \| 41.9 ± 17.4 | 42.1 \| 42.1 ± 18.8 |

## Reading

* **Most of the within-trial accuracy is trial identity, not attention.** Moving from
  B to A drops accuracy by these amounts (1–5 s windows, any model):
  * KUL: from 97–98 % to 42–59 %.
  * uhd255: from 98–99 % to 40–48 %.
  * DTU: from 67–75 % to 47–55 %.
  * esaa: from 85–92 % to 55–62 %.
* **fuglsang2020 is the one dataset where a real side signal survives** content-disjoint
  testing: 65–71 %, both subject-dependent and subject-independent. The loss from B is
  still 15–20 pp. esaa keeps a weak 55–62 %.
* **DTU is at chance** under subject-independent A (47–50 %). It is 51–55 % under
  subject-dependent A.
* **Longer windows do not help under A.** Accuracy is flat from 1 s to 5 s, which is
  what a decoder of static trial-level signatures would do, not one that accumulates
  evidence.
* **Below-chance values are structural, not noise.**
  * KUL: the attended side is fixed per story within a listener (part 1 always R,
    part 2 always L, and so on, repeats included). Every one of the 64 listener-fold
    test sets is therefore single-class, and the listener's training prior points the
    other way. A model keyed to stimulus or trial content is systematically wrong,
    which explains the huge listener sd (±14–27).
  * uhd255: each listener has 4 trials (2 L, 2 R) over 2 folds, with balanced test
    sets. The same anti-generalization appears (40–48 %). This is consistent with
    side-specific trial fingerprints that point the wrong way on the held-out stimuli.
    It is a hypothesis; we have not verified it.
* **broderick_cp was skipped.** Each listener attends the same side throughout, so
  within a listener the label is constant. A side classifier then cannot be trained
  subject-dependently, and subject-independently it could only learn listener identity.

## Deviations from the published recipes (all protocols)

1. **Input: 64 Hz upsampled to 128 Hz.** The unified bundle (0.5–32 Hz, 64 canonical
   channels, z-scored per trial) is upsampled 64 → 128 Hz with polyphase `resample_poly`.
   This keeps every kernel, pooling and FFT band at the temporal or frequency extent used
   in the papers, which ran at 128 Hz on KUL and at 128 Hz on DTU for ListenNet and
   DBPNet. DARNet's DTU code ran at 64 Hz. Upsampling adds no information above 32 Hz.
   DBPNet's gamma band (31–50 Hz) therefore carries only the 31–32 Hz edge.
2. **Class-balanced cross-entropy for train and val loss.** Under content-disjoint folds
   a listener's training prior can anti-correlate with its test label, which is the KUL
   case above. An unweighted loss would then reward a prior that is guaranteed wrong on
   test. For balanced B splits this is effectively a no-op.
3. **CSP (DARNet 64 components, DBPNet 32 components)** is MNE `CSP(cov_est='concat',
   norm_trace=True, transform_into='csp_space')` as upstream, with two changes:
   * shrinkage `reg=1e-3` instead of `None`, because the average reference and zeroed bad
     channels make the covariances rank-deficient;
   * it is fitted on non-overlapping 1-s chunks of the training data only. With concat
     covariances this is equivalent to fitting on the whole training region.
4. **ListenNet Euclidean alignment** is computed per set (train / val / test) as upstream.
   The test set is thus aligned transductively, which is label-free. Eigenvalues are
   floored at 1e-3 × their mean, because `inv(sqrtm(R))` is infinite on rank-deficient
   data. The subject-independent A runs use the `indep/main.py` learning rate (1e-3)
   but early-stop on our val set. Upstream indep has no val set.
5. **DBPNet topographic images** use MNE `standard_1005` positions of the canonical
   layout (TP9/TP10 mapped to P9/P10), projected with DBPNet's `azim_proj`, instead of
   `locs_orig.mat` (BioSemi order). Interpolation is `CloughTocher2DInterpolator`, which
   is what `griddata(method='cubic')` uses. For bad (zeroed) channels, the band power is
   replaced by the mean over good channels, since log2(0) is undefined. Image z-scoring
   is per band per set: the train set's statistics for train and val, the test set's own
   statistics for test, as upstream.
6. **Evaluation and training details.** Test scoring uses all test windows; upstream
   drops the last incomplete batch. Each model has one seed (42). The top-level DBPNet
   wrapper (TAB ‖ FRB → Linear(8, 2)) is re-stated in `run.py`, because `model.py`
   imports unrelated dependencies (dotmap); both branches are imported unmodified.
7. **Epoch caps.** ListenNet hit its 100-epoch cap in 775 of 1860 trainings, with val
   loss still falling slowly. DARNet hit its cap in 182 and DBPNet (200-epoch cap) in
   124. These caps are the papers' own.
8. **5-s windows** were not in the papers (they used 0.1 / 1 / 2 s). We use the same
   recipe for them.
