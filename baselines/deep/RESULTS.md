# Deep AAD baselines on the unified bundles (CoupleFlow stimulus-disjoint folds)

One subject-independent model per fold (all training listeners), early-stopped on
the fold's val segments, scored on held-out stimulus groups with the windowing,
margins and same-talker negatives of `baselines/linear/linear_baseline.py` (identical
rng draws). Accuracy in %, folds pooled by window count. `comp` = attended vs
competing talker, `same` = attended vs another window of the same attended
envelope, `trial` = whole-trial competing accuracy. Linear = ridge mTRF
(`linear_baseline.json`, per listener).

## vlaai

| dataset | 1s comp | 1s same | 2s comp | 2s same | 5s comp | 5s same | 10s comp | 10s same | 20s comp | 20s same | 30s comp | 30s same | 60s comp | 60s same | trial | test r | best ep (per fold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kuleuven | 56.9 | 59.2 | 59.3 | 62.2 | 63.8 | 67.5 | 68.5 | 74.1 | 73.0 | 79.9 | 77.4 | 84.5 | 82.6 | 92.2 | 88.1 | 0.105 | 2,4,3,2 |
| dtu | 57.5 | 58.7 | 60.4 | 62.0 | 64.8 | 67.9 | 69.7 | 73.4 | 74.1 | – | 78.2 | – | – | – | 82.2 | 0.118 | 2,5,6,2,6 |
| esaa | 58.3 | 60.0 | 61.5 | 63.5 | 66.4 | 71.2 | 71.8 | 76.4 | 78.8 | 83.9 | 81.7 | 81.6 | 88.9 | – | 91.0 | 0.131 | 2,4,3,4,4 |
| uhd255 | 55.9 | 57.4 | 58.0 | 59.7 | 62.1 | 65.5 | 67.0 | 69.9 | 72.1 | 76.9 | 75.4 | 79.4 | 82.2 | 86.4 | 95.0 | 0.101 | 5,4 |
| fuglsang2020 | 60.6 | 61.7 | 63.7 | 65.2 | 68.9 | 72.1 | 74.5 | 77.5 | 81.7 | 86.0 | 85.4 | – | – | – | 89.8 | 0.152 | 6,5,5,5 |
| broderick_cp | 51.0 | 54.6 | 51.3 | 57.0 | 52.2 | 60.4 | 53.1 | 64.0 | 53.1 | 68.4 | 53.9 | – | – | – | 55.4 | 0.074 | 4,2,2,4,1 |
| sparrkulee | – | 59.9 | – | 64.0 | – | 70.5 | – | 77.1 | – | 83.9 | – | 87.5 | – | 93.3 | – | 0.160 | 11,13,18,12,10 |
| megmasc | – | 71.0 | – | 78.7 | – | 89.1 | – | 95.6 | – | 98.6 | – | 99.6 | – | 100.0 | – | 0.393 | 7,6,5,5,5 |

## decaf

| dataset | 1s comp | 1s same | 2s comp | 2s same | 5s comp | 5s same | 10s comp | 10s same | 20s comp | 20s same | 30s comp | 30s same | 60s comp | 60s same | trial | test r | best ep (per fold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kuleuven | 50.9 | 50.1 | 51.6 | 50.8 | 51.3 | 49.2 | 52.0 | 50.7 | 55.4 | 51.0 | 54.1 | 51.2 | 58.3 | 48.3 | 65.0 | 0.011 | 1,3,3,1 |
| dtu | 50.0 | 49.4 | 50.0 | 49.6 | 50.3 | 49.2 | 50.1 | 48.9 | 50.1 | – | 48.8 | – | – | – | 49.6 | 0.005 | 0,1,6,5,3 |
| esaa | 50.0 | 50.4 | 49.9 | 50.4 | 49.9 | 49.1 | 50.0 | 50.1 | 50.0 | 49.9 | 50.5 | 40.1 | 50.3 | – | 49.8 | -0.010 | 1,2,1,5,1 |
| uhd255 | 50.1 | 48.9 | 50.1 | 48.9 | 50.4 | 49.0 | 50.9 | 49.7 | 49.8 | 48.1 | 50.1 | 50.3 | 51.2 | 47.1 | 52.5 | -0.011 | 0,1 |
| fuglsang2020 | 48.7 | 49.2 | 47.9 | 49.1 | 41.4 | 46.8 | 37.0 | 50.1 | 32.5 | 71.9 | 19.8 | – | – | – | 39.7 | -0.014 | 2,5,6,2 |
| broderick_cp | 50.1 | 49.4 | 50.0 | 48.6 | 50.3 | 49.2 | 50.2 | 50.2 | 50.8 | 40.8 | 50.5 | – | – | – | 50.6 | -0.014 | 4,0,1,4,1 |
| sparrkulee | – | 53.1 | – | 54.6 | – | 56.0 | – | 57.8 | – | 60.9 | – | 63.0 | – | 66.7 | – | 0.040 | 3,3,3,3,8 |
| megmasc | – | 62.9 | – | 69.9 | – | 80.6 | – | 89.0 | – | 95.8 | – | 97.4 | – | 99.7 | – | 0.233 | 1,1,1,1,1 |

## decaf_eeg

| dataset | 1s comp | 1s same | 2s comp | 2s same | 5s comp | 5s same | 10s comp | 10s same | 20s comp | 20s same | 30s comp | 30s same | 60s comp | 60s same | trial | test r | best ep (per fold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kuleuven | 51.3 | 51.3 | 51.4 | 50.9 | 49.1 | 48.6 | 50.2 | 50.1 | 51.0 | 51.5 | 51.2 | 50.3 | 50.3 | 51.6 | 55.3 | -0.000 | 1,3,3,1 |
| dtu | 51.0 | 51.0 | 51.1 | 51.5 | 49.7 | 50.2 | 50.0 | 50.1 | 51.0 | – | 51.2 | – | – | – | 52.5 | 0.001 | 0,1,6,5,3 |
| esaa | 51.1 | 50.9 | 51.5 | 51.1 | 49.7 | 49.4 | 49.4 | 50.1 | 49.7 | 51.6 | 48.9 | 47.4 | 52.2 | – | 50.7 | -0.000 | 1,2,1,5,1 |
| uhd255 | 51.0 | 50.9 | 50.9 | 50.8 | 49.9 | 49.5 | 50.9 | 51.0 | 51.1 | 48.9 | 53.1 | 51.8 | 52.2 | 50.4 | 52.5 | -0.002 | 0,1 |
| fuglsang2020 | 51.2 | 51.0 | 51.3 | 51.0 | 46.6 | 49.2 | 45.3 | 51.8 | 43.8 | 54.0 | 41.7 | – | – | – | 39.5 | -0.000 | 2,5,6,2 |
| broderick_cp | 50.6 | 51.1 | 50.6 | 50.7 | 50.3 | 49.6 | 50.0 | 49.2 | 52.2 | 39.3 | 50.8 | – | – | – | 49.2 | -0.001 | 4,0,1,4,1 |
| sparrkulee | – | 54.5 | – | 56.5 | – | 59.2 | – | 61.8 | – | 66.6 | – | 69.0 | – | 73.2 | – | 0.049 | 3,3,3,3,8 |
| megmasc | – | 63.2 | – | 69.9 | – | 80.6 | – | 89.6 | – | 95.9 | – | 97.8 | – | 99.7 | – | 0.246 | 1,1,1,1,1 |

## mm

| dataset | 1s comp | 1s same | 2s comp | 2s same | 5s comp | 5s same | 10s comp | 10s same | 20s comp | 20s same | 30s comp | 30s same | 60s comp | 60s same | trial | test r | best ep (per fold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kuleuven | 55.3 | 59.6 | 59.8 | 66.4 | 65.9 | 75.8 | 71.6 | 84.0 | 78.2 | 90.7 | 82.6 | 95.2 | 89.8 | 99.0 | 96.9 | – | 11,6,7,5 |
| dtu | 56.0 | 59.2 | 61.4 | 66.5 | 69.1 | 77.1 | 75.3 | 85.7 | 83.7 | – | 88.5 | – | – | – | 92.9 | – | 8,7,5,6,7 |
| esaa | 57.5 | 59.0 | 62.7 | 65.7 | 70.6 | 75.5 | 77.3 | 84.1 | 84.7 | 89.1 | 87.3 | 88.2 | 92.7 | – | 94.3 | – | 2,8,7,10,8 |
| uhd255 | 54.7 | 57.6 | 58.8 | 63.6 | 65.0 | 71.2 | 70.4 | 79.4 | 77.6 | 87.2 | 82.5 | 91.0 | 88.5 | 96.7 | 98.3 | – | 5,4 |
| fuglsang2020 | 57.9 | 63.4 | 64.3 | 73.3 | 73.0 | 84.9 | 80.2 | 92.5 | 89.0 | 97.8 | 92.7 | – | – | – | 95.4 | – | 3,5,5,9 |
| broderick_cp | 50.5 | 55.8 | 51.3 | 60.7 | 52.3 | 67.5 | 52.7 | 73.2 | 54.4 | 80.2 | 53.6 | – | – | – | 56.2 | – | 11,3,3,4,12 |
| sparrkulee | – | 62.6 | – | 71.5 | – | 83.3 | – | 91.0 | – | 96.4 | – | 98.3 | – | 99.7 | – | – | 31,24,18,20,32 |
| megmasc | – | 72.0 | – | 85.0 | – | 96.2 | – | 99.3 | – | 100.0 | – | 100.0 | – | 100.0 | – | – | 16,16,15,20,17 |

## Linear reference (linear_baseline.json)

| dataset | 1s comp | 1s same | 2s comp | 2s same | 5s comp | 5s same | 10s comp | 10s same | 20s comp | 20s same | 30s comp | 30s same | 60s comp | 60s same | trial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kuleuven | 57.0 | 59.7 | 60.0 | 63.8 | 64.9 | 69.7 | 69.9 | 76.5 | 75.1 | 82.5 | 79.2 | 86.2 | 84.2 | 93.6 | 87.8 |
| dtu | 59.1 | 60.9 | 63.9 | 66.7 | 72.4 | 74.8 | 79.0 | 83.0 | 86.1 | – | 90.0 | – | – | – | 93.9 |
| esaa | 61.3 | 62.6 | 65.6 | 67.9 | 74.1 | 75.9 | 80.3 | 83.3 | 87.6 | 90.4 | 90.2 | 91.4 | 93.7 | – | 94.7 |
| uhd255 | 57.0 | 58.6 | 60.5 | 62.2 | 65.5 | 67.7 | 70.9 | 73.3 | 76.2 | 80.4 | 81.5 | 85.6 | 87.4 | 90.7 | 98.3 |
| fuglsang2020 | 63.4 | 65.8 | 68.5 | 71.2 | 77.3 | 80.4 | 85.1 | 87.6 | 91.9 | 94.2 | 94.6 | – | – | – | 98.3 |
| broderick_cp | 56.5 | 56.8 | 59.1 | 59.3 | 63.7 | 64.2 | 68.8 | 68.7 | 75.9 | 75.3 | 78.8 | – | – | – | 84.3 |
| sparrkulee | – | 61.8 | – | 67.2 | – | 76.3 | – | 83.7 | – | 90.5 | – | 93.5 | – | 97.4 | – |
| megmasc | – | 72.7 | – | 81.3 | – | 92.0 | – | 97.3 | – | 99.4 | – | 99.8 | – | 100.0 | – |

## Notes

- Recipes: VLAAI paper recipe (5-s windows, 80 % overlap, -r, Adam 1e-3, patience 5);
  DECAF upstream recipe with 5-s present / 3.5-s past / 0.5-s delay, 10 epochs, patience 3,
  selected on the *recursive* val r (oracle uses the attended envelope, so it is not a
  decoder); `decaf_eeg` = its EEG branch alone, same checkpoint; MM = dilation model trained
  on 5-s windows vs a same-trial mismatch window, selected on val competing accuracy
  (same-talker on single-talker data). Epochs capped at 1500 steps; 95-min training budget
  (never reached).
- DECAF (recursive and EEG branch) sits at test r ~ 0 on all datasets, matching the
  earlier strict replication; its below-chance fuglsang2020 / sparrkulee cells come from a
  near-zero-r output tiled in per-window min-max-scaled 5-s chunks, not from a real signal.
- broderick_cp: all subject-independent deep models are at chance for `competing` while the
  per-listener linear decoder reaches 84 % trial accuracy.
- test r = mean whole-trial r(recon, attended) over test trials.

## Failures / missing folds

- none
