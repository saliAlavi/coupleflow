# Locations on the cluster the paper was run on (Ohio Supercomputer Center).
# Source this before training, preprocessing or running baselines there:
#     source scripts/env/osc.sh
# Elsewhere, export the same variables (see coupleflow/paths.py) or keep the in-repo defaults.
SCRATCH=/fs/scratch/PAS2301/alialavi
export COUPLEFLOW_DATA=$SCRATCH/datasets                 # bundles + sidecars per dataset
export COUPLEFLOW_RUNS=$SCRATCH/runs/coupleflow          # <tag>/<dataset>_fold<k>/
export COUPLEFLOW_BASELINES=$SCRATCH/baselines_tmp       # results/, results_leak/, spatial_results/
export COUPLEFLOW_THIRDPARTY=$SCRATCH/baselines_tmp/thirdparty
# raw releases (read only by coupleflow/preprocessing/readers)
export COUPLEFLOW_RAW_KULEUVEN=/fs/ess/PAS2301/Data/EEG/kuleuven
export COUPLEFLOW_RAW_DTU=/fs/ess/PAS2301/Data/EEG/dtu
export COUPLEFLOW_RAW_SPARRKULEE=/fs/ess/PAS2301/Data/EEG/icassp24
export COUPLEFLOW_RAW_FUGLSANG2020=$SCRATCH/datasets/fuglsang2020/ds-eeg-snhl
export COUPLEFLOW_RAW_ESAA=$SCRATCH/datasets/esaa
export COUPLEFLOW_RAW_UHD255=$SCRATCH/datasets/uhd255
export COUPLEFLOW_RAW_BRODERICK_CP="$SCRATCH/datasets/broderick/Cocktail Party"
export COUPLEFLOW_RAW_MEGMASC=$SCRATCH/datasets/megmasc
export COUPLEFLOW_RAW_MAESTRO=$SCRATCH/maestro-eeg-dataset
export COUPLEFLOW_RAW_AVGC=$SCRATCH/datasets/avgc/avgc
# python environment
source /users/PAS2301/alialavi/miniconda3/etc/profile.d/conda.sh
conda activate nips
