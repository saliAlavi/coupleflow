VLAAI upstream
==============

Original code: https://github.com/exporl/vlaai (GPLv3).
Commit: see `vlaai/GIT_COMMIT`.

Vendored: `vlaai/model.py` (the official TensorFlow/Keras graph),
`vlaai/tests/`, the upstream README, LICENSE and requirements.  The 200
`evaluation_datasets/DTU/*.npz` example files and the pre-trained
SavedModel/HDF5/ONNX blobs are **not** vendored (size); the parameter count of
`pretrained_models/vlaai.h5` was read once to pin down the weight-sharing
question and is recorded in `../models_upstream.py`.

TensorFlow is not installed in this environment, so `../models_upstream.py`
holds a PyTorch port of `model.py`.  It is layer-for-layer identical, including
the three things a naive rewrite gets wrong:

1. `vlaai()` builds **one** `extractor` and **one** `output_context` Keras
   `Model` and calls each four times, so those weights are *shared* across
   blocks (1 740 289 parameters — confirmed against `vlaai.h5`, which contains
   five extractor convolutions and one output-context convolution).  Only the
   `Dense(64)` is per-block.
2. `tf.keras.layers.LayerNormalization()` normalises over the **channel** axis
   at each time step and defaults to `epsilon=1e-3`.
3. `tf.keras.layers.LeakyReLU()` defaults to `alpha=0.3`.

`../test_cpu.py` asserts (1) and the padding directions implied by the Keras
`ZeroPadding1D` calls.

Reference: B. Accou, J. Vanthornhout, H. Van hamme, T. Francart, "Decoding of
the speech envelope from EEG using the VLAAI deep neural network",
*Scientific Reports* 13, 812 (2023).
