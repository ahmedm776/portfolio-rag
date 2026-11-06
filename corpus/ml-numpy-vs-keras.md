# Neural Network From Scratch vs Keras (DSAI253 Machine Learning project)

University coursework from the archive of Ahmed Khalifa. A neural
network implemented twice and compared: once from scratch in pure NumPy (forward
propagation, backpropagation, softmax cross-entropy, gradient descent all written
by hand) and once with Keras, on two datasets (Iris and MNIST).

Author: Ahmed Khalifa, DSAI253 Machine Learning.
Packaged from the preserved original notebook with AI-assistant help.
Team status: not recorded; no team or solo claim is made.

## Verified results (re-run end-to-end on Google Colab, 2026-11-04)

| Dataset | Model | Accuracy | Final loss |
|---|---|---|---|
| Iris (2-class subset) | From scratch (NumPy) | 1.0000 | 0.0032 |
| Iris | Keras | 1.0000 | 0.1143 |
| MNIST (19,700 train / 5,700 test subset) | From scratch (NumPy) | 0.9584 | 0.0122 |
| MNIST (same subset) | Keras | 0.9326 | 0.2249 |

Note: the notebook deliberately uses a reduced MNIST subset (19,700 train / 5,700
test images), not the full 60k/10k split, so these numbers are not comparable to
published full-MNIST benchmarks. Iris is used as a binary (classes 0 and 1) task
for the scratch model.

## CURRENT ADDENDA (2026-11-05)

A re-run of the same notebook procedure scored 96.51% on the full official 10,000-image MNIST test set (95.84% on the first 5,700 test images, 97.395% on the remaining tail). The original exposure of the other test images during development is not documented, so this is not an independent confirmation, and the public package ships frozen NumPy weights instantiated by repeating the deterministic notebook procedure (not proven bit-identical to the historical original weights). A matched ten-seed comparison on the same split and the 10,000-image test set gave 96.26% (sd 0.11) for the from-scratch NumPy network and 96.25% (sd 0.11) for Keras with the matched setup. Keras with an Adam recipe gave 94.78%. The gradient check covers tiny random networks only.

## Fix applied during packaging (documented, not hidden)

The preserved original notebook crashed at the MNIST section with
`NameError: name 'y_train_onehot' is not defined` - the one-hot encoding step was
missing. One line was added in front of that cell, marked with a code comment:

    y_train_onehot = np.eye(10)[y_train_mnist]

Nothing else was changed; every other cell is byte-identical to the original.
The archived verification log records the full before/after evidence.

## What is in this folder

- ml_numpy_vs_keras.ipynb - the full notebook (fixed version), runs top-to-bottom
  on Colab with no data downloads other than keras.datasets MNIST (auto).
- verification_log.md - what was run on 2026-11-04, the original failure, the fix,
  and the final metrics.
- requirements.txt - Python dependencies.

## Original preserved artifact

The original notebook is preserved privately and not redistributed.

## How to run

Open ml_numpy_vs_keras.ipynb in Google Colab, Runtime > Run all. No Drive mount,
no data files, no API keys needed. Full run takes about 2-3 minutes on a free
CPU runtime. The scratch NumPy MNIST training (100 epochs, 19,700 images) is the
slowest part.
