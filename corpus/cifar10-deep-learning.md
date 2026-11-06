# CIFAR-10 Deep Learning (DSAI308 Assignment 1)

University coursework from the archive of Ahmed Khalifa. Four models
trained and compared on CIFAR-10: a baseline CNN, a manually built ResNet, a manually
built Inception-style network, and VGG16 transfer learning.

Author: Ahmed Khalifa, DSAI308 Deep Learning course project.
Packaged from the preserved original submission with AI-assistant help.
Team status: not recorded; no team or solo claim is made.

## CURRENT RESULTS: official-split retrain (updated 2026-11-06, current headline)

The four models were retrained from scratch on the official CIFAR-10 split: 50,000 official training images (with a separate 5,000-image validation set) and the official 10,000-image test set. The protocol was locked before the test numbers were seen and no model was selected on test results. Current best: the manual ResNet reaches **89.51% test accuracy / 0.8947 macro-F1**, validation-selected from a 4-epoch continuation of the 20-epoch model with full optimizer state carried (benchmark reuse on the development-exposed official test set). The other official-test accuracies: VGG16 transfer learning 86.01%, baseline CNN 77.66%, manual Inception 67.24%. An earlier model trained partly on corrupted data was excluded and retrained cleanly. Two of the four models stay below 80%.

Superseded headline (kept for provenance): an earlier 2026-11-06 warm-start epoch-1 candidate scored 88.48% (20-epoch baseline: 87.55%); the CNN 77.66% is the validation-selected checkpoint after restoring the frozen selection protocol (last-epoch weights: 75.03%); the Inception 67.24% is a warm start bounded by a 600-second fit cap (last-epoch: 65.42%). VGG16 was not refit.

## HISTORICAL results: custom split, reloaded saved models (2026-11-04; superseded as the headline by the official-split retrain and its 2026-11-06 updates)

| Model | Architecture | Test accuracy |
|---|---|---|
| 1 | Baseline CNN | 0.7812 |
| 2 | ResNet (manual blocks) | 0.8520 |
| 3 | Inception (manual blocks) | 0.6693 |
| 4 | VGG16 transfer learning | 0.8652 |

All four reproduced EXACTLY (to 4 decimals) on the assignment's held-out test split,
with the method and full output recorded in the archived verification log.

Important (historical custom split): the assignment uses a custom stratified 48,000/6,000/6,000
train/val/test split (seed 42) over all 60,000 CIFAR-10 images, 600 test images per
class. These numbers are NOT the standard 10,000-image official-test benchmark and
must not be quoted as equivalent to it.

## Public repository

The public repository (`cifar10-resnet-classification`) ships the current-best
ResNet as an inference package: `model_best.keras`, `inference.py` with a
`--selftest` that verifies the copy end-to-end on 64 official-test samples,
plus `metrics.json` and `provenance.md` with the numbers, selection rule,
caveats and hashes. The full training project (all four models, notebooks,
report) is archived privately.

## Provenance

Original submission artifacts (notebook with outputs, 12-page report PDF, trained
models, exact split indices, training histories) are preserved privately and are
not redistributed. Dataset: CIFAR-10 via the Hugging Face mirror
`uoft-cs/cifar10`; no dataset files are redistributed.
