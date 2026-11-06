# Traffic-Sign Recognition on GTSRB (real-world computer vision)

A compact convolutional neural network trained **from scratch** to classify
real-world German traffic signs into 43 classes, evaluated on the
**GTSRB test set (12,630 images; decoded pixel and label identity with the
official test set was verified on 2026-11-05, see CURRENT STATUS)** plus the mirror's motion-blur corruption variant.
Includes a small FastAPI demo app and a pytest suite.

## CURRENT STATUS (2026-11-05, supersedes the mirror caveats below where they differ)

The official GTSRB archives were obtained (via the Wayback Machine, with independently pinned hashes) and the 12,630-image test set and 26,640-image training subset were decoded and checked: pixel and label identity with the mirror is verified. The shipped prepare script and model reproduce 98.8915% accuracy on the verified official test set (98.89%), with images resized to 32x32 and training on a 26,640-image subset of the official 39,209 training images. There are 0 exact duplicates between the original and the 32x32 images. The earlier one-duplicate audit result was not reproduced. Motion-blur provenance is still not officially verified. Training used a subset (26,640 of the official 39,209 training images), not the full set.

## Results (actual executed run, 4 Nov 2026; historical wording follows)

| Evaluation | Images | Accuracy |
|---|---|---|
| Mirror test split (same 12,630 count as the official GTSRB test set) | 12,630 | **98.89%** |
| Same test split, excluding the 1 train/test duplicate | 12,629 | **98.89%** |
| Mirror's motion_blur corruption of that test split | 12,630 | **98.21%** |

- 24 training epochs, ~10 min total on 2 CPU cores (no GPU, no paid compute).
- Saved-model reload verified deterministic (identical predictions).
- Model selection: fixed 24-epoch schedule chosen before any test use; final
  epoch kept (no best-checkpoint selection, no early stopping); the official-size
  test split was evaluated exactly once, after training. Percentages shown by the
  demo app are softmax confidence scores, not calibrated correctness
  probabilities.
- Weakest classes reported in `metrics.json` (e.g. "snow/ice warning" 82.0%,
  "empty red circle" 91.9%) - see `confusion_matrix.png` and `misclassified.png`.
- 6/6 pytest tests pass: class map, forward shape, reload determinism,
  metrics sanity, live app endpoints incl. bad-input handling, real-image sanity.

## Data provenance and honest limits

- Source: GTSRB benchmark (Stallkamp et al., 2012). The original host
  (benchmark.ini.rub.de) was unreachable from the build environment on
  2026-11-04, so the Hugging Face mirror `tanganke/gtsrb` was used.
  Downloaded files were verified by SHA-256 against the mirror's
  content-addressed metadata (all matched); the test split has the official
  12,630-image count. HISTORICAL (2026-11-04, superseded by the CURRENT STATUS
  above): at that time mirror-vs-original byte identity was not claimed.
- The mirror's train split is a 26,640-image subset of the official 39,209-image
  training set; training used 90% of it (23,976) with 2,664 for validation.
  This is disclosed, not the full official training set. HISTORICAL (2026-11-04,
  superseded): the mirror documents no split derivation, so the test split was then
  reported only as "same count as the official GTSRB test set". The CURRENT STATUS
  supersedes this: the test split was later verified identical to the official one
  (decoded pixels and labels). The byte-level identity of the mirror's files was never
  claimed; official-test identity was.
- Overlap audit (per-image SHA-1): internal validation vs test = 0 exact
  duplicates. HISTORICAL (2026-11-04 audit, not
  reproduced on 2026-11-05 - see CURRENT STATUS): mirror-train vs test showed
  exactly 1 duplicate image (test index 0, class "20 kph speed limit");
  accuracy excluding it was then unchanged at 98.89% (98.8914% vs 98.8915%).
- The motion_blur split is the mirror's precomputed corruption of the same
  12,630 test images (labels identical, per-index image correlation 0.95-0.99);
  exact official corruption provenance is not proven.
- Images were resized to 32x32. Class names come from the mirror's dataset card.
- The motion-blur variant is the mirror's precomputed corruption of the same
  test images (GTSRB-C style), used for robustness evaluation only.
- **No dataset images are redistributed in this package** (`data/` is git-ignored
  and excluded). GTSRB usage rights remain with its original publishers;
  this package is code, metrics, and a trained model for portfolio review.
- This is recognition of cropped sign images, not detection in full road scenes,
  and not a safety-critical system.

## Layout

- `src/model.py` - CompactCNN (~0.67M parameters, 32x32x3 input).
- `src/train.py` - time-budgeted, checkpoint-resumable training
  (random affine augmentation: rotation +/-10 deg, translation +/-10%, scale 0.9-1.1;
  SGD+nesterov, cosine LR, weight decay 5e-4).
- `src/evaluate.py` - full test + corruption evaluation, confusion matrix,
  misclassified grid, reload check.
- `src/prepare_data.py` - parquet -> 32x32 arrays (requires the dataset; not bundled).
- `src/app.py` - FastAPI demo: upload form, `/api/predict` JSON, `/sample?idx=N`
  live prediction on a held-out test image.
- `src/demo_grid.py` - `demo_predictions.png` (24 real test predictions).
- `tests/` - pytest suite (6 tests).
- `final_model.pt` - the trained model the metrics above come from.
- `metrics.json` - full metrics incl. per-epoch validation history.

## Reproduce

    pip install -r requirements.txt
    # download tanganke/gtsrb parquet files into data/, then:
    python3 src/prepare_data.py train && python3 src/prepare_data.py test
    python3 src/prepare_data.py motion_blur
    python3 src/train.py --seconds 100 --max-epochs 24   # repeat until done=true
    python3 src/evaluate.py
    python3 -m pytest tests/ -q
    python3 -m uvicorn app:app --app-dir src --port 8000
