# Arabic Hotel Review Sentiment (HARD dataset)

Binary sentiment classification (positive / negative) of Arabic hotel reviews,
using the HARD balanced dataset (Elnagar, Khalifa & Einea, 2018).

## Source and license
- Data: https://github.com/elnagara/HARD-Arabic-Dataset (`data/balanced-reviews.zip`),
  collected from Booking.com in June/July 2016. Modern Standard + dialectal Arabic.
- The source repository ships **no explicit LICENSE file**; the authors distribute it for
  research use with a citation request. This project is educational portfolio work
  and cites: Elnagar A., Khalifa Y.S., Einea A. (2018), Hotel Arabic-Reviews Dataset
  Construction for Sentiment Analysis Applications, Springer SCI vol 740, pp 35-52,
  doi:10.1007/978-3-319-67056-0_3.
- Raw reviews are not redistributed; the dataset is downloaded from the public
  source repository above by the project's fetch script.

## Evaluation protocol (frozen 2026-11-04 before any model fitting)
1. Labels: rating 1-2 -> negative (0), rating 4-5 -> positive (1). Ratings of 3 do
   not occur in this balanced subset.
2. Light normalization only: remove Arabic diacritics and tatweel, unify alef
   variants and yaa forms. No stemming, no external lexicons.
3. Split: **group split by hotel** (GroupShuffleSplit, test 20%, seed 42) - no hotel
   appears in both train and test, so the evaluation measures generalization to
   hotels the model never saw, not memorized property-specific wording.
4. The TF-IDF vectorizer is fit on the training reviews only.
5. Models compared on the untouched test set exactly once: majority-class dummy,
   MultinomialNB, and logistic regression. No post-hoc tuning after viewing test
   results. All metrics reported as-is, including any baseline we fail to beat.
6. Limitations stated plainly: reviews are from 2016 Booking.com hotels; the label
   is derived from star ratings; single frozen split, no cross-validation claim.

## CURRENT RESULTS (2026-11-05)

A fresh full training run on 22,396 reviews after removing duplicates reached 94.6% accuracy, F1 0.948 and AUC 0.983, with 5 tests passing. The number is measured on the same test set used during development (benchmark reuse, not a new independent confirmation).

## Layout
- `src/prepare.py` - load, normalize, label, group-split; writes data/train.pkl, data/test.pkl
- `src/train.py` - fit vectorizer + models, evaluate once, write metrics.json + model.joblib
- `src/app.py` - FastAPI serving the saved pipeline (Arabic input)
- `tests/` - pytest suite (split integrity, label mapping, train-only fitting, prediction sanity, API)
