# Image-caption BM25 search

A DSAI201 image-caption search implementation from the coursework archive of
Ahmed Khalifa, plus a separate AI-assisted post-coursework improvement. Ahmed's
personal contribution is unconfirmed (original coursework); no unaided
authorship or university-submission claim is made. Not computer vision:
search ranks caption TEXT, not image pixels. No external images fetched or verified.

## Provenance
Originals (34-cell notebook, 2-page report) are preserved privately and not
redistributed. The separate BBC Phase 1 is another source, NOT this project's
implementation.
Current public data: https://www.kaggle.com/datasets/thedevastator/220k-gpt4vision-image-captions
Upstream: https://huggingface.co/datasets/laion/220k-GPT4Vision-captions-from-LIVIS
Upstream card: captions are Mistral-7B-OpenOrca summaries of LVIS-Instruct4V
dialogues, not a claim all final captions were directly written by GPT4Vision.
Kaggle reports CC0; upstream metadata Apache 2.0. External COCO image rights remain
separate. No raw captions/URLs, dataset, generated index or model state is
redistributed. Included screenshots use invented synthetic captions only.

## Actual execution and limits
Original CORE cells 1,4,5,7,8,9,11,12,13,16,17,18,19,23,24,25,27,28,29,33 executed
on current downloaded v2 data, changing only the local data path. 217,868 rows / 21,277 terms /
8,078,942 tokens, 88.8 seconds in the build environment. Gradio launch, download cell and pickle
save/load omitted. This is NOT a full original run or proof of historical-file identity.

Final improved source whole-corpus runs repeat the same acceptance, matrix, result IDs
and scores except timings. 217,868 input; 0 invalid/blank captions or URLs; 1 exact
caption+URL duplicate removed; 217,867 accepted. BM25 217,867 x 277,468 stemmed unigram+bigram features.
Source CSV SHA256 d23256511d8ca679e8dd34b2df94e8186a8acb93b0bb6163055a5e6adf5d617e .
Retrieval quality is measured in eval/RESULTS.md with a frozen before/after
protocol. Current ranker (BM25+bigram): known-item full-caption lookup 150/150,
realistic partial-word recall top-1 140/150, top-5 150/150 (MRR 0.963), judged
topical queries P@1 0.9 / P@6 0.800 with named failure modes. Before this change
(TF-IDF cosine): partial-word top-1 67/150, top-5 101/150 (MRR 0.547), judged
P@1 1.0 / P@6 0.767. Judgments are author-built and shipped for re-checking, not
an independent benchmark; the new pool was re-judged under
the same rubric. Scores are BM25 scores, NOT probabilities; expansion remains
unbenchmarked. No speed-superiority claim. Current corpus build takes about 8 seconds on
the reference machine only; not comparable to the original core time, which includes
different steps. No paid APIs.

## Changes in separate improved version
- Explicit CSV path/schema checks, no nondeterministic last-CSV selection.
- Missing captions rejected rather than turning NaN into literal text; URL validation;
  sequential reject-union then exact-pair dedup, stable original row IDs.
- Cached consistent stemming across corpus, query and OPTIONAL manual expansion.
  Expansion integrated into the UI, off by default, unbenchmarked and may reduce relevance.
- Sublinear TF / smoothed IDF / L2-normalized cosine replaced the original raw TF*log(N/(df+1)).
- BM25 (k1=1.5, b=0.75, Robertson DF smoothing) over stemmed unigram+bigram features
  then replaced TF-IDF cosine after measured comparison on the frozen eval
  protocol: partial known-item top-1 67->140/150, top-5 101->150, MRR
  0.547->0.963; judged topical P@1 1.0->0.9, mean P@6 0.767->0.800 (comparable).
  Details, candidate-variant comparison, and an untouched held-out
  confirmation (disjoint 150-caption sample: TF-IDF 69/105/0.553 vs BM25+bigram
  137/146/0.939 top-1/top-5/MRR) in eval/RESULTS.md. The development 150-sample
  and 10 judged queries were used for variant selection and are labeled
  development-only. Original TF-IDF pool and judgments preserved at
  eval/judged_pool_tfidf.json / eval/judgments_tfidf.json; the new pool was
  re-judged under the same rubric (see RESULTS.md).
- Auto-escaped template, restricted HTTP(S) link scheme, no credentials in URL,
  noreferrer/noopener links, CSP prevents automatic remote images/scripts.
- HTTP source links remain labeled HTTP, NOT silently rewritten to HTTPS. Links may
  be stale/inappropriate; opening a link visits an outside site. No automatic fetch.
- Empty/OOV state, 500-character query limit, deterministic ties, no pickle loading.
  Rebuilds the sparse index from CSV on startup; no persisted trusted-model claim.

## Reproduce
Python 3.10 verified; requirements.lock.txt is the exact tested environment but includes unrelated prior
verification dependencies, not a minimal specification.
python -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/python test_search.py
.venv/bin/python download_data.py
Use the returned exact train.csv path:
.venv/bin/python image_search.py /path/to/train.csv --output local-output --serve
Open http://127.0.0.1:8052 . Local-only, no deployment/tunnel.
Synthetic demonstration without the dataset:
.venv/bin/python image_search.py synthetic_fixture.csv --output demo-output --serve --demo

## Verified tests and pixels
11 unit tests: ranking, empty, OOV, integrated expansion, matching stems, bad URL,
HTML escape/CSP, missing-caption/duplicate accounting, missing column, query limit,
stable ties. Actual browser: synthetic search/empty/expansion and a full-corpus query
"kitchen cabinet" -> IMG_204406 first (cosine .723995), five results. Desktop and 390px
mobile inspected; no overflow, mobile width 390px. All browser requests stayed local,
no outside images fetched. Synthetic screenshots included, not original data output.

Intermediate fix: a first strict HTTPS-only data run accepted 0 because source URLs use
HTTP. Allowed labeled HTTP(S), no rewriting, reran the final source and tests. A UI fixture
run overwrote a local diagnostic output folder; corrected by rerunning the full corpus
into a distinct final-full-output and comparing rerun results. The delivered aggregate
report comes only from the final full-corpus run. No silent mixing of fixture metrics.

Secret-pattern scan and source/output/config review done before packaging.
Raw baseline logs containing source caption excerpts intentionally NOT included.
