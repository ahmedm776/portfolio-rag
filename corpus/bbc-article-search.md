# BBC News Article Indexing and Retrieval

DSAI201 coursework from the archive of Ahmed Khalifa: an inverted index over BBC
text news (2,225 articles), plus a separate AI-assisted post-coursework
improvement that adds TF-IDF and then a hybrid semantic ranker as explicit
extensions. Personal contribution to the original coursework is unconfirmed; no
unaided-authorship or submission claim is made.

## What actually ran
Original cells 3 and 4 only, using the current downloaded public data path: cell 4 fails
with NameError (preprocess not defined) because the original calls the function before
a later cell defines it. No full original or repaired-original run is claimed. The native
original had 10 historical outputs; the portfolio copy strips them to avoid
redistributing source excerpts.
The improved final source executed on all 2,225 articles twice; postings and run report
byte-identical across reruns. 10 tests passed. CLI query also executed.

Downloaded source: https://www.kaggle.com/datasets/pariza/bbc-news-summary/data
Kagglehub handle pariza/bbc-news-summary, version 2, downloaded 2026-11-04.
Rights reference: http://mlg.ucd.ie/datasets/bbc.html . UCD states non-commercial
and research use only; BBC owns original article copyright. Kaggle CC0 metadata
conflicts with these terms and is NOT treated as permission to redistribute.
No raw articles/summaries, index/postings, trained TF-IDF state or copied excerpts
are redistributed in this archive. Runtime artifacts stay local.

## CURRENT IMPROVED VERSION (2026-11-05)

The improved search is a hybrid of TF-IDF and semantic embeddings, evaluated before/after under a frozen protocol: the known article is the first result for 135 of 150 queries, and topical-relevance precision at the first 5 results rose from 0.68 to 0.80. On the held-out sample, known-article lookup did not improve (128 versus 131 for the baseline), so the real gain is topical relevance. The relevance judgments are the authors' own, not an independent benchmark, and the topical-relevance stage had no held-out check.

## Substantive improvements
- Correct run order in separate standalone module; original retained unchanged.
- Article-only root discovery, excludes summaries. Download includes 8,900 TXT files
  due to TWO duplicate trees, each 2,225 articles + 2,225 summaries. Compare relative
  paths + SHA256 for every article in roots, fail if they disagree, select one.
- Deterministic category/filename IDs; file basenames alone collide across topics.
- All 2,225 articles used, no traversal-first-350 selection or category-label reranking.
- 2,127 unique content hashes: 98 exact-content duplicates beyond first occurrences,
  retained as distinct source IDs and reported. Search hits may repeat content;
  they are not independent evidence. No claim of a deduplicated corpus.
- Explicit encoding: 2,224 UTF8, 1 cp1252, errors do not silently yield empty text.
- Inverted postings frequencies, deterministic order and tie-breaking.
- Two selectable rankers. Search(rows) defaults to hybrid: 0.65 * TF-IDF cosine
  + 0.35 * MiniLM-L6-v2 embedding cosine (adopted 2026-11-04, eval/EXPERIMENTS.md)
  - intended for topical queries (best judged relevance). Search(rows,
  mode='tfidf') keeps the original sparse baseline runnable - intended for exact
  known-item lookup (slightly better held-out strict findability, no model
  download). Empty/OOV queries yield no hits in both modes. Hybrid first run
  downloads the ~90 MB model from Hugging Face (cached afterwards); indexing
  encodes the corpus once (~100 s CPU).
- Manifest hashes/counters, source provenance, invalid-root guards and tests.

## Results and retrieval evaluation
Categories: business 510 / entertainment 386 / politics 417 / sport 511 / tech 401.
Inverted vocabulary 27,906; TF-IDF 2,225 x 27,609 after English stopword removal.
Retrieval quality is measured in eval/RESULTS.md (added 2026-11-04): known-item
top-5 150/150 (MRR 0.954), judged topical queries P@5 0.80 / P@10 0.70, both
development-set numbers with an honest held-out known-item check reported.
Judgments are author-built (two judges, originals preserved) and shipped for
re-checking, not an independent benchmark; no personal-skill claim.
Article-derived query IDs/scores in run_report.json are diagnostics, not a scorecard.

## Reproduce
Python 3.10 verified. sentence-transformers 6.2.0 + torch 2.14.1+cpu added for the
hybrid ranker (CPU-only, no GPU needed). requirements.lock.txt is the exact tested environment (includes
some unrelated prior verification dependencies, conservative not minimal).
python -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/python test_news_index.py
.venv/bin/python news_index.py --download --output local-output
.venv/bin/python news_index.py --download --query "football world cup"
Or give local dataset root as positional argument instead of --download.
No Kaggle account/auth or paid API used. Local output may contain copyrighted terms;
do not redistribute generated postings, data or article text without clearance.

## Tests
Unique category IDs; summary exclusion; agreeing roots not doubled; disagreeing
roots rejected; missing article root; term counts; duplicate IDs rejected; known
synthetic search; deterministic manifest/index; deterministic tie order. Tests use
invented tiny text fixtures only. Whole-data execution is separate from unit tests.
Source/output/config secret scan completed before packaging, no literals flagged.
