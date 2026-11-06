# Ask my projects - a document RAG over the portfolio READMEs

A small retrieval-augmented assistant that answers questions about Ahmed's
portfolio projects using **only** the project documents in `corpus/`. Every
answer names the chunks it was grounded on. Questions outside the documents'
scope get an explicit "not covered" answer when no project document is relevant;
when a project's document is relevant but does not carry the asked detail, the
answer quotes that document's closest verbatim sentences (never another project's
numbers) rather than inventing an answer.

The public corpus covers the eight published project repositories. Private,
partial, or unpublished project notes are deliberately excluded.

## What it does

1. **Ingest** (`src/ingest.py`): splits each project document into one chunk per
   markdown section (max 900 chars, split on paragraphs), carrying the
   project title into every chunk. 8 documents -> 57 chunks.
2. **Retrieve** (`src/retriever.py`): TF-IDF over word 1-2 grams plus
   character 3-5 grams (char grams catch partial technical terms like
   "GTSRB" or "TF-IDF"), cosine similarity, top-3 chunks. English stop words
   are stripped before vectorization: short out-of-corpus queries made mostly
   of stop words ("what is the capital of france") otherwise matched short
   chunks on "what is the" word/char grams and could beat real in-corpus
   questions (regression found during the 2026-11-06 public-corpus curation).
3. **Abstain**: if the best cosine score is below 0.15, the question is
   judged out-of-corpus and the answer is a fixed honest refusal. The LLM
   is never called on abstained questions, so it cannot fill gaps from
   its own priors. Caveat: the threshold is fitted on the same test
   questions that verify it - `tests/test_rag.py::test_threshold_margin`
   is a development sanity check on those probes, not independent
   retrieval-quality evidence.
4. **Answer** (`src/extractive.py`): the reply is built from **verbatim
   sentences** of the retrieved chunks - sentences (and metric table rows,
   rendered as readable lines) are ranked by TF-IDF cosine against the
   question and the best 1-2 are quoted across different chunks where
   possible. Numeric questions ("what accuracy...") get a boost toward
   sentences carrying numbers. Nothing is paraphrased and every citation
   is exact by construction - but note what this does NOT guarantee: a
   verbatim quote faithfully repeats whatever the corpus says, including
   stale or under-qualified wording. (This happened in practice: a
   superseded "official test set" sentence in the traffic-sign document was
   quoted until the corpus was reconciled and regression tests were added
   to pin the corrected mirror-provenance wording.) Corpus freshness is a
   maintenance requirement, and `tests/test_rag.py` carries regression
   coverage for the qualifier wording that must survive.
5. **Serve** (`src/app.py`): FastAPI app with a minimal chat UI at `/`,
   `GET /api/docs` (indexed documents) and `POST /api/ask`
   (answer + citations + retrieved chunks with scores).

## Status: extractive document-QA/search assistant

This package is a working **extractive** document-QA/search assistant over
the portfolio documents. It is not a completed generative RAG project.

## The LLM experiment, honestly

A generative compose step was built and evaluated - kept clearly separate
from the default path in `src/generator.py`, behind `{"llm": true}` in
`/api/ask`, never loaded unless requested: `google/flan-t5-small` (80M) and
`flan-t5-base` (250M) on CPU. Both produced fragment answers
("Accuracy", "0.5 threshold") and blended unrelated cross-project chunks
into misleading sentences. At this model size, generation was less
trustworthy than extraction, so the default path quotes the documents
verbatim. The LLM path remains in the code for experimentation with this
limit documented; model weights are not in this package (downloaded from
the Hugging Face hub on first use, ~900 MB).

## Held-out probe evaluation (historical, 13-document internal corpus)

An earlier internal revision of this project ran over 13 project documents
(including private, partial and unpublished notes that the public corpus now
excludes). On that corpus, a held-out probe set (24 in-corpus probes with
fresh phrasings written after the abstain threshold was frozen, plus 10
out-of-corpus probes) measured: raw top-1 23/24 = 0.96, raw top-3 24/24 = 1.00;
answer coverage 20/24 = 0.83 (4 over-abstentions, 0 wrong-document answers);
out-of-corpus abstention 10/10. Limits stated then still apply: the probes were
hand-authored by the same builder over a corpus it already knew, and the set
later became development-exposed, so those numbers are development evidence on
fresh phrasings, not independent external validation. The eval harness and
probe sets are development material and are not included in this public
repository.

## Honest limits

- The corpus is 8 project documents. This is a document assistant over those
  files, not a general assistant and not a claim about any project's quality;
  what a document does not say, this cannot answer.
- Answers are verbatim quotes: they cannot lie, but they can be the
  *wrong* quote - retrieval picks the most question-like sentences, not
  necessarily the best answer. Read the cited chunk, which is always
  returned, before trusting a wording.
- The abstention threshold is calibrated on the test question sets in
  `tests/test_rag.py`; a differently-phrased borderline question can land
  on either side. Cross-document questions are not this retriever's strength.
- Percentages and metrics quoted from documents are the projects' reported
  numbers with the qualifiers those documents carry (e.g. benchmark reuse,
  mirror-dataset provenance). This app repeats documents; it does not
  re-verify them.

## Run it

```bash
pip install -r requirements.txt
python src/ingest.py          # rebuild the chunk index from corpus/
python -m pytest tests/ -q    # 32 tests: routing, abstention, margin, citation, regressions
uvicorn src.app:app --port 8000
# open http://localhost:8000
```

## Verified

- `python -m pytest tests/ -q`: **32 passed** (in-corpus routing, out-of-corpus
  abstention, threshold separation, verbatim-and-cited answer contract,
  abstained-answer text, scope guard, comparison answers, plus regression
  coverage: GTSRB mirror/subset/overlap qualifier wording in both answers and
  corpus, food-ML limitations retrievable, coursework provenance labels,
  no wrong-project citations, no cross-project metric leaks).
- App smoke-tested: `/` 200, `/api/docs` lists the indexed documents,
  `/api/ask` answers with citations and abstains out-of-corpus.

## Development history (v2.5, 2026-11-06)

Project-aware routing, scope guard, current-over-historical selection,
wrapped-bullet fix, two-project comparison answers (juxtaposition only - no
comparative judgment is ever synthesized, sentences are never joined across
documents, and authorship/contribution questions keep the single-doc path),
comparison answers restricted to current-status sentences, corpus currency
corrections, and concurrency hardening (the comparison current-status filter
is a plain local parameter, not a mutable instance attribute). Probe results
quoted during development were development-exposed benchmark reuse, not
independent validation.

**Rollback.** Earlier revisions are preserved privately; each package is
self-contained with no shared state.
