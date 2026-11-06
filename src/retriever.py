"""TF-IDF retrieval over portfolio chunks.

Word unigrams+bigrams and character 3-5 grams are combined: word features
match vocabulary, character features catch partial terms (GTSRB, TF-IDF,
flan). The abstention threshold is calibrated by tests/test_rag.py.
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN = re.compile(r"[A-Za-z0-9%.\-/]+")

def _strip_stops(text: str) -> str:
    """Remove English stop words before vectorization (both channels).

    Short out-of-corpus queries made mostly of stop words ("what is the
    capital of france") otherwise matched short chunks on "what is the"
    word/char grams and beat real in-corpus questions (regression found
    during the 2026-11-06 public-corpus curation)."""
    return " ".join(w for w in _TOKEN.findall(text) if w.lower() not in ENGLISH_STOP_WORDS)

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "index" / "chunks.jsonl"
ABSTAIN_THRESHOLD = 0.15  # cosine on the combined sparse space; tests pin this

class Retriever:
    def __init__(self):
        self.chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
        texts = [_strip_stops(c["text"]) for c in self.chunks]
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.W = sparse.hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def search(self, query: str, k: int = 3):
        query = _strip_stops(query)
        q = sparse.hstack([self.word.transform([query]), self.char.transform([query])])
        sims = cosine_similarity(q, self.W)[0]
        order = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.chunks[i]) for i in order]

    def retrieve(self, query: str, k: int = 3):
        """Return (hits, abstained). Abstains when the best match is weak."""
        hits = self.search(query, k)
        return hits, (not hits or hits[0][0] < ABSTAIN_THRESHOLD)

if __name__ == "__main__":
    import sys
    r = Retriever()
    for score, ch in r.search(" ".join(sys.argv[1:]) or "food prices"):
        print(f"{score:.3f}  {ch['chunk_id']}  {ch['section']}")
