"""Extractive answering: verbatim sentences from the retrieved chunks.

The answer is built from the chunks' own words - sentences are ranked by
TF-IDF cosine against the question and the best 1-2 are quoted. Nothing is
paraphrased, so the answer can never invent a fact; the citation is exact.

Why not generation: google/flan-t5-small and -base were evaluated for the
compose step (see src/generator.py, kept behind --llm). At 80M/250M params
on CPU they produced fragment answers and blended unrelated chunks, so the
default path is extractive. The LLM path remains for experimentation with
its limits documented in the README.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ABSTAIN_ANSWER = (
    "I don't have that in the portfolio documents. The indexed READMEs "
    "cover only the projects in the corpus; try asking about one of them."
)

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

def sentences(text: str):
    out = []
    lines = text.split("\n")[1:]  # drop synthetic "title - section" header
    paras, buf = [], []
    def flush():
        nonlocal buf
        if buf:
            paras.append(" ".join(buf)); buf = []
    for line in lines:
        ls = line.strip()
        if not ls:
            flush(); continue
        if ls.count("|") >= 2 or ls.startswith(("#", "!", "```")):
            flush(); continue  # tables, headers, images, fences
        if ls.startswith("- ") or re.match(r"\d+[.)] ", ls):
            flush()
            buf.append(re.sub(r"^(- |\d+[.)] )", "", ls).replace("`", ""))  # wrapped bullet lines continue this item
            continue
        buf.append(ls)
    flush()
    # table rows carry the metrics; turn each row into a readable sentence
    table, header = [], None
    for line in lines + [""]:
        ls = line.strip()
        if ls.startswith("|") and ls.count("|") >= 2:
            cells = [c.strip().replace("**", "") for c in ls.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # separator row
            if header is None:
                header = cells
            else:
                table.append(cells)
        else:
            for row in table:
                bits = [f"{h}: {v}" for h, v in zip(header[1:], row[1:]) if v and v != "-"]
                sent = f"{row[0]} - " + "; ".join(bits) + "." if bits else row[0] + "."
                if 25 <= len(sent) <= 400:
                    out.append(sent)
            table, header = [], None
    for para in paras:
        para = para.replace("**", "")
        for s in _SENT.split(para):
            s = s.strip()
            if 25 <= len(s) <= 600 and not s.startswith("http"):
                low = s.lower()
                if low.startswith(("this block is", "the text below is", "the text above is", "remaining sections")):
                    continue  # overlay meta-commentary, not project content
                out.append(s)
    return out

def answer(question: str, hits, abstained: bool):
    if abstained:
        return {"answer": ABSTAIN_ANSWER, "citations": [], "abstained": True}
    top = hits[0][0]
    hits = [(s_, c_) for s_, c_ in hits if s_ >= 0.6 * top] or hits[:1]
    # Project-specific questions must quote the leading document: a question
    # about project X must not be answered from project Y's metric table just
    # because Y's table is lexically richer (regression: after a corpus
    # refresh, "what accuracy did the traffic sign model reach" quoted the
    # food project's accuracy row). Pool from the top-ranked document first;
    # fall back to all retrieved chunks only if it has nothing relevant.
    top_doc = hits[0][1]["chunk_id"].split("#")[0]
    pool = []  # (sentence, chunk_id)
    for _, ch in hits:
        if ch["chunk_id"].split("#")[0] == top_doc:
            for s in sentences(ch["text"]):
                pool.append((s, ch["chunk_id"]))
    pool_all = []  # (sentence, chunk_id)
    for _, ch in hits:
        for s in sentences(ch["text"]):
            pool_all.append((s, ch["chunk_id"]))
    if not pool:
        body = hits[0][1]["text"][:300]
        return {"answer": body, "citations": [hits[0][1]["chunk_id"]], "abstained": False}
    def rank(pool_):
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        M = vec.fit_transform([question] + [s for s, _ in pool_])
        return cosine_similarity(M[0:1], M[1:])[0]
    sims = rank(pool)
    if sims.max() <= 0 and pool_all:
        # The leading document has nothing lexically relevant. Quoting another
        # project's chunk here would attribute a foreign fact to the asked
        # project (e.g. a metric the asked project's README does not carry),
        # so instead of silently falling back across documents, say plainly
        # that the leading document does not cover the question.
        return {"answer": (f"The {top_doc} README does not cover that. "
                           "Try rephrasing, or ask about a different project."),
                "citations": [], "abstained": True}
    if re.search(r"accuracy|score|how (many|much)|what.*(result|metric|auc|percent|accuracy|rate)", question, re.I):
        # numeric questions: prefer sentences that actually carry a number
        sims = sims + 0.15 * np.array([1 if re.search(r"[0-9%]", s) else 0 for s, _ in pool])
    order = np.argsort(-sims)
    picked, seen_chunks = [], set()
    for i in order:
        s, cid = pool[i]
        if sims[i] <= 0:
            break
        if cid in seen_chunks and len(picked) >= 1:
            continue  # prefer coverage across chunks
        picked.append((s, cid)); seen_chunks.add(cid)
        if len(picked) == 2:
            break
    if not picked:
        picked = [(pool[0][0], pool[0][1])]
    text = " ".join(f"{s}" for s, _ in picked)
    citations = sorted({cid for _, cid in picked})
    return {"answer": text, "citations": citations, "abstained": False}
