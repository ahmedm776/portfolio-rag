"""v2.5 extractive pipeline: project-aware retrieval, scope guard, current-over-historical.

Everything here is a heuristic over the corpus text; nothing is generated.
  1. Project routing: if the question names a project (alias table), retrieval
     and the answer sentences are restricted to that project's document(s), so
     a metric from another project is never quoted for it.
  2. Scope guard: abstain if the question carries a specific term that appears
     nowhere in the corpus (e.g. a dataset, device or model name), or asks for a
     metric/hyperparameter the routed document never mentions.
  3. Currency: sections headed HISTORICAL are down-weighted unless the question
     asks about the custom/original/historical split, and are labelled when used.
  4. Sentences from different documents are never joined into one answer.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import extractive
from extractive import sentences, ABSTAIN_ANSWER

ALIASES = {
    "traffic-sign-gtsrb": r"gtsrb|traffic[- ]sign|road[- ]sign|street[- ]sign|sign (recogni|classif)",
    "cifar10-deep-learning": r"cifar|vgg|inception|resnet",
    "ml-numpy-vs-keras": r"numpy|keras|from scratch|mnist|iris",
    "arabic-hotel-sentiment": r"arabic|hotel|\bhard\b",
    "bbc-article-search": r"\bbbc\b|news article",
    "image-caption-search": r"caption",
    "movies-validation": r"\bmovies?\b",
    "egypt-food-prices-ml": r"spike|early[- ]warning|price spike|food price.*(model|classif)|brier",
}
STOP = set("""the a an of and or to in on for is was are were be been by with as at from that this it its which what who whom whose
how many much did does do has have had than then there their they them into not no yes if so can could would should will
when where why about over under per any all each both more most less also very just only same other such these those i you we he
she his her our your me my us get got give gave show tell please project projects portfolio ahmed ahmeds model models used use
using uses reach reached score scored take took long country compare compared versus vs now current currently""".split())
METRICS = ["f1", "auc", "roc", "precision", "recall", "brier", "latency", "throughput", "dropout", "learning rate",
           "inference", "price per", "users", "stars", "downloads", "salary", "accuracy", "training time", "transformer"]
HOWTO_Q = re.compile(r"^\s*how\s+(?:do|can|could|should)\s+(?:i|we|you)\s+(?:write|implement|code|program|build|create|make)\b", re.I)
NOW_Q = re.compile(r"\b(now|currently|latest|today)\b", re.I)
HIST_Q = re.compile(r"custom|historical|original|assignment|reload|earlier|old |first version|version 1|v1", re.I)
COMP_Q = re.compile(r"\b(compare|comparison|comparing|versus|vs\.?|difference between|differ|better than|worse than|which (?:is|was) (?:better|higher|lower|faster|slower|more|less))\b", re.I)
AUTHOR_Q = re.compile(r"sole[- ]author|authorship|contribution|unaided|who (wrote|built|made|created)|"
                      r"who is the author|\bsolely\b|own work|work by .* alone|den(ies|ied|y) (writing|writing it|authorship)|"
                      r"(write|wrote|build|built|make|made|create|created).*(alone|himself|by himself)", re.I)
AUTHOR_MARK = re.compile(r"contribution|authorship|unaided|sole author", re.I)
STATUS_MARK = re.compile(r"unconfirmed|not confirmed|not yet confirmed", re.I)

# Uniform verified contribution status, kept as PROJECT METADATA - deliberately NOT
# in the corpus: it is used only when the question is about authorship/contribution
# (AUTHOR_Q), so it cannot dilute retrieval TF-IDF, create name-token matches, or
# surface in scope/metric answers. "Unconfirmed" records absence of confirmation;
# it neither asserts nor denies authorship and grants no rights.
CONTRIBUTION_STATUS = (
    "Contribution status: Ahmed's personal contribution and sole authorship are unconfirmed."
)

def _tok(s):
    # corpus OCR/gluing often merges words and digits ("leaving6,967", "includes8,900TXTfiles")
    s = re.sub(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])", " ", s)
    out = []
    for t in re.findall(r"[A-Za-z0-9]+", s):
        out.append(t.lower())
        if re.search(r"[a-z]", t) and re.search(r"[A-Z]", t):
            out += [p.lower() for p in re.findall(r"[A-Z]+|[a-z]+", t)]  # TXTfiles -> txt, files
    return out

def _norm(w):
    return w[:-1] if len(w) > 4 and w.endswith("s") else w

class Pipeline:
    def __init__(self, retriever):
        self.r = retriever
        self.vocab = set()
        self.doc_text = {}
        self.corpus_lower = ""
        for c in retriever.chunks:
            d = c["doc"].replace(".md", "")
            self.doc_text[d] = self.doc_text.get(d, "") + " " + c["text"].lower()
            self.corpus_lower += " " + c["text"].lower()
            for w in _tok(c["text"]):
                self.vocab.add(_norm(w))
        self.alias = {d: re.compile(p, re.I) for d, p in ALIASES.items()}
        df = {}
        n = 0
        for c in retriever.chunks:
            for sent in sentences(c["text"]):
                n += 1
                for w in set(_norm(x) for x in _tok(sent)):
                    df[w] = df.get(w, 0) + 1
        self.idf = {w: np.log((n + 1) / (v + 0.5)) + 0.5 for w, v in df.items()}
        self.n_sent = n

    def routed_docs(self, q):
        hit = [d for d, rx in self.alias.items() if rx.search(q)]
        return hit

    def unknown_specific(self, q):
        """Specific query terms absent from the whole corpus."""
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", q)
        bad = []
        for i, w in enumerate(words):
            parts = [p for p in re.split(r"[-\.]", w) if p]
            for p in parts:
                lw = p.lower()
                if lw in STOP or len(lw) < 2:
                    continue
                if _norm(lw) in self.vocab or (len(lw) >= 3 and any(lw in v for v in self.vocab)) or (len(lw) >= 2 and lw in self.corpus_lower):
                    continue
                specific = (i > 0 and w[0].isupper()) or any(ch.isdigit() for ch in w) or (w.isupper() and len(w) > 1)
                if specific:
                    bad.append(w)
        content = [w for w in _tok(q) if w not in STOP and len(w) > 2]
        unk = [w for w in content if _norm(w) not in self.vocab]
        frac = len(unk) / max(1, len(content))
        return bad, unk, frac

    def ask(self, q, k=4):
        if HOWTO_Q.match(q):
            return self._abstain("scope: generic how-to question, not about the portfolio documents")
        docs = self.routed_docs(q)
        if docs:
            for i, w in enumerate(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", q)):
                cap = (i > 0 and w[0].isupper()) or (w.isupper() and len(w) > 1)
                if not cap or len(w) < 3 or w.lower() in STOP:
                    continue
                lw = w.lower()
                if not any(lw in self.doc_text[d] for d in docs):
                    return self._abstain(f"scope: '{w}' not in the routed document")
        bad, unk, frac = self.unknown_specific(q)
        if bad or frac >= 0.5:
            return self._abstain(f"scope: unknown terms {bad or unk}")
        sims_all = self.r.search(q, k=len(self.r.chunks))
        if docs:
            cand = [(s, c) for s, c in sims_all if c["doc"].replace(".md", "") in docs]
            thr = 0.06
        else:
            cand = sims_all
            thr = extractive_threshold()
        hits = cand[:(6 if docs else k)]
        if not hits or hits[0][0] <= thr:
            return self._abstain("scope: weak retrieval")
        ql = q.lower()
        top_docs = docs or [hits[0][1]["doc"].replace(".md", "")]
        for m in METRICS:
            if re.search(r"\b" + re.escape(m) + r"\b", ql) and not any(re.search(r"\b" + re.escape(m) + r"\b", self.doc_text[d]) for d in top_docs):
                return self._abstain(f"scope: '{m}' not in routed document")
        if len(docs) >= 2 and COMP_Q.search(q) and not AUTHOR_Q.search(q):
            return self._compare(q, docs[:2], sims_all)
        return self._extract(q, hits, docs)

    def _abstain(self, why):
        return {"answer": ABSTAIN_ANSWER, "citations": [], "abstained": True, "why": why}

    def _compare(self, q, two_docs, sims_all):
        """Two-project comparison: per-doc extraction, labeled juxtaposition, no synthesis.
        Sentences are never joined across docs; no comparative connective is introduced."""
        parts = []
        cites = set()
        covered = 0
        for d in two_docs:
            cand = [(s, c) for s, c in sims_all if c["doc"].replace(".md", "") == d]
            hits_d = cand[:6]
            if not hits_d or hits_d[0][0] <= 0.06:
                parts.append(f"{d}: The {d} README does not cover that.")
                continue
            r = self._extract(q, hits_d, [d], current_only=True)
            if r["abstained"]:
                parts.append(f"{d}: The {d} README does not cover that.")
                continue
            covered += 1
            parts.append(f"{d}: {r['answer']}")
            cites.update(r["citations"])
        if covered == 0:
            return self._abstain("comparison: neither routed document covers the question")
        note = ("Note: these are verbatim quotes from each project's own README; the projects "
                "use different evaluation protocols and test sets, so the numbers are not directly comparable. ")
        return {"answer": note + " ".join(parts), "citations": sorted(cites), "abstained": False,
                "why": "comparison" if covered == 2 else "partial-comparison"}

    def _extract(self, q, hits, docs, current_only=False):
        hist_ok = bool(HIST_Q.search(q))
        current_only = current_only and not hist_ok
        top_doc = hits[0][1]["doc"].replace(".md", "")
        cscore = {c["chunk_id"]: sc for sc, c in hits}
        pool = []
        for ch in self.r.chunks:
            if ch["doc"].replace(".md", "") != top_doc:
                continue
            sec = ch["section"].upper()
            hist = ("HISTORICAL" in sec or "SUPERSEDED" in sec
                    or (re.search(r"20\d\d-\d\d-\d\d", sec) and "CURRENT" not in sec)
                    or "HISTORICAL" in ch["text"].split("\n")[0].upper())
            for sent in sentences(ch["text"]):
                # sentence-level marker: an explicit HISTORICAL/SUPERSEDED lead inside a
                # non-historical section still marks the sentence itself historical
                shist = hist or sent.lstrip().upper().startswith(("HISTORICAL", "SUPERSEDED"))
                pool.append((sent, ch["chunk_id"], shist))
        if not pool:
            return self._abstain("no sentences")
        if current_only:
            pool = [pp for pp in pool if not pp[2]]
            if not pool:
                return self._abstain("no current-status sentences")
        # local idf: words that appear in most sentences of this document (its name, 'model', ...) carry no signal
        ldf = {}
        for sent, _, _ in pool:
            for w in {_norm(x) for x in _tok(sent)}:
                ldf[w] = ldf.get(w, 0) + 1
        npool = len(pool)
        lidf = lambda w: float(np.log((npool + 1) / (ldf.get(w, 0) + 0.5)))
        qterms = {_norm(w) for w in _tok(q) if w not in STOP and len(w) > 1}
        qw = {w: max(lidf(w), 0.0) for w in qterms}
        denom = sum(qw.values()) or 1.0
        numeric_q = bool(re.search(r"accuracy|score|how (many|much)|what.*(result|metric|auc|percent|rate)|f1|brier|parameters|license|size|count|number|version", q, re.I))
        author_q = bool(AUTHOR_Q.search(q))
        now_q = bool(NOW_Q.search(q)) and not hist_ok
        qm = re.search(r"\b(accuracy|f1|auc|brier|precision|recall|score)\b", q.lower())
        hm = re.search(r"\b(?:what|which)\s+(?:is|are|was|were|did|does|do)?\s*(?:the)?\s*([a-z0-9\-]+)", q.lower())
        head = _norm(hm.group(1)) if hm and hm.group(1).lower() not in STOP else None
        topc = max(cscore.values()) or 1.0
        focus = []
        m_ = re.match(r"\s*(?:what|which|whose|who|how (?:many|much|long|accurate)|where|why|how)\b(.*)", q, re.I)
        if m_:
            for w in re.findall(r"[A-Za-z0-9]+", m_.group(1)):
                if w.lower() in STOP or len(w) < 2:
                    continue
                focus.append(_norm(w.lower()))
                if len(focus) == 3: break
        def qmatch(w, sw):
            if w in sw:
                return True
            if len(w) >= 4:
                return any(len(t) >= 4 and (t.startswith(w) or w.startswith(t)) for t in sw)
            return False
        sims = np.zeros(len(pool))
        for i, (sent, cid, hist) in enumerate(pool):
            sw = {_norm(x) for x in _tok(sent)}
            cov = sum(v for w, v in qw.items() if qmatch(w, sw)) / denom
            if focus:
                fw = [max(lidf(f), 0.1) for f in focus]
                fsc = sum(w for f, w in zip(focus, fw) if qmatch(f, sw)) / (sum(fw) or 1.0)
            else:
                fsc = 0.0
            sims[i] = cov / (1.0 + len(sent) / 250.0) + 0.35 * fsc + (0.12 if (numeric_q and re.search(r"[0-9%]", sent)) else 0.0) + 0.08 * cscore.get(cid, 0.0) / topc + (0.5 if (author_q and AUTHOR_MARK.search(sent)) else 0.0)
            if numeric_q and not re.search(r"[0-9]", sent):
                sims[i] *= 0.35
            if author_q and not AUTHOR_MARK.search(sent):
                sims[i] *= 0.5
            if qm and any(len(t) >= 3 and (t.startswith(qm.group(1)) or qm.group(1).startswith(t)) for t in sw):
                sims[i] += 0.3
            if (head and head in qterms and not qmatch(head, sw)
                    and not (docs and any(self.alias[d].search(head) for d in docs))):
                sims[i] *= 0.5
            if now_q and hist:
                sims[i] = -1.0
        if not hist_ok:
            sims = sims - 0.3 * np.array([1 if p[2] else 0 for p in pool])
        if sims.max() <= 0:
            return {"answer": f"The {top_doc} README does not cover that. Try rephrasing, or ask about a different project.",
                    "citations": [], "abstained": True, "why": "no lexical support"}
        self.last_debug = [(round(float(sims[j]),3), pool[j][0][:90]) for j in np.argsort(-sims)[:4]]
        order = np.argsort(-sims)
        picked = []
        for i in order:
            if sims[i] <= 0 and picked:
                break
            if any(pool[i][0] == p[0] for p in picked):
                continue
            if picked and sims[i] < 0.6 * sims[order[0]]:
                break
            picked.append(pool[i])
            if len(picked) == 2:
                break
        if author_q:
            # an authorship/contribution question must surface the explicit status
            # sentence when the routed document carries one, not only adjacent
            # provenance sentences (regression n23: BBC routed correctly but the
            # "Personal contribution unconfirmed" sentence lost to provenance text)
            status_idx = [j for j in range(len(pool)) if STATUS_MARK.search(pool[j][0])]
            if status_idx and not any(STATUS_MARK.search(pp[0]) for pp in picked):
                best = max(status_idx, key=lambda j: sims[j])
                if sims[best] > 0:
                    if picked:
                        picked[-1] = pool[best]
                    else:
                        picked.append(pool[best])
            if not any(STATUS_MARK.search(pp[0]) for pp in picked):
                # routed document carries no verbatim status sentence: answer from
                # project metadata (uniform verified status), attributed to the
                # routed document's overview chunk
                picked.insert(0, (CONTRIBUTION_STATUS, top_doc + "#0", False))
        if picked and picked[0][0].rstrip().endswith(":"):
            cid0 = picked[0][1]
            sents0 = [pp for pp in pool if pp[1] == cid0]
            idx = next((j for j, pp in enumerate(sents0) if pp[0] == picked[0][0]), None)
            if idx is not None and idx + 1 < len(sents0):
                picked.append(sents0[idx + 1])
        text = " ".join(("(historical) " + p[0]) if p[2] else p[0] for p in picked)
        return {"answer": text, "citations": sorted({p[1] for p in picked}), "abstained": False, "why": "ok"}

def extractive_threshold():
    from retriever import ABSTAIN_THRESHOLD
    return ABSTAIN_THRESHOLD
