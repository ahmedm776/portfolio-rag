"""Grounded answer generation with flan-t5-small (CPU).

The model sees ONLY retrieved chunks and must cite the chunk ids it used.
On retrieval abstention this module never calls the model - the answer is
a fixed honest refusal, so the LLM can never fill gaps from its priors.
"""
import re
from pathlib import Path

MODEL_NAME = "google/flan-t5-base"
MAX_CONTEXT_CHARS = 1600

ABSTAIN_ANSWER = (
    "I don't have that in the portfolio documents. The indexed READMEs "
    "cover only the projects in the corpus; try asking about one of them."
)

class Generator:
    def __init__(self):
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        import torch
        torch.set_num_threads(2)
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.model.eval()

    def answer(self, question: str, hits, abstained: bool):
        if abstained:
            return {"answer": ABSTAIN_ANSWER, "citations": [], "abstained": True}
        # Keep only chunks close to the top score: weaker cross-project
        # chunks in the context make the small model blend documents.
        top = hits[0][0]
        hits = [(s_, c_) for s_, c_ in hits if s_ >= 0.6 * top] or hits[:1]
        ctx_parts, total = [], 0
        for score, ch in hits:
            piece = f"[{ch['chunk_id']}] {ch['text']}"
            if total + len(piece) > MAX_CONTEXT_CHARS:
                piece = piece[: MAX_CONTEXT_CHARS - total]
            ctx_parts.append(piece)
            total += len(piece)
        context = "\n\n".join(ctx_parts)
        prompt = (
            "Answer the question in one or two sentences using only the "
            "context below. If the context does not contain the answer, "
            "say it is not covered.\n\n"
            f"Context (from the README of {hits[0][1]['title']}):\n{context}\n\nQuestion: {question}\nAnswer:" 
        )
        import torch
        ids = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=160, do_sample=False)
        text = self.tok.decode(out[0], skip_special_tokens=True).strip()
        # Citations are retrieval provenance: the chunks the answer was
        # grounded on. The 80M model is not relied on to echo bracket ids.
        citations = [ch["chunk_id"] for _, ch in hits[:2]]
        return {"answer": text, "citations": citations, "abstained": False}

if __name__ == "__main__":
    import sys
    from retriever import Retriever
    r, g = Retriever(), Generator()
    q = " ".join(sys.argv[1:]) or "What accuracy did the traffic sign model reach?"
    hits, ab = r.retrieve(q)
    print(g.answer(q, hits, ab))
