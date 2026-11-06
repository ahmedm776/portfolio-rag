"""FastAPI demo for the portfolio document RAG.

GET  /         minimal chat UI
GET  /api/docs list of indexed documents
POST /api/ask  {question} -> {answer, citations, abstained, retrieved}
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import Retriever
import extractive
from pipeline import Pipeline

app = FastAPI(title="Portfolio document assistant")
retriever = Retriever()
pipeline = Pipeline(retriever)
generator = None  # lazy: only loaded when llm=1 is requested

def get_generator():
    global generator
    if generator is None:
        from generator import Generator
        generator = Generator()
    return generator

class Ask(BaseModel):
    question: str
    llm: bool = False  # experimental flan-t5-base compose step (see README limits)

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Portfolio document assistant</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.3rem} .note{color:#666;font-size:.85rem;margin-bottom:1rem}
#log div{border:1px solid #ddd;border-radius:8px;padding:.7rem;margin:.6rem 0;white-space:pre-wrap}
.q{background:#f0f4ff}.cite{font-size:.8rem;color:#555}
input{width:70%;padding:.5rem;font-size:1rem}button{padding:.5rem 1rem;font-size:1rem}
</style></head><body>
<h1>Ask about Ahmed's portfolio projects</h1>
<div class=note>Answers come only from the indexed project READMEs, with citations.
Questions outside the documents' scope get an explicit "not covered" answer when no project README is relevant.
Answers quote the cited README verbatim; percentages are the models' reported metrics or uncalibrated scores, as labeled there.</div>
<div id=log></div>
<form id=f><input id=q placeholder="e.g. how does the food price model prevent leakage?" autofocus>
<button>Ask</button></form>
<script>
const log=document.getElementById('log');
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const q=document.getElementById('q');const d=document.createElement('div');
d.className='q';d.textContent='Q: '+q.value;log.appendChild(d);
const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q.value})});
const j=await r.json();const a=document.createElement('div');
a.innerHTML='A: '+j.answer+(j.citations.length?'<div class=cite>sources: '+j.citations.join(', ')+'</div>':'')+(j.abstained?'<div class=cite>(abstained: not covered by the documents)</div>':'');
log.appendChild(a);q.value='';q.focus();};
const qp=new URLSearchParams(location.search).get('q');
if(qp){document.getElementById('q').value=qp;document.getElementById('f').requestSubmit();}
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE

@app.get("/api/docs")
def docs():
    seen, out = set(), []
    for c in retriever.chunks:
        if c["doc"] not in seen:
            seen.add(c["doc"]); out.append({"doc": c["doc"], "title": c["title"]})
    return {"documents": out, "chunks": len(retriever.chunks)}

@app.post("/api/ask")
def ask(body: Ask):
    hits, abstained = retriever.retrieve(body.question)
    if body.llm:
        result = get_generator().answer(body.question, hits, abstained)
    else:
        result = pipeline.ask(body.question)
        result.pop("why", None)
    result["retrieved"] = [
        {"chunk_id": c["chunk_id"], "doc": c["doc"], "section": c["section"], "score": round(s, 3)}
        for s, c in hits
    ]
    return result
