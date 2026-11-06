"""Chunk the portfolio README corpus into retrieval units.

One chunk per markdown section (## ...), with the document title carried
into every chunk so a section never floats free of its project. Sections
longer than MAX_CHARS are split on paragraph boundaries.
"""
import json, re, sys
from pathlib import Path

MAX_CHARS = 900
CORPUS = Path(__file__).resolve().parent.parent / "corpus"
OUT = Path(__file__).resolve().parent.parent / "index" / "chunks.jsonl"

def chunks_for(path: Path):
    text = path.read_text(encoding="utf-8")
    title_m = re.search(r"^#\s+(.+)$", text, re.M)
    title = title_m.group(1).strip() if title_m else path.stem
    # split on section headers, keep the header with its body
    parts = re.split(r"(?m)^(##+ .+)$", text)
    sections = []
    lead = parts[0].strip()
    if lead:
        sections.append(("Overview", lead))
    for i in range(1, len(parts) - 1, 2):
        head = parts[i].lstrip("#").strip()
        body = parts[i + 1].strip()
        if body:
            sections.append((head, body))
    out = []
    for head, body in sections:
        body = re.sub(r"\n{3,}", "\n\n", body)
        if len(body) <= MAX_CHARS:
            out.append((head, body))
        else:
            buf = ""
            for para in body.split("\n\n"):
                if buf and len(buf) + len(para) + 2 > MAX_CHARS:
                    out.append((head, buf)); buf = para
                else:
                    buf = f"{buf}\n\n{para}" if buf else para
            if buf:
                out.append((head, buf))
    for n, (head, body) in enumerate(out):
        yield {
            "doc": path.name,
            "title": title,
            "section": head,
            "chunk_id": f"{path.stem}#{n}",
            "text": f"{title} - {head}\n{body}",
        }

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for path in sorted(CORPUS.glob("*.md")):
            for ch in chunks_for(path):
                f.write(json.dumps(ch, ensure_ascii=False) + "\n"); n += 1
    print(f"{n} chunks from {len(list(CORPUS.glob('*.md')))} docs -> {OUT}")

if __name__ == "__main__":
    main()
