"""
Retrieval for Miss Pedant.

    upload -> parse -> sentence-aware chunks -> BGE embeddings -> Chroma
    query  -> dense search + BM25 -> reciprocal rank fusion -> cross-encoder rerank

Papers are stored on disk (data/papers/<id>.txt) plus a small JSON registry,
so the library survives restarts. With at most 10 papers, BM25 is rebuilt per
query over the selected papers' chunks; that's fast enough and keeps the
keyword index perfectly in sync with whatever the user has selected.
"""
import io
import json
import math
import os
import re
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import chromadb
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PAPERS_DIR = DATA / "papers"
REGISTRY = DATA / "papers.json"
SAMPLES = ROOT / "sample_papers"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
USE_RERANK = os.getenv("USE_RERANK", "1") == "1"
MAX_PAPERS = 10

_lock = threading.Lock()  # uploads and deletes touch the registry; keep them serial


# ---------- models + store (loaded lazily, once) ----------

@lru_cache
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)  # picks the Mac GPU (MPS) automatically


@lru_cache
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


@lru_cache
def collection():
    DATA.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DATA / "chroma"))
    # cosine distance matches normalized BGE embeddings
    return client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})


# ---------- registry ----------

def _load() -> list[dict]:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []


def _save(items: list[dict]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(items, indent=2))


def list_papers(with_text: bool = True) -> list[dict]:
    out = []
    for p in _load():
        item = dict(p)
        if with_text:
            path = PAPERS_DIR / f"{p['id']}.txt"
            item["text"] = path.read_text(encoding="utf-8") if path.exists() else ""
        out.append(item)
    return out


# ---------- parsing + chunking ----------

def parse(filename: str, data: bytes) -> str:
    """Pull plain text out of a PDF / TXT / Markdown upload."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:80])
    else:
        text = data.decode("utf-8", errors="ignore")
    text = re.sub(r"-\n(\w)", r"\1", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk(text: str, target_chars: int = 900, overlap_sentences: int = 1) -> list[str]:
    """Pack whole sentences into ~900-character chunks, carrying one sentence
    over between chunks so an idea split across the boundary isn't lost."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]
    chunks, current, fresh = [], [], 0
    for s in sentences:
        current.append(s)
        fresh += 1
        if sum(len(x) for x in current) >= target_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            fresh = 0
    if fresh:  # leftover sentences that haven't been emitted yet
        chunks.append(" ".join(current))
    return chunks


# ---------- library operations ----------

def add_paper(filename: str, data: bytes) -> dict:
    with _lock:
        items = _load()
        if len(items) >= MAX_PAPERS:
            raise ValueError(f"The library is full ({MAX_PAPERS} papers max). Remove one first.")

        text = parse(filename, data)
        if len(text) < 40:
            raise ValueError("No readable text found. Scanned PDFs need OCR before upload.")

        pid = uuid.uuid4().hex[:10]
        chunks = chunk(text)
        embeddings = embedder().encode(chunks, batch_size=32, normalize_embeddings=True).tolist()
        collection().add(
            ids=[f"{pid}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"paper_id": pid, "paper": filename, "idx": i} for i in range(len(chunks))],
        )

        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        (PAPERS_DIR / f"{pid}.txt").write_text(text, encoding="utf-8")

        meta = {
            "id": pid,
            "name": filename,
            "kind": "pdf" if filename.lower().endswith(".pdf") else "txt",
            "words": len(text.split()),
            "chunks": len(chunks),
        }
        items.append(meta)
        _save(items)
        return {**meta, "text": text}


def delete_paper(pid: str) -> bool:
    with _lock:
        items = _load()
        if not any(p["id"] == pid for p in items):
            return False
        collection().delete(where={"paper_id": pid})
        (PAPERS_DIR / f"{pid}.txt").unlink(missing_ok=True)
        _save([p for p in items if p["id"] != pid])
        return True


def seed_samples() -> None:
    """Index the two sample papers on first run so the app works immediately."""
    if _load() or not SAMPLES.exists():
        return
    for f in sorted(SAMPLES.glob("*.txt")):
        add_paper(f.name, f.read_bytes())


# ---------- retrieval ----------

STOPWORDS = set("""a an the and or of to in on for with by from at as is are was were be been it its this that
these those how what which who why when does do did can could would should into about than then there their
they them we our you your i me my not no""".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def retrieve(query: str, paper_ids: list[str] | None = None, k: int = 6, candidates: int = 20) -> list[dict]:
    ids = [p["id"] for p in _load() if not paper_ids or p["id"] in paper_ids]
    if not ids or not query.strip():
        return []

    where = {"paper_id": {"$in": ids}}
    got = collection().get(where=where, include=["documents", "metadatas"])
    docs, metas, chunk_ids = got["documents"], got["metadatas"], got["ids"]
    if not docs:
        return []
    position = {cid: i for i, cid in enumerate(chunk_ids)}
    n = min(candidates, len(docs))

    # 1) dense: meaning-based nearest neighbours
    q_emb = embedder().encode(query, normalize_embeddings=True).tolist()
    res = collection().query(query_embeddings=[q_emb], n_results=n, where=where)
    dense = [position[c] for c in res["ids"][0] if c in position]

    # 2) sparse: BM25 keywords, which catches exact names and acronyms
    #    Relevance is decided by real term overlap, and BM25 orders the matches.
    #    (With very few chunks BM25's IDF can be exactly zero, so a score > 0
    #    filter alone would silently drop genuine keyword hits.)
    doc_tokens = [_tokens(d) for d in docs]
    q_tokens = set(_tokens(query))
    overlap = [len(q_tokens & set(t)) for t in doc_tokens]
    scores = BM25Okapi(doc_tokens).get_scores(list(q_tokens)) if q_tokens else [0.0] * len(docs)
    sparse = sorted((i for i in range(len(docs)) if overlap[i] > 0),
                    key=lambda i: (scores[i], overlap[i]), reverse=True)[:n]

    # 3) reciprocal rank fusion: anything ranked well by either list rises
    fused_score: dict[int, float] = {}
    for rank, i in enumerate(dense):
        fused_score[i] = fused_score.get(i, 0) + 1 / (60 + rank)
    for rank, i in enumerate(sparse):
        fused_score[i] = fused_score.get(i, 0) + 1 / (60 + rank)
    fused = sorted(fused_score, key=fused_score.get, reverse=True)[:n]

    # 4) rerank the shortlist with a cross-encoder that reads query + chunk together
    if USE_RERANK and len(fused) > 1:
        rr = reranker().predict([(query, docs[i]) for i in fused])
        order = sorted(range(len(fused)), key=lambda j: rr[j], reverse=True)
        ranked = [(fused[j], 1 / (1 + math.exp(-float(rr[j])))) for j in order]  # sigmoid -> 0..1
    else:
        ranked = [(i, fused_score[i]) for i in fused]

    dense_set, sparse_set = set(dense), set(sparse)
    out = []
    for i, score in ranked[:k]:
        out.append({
            "paper": metas[i]["paper"],
            "paper_id": metas[i]["paper_id"],
            "text": docs[i],
            "score": round(score, 3),
            # provenance: which retriever found this chunk (shown in the Sources panel)
            "retriever": "both" if i in dense_set and i in sparse_set else ("dense" if i in dense_set else "bm25"),
        })
    return out
