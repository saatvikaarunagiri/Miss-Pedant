"""
Test fixtures.

The real app downloads ~5 GB of models, so the tests swap in two stand-ins:
  - a tiny bag-of-words embedder instead of BGE
  - a scripted model stream instead of mlx / Claude
Everything else (parsing, chunking, Chroma, BM25, fusion, the API) is real.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import llm  # noqa: E402
import rag  # noqa: E402


class HashEmbedder:
    """Deterministic 128-dim bag-of-words vectors. Good enough to rank by word overlap."""

    def encode(self, texts, batch_size=32, normalize_embeddings=True):
        single = isinstance(texts, str)
        rows = []
        for t in [texts] if single else texts:
            v = np.zeros(128)
            for w in t.lower().split():
                v[int(hashlib.md5(w.strip(".,()").encode()).hexdigest(), 16) % 128] += 1
            rows.append(v / (np.linalg.norm(v) or 1))
        arr = np.array(rows)
        return arr[0] if single else arr


def scripted_stream(messages, tier="default", max_tokens=1800):
    yield "Cas9 is guided by an sgRNA [1]"
    yield " and requires a PAM [1]."


@pytest.fixture
def client(tmp_path, monkeypatch):
    # point all storage at a throwaway folder
    monkeypatch.setattr(rag, "DATA", tmp_path)
    monkeypatch.setattr(rag, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(rag, "REGISTRY", tmp_path / "papers.json")
    monkeypatch.setattr(rag, "USE_RERANK", False)
    monkeypatch.setattr(rag, "embedder", lambda: HashEmbedder())
    monkeypatch.setattr(llm, "stream_chat", scripted_stream)
    rag.collection.cache_clear()

    from fastapi.testclient import TestClient
    import app

    with TestClient(app.app) as c:  # runs startup, which indexes the sample papers
        yield c
    rag.collection.cache_clear()
