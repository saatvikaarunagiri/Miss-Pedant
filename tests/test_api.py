def test_health_reports_seeded_samples(client):
    body = client.get("/api/health").json()
    assert body["papers"] == 2
    assert "provider" in body and "model" in body


def test_papers_include_text(client):
    papers = client.get("/api/papers").json()["papers"]
    assert len(papers) == 2
    assert all(p["text"] and p["chunks"] >= 1 for p in papers)


def test_retrieve_finds_the_right_paper(client):
    ids = [p["id"] for p in client.get("/api/papers").json()["papers"]]
    cases = {
        "How does Cas9 find its target PAM?": "CRISPR",
        "What do lipid nanoparticles do for mRNA?": "mRNA",
    }
    for query, expected in cases.items():
        top = client.post("/api/retrieve", json={"query": query, "paper_ids": ids, "k": 2}).json()["passages"][0]
        assert top["paper"].startswith(expected)
        assert top["retriever"] in {"dense", "bm25", "both"}


def test_retrieve_respects_selection(client):
    papers = client.get("/api/papers").json()["papers"]
    mrna = next(p for p in papers if p["name"].startswith("mRNA"))
    got = client.post("/api/retrieve", json={"query": "Cas9 PAM", "paper_ids": [mrna["id"]]}).json()["passages"]
    assert got and all(p["paper_id"] == mrna["id"] for p in got)


def test_upload_and_delete(client):
    r = client.post("/api/papers", files={"file": ("notes.md", b"Insulin is secreted by pancreatic beta cells. It lowers blood glucose.", "text/markdown")})
    assert r.status_code == 200
    pid = r.json()["id"]
    assert len(client.get("/api/papers").json()["papers"]) == 3
    assert client.delete(f"/api/papers/{pid}").json() == {"ok": True}
    assert len(client.get("/api/papers").json()["papers"]) == 2
    assert client.delete(f"/api/papers/{pid}").status_code == 404


def test_upload_rejects_bad_files(client):
    assert client.post("/api/papers", files={"file": ("x.exe", b"zz", "application/octet-stream")}).status_code == 400
    assert client.post("/api/papers", files={"file": ("empty.txt", b"   ", "text/plain")}).status_code == 400


def test_library_caps_at_ten(client):
    for i in range(8):
        assert client.post("/api/papers", files={"file": (f"p{i}.txt", f"Paper {i} studies protein folding in yeast cells.".encode(), "text/plain")}).status_code == 200
    r = client.post("/api/papers", files={"file": ("one-too-many.txt", b"Another paper about ribosomes and translation.", "text/plain")})
    assert r.status_code == 400 and "full" in r.json()["detail"]


def test_generate_streams_text(client):
    r = client.post("/api/generate", json={"messages": [{"role": "user", "content": "rules"}, {"role": "user", "content": "q"}]})
    assert r.status_code == 200
    assert r.text == "Cas9 is guided by an sgRNA [1] and requires a PAM [1]."


def test_generate_returns_503_when_model_is_down(client, monkeypatch):
    import llm

    def down(*args, **kwargs):
        raise ConnectionError("connection refused")
        yield  # makes this a generator

    monkeypatch.setattr(llm, "stream_chat", down)
    assert client.post("/api/generate", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 503


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Miss Pedant" in r.text
