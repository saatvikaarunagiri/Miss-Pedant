"""
Miss Pedant: FastAPI backend.

Endpoints
  GET    /api/health          which model is answering, how many papers
  GET    /api/papers          the library (with full text)
  POST   /api/papers          upload one paper (multipart "file")
  DELETE /api/papers/{id}     remove a paper
  POST   /api/retrieve        hybrid retrieval over selected papers
  POST   /api/generate        stream a model answer as plain text
  GET    /                    the frontend
"""
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing modules that read environment variables at import time.
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import llm  # noqa: E402
import rag  # noqa: E402

FRONTEND = ROOT / "frontend"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED = (".pdf", ".txt", ".md", ".markdown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # First run downloads the embedding model (~130 MB) and indexes the samples.
    try:
        await run_in_threadpool(rag.seed_samples)
    except Exception as e:  # don't block startup on a seeding problem
        print(f"[miss-pedant] could not seed sample papers: {e}")
    print(f"[miss-pedant] model: {llm.describe()}  ·  open http://127.0.0.1:8000")
    yield


app = FastAPI(title="Miss Pedant", lifespan=lifespan)


# ---------- request shapes ----------

class Turn(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: list[Turn] = Field(min_length=1)
    tier: str = "default"
    max_tokens: int = Field(default=1800, le=4096)


class RetrieveRequest(BaseModel):
    query: str
    paper_ids: list[str] = []
    k: int = Field(default=6, ge=1, le=12)


# ---------- routes ----------

@app.get("/api/health")
def health():
    return {**llm.describe(), "papers": len(rag.list_papers(with_text=False))}


@app.get("/api/papers")
def papers():
    return {"papers": rag.list_papers()}


@app.post("/api/papers")
async def upload(file: UploadFile = File(...)):
    name = file.filename or "paper"
    if not name.lower().endswith(ALLOWED):
        raise HTTPException(400, "Upload a PDF, TXT or Markdown file.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is over 25 MB.")
    try:
        # parsing + embedding is CPU/GPU work; keep it off the event loop
        return await run_in_threadpool(rag.add_paper, name, data)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/papers/{paper_id}")
def delete(paper_id: str):
    if not rag.delete_paper(paper_id):
        raise HTTPException(404, "No such paper.")
    return {"ok": True}


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest):
    return {"passages": rag.retrieve(req.query, req.paper_ids, req.k)}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    stream = llm.stream_chat([t.model_dump() for t in req.messages], req.tier, req.max_tokens)

    # Pull the first piece before responding, so "model server isn't running"
    # becomes a clean 503 instead of a half-open stream.
    try:
        first = next(stream)
    except StopIteration:
        first = ""
    except Exception as e:
        raise HTTPException(503, f"Model unavailable: {e}")

    def body():
        yield first
        try:
            yield from stream
        except Exception as e:  # connection dropped mid-answer
            yield f"\n\n[stream interrupted: {e}]"

    return StreamingResponse(body(), media_type="text/plain; charset=utf-8")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
