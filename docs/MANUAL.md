# Miss Pedant: User and Developer Manual

This manual covers installing Miss Pedant, using every feature, configuring it, and changing it. For a one-page overview, see the [README](../README.md).

## Contents

1. [What Miss Pedant is](#1-what-miss-pedant-is)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Choosing a model](#4-choosing-a-model)
5. [Running the app](#5-running-the-app)
6. [Using Miss Pedant](#6-using-miss-pedant)
7. [Configuration reference](#7-configuration-reference)
8. [How it works](#8-how-it-works)
9. [API reference](#9-api-reference)
10. [Customizing](#10-customizing)
11. [Testing](#11-testing)
12. [Troubleshooting](#12-troubleshooting)
13. [Known limitations](#13-known-limitations)
14. [Data and privacy](#14-data-and-privacy)

---

## 1. What Miss Pedant is

Miss Pedant is a retrieval-augmented generation (RAG) application for reading scientific papers. You upload up to ten papers, and she answers questions using only what those papers say, citing each claim back to the exact passage it came from.

Around that core are tools for studying and writing. Red Pen fact-checks your own drafts, Pop Quiz tests your understanding, Concept Map shows how ideas connect, and Reviewer #2 critiques a paper.

The app has two parts:

- **A Python backend** (FastAPI). It parses and indexes papers, runs hybrid retrieval, and streams model output.
- **A single-page frontend** (`frontend/index.html`). It is plain HTML, CSS and JavaScript with no build step, and the backend serves it.

---

## 2. Requirements

| Requirement | Notes |
| --- | --- |
| Python 3.11 | 3.10–3.12 should also work; 3.11 is what it's tested on. |
| Apple Silicon Mac (M1 or later) | Only needed for the default local model. Any OS works with the Claude or OpenAI API. |
| 16 GB RAM | Comfortable for the default 4-bit 7B model. See [memory tips](#memory-on-a-16-gb-mac). |
| About 6 GB of free disk | For the language model, embedding model and reranker. |
| A modern browser | Chrome, Safari, Firefox or Edge. |
| Internet on first launch | To download models. After that, the local setup works offline (except the `.docx` export; see [limitations](#13-known-limitations)). |

---

## 3. Installation

### 3.1 Get the code

```bash
git clone https://github.com/<your-username>/miss-pedant.git
cd miss-pedant
```

### 3.2 Create a virtual environment

A virtual environment keeps this project's packages separate from the rest of your system.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. Run `source .venv/bin/activate` again whenever you open a new terminal for this project.

If `python3.11` isn't found on a Mac, install it with `brew install python@3.11`.

### 3.3 Install dependencies

```bash
pip install -r requirements.txt
```

This installs the web server, the retrieval stack (Chroma, sentence-transformers, BM25), both API clients, and `mlx-lm`. The `mlx-lm` package is only installed on Apple Silicon and is skipped automatically elsewhere.

`sentence-transformers` pulls in PyTorch, so expect this step to take a few minutes.

### 3.4 Create your settings file

```bash
cp .env.example .env
```

The defaults use a local model, so you don't need to change anything to get started. `.env` is listed in `.gitignore`, so your API keys will never be committed.

---

## 4. Choosing a model

Set `LLM_PROVIDER` in `.env` to one of three options.

### Local with mlx (default)

```
LLM_PROVIDER=mlx
MLX_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit
```

This runs entirely on your Mac and costs nothing per question. `run.sh` starts the model server for you.

### Claude API

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Use this for the best results. It's noticeably better at the features that need structured output: Red Pen, Pop Quiz, Concept Map and paper summaries. `ANTHROPIC_MODEL` is used for answers and `ANTHROPIC_MODEL_QUICK` for lighter tasks such as summaries and follow-up suggestions. You can get a key at console.anthropic.com.

### OpenAI API

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

This also works with any service that speaks the OpenAI protocol. Set `OPENAI_BASE_URL` in `.env` to point at it.

### Which should I pick?

| If you want… | Use |
| --- | --- |
| Free, private, offline after setup | `mlx` |
| The most reliable answers and structured features, for a demo | `anthropic` |
| A model you already pay for | `openai` |

Retrieval always runs locally, whichever provider you choose. Only the passages relevant to each question are sent to an API.

---

## 5. Running the app

### 5.1 The one-command way

```bash
./run.sh
```

The script:

1. creates `.env` from `.env.example` if it doesn't exist,
2. activates `.venv` if present,
3. starts `mlx_lm.server` on port 8080 if you're using `mlx` and it isn't already running,
4. waits for the model server to be ready, and
5. starts the web app on port 8000.

Open **http://127.0.0.1:8000** in your browser. Press `Ctrl+C` in the terminal to stop everything.

**The first launch is slow, once.** It downloads the language model (about 4 GB), the embedding model (about 130 MB) and the reranker (about 1 GB), then indexes the two sample papers. Later launches take seconds. Model server output goes to `mlx.log`.

If you see `permission denied` when running the script, make it executable first: `chmod +x run.sh`.

### 5.2 Running the servers separately

Running the two servers separately is useful while developing, because `--reload` restarts the backend whenever you edit Python code.

```bash
# terminal 1: the model (skip this if you use an API provider)
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080

# terminal 2: the app
source .venv/bin/activate
cd backend
uvicorn app:app --reload --port 8000
```

### 5.3 Checking it's working

```bash
curl http://127.0.0.1:8000/api/health
```

You should see the provider, the model, and `"papers": 2`.

---

## 6. Using Miss Pedant

The icons on the left rail switch between views. From top to bottom: Ask, Library, Red Pen, Pop Quiz, Concept Map and History. The moon icon at the bottom toggles dark mode. On a phone, the rail sits across the top.

### 6.1 Library: adding papers

Click **Library**, then either click **Add papers** or drag files onto the drop area.

- **Formats:** PDF, `.txt`, `.md`.
- **Limits:** 10 papers and 25 MB per file.
- **Status badges:** `indexing` while a file is parsed and embedded, `ready` when it's searchable, `error` if no text could be extracted.

Each paper card has these controls:

| Control | What it does |
| --- | --- |
| Checkbox | Include or exclude the paper from answers, quizzes and maps. Unchecked papers stay in the library. |
| Sparkle icon (Analyze) | Writes a two-sentence summary and the paper's three key findings. |
| **R2** | Opens a Reviewer #2 critique (see [6.7](#67-reviewer-2)). |
| Quote icon (Cite) | Infers an APA citation and a BibTeX entry from the paper's text, with copy buttons. |
| Trash icon (Remove) | Deletes the paper and its index entries. |

Two sample papers, on CRISPR-Cas9 and mRNA vaccines, are added on first run. Remove them whenever you like. They won't come back unless you delete the `data/` folder.

**A scanned PDF that shows `error`** has no text layer. See [troubleshooting](#12-troubleshooting) for the OCR fix.

### 6.2 Ask: questions and answers

Type a question and press **Enter** or click the arrow. Use **Shift+Enter** for a new line.

Miss Pedant searches your selected papers, picks the most relevant passages, and streams an answer that cites them as `[1]`, `[2]`, and so on.

- **Click a citation** to jump to and highlight its passage in the **Sources** panel.
- **The Sources panel** lists each passage with its paper and a badge showing which search found it:
  - `dense`: matched by meaning
  - `bm25`: matched by keywords
  - `dense + bm25`: found by both

  On a phone, open the panel with the ☰ button in the header.
- **Follow-up questions** appear under each answer. Click one to ask it.
- **Copy** puts the answer on your clipboard. **Regenerate** asks the same question again.
- **Conversation memory:** the last six messages are sent along with each new question, so follow-ups like "and in humans?" work.

If the passages don't contain the answer, Miss Pedant is instructed to say so rather than guess.

**Writing good questions.** Specific questions retrieve better passages. "What repair pathways follow a Cas9 cut?" beats "Tell me about CRISPR." Naming the exact term you care about also helps the keyword search.

### 6.3 Level and Pedantry

The two switches above the chat change how answers are written, not what they're based on.

**Level**
- *Undergrad:* defines jargon and explains clearly.
- *Grad:* precise but readable. This is the default.
- *Expert:* technical and concise.

**Pedantry**
- *Gentle:* warm and encouraging.
- *Strict:* corrects loose terminology in your question in one sentence, then answers. This is the default.
- *Insufferable:* corrects your terminology, adds one dry aside, and stays fully cited.

### 6.4 Synthesize

The **Synthesize** button in the Ask header compares all selected papers. It reports shared themes, where they agree, and any contradictions or gaps, with citations. You need at least two papers selected.

### 6.5 Red Pen

Red Pen fact-checks writing against your papers. It's useful for checking a thesis paragraph, an abstract, lecture notes, or a claim from a news article.

1. Open **Red Pen**. A sample draft with deliberate mistakes is preloaded.
2. Paste your own text, or keep the sample.
3. Click **Mark it up**.

Every sentence comes back marked:

| Mark | Meaning |
| --- | --- |
| Green underline | **Supported** by your papers |
| Red strikethrough | **Contradicted** by your papers |
| Amber squiggle | **Not in your papers**: they say nothing either way |

Numbered margin notes explain each mark and name the paper involved. Click a sentence to jump to its note, or click a note to find its sentence. The stamp gives an overall letter grade for factual accuracy.

"Not in your papers" doesn't mean false. It means your sources can't confirm it.

### 6.6 Pop Quiz

1. Open **Pop Quiz** and choose **Easy**, **Medium** or **Hard**.
2. Click **New quiz**. Miss Pedant writes five multiple-choice questions from your selected papers.
3. Pick an answer. You'll see whether it was right, a one-line explanation, and the source paper.
4. After the last question you get a grade and a verdict. **Retake** repeats the same questions, and **New quiz** writes fresh ones.

A perfect score earns confetti.

### 6.7 Reviewer #2

Click **R2** on a paper in the Library. You get a peer review with a verdict (Accept, Minor revisions, Major revisions or Reject), two strengths, three weaknesses, and two questions for the authors. The tone is sharp but the points are meant to be useful. **Copy review** puts it on your clipboard.

### 6.8 Concept Map

1. Open **Concept Map** and click **Build map**.
2. Miss Pedant extracts 12–20 concepts from your selected papers and the relationships between them. The concepts are color-coded by type: gene, protein, molecule, process, disease, method, organism or concept.

You can then work with the map:

- **Drag** nodes to rearrange them. The layout settles itself.
- **Click a node** to highlight its connections, see each relationship labeled, and open a detail card.
- **Ask Miss Pedant about this** in the detail card sends a question about that concept and its neighbors to the chat.
- **Click empty space** to clear the selection.

Larger nodes have more connections.

### 6.9 Highlight to ask

Select any text in an answer, a source passage, or a Red Pen result. A small **Ask Miss Pedant about this** button appears above your selection. Click it to ask for a deeper explanation of exactly that phrase.

### 6.10 History

Every conversation is saved automatically. Open **History** to see past chats with their dates and message counts.

- Click a chat to reopen it, citations included.
- Click the trash icon to delete one chat, or **Clear all** to delete everything.

History is stored in your browser on this computer and keeps the 40 most recent chats. It won't appear in a different browser or on another device.

### 6.11 Downloading a chat

Click **Download** in the Ask header and choose a format:

| Format | Best for |
| --- | --- |
| `.md` | Notes apps, GitHub, Obsidian |
| `.txt` | Anything; plain text with no formatting |
| `.docx` | Microsoft Word or Google Docs |

Each export includes every question, every answer, and the list of sources cited under each answer.

---

## 7. Configuration reference

All settings live in `.env`. After changing it, restart the app.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `mlx` | `mlx`, `anthropic` or `openai` |
| `MLX_BASE_URL` | `http://127.0.0.1:8080/v1` | Where `mlx_lm.server` listens |
| `MLX_MODEL` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Any MLX model from Hugging Face |
| `ANTHROPIC_API_KEY` | empty | Your Claude API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Model for answers and structured features |
| `ANTHROPIC_MODEL_QUICK` | `claude-haiku-4-5-20251001` | Model for summaries and follow-ups |
| `OPENAI_API_KEY` | empty | Your OpenAI key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `OPENAI_BASE_URL` | unset | Optional: any OpenAI-compatible endpoint |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model for dense search |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder for reranking |
| `USE_RERANK` | `1` | Set to `0` to skip reranking |

**If you change `EMBED_MODEL`, delete the `data/` folder** and re-upload your papers. Vectors from different embedding models aren't comparable.

### Memory on a 16 GB Mac

The default 4-bit 7B model uses about 5 GB, the reranker about 1 GB, and the embedder very little. If your Mac slows down or the model server crashes:

- set `USE_RERANK=0`, which saves about 1 GB at a small cost to ranking quality, or
- switch to a smaller model: `MLX_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit`, or
- use an API provider, which needs no local model at all.

---

## 8. How it works

```
 Browser (frontend/index.html)
    │  build prompts, render answers, citations, quiz, map, history, exports
    │
    ▼
 FastAPI (backend/app.py)
    ├── rag.py ── parse ─ chunk ─ embed ─ Chroma (data/chroma)
    │             query: dense ┐
    │                    BM25  ┴─ reciprocal rank fusion ─ cross-encoder rerank
    │
    └── llm.py ── stream text from mlx_lm.server | Claude API | OpenAI API
```

### 8.1 Ingestion (`rag.py`)

1. **Parse.** `pypdf` extracts text from the first 80 pages of a PDF. Plain text and Markdown are read directly. Words hyphenated across line breaks are re-joined.
2. **Chunk.** Text is split into sentences, and whole sentences are packed into chunks of about 900 characters. The last sentence of each chunk is repeated at the start of the next, so an idea that spans a boundary isn't lost.
3. **Embed.** Each chunk is embedded with BGE-small (384 dimensions, normalized). On a Mac this runs on the GPU automatically.
4. **Store.** Vectors go into a persistent Chroma collection in `data/chroma`, tagged with the paper's ID. The full text goes to `data/papers/<id>.txt`, and metadata to `data/papers.json`.

### 8.2 Retrieval

For each question, restricted to the selected papers:

1. **Dense search.** Chroma returns the 20 chunks closest in meaning to the question.
2. **Keyword search.** BM25 ranks chunks that share words with the question, after stopwords are removed. Only chunks that contain at least one query term qualify. This guard matters for small libraries, where BM25's IDF term can be zero and silently hide genuine matches.
3. **Fusion.** Reciprocal rank fusion combines the two lists. A chunk ranked highly by either search rises.
4. **Rerank.** A cross-encoder reads the question and each shortlisted chunk together and re-scores them. The top six are returned, each tagged with the retriever that found it.

### 8.3 Generation (`llm.py`)

The frontend numbers the retrieved passages and sends them with instructions: answer only from these excerpts, cite with `[n]`, and say so if the answer isn't there. `llm.py` streams the model's reply back.

Before sending, consecutive messages from the same speaker are merged and the conversation is made to start with a user turn, because some APIs reject anything else. Temperature is set low (0.3) to favor faithfulness over creativity.

If the model can't be reached, the backend returns a clean `503` before any streaming starts, and the UI shows a helpful message.

### 8.4 Structured features

Red Pen, Pop Quiz, Concept Map, summaries and citations ask the model for JSON. The frontend tolerates replies wrapped in prose or code fences, extracting the first JSON object or array it finds. If parsing still fails, the feature shows an in-character "try again" message.

### 8.5 Project layout

```
miss-pedant/
├── backend/
│   ├── app.py            API routes, upload limits, streaming, serves the UI
│   ├── rag.py            parsing, chunking, indexing, hybrid retrieval
│   └── llm.py            one streaming interface over three providers
├── frontend/
│   └── index.html        the entire interface
├── sample_papers/        indexed on first run
├── tests/                pytest suite with stand-in models
├── docs/
│   ├── MANUAL.md         this file
│   └── images/           screenshots
├── data/                 created at runtime (gitignored)
├── .github/workflows/    CI: runs the tests on every push
├── requirements.txt      full app
├── requirements-dev.txt  lightweight set for tests
├── .env.example
└── run.sh
```

---

## 9. API reference

All endpoints are under `http://127.0.0.1:8000`. FastAPI also serves interactive docs at **/docs**, where you can try every endpoint from the browser.

### `GET /api/health`

```json
{ "provider": "mlx", "model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "papers": 2 }
```

### `GET /api/papers`

Returns the library, including each paper's full extracted text.

```json
{ "papers": [ { "id": "a1b2c3d4e5", "name": "paper.pdf", "kind": "pdf", "words": 5120, "chunks": 38, "text": "…" } ] }
```

### `POST /api/papers`

Uploads one paper as multipart form data, in a field named `file`.

```bash
curl -F "file=@my-paper.pdf" http://127.0.0.1:8000/api/papers
```

| Status | Meaning |
| --- | --- |
| `200` | Indexed. Returns the paper, as above. |
| `400` | Wrong file type, no readable text, or the library is full. |
| `413` | File larger than 25 MB. |

### `DELETE /api/papers/{id}`

Removes a paper and its chunks. Returns `{"ok": true}`, or `404` if the ID doesn't exist.

### `POST /api/retrieve`

```json
{ "query": "How does Cas9 find its target?", "paper_ids": ["a1b2c3d4e5"], "k": 6 }
```

An empty `paper_ids` list searches every paper. `k` can be 1–12. The response:

```json
{ "passages": [ { "paper": "paper.pdf", "paper_id": "a1b2c3d4e5", "text": "…", "score": 0.91, "retriever": "both" } ] }
```

### `POST /api/generate`

```json
{ "messages": [ { "role": "user", "content": "…" } ], "tier": "default", "max_tokens": 1800 }
```

`tier` is `quick` or `default`. The response is streamed plain text. It returns `503` if the model is unreachable.

---

## 10. Customizing

### Colors and font

All colors are CSS variables at the top of `frontend/index.html`, in the `:root` block (and in `:root[data-theme="dark"]` for dark mode). The font is one variable:

```css
--font: "Times New Roman", Times, "Liberation Serif", serif;
```

Change it there and every piece of text follows. Liberation Serif is a metric-compatible stand-in for Times New Roman on Linux machines.

### The molecular background

The slowly rotating protein backbone is drawn on a canvas. To make it fainter, lower `--mol-a`, which defaults to `.32`. To remove it entirely, delete the `<canvas id="mol">` element.

### Miss Pedant's personality

The tone instructions live in the `PED` object in `frontend/index.html`, one string per Pedantry setting. Reading-level instructions are next to them in `sendMsg()`. The quiz verdicts and correct/incorrect phrases are in `quizEnd()` and the `RIGHT` and `WRONG` arrays.

### Retrieval

In `backend/rag.py`:

- `chunk(target_chars=900, overlap_sentences=1)` controls chunk size and overlap.
- `retrieve(k=6, candidates=20)` controls how many passages are returned and how many are considered before reranking.
- `MAX_PAPERS = 10` sets the library limit.

### Adding a model provider

Add a branch to `stream_chat()` in `backend/llm.py` that yields text pieces, and a case in `model_for()`. Nothing else needs to change.

---

## 11. Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The tests swap in a tiny bag-of-words embedder and a scripted model, so they run in about a second with no downloads. Everything else is real: parsing, chunking, Chroma, BM25, fusion, and every API route.

They cover:

- sentence-aware chunking and hyphen repair,
- retrieval accuracy and paper selection,
- upload, delete, file-type checks and the 10-paper limit,
- streaming, and the clean `503` when the model is down,
- serving the frontend.

GitHub Actions runs the suite on every push and pull request (`.github/workflows/tests.yml`).

---

## 12. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| **"Can't reach the model."** | The model server isn't running. Check `mlx.log`, or run `mlx_lm.server --model <MLX_MODEL> --port 8080` in a terminal and read its output. With an API provider, check your key in `.env`. |
| **"Can't reach the backend."** | The web app isn't running, or you opened the HTML file directly. Start `./run.sh` and use http://127.0.0.1:8000. |
| **A PDF shows `error`.** | It's a scanned image with no text layer. Run `brew install ocrmypdf`, then `ocrmypdf in.pdf out.pdf`, and upload `out.pdf`. |
| **Red Pen, Quiz or Map says "try again".** | The model didn't return valid JSON. Small local models do this sometimes. Retry, or switch to `LLM_PROVIDER=anthropic`. |
| **Answers ignore my paper.** | Check that the paper's checkbox is ticked in the Library. Then try naming the specific term you're asking about. |
| **`address already in use`** | Something else is on port 8000 or 8080. Stop it, or change the port in `run.sh` (and `MLX_BASE_URL` if you move the model server). |
| **`permission denied: ./run.sh`** | Run `chmod +x run.sh`. |
| **The Mac becomes slow, or the model server dies.** | You're out of memory. See [memory tips](#memory-on-a-16-gb-mac). |
| **Downloads fail during first launch.** | A network or Hugging Face hiccup. Run `./run.sh` again; finished downloads are cached. |
| **The `.docx` export does nothing.** | It loads a small zip library from a CDN, so it needs an internet connection. `.md` and `.txt` work offline. |
| **I want a clean slate.** | Stop the app and delete the `data/` folder. The sample papers come back on the next launch. |

---

## 13. Known limitations

- **Answers depend on retrieval.** If the right passage isn't retrieved, Miss Pedant should say the papers don't cover the question, but a small local model occasionally answers anyway.
- **Structured features read the start of each paper.** Red Pen, Pop Quiz and Concept Map use roughly the first 3,000–3,500 characters of each selected paper, and Synthesize the first 1,400. Long papers are represented by their opening sections, usually the abstract and introduction. Chat questions are not limited this way, because they search every chunk.
- **Citations are inferred.** APA and BibTeX entries are generated from the paper's text and should be checked against the published source.
- **History is per browser.** It's stored in your browser, not on the server.
- **Single user.** There are no accounts. Anyone who can reach port 8000 sees the same library, so keep the server on `127.0.0.1`.
- **No OCR.** Scanned PDFs must be converted first.

---

## 14. Data and privacy

- Uploaded papers, extracted text and the vector index stay in the `data/` folder on your computer.
- With `LLM_PROVIDER=mlx`, nothing leaves your machine after the models are downloaded.
- With an API provider, the retrieved passages and your question are sent to that provider for each request. Their data policies apply.
- Chat history stays in your browser's local storage.
- `data/` and `.env` are gitignored, so pushing the repository never uploads your papers or keys.
