# RAG Document Assistant

A production-ready, NotebookLM/ChatGPT-inspired document chatbot: a **Streamlit** frontend
talking to a **FastAPI** backend that runs a **LangGraph**-orchestrated RAG pipeline over a
**ChromaDB** vector store. Upload files, add a website URL, or drop in a YouTube link, then ask
questions - every answer is grounded strictly in what you provided, with citations.

## Architecture

```
Browser <-> Streamlit (frontend/streamlit_app.py) <-- HTTP/SSE --> FastAPI (backend/main.py)
                                                                          |
                                                        LangGraph (backend/graph.py)
                                                        rewrite -> retrieve -> rerank -> prompt
                                                                          |
                                                    ChromaDB (backend/vector_store.py, per-session)
                                                                          |
                                  Local (free) embeddings + Groq/OpenRouter/OpenAI chat (fallback)
```

The frontend holds **no** LLM/vector-store logic - it only renders UI and calls the backend over
HTTP (ingestion, source management, settings) and SSE (streamed chat tokens). All API keys live
server-side in the backend's `.env`; the browser never sees them.

## Features

- Premium dark/light glassmorphism UI with streaming "typing" responses
- 11 source types: PDF, DOCX, TXT, CSV, XLSX, PPTX, PNG/JPG/JPEG/WEBP (OCR), YouTube URLs,
  website URLs, Markdown, JSON
- Drag & drop multi-file upload, dedicated "Add Website" / "Add YouTube" inputs
- LangGraph pipeline: follow-up question rewriting -> vector search -> hybrid lexical+vector
  rerank -> grounded generation, with a strict "not found in documents" fallback
- Free by default: local (keyless) embeddings + Groq's free chat tier, with automatic fallback
  Groq -> OpenRouter -> OpenAI (whichever have keys configured)
- Duplicate-source detection (content hash), per-source and bulk delete, source count indicator
- Conversation memory, chat history panel, token usage & cost tracking, session persistence
- Export chat to PDF / TXT / JSON
- Structured logging (`logs/app.log`) and centralized error handling

## Project Structure

Backend and frontend are two fully independent, self-contained projects - each has its own
virtual environment, its own `requirements.txt`, and its own copy of the small shared `utils/`
helpers (no cross-folder imports). Run/deploy either one on its own.

```
project/
├── frontend/
│   ├── streamlit_app.py        # UI only - talks to the backend over HTTP/SSE
│   ├── pages/                   # Additional Streamlit pages (e.g. Chat History)
│   ├── assets/                  # Mascot image + style.css (frontend-only)
│   ├── .streamlit/config.toml   # Theme + server config
│   ├── utils/                    # export_chat.py, helpers.py (frontend's own copy)
│   ├── requirements.txt
│   └── Dockerfile
├── backend/
│   ├── main.py                 # FastAPI app (async endpoints, SSE chat stream)
│   ├── rag_pipeline.py         # Orchestrates ingestion + chat end-to-end
│   ├── graph.py                # LangGraph: rewrite -> retrieve -> rerank -> prompt
│   ├── retriever.py            # Vector search + hybrid lexical rerank
│   ├── vector_store.py         # ChromaDB per-session collection wrapper
│   ├── llm_provider.py         # Local embeddings + Groq/OpenRouter/OpenAI chat with fallback
│   ├── mcp_math_client.py       # Routes math questions to an external MCP math server
│   ├── session_manager.py      # Per-session JSON persistence (sources/chat/usage/settings)
│   ├── config.py                # Env-driven settings (pydantic-settings)
│   ├── models.py                # Pydantic request/response schemas
│   ├── loaders/                 # One module per source type + shared chunking logic
│   ├── utils/                    # logger.py, helpers.py (backend's own copy)
│   ├── tests/                    # pytest suite (run with `python -m pytest` from here)
│   ├── chroma_db/ uploads/ .sessions/ logs/   # gitignored, auto-created at runtime
│   ├── .env.example / .env
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Setup (free, no OpenAI account required)

### 1. Get a free Groq API key

Go to https://console.groq.com/keys, sign up (no credit card), and create a key. This powers
chat. Embeddings are local and need **no key at all** (see step 3).

### 2. Install Tesseract OCR (only if you plan to upload images)

- **Windows**: install from https://github.com/UB-Mannheim/tesseract/wiki, then either add it to
  PATH or set `TESSERACT_CMD` in `.env` to the full `tesseract.exe` path.
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

(Skip this if you don't need image ingestion - every other source type works without it.)

### 3. Set up the backend (its own venv + dependencies)

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

This installs `sentence-transformers` (local embeddings) which pulls in PyTorch - a larger,
slower install than a pure API-only setup, but it's what makes embeddings free and keyless.

### 4. Configure environment variables (backend only - the frontend needs no API keys)

```powershell
copy .env.example .env
```

Edit `.env` and just set:
```
GROQ_API_KEY=gsk_your_key_here
```
Everything else already defaults to the free setup: `EMBEDDING_PROVIDER=local` (no key) and
`LLM_PROVIDER=groq`. OpenAI/OpenRouter stay fully optional - only set those if you want them as
an additional fallback or prefer their models.

The local embedding model (~80MB) downloads automatically the first time the backend embeds a
document - that first upload will take longer than subsequent ones.

### 5. Set up the frontend (its own venv + dependencies)

In a separate terminal:
```powershell
cd frontend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 6. Run both processes

Terminal 1 (backend - from inside `backend/`, venv activated):
```powershell
uvicorn main:app --reload --port 8000
```

Terminal 2 (frontend - from inside `frontend/`, venv activated):
```powershell
streamlit run streamlit_app.py
```

Open the Streamlit URL (default `http://localhost:8501`). The sidebar shows backend connection
status and which providers are configured - no API key entry in the browser. If the backend runs
somewhere other than `http://localhost:8000`, set `BACKEND_URL` as an environment variable before
starting the frontend.

### Docker (alternative to steps 3-6)

```bash
docker compose up --build
```
Frontend on `http://localhost:8501`, backend on `http://localhost:8000`. Backend reads
`backend/.env`; the frontend needs no secrets.

## Notes

- **"GPT-5.5" doesn't exist** as a real OpenAI model name; `OPENAI_CHAT_MODEL` is fully
  configurable in `.env` so you can point it at whatever real model you have access to.
- **Embeddings are local by default** (`backend/llm_provider.py`, `sentence-transformers/all-MiniLM-L6-v2`)
  - free and keyless, but noticeably lower-quality than OpenAI's hosted embeddings and it pulls
  in PyTorch. Set `EMBEDDING_PROVIDER=openai` (+ `OPENAI_API_KEY`) if you want better retrieval
  quality and don't mind the cost.
- **Re-ranking is heuristic, not a second ML model** (0.75×vector-distance + 0.25×lexical-overlap,
  see `backend/retriever.py`) - since embeddings already pull in PyTorch, a cross-encoder reranker
  would work fine here too if you want to add one later.
- Each browser tab gets its own session, with its own ChromaDB collection and persisted
  chat/source/usage state under `.sessions/`. Session identity lives only in Streamlit's
  `session_state`, not the URL, so refreshing the page always starts a brand-new, empty
  conversation - to resume a past one, open it from the Chat History page instead.
- Streamlit's native chrome follows `.streamlit/config.toml`'s dark base theme; the in-app
  light/dark toggle re-themes all custom chat/sidebar UI at runtime.
