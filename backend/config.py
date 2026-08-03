"""Centralized, environment-driven configuration for the backend.

Every tunable lives here so the rest of the codebase never reads `os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider selection -------------------------------------------------
    # "groq" is primary by default (free tier, no OpenAI account needed). If its call fails
    # (quota/auth/connection) or no key is set, the app automatically falls back through
    # "openrouter" then "openai" (whichever have keys set).
    llm_provider: str = "groq"

    # OpenAI is now OPTIONAL - only needed if you set EMBEDDING_PROVIDER=openai below, or if you
    # want it available as a chat fallback. NOTE: there is no model literally named "gpt-5.5" -
    # set this to whatever real model your account has access to (e.g. gpt-4.1, gpt-4o).
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4.1"
    openai_embedding_model: str = "text-embedding-3-small"

    # OpenRouter (OpenAI-compatible endpoint, many free-tier models available).
    openrouter_api_key: str = ""
    openrouter_chat_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Groq (OpenAI-compatible endpoint, fast free-tier inference, no credit card required).
    # Get a free key at https://console.groq.com/keys
    groq_api_key: str = ""
    groq_chat_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Claude (Anthropic) - not OpenAI-compatible, uses its own SDK/client (see llm_provider.build_llm).
    # Get a key at https://console.anthropic.com/settings/keys
    anthropic_api_key: str = ""
    anthropic_chat_model: str = "claude-sonnet-5"

    # Gemini (Google) - not OpenAI-compatible, uses its own SDK/client (see llm_provider.build_llm).
    # Get a free key at https://aistudio.google.com/apikey
    google_api_key: str = ""
    google_chat_model: str = "gemini-2.0-flash"

    # --- Speech-to-text (voice input) -------------------------------------------------
    # Groq's Whisper endpoint is free-tier and OpenAI-compatible, so it reuses GROQ_API_KEY/
    # groq_base_url above - no separate key needed. Falls back to OpenAI's Whisper (openai_api_key)
    # if Groq has no key or the request fails. OpenRouter has no audio-transcription endpoint.
    # "large-v3" (not "-turbo") - turbo trades accuracy for speed, and a few seconds of extra
    # latency on a short voice clip is a much better trade than garbled transcriptions.
    groq_whisper_model: str = "whisper-large-v3"
    openai_whisper_model: str = "whisper-1"
    # Without a language hint, Whisper auto-detects language from the clip - on short recordings
    # that's a common cause of garbled/wrong-language output. Set blank to restore auto-detect
    # (needed for non-English speakers).
    whisper_language: str = "en"

    # --- Embeddings ------------------------------------------------------------------
    # "local" (default): a small HuggingFace sentence-transformers model runs on your own CPU.
    # No account, no API key, no cost - the first run downloads the model (~80MB) once.
    # "openai": use OpenAI's hosted embedding API instead (requires OPENAI_API_KEY).
    embedding_provider: str = "local"
    # Multilingual (not just English all-MiniLM-L6-v2) because sources aren't guaranteed to be
    # English - e.g. a YouTube video whose only available transcript is auto-generated in another
    # language (see backend/loaders/youtube_loader.py's non-English fallback). An English-only
    # model puts English queries and non-English documents far apart in embedding space, which
    # silently fails retrieval (everything scores above the relevance cutoff) even though the
    # right content is right there.
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # --- Storage -----------------------------------------------------------------
    chroma_persist_dir: str = "chroma_db"
    uploads_dir: str = "uploads"
    sessions_dir: str = ".sessions"

    # --- RAG tuning ----------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # Wider initial candidate pool (vector search is cheap - no LLM cost) so a broad question like
    # "summarize each document" has a real chance of surfacing at least one chunk per uploaded
    # source before reranking/diversification narrows it down.
    retrieval_top_k: int = 20
    rerank_top_k: int = 8
    max_distance: float = 0.9  # chroma cosine distance cutoff; above this a chunk is "irrelevant"
    max_history_turns: int = 4

    # --- Server --------------------------------------------------------------------
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_url: str = "http://localhost:8000"  # used by the frontend to reach the backend
    cors_origins: str = "*"

    # --- OCR -------------------------------------------------------------------------
    tesseract_cmd: str = ""  # optional explicit path to tesseract.exe on Windows

    # --- MCP math server -----------------------------------------------------------
    mcp_math_server_command: str = ""
    mcp_math_server_args: str = ""
    mcp_math_timeout_seconds: float = 10.0

    # --- Misc --------------------------------------------------------------------
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
