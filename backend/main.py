"""FastAPI backend entrypoint: async REST + SSE-streaming endpoints for the RAG pipeline.

Run with (from inside this backend/ folder): uvicorn main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import history_store
import rag_pipeline
from api.chat_routes import router as chat_router
from config import get_settings
from database.mongodb import close_mongo_connection, connect_to_mongo
from llm_provider import fallback_order, transcribe_audio, validate_api_key
from models import (
    AddUrlRequest,
    AddYoutubeRequest,
    IngestResponse,
    SessionSettings,
    SourceInfo,
    SourceListResponse,
    TokenUsage,
    TranscriptionResponse,
)
from session_manager import get_session
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(title="RAG Document Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.on_event("startup")
async def on_startup() -> None:
    Path(settings.uploads_dir).mkdir(exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(exist_ok=True)
    Path(settings.sessions_dir).mkdir(exist_ok=True)

    await run_in_threadpool(history_store.init_db)
    await run_in_threadpool(history_store.migrate_legacy_json_sessions)

    # MongoDB is the only source of truth for chat/message history - fail fast at startup rather
    # than surfacing a confusing error on the user's first chat message.
    await run_in_threadpool(connect_to_mongo)

    logger.info("Backend started. Provider order: %s", settings.llm_provider)

    if settings.embedding_provider == "local":
        # Load (and on first-ever run, download) the local embedding model now, in a thread so it
        # doesn't block the event loop, rather than paying that cost on a user's first message.
        from llm_provider import get_embeddings_client

        logger.info("Warming local embedding model...")
        await run_in_threadpool(lambda: get_embeddings_client().embed_query("warmup"))
        logger.info("Local embedding model ready.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await run_in_threadpool(close_mongo_connection)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/validate-key")
async def validate_key(payload: dict) -> dict:
    is_valid, message = await run_in_threadpool(validate_api_key, payload.get("api_key", ""))
    return {"valid": is_valid, "message": message}


@app.get("/providers")
async def providers() -> dict:
    """Which LLM providers have an API key configured server-side (never returns the keys)."""
    configured = {c.name for c in fallback_order(None)}
    return {
        "openai": "openai" in configured,
        "openrouter": "openrouter" in configured,
        "groq": "groq" in configured,
        "anthropic": "anthropic" in configured,
        "google": "google" in configured,
        "default_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "embeddings_ready": settings.embedding_provider == "local" or bool(settings.openai_api_key),
    }


# ----------------------------------------------------------------------------
# Sources: upload / website / youtube / list / delete
# ----------------------------------------------------------------------------
@app.post("/sources/{session_id}/upload", response_model=list[IngestResponse])
async def upload_sources(session_id: str, files: list[UploadFile] = File(...)) -> list[IngestResponse]:
    responses = []
    for upload in files:
        raw_bytes = await upload.read()
        outcome = await run_in_threadpool(rag_pipeline.ingest_file, session_id, upload.filename, raw_bytes)
        responses.append(
            IngestResponse(
                success=outcome.success,
                source=SourceInfo(**outcome.source) if outcome.source else None,
                message=outcome.message,
                duplicate=outcome.duplicate,
            )
        )
    return responses


@app.post("/sources/{session_id}/website", response_model=IngestResponse)
async def add_website(session_id: str, payload: AddUrlRequest) -> IngestResponse:
    outcome = await run_in_threadpool(rag_pipeline.ingest_website, session_id, payload.url)
    return IngestResponse(
        success=outcome.success,
        source=SourceInfo(**outcome.source) if outcome.source else None,
        message=outcome.message,
        duplicate=outcome.duplicate,
    )


@app.post("/sources/{session_id}/youtube", response_model=IngestResponse)
async def add_youtube(session_id: str, payload: AddYoutubeRequest) -> IngestResponse:
    outcome = await run_in_threadpool(rag_pipeline.ingest_youtube, session_id, payload.url)
    return IngestResponse(
        success=outcome.success,
        source=SourceInfo(**outcome.source) if outcome.source else None,
        message=outcome.message,
        duplicate=outcome.duplicate,
    )


@app.get("/sources/{session_id}", response_model=SourceListResponse)
async def list_sources(session_id: str) -> SourceListResponse:
    sources = get_session(session_id).sources
    return SourceListResponse(sources=[SourceInfo(**s) for s in sources], count=len(sources))


@app.delete("/sources/{session_id}/{source_id}")
async def delete_source(session_id: str, source_id: str) -> dict:
    await run_in_threadpool(rag_pipeline.remove_source, session_id, source_id)
    return {"success": True}


@app.delete("/sources/{session_id}")
async def delete_all_sources(session_id: str) -> dict:
    await run_in_threadpool(rag_pipeline.clear_sources, session_id)
    return {"success": True}


# Chat endpoints (POST /chat, GET/PATCH/DELETE /chat/{chat_id}, GET /chats/{user_id}) are mounted
# via `api.chat_routes.router` above - MongoDB-backed, see services/chat_service.py.


def _wav_loudness(raw_bytes: bytes) -> str:
    """RMS/peak amplitude of a 16-bit WAV clip, as a fraction of full scale - a quick, dependency-free
    way to tell "silent/near-silent recording" (mic muted or wrong input device) apart from "audio is
    fine, the model just mis-transcribed it" when voice input produces garbage text."""
    import audioop
    import io
    import wave

    try:
        with wave.open(io.BytesIO(raw_bytes)) as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sampwidth = wav_file.getsampwidth()
        if not frames:
            return "empty (0 frames)"
        rms = audioop.rms(frames, sampwidth)
        peak = audioop.max(frames, sampwidth)
        full_scale = 2 ** (8 * sampwidth - 1)
        return f"rms={rms} ({rms / full_scale:.1%} FS), peak={peak} ({peak / full_scale:.1%} FS)"
    except Exception as exc:  # noqa: BLE001 - diagnostic only, never break transcription over it
        return f"could not analyze ({exc})"


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    """Speech-to-text for the chat input's voice recorder (see Groq/OpenAI fallback in llm_provider)."""
    raw_bytes = await file.read()
    logger.info(
        "Transcribe request: %d bytes (%s), loudness: %s",
        len(raw_bytes), file.content_type, _wav_loudness(raw_bytes),
    )
    try:
        text = await run_in_threadpool(transcribe_audio, raw_bytes, file.filename or "audio.wav")
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a normal error toast
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    logger.info("Transcribe result: %r", text)
    return TranscriptionResponse(text=text)


# ----------------------------------------------------------------------------
# Usage & settings
# ----------------------------------------------------------------------------
@app.get("/usage/{session_id}", response_model=TokenUsage)
async def usage(session_id: str) -> TokenUsage:
    data = get_session(session_id).usage
    return TokenUsage(
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
        estimated_cost_usd=data["estimated_cost_usd"],
        provider=data.get("last_provider", ""),
        model=data.get("last_model", ""),
    )


@app.get("/settings/{session_id}", response_model=SessionSettings)
async def get_session_settings(session_id: str) -> SessionSettings:
    return SessionSettings(**get_session(session_id).settings_overrides)


@app.post("/settings/{session_id}", response_model=SessionSettings)
async def update_session_settings(session_id: str, payload: SessionSettings) -> SessionSettings:
    session = get_session(session_id)
    session.update_settings(payload.model_dump(exclude_none=True))
    return SessionSettings(**session.settings_overrides)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):  # noqa: ANN001 - starlette signature
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})
