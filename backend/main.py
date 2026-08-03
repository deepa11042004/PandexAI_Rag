"""FastAPI backend entrypoint: async REST + SSE-streaming endpoints for the RAG pipeline.

Run with (from inside this backend/ folder): uvicorn main:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import history_store
import rag_pipeline
from config import get_settings
from llm_provider import fallback_order, transcribe_audio, validate_api_key
from models import (
    AddUrlRequest,
    AddYoutubeRequest,
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ConversationDetail,
    ConversationListResponse,
    ConversationSummary,
    ConversationUpdateRequest,
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


@app.on_event("startup")
async def on_startup() -> None:
    Path(settings.uploads_dir).mkdir(exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(exist_ok=True)
    Path(settings.sessions_dir).mkdir(exist_ok=True)

    await run_in_threadpool(history_store.init_db)
    await run_in_threadpool(history_store.migrate_legacy_json_sessions)

    logger.info("Backend started. Provider order: %s", settings.llm_provider)

    if settings.embedding_provider == "local":
        # Load (and on first-ever run, download) the local embedding model now, in a thread so it
        # doesn't block the event loop, rather than paying that cost on a user's first message.
        from llm_provider import get_embeddings_client

        logger.info("Warming local embedding model...")
        await run_in_threadpool(lambda: get_embeddings_client().embed_query("warmup"))
        logger.info("Local embedding model ready.")


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
# Chat history page: search/sort/filter/paginate past conversations, view one in full,
# create/pin/delete. Metadata-only list (no message bodies) for a cheap list call; the full
# conversation (including messages) is only fetched on GET /api/chat-history/{id} ("View").
# ----------------------------------------------------------------------------
@app.get("/api/chat-history", response_model=ConversationListResponse)
async def list_chat_history(
    search: str = "",
    sort: str = "updated",
    model: str = "",
    date_from: str = "",
    date_to: str = "",
    min_tokens: int = 0,
    max_tokens: int = 0,
    page: int = 1,
    page_size: int = 9,
) -> ConversationListResponse:
    result = await run_in_threadpool(
        history_store.list_conversations,
        search, sort, model, date_from, date_to, min_tokens, max_tokens, page, page_size,
    )
    return ConversationListResponse(items=[ConversationSummary(**s) for s in result["items"]], **{
        k: v for k, v in result.items() if k != "items"
    })


@app.get("/api/chat-history/{session_id}", response_model=ConversationDetail)
async def get_chat_history_item(session_id: str) -> ConversationDetail:
    conversation = await run_in_threadpool(history_store.get_conversation, session_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail(**conversation)


@app.post("/api/chat-history", response_model=ConversationSummary)
async def create_chat_history_item() -> ConversationSummary:
    conversation = await run_in_threadpool(history_store.create_conversation)
    return ConversationSummary(**conversation)


@app.put("/api/chat-history/{session_id}", response_model=ConversationSummary)
async def update_chat_history_item(session_id: str, payload: ConversationUpdateRequest) -> ConversationSummary:
    conversation = await run_in_threadpool(history_store.update_conversation, session_id, payload.pinned)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationSummary(**conversation)


@app.delete("/api/chat-history/{session_id}")
async def delete_chat_history_item(session_id: str) -> dict:
    await run_in_threadpool(rag_pipeline.delete_session, session_id)
    return {"success": True}


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


# ----------------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------------
@app.get("/chat/{session_id}/history", response_model=ChatHistoryResponse)
async def chat_history(session_id: str) -> ChatHistoryResponse:
    messages = get_session(session_id).chat_history
    return ChatHistoryResponse(session_id=session_id, messages=[ChatMessage(**m) for m in messages])


@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str) -> dict:
    get_session(session_id).clear_chat()
    return {"success": True}


@app.post("/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    """Server-Sent Events stream: one `data: {...}` line per token, then a final `done` event."""

    async def event_stream():
        try:
            async for event in rag_pipeline.stream_chat(payload.session_id, payload.message):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001 - never let an unhandled error hang the stream
            logger.exception("Unhandled error streaming chat for session %s", payload.session_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
