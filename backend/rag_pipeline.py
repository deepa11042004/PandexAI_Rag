"""Orchestrates the end-to-end RAG pipeline: ingestion (extract -> chunk -> embed -> store) and
chat (LangGraph retrieval -> streamed, provider-fallback generation), wiring together the vector
store, session state, and LangGraph defined elsewhere in this package.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from config import get_settings
from graph import NOT_FOUND_MESSAGE, run_retrieval_graph
from llm_provider import astream_with_fallback, get_embeddings_client
from loaders import base as loaders
from loaders.website_loader import WebsiteExtractionError
from loaders.youtube_loader import YoutubeExtractionError
from mcp_math_client import classify_query, evaluate_math_query
from session_manager import SessionManager, delete_session_file, get_session
from vector_store import VectorStoreManager, delete_session_collection
from utils.helpers import content_hash, count_tokens, estimate_cost, new_id, now_iso
from utils.logger import get_logger

logger = get_logger(__name__)

# Short greetings/small-talk shouldn't hit the "I couldn't find this in the uploaded documents"
# wall just because no source has been added yet - a document assistant should still be able to
# say hello. A word-set check (rather than a fixed phrase regex) naturally handles combinations
# like "hi how are you" or "hey thanks bye" without needing to enumerate every phrasing, while
# still rejecting real questions ("what is the capital of France") whose words aren't all in the
# small-talk vocabulary.
_SMALL_TALK_WORDS = {
    "hi", "hii", "hiii", "hiiii", "hello", "helo", "hey", "heyy", "hiya", "yo", "sup", "howdy",
    "good", "morning", "afternoon", "evening", "night",
    "how", "are", "r", "you", "u", "doing", "going", "it", "whats", "what's", "up",
    "thanks", "thank", "ty", "very", "much", "appreciated", "a", "lot",
    "bye", "goodbye", "see", "later", "cya",
    "who", "your", "name", "can", "what", "do", "yourself", "there",
    "ok", "okay", "yes", "yeah", "yep", "no", "nope", "please", "cool", "nice", "great", "awesome",
}
_MAX_SMALL_TALK_WORDS = 8


def _is_small_talk(question: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", question.lower())
    if not words or len(words) > _MAX_SMALL_TALK_WORDS:
        return False
    return all(word in _SMALL_TALK_WORDS for word in words)

_SMALL_TALK_SYSTEM_PROMPT = (
    "You are the assistant for a document Q&A app. The user just sent a greeting or small-talk "
    "message, not a question about their documents. Reply warmly and briefly (one short sentence "
    "or two), and if it fits naturally, mention you're ready to answer questions once they upload "
    "a document, website, or YouTube link. Do not mention 'context' or apologize for lacking "
    "information - this isn't a document question."
)

_vector_stores: dict[str, VectorStoreManager] = {}
_vector_stores_guard = threading.Lock()


def get_vector_store(session_id: str) -> VectorStoreManager:
    with _vector_stores_guard:
        if session_id not in _vector_stores:
            _vector_stores[session_id] = VectorStoreManager(session_id, get_embeddings_client())
        return _vector_stores[session_id]


@dataclass
class IngestOutcome:
    success: bool
    source: dict | None = None
    message: str = ""
    duplicate: bool = False


def _register_source(
    session: SessionManager, manager: VectorStoreManager, result: loaders.LoadResult, hash_: str, size_bytes: int
) -> dict:
    source_id = new_id()
    manager.add_documents(result.chunks, source_id=source_id, content_hash=hash_)
    source = {
        "id": source_id,
        "name": result.name,
        "type": result.source_type,
        "chunks": len(result.chunks),
        "size_bytes": size_bytes,
        "content_hash": hash_,
        "added_at": now_iso(),
    }
    session.add_source(source)
    return source


def ingest_file(session_id: str, filename: str, raw_bytes: bytes) -> IngestOutcome:
    session = get_session(session_id)
    manager = get_vector_store(session_id)

    hash_ = content_hash(raw_bytes)
    if hash_ in session.known_content_hashes():
        return IngestOutcome(success=False, message=f"'{filename}' is a duplicate of an existing source.", duplicate=True)

    try:
        result = loaders.load_file(filename, raw_bytes)
    except loaders.UnsupportedFileTypeError as exc:
        return IngestOutcome(success=False, message=str(exc))
    except Exception as exc:  # noqa: BLE001 - isolate any parser failure to this one file
        logger.exception("Failed to parse '%s'", filename)
        return IngestOutcome(success=False, message=f"Failed to process '{filename}': {exc}")

    if not result.success:
        return IngestOutcome(success=False, message=result.error or "No extractable text found.")

    source = _register_source(session, manager, result, hash_, size_bytes=len(raw_bytes))
    return IngestOutcome(success=True, source=source, message=f"Processed '{filename}' ({source['chunks']} chunks).")


def ingest_website(session_id: str, url: str) -> IngestOutcome:
    session = get_session(session_id)
    manager = get_vector_store(session_id)

    try:
        result = loaders.load_website(url)
    except WebsiteExtractionError as exc:
        return IngestOutcome(success=False, message=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to scrape '%s'", url)
        return IngestOutcome(success=False, message=f"Failed to scrape '{url}': {exc}")

    if not result.success:
        return IngestOutcome(success=False, message=result.error or "No extractable text found on that page.")

    hash_ = content_hash(url.strip().lower().encode())
    if hash_ in session.known_content_hashes():
        return IngestOutcome(success=False, message=f"'{url}' looks like a duplicate of an existing source.", duplicate=True)

    source = _register_source(session, manager, result, hash_, size_bytes=0)
    return IngestOutcome(success=True, source=source, message=f"Scraped '{result.name}' ({source['chunks']} chunks).")


def ingest_youtube(session_id: str, url: str) -> IngestOutcome:
    session = get_session(session_id)
    manager = get_vector_store(session_id)

    try:
        result = loaders.load_youtube(url)
    except YoutubeExtractionError as exc:
        return IngestOutcome(success=False, message=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch transcript for '%s'", url)
        return IngestOutcome(success=False, message=f"Failed to fetch transcript: {exc}")

    if not result.success:
        return IngestOutcome(success=False, message=result.error or "No transcript text found.")

    hash_ = content_hash(url.strip().encode())
    if hash_ in session.known_content_hashes():
        return IngestOutcome(success=False, message="This video is a duplicate of an existing source.", duplicate=True)

    source = _register_source(session, manager, result, hash_, size_bytes=0)
    return IngestOutcome(success=True, source=source, message=f"Transcribed '{result.name}' ({source['chunks']} chunks).")


def remove_source(session_id: str, source_id: str) -> None:
    get_vector_store(session_id).delete_source(source_id)
    get_session(session_id).remove_source(source_id)


def clear_sources(session_id: str) -> None:
    get_vector_store(session_id).clear()
    get_session(session_id).clear_sources()


def delete_session(session_id: str) -> None:
    """Permanently remove a whole conversation: its vector store collection and its session file,
    used by the chat-history page's Delete action (as opposed to `clear_sources`/`clear_chat`,
    which just empty parts of a session that's still in use)."""
    with _vector_stores_guard:
        _vector_stores.pop(session_id, None)
    delete_session_collection(session_id)
    delete_session_file(session_id)


async def _stream_and_record(
    session: SessionManager,
    messages: list[BaseMessage],
    preferred_provider: str | None,
    model_override: str | None,
    citations: list[str],
) -> AsyncIterator[dict[str, Any]]:
    """Shared tail: stream tokens from the fallback-ordered LLM, then emit the final event.

    Persisting the resulting message is the caller's job now (`services.chat_service`, backed by
    MongoDB) - this function only tracks token/cost usage, which stays in the SQLite session store
    since it isn't "chat history".
    """
    collected = ""
    provider_used = ""
    model_used = ""
    try:
        async for token, provider_name, model_name in astream_with_fallback(
            messages, preferred_provider, model_override=model_override
        ):
            collected += token
            provider_used, model_used = provider_name, model_name
            yield {"type": "token", "data": token}
    except Exception as exc:  # noqa: BLE001 - every provider failed
        logger.exception("All LLM providers failed for session %s", session.session_id)
        yield {"type": "error", "message": f"All configured LLM providers failed: {exc}"}
        return

    input_tokens = sum(count_tokens(m.content, model_used or "gpt-4o") for m in messages)
    output_tokens = count_tokens(collected, model_used or "gpt-4o")
    cost = estimate_cost(model_used or "", input_tokens, output_tokens)
    session.add_usage(input_tokens, output_tokens, cost, provider_used, model_used)

    yield {
        "type": "done",
        "citations": citations,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "estimated_cost_usd": cost},
        "provider": provider_used,
        "model": model_used,
        "tool_used": None,
    }


async def stream_chat(
    chat_id: str, question: str, history_pairs: list[tuple[str, str]]
) -> AsyncIterator[dict[str, Any]]:
    """Run the LangGraph retrieval pipeline, then stream the grounded answer token-by-token.

    `history_pairs` (most recent last) is the persistent chat history for this conversation,
    supplied by the caller (`services.chat_service`, reading from MongoDB) - this function itself
    holds no chat history in memory across calls.

    Yields dict events: {"type": "token", "data": str} while generating, then a final
    {"type": "done", "citations": [...], "usage": {...}, "provider": str, "model": str,
    "tool_used": str | None} event, or {"type": "error", "message": str} if something goes wrong.
    """
    settings = get_settings()
    session = get_session(chat_id)
    manager = get_vector_store(chat_id)

    preferred_provider = session.settings_overrides.get("llm_provider") or settings.llm_provider
    model_override = session.settings_overrides.get("chat_model")

    if _is_small_talk(question):
        messages: list[BaseMessage] = [SystemMessage(content=_SMALL_TALK_SYSTEM_PROMPT), HumanMessage(content=question)]
        async for event in _stream_and_record(session, messages, preferred_provider, model_override, citations=[]):
            yield event
        return

    mcp_math_enabled = session.settings_overrides.get("mcp_math_enabled", True)
    route = classify_query(question) if mcp_math_enabled else "rag"
    if route in {"math", "mixed"}:
        if route == "math":
            answer = await evaluate_math_query(question)
            if answer is not None:
                yield {"type": "token", "data": answer}
                yield {"type": "done", "citations": [], "usage": {}, "provider": "", "model": "", "tool_used": "calculator_mcp"}
                return
        else:
            rerank_top_k_override = session.settings_overrides.get("rerank_top_k")
            source_ids = [s["id"] for s in session.sources]

            try:
                state = await run_retrieval_graph(
                    question, history_pairs, manager, preferred_provider, rerank_top_k_override, source_ids
                )
            except Exception as exc:  # noqa: BLE001 - surface retrieval failures cleanly to the client
                logger.exception("Retrieval graph failed for chat %s", chat_id)
                yield {"type": "error", "message": f"Retrieval failed: {exc}"}
                return

            if state["has_context"]:
                answer = await evaluate_math_query(question, state["context"])
                if answer is not None:
                    yield {"type": "token", "data": answer}
                    yield {
                        "type": "done", "citations": state["citations"], "usage": {},
                        "provider": "", "model": "", "tool_used": "calculator_mcp",
                    }
                    return

    if manager.count() == 0:
        answer = NOT_FOUND_MESSAGE
        yield {"type": "token", "data": answer}
        yield {"type": "done", "citations": [], "usage": {}, "provider": "", "model": "", "tool_used": None}
        return

    rerank_top_k_override = session.settings_overrides.get("rerank_top_k")
    source_ids = [s["id"] for s in session.sources]

    try:
        state = await run_retrieval_graph(
            question, history_pairs, manager, preferred_provider, rerank_top_k_override, source_ids
        )
    except Exception as exc:  # noqa: BLE001 - surface retrieval failures cleanly to the client
        logger.exception("Retrieval graph failed for chat %s", chat_id)
        yield {"type": "error", "message": f"Retrieval failed: {exc}"}
        return

    if not state["has_context"]:
        answer = NOT_FOUND_MESSAGE
        yield {"type": "token", "data": answer}
        yield {"type": "done", "citations": [], "usage": {}, "provider": "", "model": "", "tool_used": None}
        return

    async for event in _stream_and_record(session, state["messages"], preferred_provider, model_override, state["citations"]):
        yield event
