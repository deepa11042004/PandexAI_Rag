"""Chat orchestration: the only place that wires MongoDB persistence together with the existing
RAG pipeline. Implements the STEP 1-9 flow from the spec:

    receive (user_id, chat_id, question)
    -> create chat if missing
    -> store the user message
    -> load prior messages, turn them into LangChain-compatible history
    -> run the (unmodified) retrieval graph + LLM generation, passing that history in
    -> store the assistant response (with tool_used, if the MCP math server answered)
    -> return the answer

MongoDB (via the repositories) is the only source of truth for chat/message history - nothing
here is cached in memory, session state, or a LangGraph checkpointer.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi.concurrency import run_in_threadpool

import rag_pipeline
from config import get_settings
from database.models import ChatDetail, ChatSummary, MessageOut
from database.repository import ChatRepository, MessageRepository, UserRepository, history_pairs_from_messages
from utils.logger import get_logger

logger = get_logger(__name__)

_TITLE_MAX_WORDS = 5
_DEFAULT_TITLE = "New chat"


class InvalidChatIdError(ValueError):
    pass


class ChatNotFoundError(LookupError):
    pass


class EmptyMessageError(ValueError):
    pass


def is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def generate_title(question: str, max_words: int = _TITLE_MAX_WORDS) -> str:
    """First `max_words` words of the question, stripped of trailing punctuation - e.g.
    "What is Artificial Intelligence?" -> "What Is Artificial Intelligence"."""
    words = re.findall(r"\S+", question.strip())
    if not words:
        return _DEFAULT_TITLE
    title = " ".join(words[:max_words]).strip(" ?.!,;:")
    return title.title() if title else _DEFAULT_TITLE


class ChatService:
    def __init__(
        self,
        user_repo: UserRepository,
        chat_repo: ChatRepository,
        message_repo: MessageRepository,
    ) -> None:
        self.users = user_repo
        self.chats = chat_repo
        self.messages = message_repo

    # --- read-side ---------------------------------------------------------------------

    async def list_chats(self, user_id: str) -> list[ChatSummary]:
        await run_in_threadpool(self.users.ensure_user, user_id)
        chats = await run_in_threadpool(self.chats.list_chats_for_user, user_id)
        summaries = []
        for chat in chats:
            count = await run_in_threadpool(self.messages.count_messages, chat.chat_id)
            summaries.append(
                ChatSummary(
                    chat_id=chat.chat_id, user_id=chat.user_id, title=chat.title,
                    created_at=chat.created_at, updated_at=chat.updated_at, message_count=count,
                )
            )
        return summaries

    async def get_chat_detail(self, chat_id: str) -> ChatDetail:
        if not is_valid_uuid(chat_id):
            raise InvalidChatIdError(f"'{chat_id}' is not a valid chat id.")
        chat = await run_in_threadpool(self.chats.get_chat, chat_id)
        if chat is None:
            raise ChatNotFoundError(f"Chat '{chat_id}' does not exist.")
        messages = await run_in_threadpool(self.messages.get_messages, chat_id)
        return ChatDetail(
            chat_id=chat.chat_id, user_id=chat.user_id, title=chat.title,
            created_at=chat.created_at, updated_at=chat.updated_at,
            messages=[
                MessageOut(role=m.role, content=m.content, timestamp=m.timestamp, tool_used=m.tool_used, metadata=m.metadata)
                for m in messages
            ],
        )

    # --- write-side ----------------------------------------------------------------------

    async def rename_chat(self, chat_id: str, title: str) -> ChatSummary:
        if not is_valid_uuid(chat_id):
            raise InvalidChatIdError(f"'{chat_id}' is not a valid chat id.")
        if not title.strip():
            raise EmptyMessageError("Title cannot be empty.")
        chat = await run_in_threadpool(self.chats.set_title, chat_id, title.strip())
        if chat is None:
            raise ChatNotFoundError(f"Chat '{chat_id}' does not exist.")
        count = await run_in_threadpool(self.messages.count_messages, chat_id)
        return ChatSummary(
            chat_id=chat.chat_id, user_id=chat.user_id, title=chat.title,
            created_at=chat.created_at, updated_at=chat.updated_at, message_count=count,
        )

    async def delete_chat(self, chat_id: str) -> None:
        if not is_valid_uuid(chat_id):
            raise InvalidChatIdError(f"'{chat_id}' is not a valid chat id.")
        chat = await run_in_threadpool(self.chats.get_chat, chat_id)
        if chat is None:
            raise ChatNotFoundError(f"Chat '{chat_id}' does not exist.")
        await run_in_threadpool(self.messages.delete_messages, chat_id)
        await run_in_threadpool(self.chats.delete_chat, chat_id)
        # Sources/settings/usage bookkeeping (SQLite) and the Chroma vector-store collection are
        # keyed by the same id and unrelated to chat *history* - still need cleaning up so a
        # deleted chat doesn't leave orphaned uploaded-document embeddings behind.
        await run_in_threadpool(rag_pipeline.delete_session, chat_id)

    async def send_message(
        self, user_id: str, chat_id: str | None, question: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Yields SSE-ready event dicts: one `meta` event first (resolved chat_id/title), then
        `token`/`done`/`error` events forwarded from `rag_pipeline.stream_chat`."""
        if not question or not question.strip():
            raise EmptyMessageError("Message cannot be empty.")
        if not user_id or not user_id.strip():
            raise EmptyMessageError("user_id is required.")
        if chat_id is not None and not is_valid_uuid(chat_id):
            raise InvalidChatIdError(f"'{chat_id}' is not a valid chat id.")

        await run_in_threadpool(self.users.ensure_user, user_id)

        resolved_chat_id = chat_id or str(uuid.uuid4())
        chat = await run_in_threadpool(self.chats.get_chat, resolved_chat_id)
        is_new_chat = chat is None
        if is_new_chat:
            chat = await run_in_threadpool(self.chats.create_chat, resolved_chat_id, user_id, _DEFAULT_TITLE)

        prior_messages = await run_in_threadpool(self.messages.get_messages, resolved_chat_id)
        settings = get_settings()
        history_pairs = history_pairs_from_messages(prior_messages, settings.max_history_turns)

        await run_in_threadpool(
            self.messages.add_message, resolved_chat_id, user_id, "user", question.strip()
        )

        title = chat.title
        if not prior_messages:  # first message in this chat -> auto-title it
            title = generate_title(question)
            await run_in_threadpool(self.chats.set_title, resolved_chat_id, title)
        else:
            await run_in_threadpool(self.chats.touch_updated_at, resolved_chat_id)

        yield {"type": "meta", "chat_id": resolved_chat_id, "title": title, "is_new_chat": is_new_chat}

        collected = ""
        tool_used: str | None = None
        citations: list[str] = []
        try:
            async for event in rag_pipeline.stream_chat(resolved_chat_id, question.strip(), history_pairs):
                if event["type"] == "token":
                    collected += event["data"]
                elif event["type"] == "done":
                    tool_used = event.get("tool_used")
                    citations = event.get("citations", [])
                yield event
        except Exception as exc:  # noqa: BLE001 - persist whatever we generated, then surface the error
            logger.exception("stream_chat failed for chat %s", resolved_chat_id)
            yield {"type": "error", "message": str(exc)}
            collected = collected or f"[Error: {exc}]"

        if collected:
            await run_in_threadpool(
                self.messages.add_message,
                resolved_chat_id, user_id, "assistant", collected,
                tool_used, {"citations": citations} if citations else {},
            )
            await run_in_threadpool(self.chats.touch_updated_at, resolved_chat_id)


__all__ = ["ChatService", "InvalidChatIdError", "ChatNotFoundError", "EmptyMessageError", "is_valid_uuid", "generate_title"]
