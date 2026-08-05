"""Pydantic models for the MongoDB documents (`UserModel`/`ChatModel`/`MessageModel`) and the
request/response schemas used by the chat-history API surface (`backend/api/chat_routes.py`).

Timestamps are stored as ISO-8601 strings (`utils.helpers.now_iso`), matching the convention
already used everywhere else in this codebase (SQLite session rows, ChatMessage, etc.) rather
than introducing a second, BSON-datetime timestamp format.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------------------------
# MongoDB document models
# ----------------------------------------------------------------------------


class UserModel(BaseModel):
    """`users` collection: one document per (locally persisted) user."""

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, alias="_id")
    user_id: str
    name: str = ""
    email: str = ""
    created_at: str


class ChatModel(BaseModel):
    """`chats` collection: one document per conversation, keyed by a UUID `chat_id`."""

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, alias="_id")
    chat_id: str
    user_id: str
    title: str = "New chat"
    created_at: str
    updated_at: str


class MessageModel(BaseModel):
    """`messages` collection: one document per individual chat message (user or assistant)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, alias="_id")
    chat_id: str
    user_id: str
    role: str  # "user" | "assistant"
    content: str
    timestamp: str
    tool_used: str | None = None  # e.g. "calculator_mcp" when the MCP math server answered
    metadata: dict = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# API request/response schemas
# ----------------------------------------------------------------------------


class ChatMessageRequest(BaseModel):
    """POST /chat body. `chat_id` is optional - omit it (or pass one that doesn't exist yet)
    to have a new conversation created automatically."""

    user_id: str = Field(..., min_length=1)
    chat_id: str | None = None
    question: str = Field(..., min_length=1)


class MessageOut(BaseModel):
    role: str
    content: str
    timestamp: str
    tool_used: str | None = None
    metadata: dict = Field(default_factory=dict)


class ChatSummary(BaseModel):
    chat_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ChatDetail(ChatSummary):
    messages: list[MessageOut] = Field(default_factory=list)


class ChatListResponse(BaseModel):
    chats: list[ChatSummary]


class RenameChatRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


__all__ = [
    "UserModel",
    "ChatModel",
    "MessageModel",
    "ChatMessageRequest",
    "MessageOut",
    "ChatSummary",
    "ChatDetail",
    "ChatListResponse",
    "RenameChatRequest",
]
