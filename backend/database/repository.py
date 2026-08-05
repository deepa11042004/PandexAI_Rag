"""Repository pattern: the only place in the codebase that talks pymongo query syntax.

Each repository wraps one collection and returns plain dicts / Pydantic models - callers
(`services.chat_service`) never see a pymongo cursor or an `ObjectId` directly. Kept synchronous
(matching `pymongo`, per the project's requirement) - the async FastAPI layer calls these via
`run_in_threadpool`, the same pattern already used for the existing SQLite `history_store` module.
"""

from __future__ import annotations

from typing import Any

from pymongo.database import Database
from pymongo.errors import PyMongoError

from database.models import ChatModel, MessageModel, UserModel
from utils.helpers import now_iso
from utils.logger import get_logger

logger = get_logger(__name__)


def _stringify_id(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Mongo returns `_id` as an `ObjectId`; Pydantic models want a plain string."""
    if doc is None:
        return None
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


class RepositoryError(RuntimeError):
    """Wraps a pymongo failure (e.g. connection drop mid-query) into a domain-level error."""


def _guard(operation: str):
    """Decorator: turn any PyMongoError into a RepositoryError with context, so the service/API
    layers can catch one exception type instead of depending on pymongo internals."""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except PyMongoError as exc:
                logger.exception("MongoDB operation failed: %s", operation)
                raise RepositoryError(f"MongoDB operation failed ({operation}): {exc}") from exc

        return wrapper

    return decorator


class UserRepository:
    def __init__(self, db: Database) -> None:
        self._collection = db["users"]

    @_guard("ensure_user")
    def ensure_user(self, user_id: str, name: str = "", email: str = "") -> UserModel:
        """Idempotent get-or-create: returns the existing user, or creates one on first sight."""
        existing = self._collection.find_one({"user_id": user_id})
        if existing:
            return UserModel.model_validate(_stringify_id(existing))

        doc = {"user_id": user_id, "name": name, "email": email, "created_at": now_iso()}
        result = self._collection.insert_one(doc)
        doc["_id"] = result.inserted_id
        return UserModel.model_validate(_stringify_id(doc))

    @_guard("get_user")
    def get_user(self, user_id: str) -> UserModel | None:
        doc = self._collection.find_one({"user_id": user_id})
        return UserModel.model_validate(_stringify_id(doc)) if doc else None


class ChatRepository:
    def __init__(self, db: Database) -> None:
        self._collection = db["chats"]

    @_guard("create_chat")
    def create_chat(self, chat_id: str, user_id: str, title: str = "New chat") -> ChatModel:
        timestamp = now_iso()
        doc = {
            "chat_id": chat_id,
            "user_id": user_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        result = self._collection.insert_one(doc)
        doc["_id"] = result.inserted_id
        return ChatModel.model_validate(_stringify_id(doc))

    @_guard("get_chat")
    def get_chat(self, chat_id: str) -> ChatModel | None:
        doc = self._collection.find_one({"chat_id": chat_id})
        return ChatModel.model_validate(_stringify_id(doc)) if doc else None

    @_guard("list_chats_for_user")
    def list_chats_for_user(self, user_id: str) -> list[ChatModel]:
        cursor = self._collection.find({"user_id": user_id}).sort("updated_at", -1)
        return [ChatModel.model_validate(_stringify_id(doc)) for doc in cursor]

    @_guard("touch_chat")
    def touch_updated_at(self, chat_id: str) -> None:
        self._collection.update_one({"chat_id": chat_id}, {"$set": {"updated_at": now_iso()}})

    @_guard("set_title")
    def set_title(self, chat_id: str, title: str) -> ChatModel | None:
        self._collection.update_one(
            {"chat_id": chat_id}, {"$set": {"title": title, "updated_at": now_iso()}}
        )
        return self.get_chat(chat_id)

    @_guard("delete_chat")
    def delete_chat(self, chat_id: str) -> bool:
        result = self._collection.delete_one({"chat_id": chat_id})
        return result.deleted_count > 0


class MessageRepository:
    def __init__(self, db: Database) -> None:
        self._collection = db["messages"]

    @_guard("add_message")
    def add_message(
        self,
        chat_id: str,
        user_id: str,
        role: str,
        content: str,
        tool_used: str | None = None,
        metadata: dict | None = None,
    ) -> MessageModel:
        doc = {
            "chat_id": chat_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": now_iso(),
            "tool_used": tool_used,
            "metadata": metadata or {},
        }
        result = self._collection.insert_one(doc)
        doc["_id"] = result.inserted_id
        return MessageModel.model_validate(_stringify_id(doc))

    @_guard("get_messages")
    def get_messages(self, chat_id: str) -> list[MessageModel]:
        cursor = self._collection.find({"chat_id": chat_id}).sort("timestamp", 1)
        return [MessageModel.model_validate(_stringify_id(doc)) for doc in cursor]

    @_guard("count_messages")
    def count_messages(self, chat_id: str) -> int:
        return self._collection.count_documents({"chat_id": chat_id})

    @_guard("delete_messages")
    def delete_messages(self, chat_id: str) -> int:
        result = self._collection.delete_many({"chat_id": chat_id})
        return result.deleted_count


def history_pairs_from_messages(messages: list[MessageModel], max_turns: int) -> list[tuple[str, str]]:
    """Fold a flat, timestamp-ordered message list into (question, answer) turns for LangChain -
    mirrors `session_manager.SessionManager.chat_history_pairs` so `graph.py` needs no changes."""
    pairs: list[tuple[str, str]] = []
    pending_question: str | None = None
    for message in messages:
        if message.role == "user":
            pending_question = message.content
        elif message.role == "assistant" and pending_question is not None:
            pairs.append((pending_question, message.content))
            pending_question = None
    return pairs[-max_turns:] if max_turns > 0 else pairs


__all__ = [
    "UserRepository",
    "ChatRepository",
    "MessageRepository",
    "RepositoryError",
    "history_pairs_from_messages",
]
