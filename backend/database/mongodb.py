"""MongoDB connection management (pymongo).

A single process-wide `MongoClient` is created lazily and reused (pymongo pools connections
internally, so - like `session_manager.connect()` for SQLite - there's no reason to open a new
client per request). `connect_to_mongo()` is called once from the FastAPI startup event to fail
fast with a clear error if the database is unreachable, and to create the indexes required by
the DATABASE INDEXES requirement (chat_id, user_id, timestamp).
"""

from __future__ import annotations

from functools import lru_cache

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

USERS_COLLECTION = "users"
CHATS_COLLECTION = "chats"
MESSAGES_COLLECTION = "messages"


class MongoConnectionError(RuntimeError):
    """Raised when MongoDB cannot be reached or the client hasn't been initialized yet."""


@lru_cache
def _get_client() -> MongoClient:
    settings = get_settings()
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)


def get_database() -> Database:
    """Return the `PandexAI` database handle. Raises MongoConnectionError if unreachable."""
    settings = get_settings()
    try:
        client = _get_client()
        return client[settings.mongodb_db_name]
    except PyMongoError as exc:
        raise MongoConnectionError(f"Could not connect to MongoDB: {exc}") from exc


def connect_to_mongo() -> None:
    """Verify connectivity and ensure indexes exist. Call once at application startup."""
    settings = get_settings()
    try:
        client = _get_client()
        client.admin.command("ping")
    except PyMongoError as exc:
        raise MongoConnectionError(
            f"Could not connect to MongoDB at '{settings.mongodb_uri}'. "
            f"Is MongoDB running and is MONGODB_URI correct in backend/.env? ({exc})"
        ) from exc

    db = client[settings.mongodb_db_name]
    _ensure_indexes(db)
    logger.info("Connected to MongoDB database '%s' at %s", settings.mongodb_db_name, settings.mongodb_uri)


def _ensure_indexes(db: Database) -> None:
    db[USERS_COLLECTION].create_index([("user_id", ASCENDING)], unique=True, name="uniq_user_id")

    db[CHATS_COLLECTION].create_index([("chat_id", ASCENDING)], unique=True, name="uniq_chat_id")
    db[CHATS_COLLECTION].create_index([("user_id", ASCENDING)], name="idx_chats_user_id")
    db[CHATS_COLLECTION].create_index([("updated_at", ASCENDING)], name="idx_chats_updated_at")

    db[MESSAGES_COLLECTION].create_index([("chat_id", ASCENDING)], name="idx_messages_chat_id")
    db[MESSAGES_COLLECTION].create_index([("user_id", ASCENDING)], name="idx_messages_user_id")
    db[MESSAGES_COLLECTION].create_index([("timestamp", ASCENDING)], name="idx_messages_timestamp")
    # Compound index: the hot path is "all messages of one chat, oldest first".
    db[MESSAGES_COLLECTION].create_index(
        [("chat_id", ASCENDING), ("timestamp", ASCENDING)], name="idx_messages_chat_id_timestamp"
    )


def close_mongo_connection() -> None:
    try:
        _get_client().close()
    except Exception:  # noqa: BLE001 - best-effort on shutdown
        pass
    _get_client.cache_clear()


__all__ = [
    "get_database",
    "connect_to_mongo",
    "close_mongo_connection",
    "MongoConnectionError",
    "USERS_COLLECTION",
    "CHATS_COLLECTION",
    "MESSAGES_COLLECTION",
]
