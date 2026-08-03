"""Per-session persistence: source metadata, chat history, token usage, and settings.

Backed by a single SQLite database (`.sessions/history.db`, one row per conversation in the
`conversations` table - see `backend.history_store` for the table DDL and the list/search/filter
query surface used by the chat-history page). A short-lived connection is opened per operation
(WAL mode + a busy timeout) rather than held open, which is simple and safe across FastAPI's
threadpool without any manual locking.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from config import get_settings
from utils.helpers import now_iso

_TITLE_MAX_LEN = 60


def db_path() -> Path:
    return Path(get_settings().sessions_dir) / "history.db"


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


def _default_state() -> dict[str, Any]:
    return {
        "sources": [],  # list[SourceInfo-shaped dict]
        "chat_history": [],  # list[ChatMessage-shaped dict]
        "settings": {},  # SessionSettings-shaped dict, sparse overrides only
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "last_provider": "",
            "last_model": "",
        },
    }


def derive_title(chat_history: list[dict]) -> str:
    first_user_message = next((m["content"] for m in chat_history if m.get("role") == "user"), "")
    if not first_user_message:
        return "Untitled conversation"
    title = first_user_message[:_TITLE_MAX_LEN]
    if len(first_user_message) > _TITLE_MAX_LEN:
        title += "..."
    return title


def _row_to_state(row: sqlite3.Row) -> dict[str, Any]:
    state = _default_state()
    state["sources"] = json.loads(row["sources"] or "[]")
    state["chat_history"] = json.loads(row["messages"] or "[]")
    state["settings"] = json.loads(row["settings"] or "{}")
    state["usage"] = {
        "input_tokens": row["prompt_tokens"],
        "output_tokens": row["completion_tokens"],
        "estimated_cost_usd": row["estimated_cost_usd"],
        "last_provider": row["provider"],
        "last_model": row["model"],
    }
    return state


class SessionManager:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._created_at: str | None = None
        self._pinned = False
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        with connect() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (self.session_id,)).fetchone()
        if row is None:
            return _default_state()
        self._created_at = row["created_at"]
        self._pinned = bool(row["pinned"])
        return _row_to_state(row)

    def save(self) -> None:
        created_at = self._created_at or now_iso()
        title = derive_title(self._state["chat_history"])
        usage = self._state["usage"]
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    id, title, created_at, updated_at, pinned, provider, model,
                    prompt_tokens, completion_tokens, estimated_cost_usd, message_count,
                    messages, sources, settings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title, updated_at=excluded.updated_at, provider=excluded.provider,
                    model=excluded.model, prompt_tokens=excluded.prompt_tokens,
                    completion_tokens=excluded.completion_tokens, estimated_cost_usd=excluded.estimated_cost_usd,
                    message_count=excluded.message_count, messages=excluded.messages,
                    sources=excluded.sources, settings=excluded.settings
                """,
                (
                    self.session_id, title, created_at, now_iso(), int(self._pinned),
                    usage["last_provider"], usage["last_model"],
                    usage["input_tokens"], usage["output_tokens"], usage["estimated_cost_usd"],
                    len(self._state["chat_history"]),
                    json.dumps(self._state["chat_history"], ensure_ascii=False),
                    json.dumps(self._state["sources"], ensure_ascii=False),
                    json.dumps(self._state["settings"], ensure_ascii=False),
                ),
            )
            conn.commit()
        self._created_at = created_at

    # --- sources -----------------------------------------------------------------
    @property
    def sources(self) -> list[dict]:
        return self._state["sources"]

    def add_source(self, source: dict) -> None:
        self._state["sources"].append(source)
        self.save()

    def remove_source(self, source_id: str) -> None:
        self._state["sources"] = [s for s in self._state["sources"] if s["id"] != source_id]
        self.save()

    def clear_sources(self) -> None:
        self._state["sources"] = []
        self.save()

    def known_content_hashes(self) -> set[str]:
        return {s["content_hash"] for s in self._state["sources"] if s.get("content_hash")}

    # --- chat ----------------------------------------------------------------------
    @property
    def chat_history(self) -> list[dict]:
        return self._state["chat_history"]

    def chat_history_pairs(self, max_turns: int) -> list[tuple[str, str]]:
        """Recent (question, answer) tuples for conversation memory, most recent last."""
        messages = self._state["chat_history"]
        pairs: list[tuple[str, str]] = []
        pending_question: str | None = None
        for msg in messages:
            if msg["role"] == "user":
                pending_question = msg["content"]
            elif msg["role"] == "assistant" and pending_question is not None:
                pairs.append((pending_question, msg["content"]))
                pending_question = None
        return pairs[-max_turns:]

    def append_message(self, message: dict) -> None:
        self._state["chat_history"].append(message)
        self.save()

    def clear_chat(self) -> None:
        self._state["chat_history"] = []
        self.save()

    # --- usage -----------------------------------------------------------------------
    @property
    def usage(self) -> dict:
        return self._state["usage"]

    def add_usage(self, input_tokens: int, output_tokens: int, cost: float, provider: str, model: str) -> None:
        self._state["usage"]["input_tokens"] += input_tokens
        self._state["usage"]["output_tokens"] += output_tokens
        self._state["usage"]["estimated_cost_usd"] += cost
        self._state["usage"]["last_provider"] = provider
        self._state["usage"]["last_model"] = model
        self.save()

    # --- settings --------------------------------------------------------------------
    @property
    def settings_overrides(self) -> dict:
        return self._state["settings"]

    def update_settings(self, overrides: dict) -> None:
        self._state["settings"].update({k: v for k, v in overrides.items() if v is not None})
        self.save()


_session_cache: dict[str, SessionManager] = {}
_cache_guard = threading.Lock()


def get_session(session_id: str) -> SessionManager:
    with _cache_guard:
        if session_id not in _session_cache:
            _session_cache[session_id] = SessionManager(session_id)
        return _session_cache[session_id]


def delete_session_file(session_id: str) -> None:
    with _cache_guard:
        _session_cache.pop(session_id, None)
    with connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (session_id,))
        conn.commit()


__all__ = ["SessionManager", "get_session", "delete_session_file", "db_path", "now_iso"]
