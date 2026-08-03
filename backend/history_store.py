"""Chat-history page backing store: schema setup, one-time legacy-JSON migration, and the
search/sort/filter/paginate query surface behind `GET /api/chat-history`.

Reads/writes the same `conversations` SQLite table that `backend.session_manager.SessionManager`
uses for per-session read/write during chat - this module is purely the list-oriented query layer
on top of it, so there's exactly one source of truth for conversation data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from config import get_settings
from session_manager import connect, derive_title, db_path
from utils.helpers import new_id, now_iso
from utils.logger import get_logger

logger = get_logger(__name__)

_SORT_COLUMNS = {
    "newest": "created_at DESC",
    "oldest": "created_at ASC",
    "updated": "updated_at DESC",
}


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0,
                message_count INTEGER NOT NULL DEFAULT 0,
                messages TEXT NOT NULL DEFAULT '[]',
                sources TEXT NOT NULL DEFAULT '[]',
                settings TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_model ON conversations(model)")
        conn.commit()


def migrate_legacy_json_sessions() -> None:
    """One-time import of `.sessions/<id>.json` files into the `conversations` table.

    Migrates every file regardless of whether it has chat history - a sources-only session still
    needs its data preserved now that `SessionManager` no longer reads JSON at all. Successfully
    migrated files are moved into `.sessions/_migrated/` so this never reprocesses them; a file that
    fails to parse is logged and left in place rather than migrated or deleted.
    """
    sessions_dir = Path(get_settings().sessions_dir)
    if not sessions_dir.exists():
        return

    migrated_dir = sessions_dir / "_migrated"

    with connect() as conn:
        for path in sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping unreadable legacy session file: %s", path)
                continue

            session_id = path.stem
            chat_history = data.get("chat_history", [])
            usage = data.get("usage", {})
            stat = path.stat()
            created_at = now_iso() if stat.st_ctime <= 0 else _iso_from_epoch(stat.st_ctime)
            updated_at = _iso_from_epoch(stat.st_mtime)

            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO conversations (
                        id, title, created_at, updated_at, pinned, provider, model,
                        prompt_tokens, completion_tokens, estimated_cost_usd, message_count,
                        messages, sources, settings
                    ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id, derive_title(chat_history), created_at, updated_at,
                        usage.get("last_provider", ""), usage.get("last_model", ""),
                        usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                        usage.get("estimated_cost_usd", 0.0), len(chat_history),
                        json.dumps(chat_history, ensure_ascii=False),
                        json.dumps(data.get("sources", []), ensure_ascii=False),
                        json.dumps(data.get("settings", {}), ensure_ascii=False),
                    ),
                )
            except Exception:  # noqa: BLE001 - never let one bad file block startup
                logger.exception("Failed to migrate legacy session file: %s", path)
                continue

            migrated_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(migrated_dir / path.name))
            logger.info("Migrated legacy session %s into %s", session_id, db_path())

        conn.commit()


def _iso_from_epoch(epoch: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


def _row_to_summary(row: Any) -> dict[str, Any]:
    prompt_tokens = row["prompt_tokens"]
    completion_tokens = row["completion_tokens"]
    duration_seconds: float | None = None
    if row["message_count"] > 1:
        from datetime import datetime

        try:
            delta = datetime.fromisoformat(row["updated_at"]) - datetime.fromisoformat(row["created_at"])
            duration_seconds = max(delta.total_seconds(), 0.0)
        except ValueError:
            duration_seconds = None

    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "pinned": bool(row["pinned"]),
        "provider": row["provider"],
        "model": row["model"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": row["estimated_cost_usd"],
        "message_count": row["message_count"],
        "duration_seconds": duration_seconds,
    }


_SUMMARY_COLUMNS = (
    "id, title, created_at, updated_at, pinned, provider, model, "
    "prompt_tokens, completion_tokens, estimated_cost_usd, message_count"
)


def list_conversations(
    search: str = "",
    sort: str = "updated",
    model: str = "",
    date_from: str = "",
    date_to: str = "",
    min_tokens: int = 0,
    max_tokens: int = 0,
    page: int = 1,
    page_size: int = 9,
) -> dict[str, Any]:
    where = ["message_count > 0"]
    params: list[Any] = []

    if search:
        where.append("(LOWER(title) LIKE ? OR LOWER(messages) LIKE ?)")
        needle = f"%{search.lower()}%"
        params.extend([needle, needle])
    if model:
        where.append("model = ?")
        params.append(model)
    if date_from:
        where.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        where.append("created_at <= ?")
        params.append(date_to + "T23:59:59")
    if min_tokens:
        where.append("(prompt_tokens + completion_tokens) >= ?")
        params.append(min_tokens)
    if max_tokens:
        where.append("(prompt_tokens + completion_tokens) <= ?")
        params.append(max_tokens)

    where_sql = " AND ".join(where)
    order_sql = _SORT_COLUMNS.get(sort, _SORT_COLUMNS["updated"])
    page = max(page, 1)
    page_size = max(page_size, 1)
    offset = (page - 1) * page_size

    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM conversations WHERE {where_sql}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM conversations WHERE {where_sql} "
            f"ORDER BY pinned DESC, {order_sql} LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        ).fetchall()
        available_models = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT model FROM conversations WHERE model != '' AND message_count > 0 ORDER BY model"
            ).fetchall()
        ]

    items = [_row_to_summary(row) for row in rows]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": offset + len(items) < total,
        "available_models": available_models,
    }


def get_conversation(session_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    summary = _row_to_summary(row)
    summary["messages"] = json.loads(row["messages"] or "[]")
    return summary


def create_conversation() -> dict[str, Any]:
    session_id = new_id()
    timestamp = now_iso()
    with connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, "Untitled conversation", timestamp, timestamp),
        )
        conn.commit()
    return get_conversation(session_id)  # type: ignore[return-value]


def update_conversation(session_id: str, pinned: bool | None) -> dict[str, Any] | None:
    if pinned is None:
        return get_conversation(session_id)
    with connect() as conn:
        cursor = conn.execute(
            "UPDATE conversations SET pinned = ? WHERE id = ?", (int(pinned), session_id)
        )
        conn.commit()
        if cursor.rowcount == 0:
            return None
    return get_conversation(session_id)


__all__ = [
    "init_db",
    "migrate_legacy_json_sessions",
    "list_conversations",
    "get_conversation",
    "create_conversation",
    "update_conversation",
]
