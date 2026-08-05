"""SQLite schema setup + one-time legacy-JSON migration for the `conversations` table.

Chat/message history now lives in MongoDB (see `backend/database/`) - this table is only used by
`backend.session_manager.SessionManager` for the non-chat-history parts of a conversation (sources,
token usage, per-chat settings overrides), keyed by the same id as the Mongo `chat_id`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from config import get_settings
from session_manager import connect, derive_title, db_path
from utils.logger import get_logger

logger = get_logger(__name__)


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
            created_at = _iso_from_epoch(stat.st_ctime) if stat.st_ctime > 0 else _iso_from_epoch(stat.st_mtime)
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


__all__ = ["init_db", "migrate_legacy_json_sessions"]
