"""Persistent local user identity.

There's no login/auth system in this app - it's a single-user local deployment. To still give
each user their own permanent chat history in MongoDB (the `users`/`chats` collections are keyed
by `user_id`), a UUID is generated once and cached in a small JSON file next to this package. Every
future launch of the Streamlit app (even after a full restart) reads the same file and reuses the
same `user_id`, so `GET /chats/{user_id}` keeps returning the same conversations.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

_USER_FILE = Path(__file__).resolve().parent.parent / ".local_user.json"
_DEFAULT_NAME = "Local User"


def get_or_create_user() -> dict:
    """Returns {"user_id": ..., "name": ...}, creating and persisting one on first run."""
    if _USER_FILE.exists():
        try:
            data = json.loads(_USER_FILE.read_text(encoding="utf-8"))
            if data.get("user_id"):
                return data
        except (json.JSONDecodeError, OSError):
            pass  # fall through and recreate a fresh identity rather than crashing the app

    data = {"user_id": str(uuid.uuid4()), "name": _DEFAULT_NAME}
    try:
        _USER_FILE.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass  # worst case: a new user_id is generated on every restart instead of persisting
    return data


def get_or_create_user_id() -> str:
    return get_or_create_user()["user_id"]
