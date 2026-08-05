"""PandexAI - Chat History page: every past conversation for this local user, backed by MongoDB
(`GET /chats/{user_id}`), with search, open/resume, rename, and delete.

Kept deliberately separate from the (already large) main chat script - this page only ever talks
to the backend's `/chats/{user_id}` and `/chat/{chat_id}` endpoints, never the RAG/ingestion ones.
A conversation's full message history is only fetched when the user actually clicks Open (which
hands off to the main chat page, reusing its existing history-loading + rendering code).
"""

from __future__ import annotations

import html
import os
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

from utils.user_store import get_or_create_user

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HTTP_TIMEOUT = 10.0

st.set_page_config(page_title="PandexAI - History", page_icon="🐼", layout="wide", initial_sidebar_state="expanded")

st.session_state.setdefault("theme", "dark")
st.session_state.setdefault("user_id", get_or_create_user()["user_id"])


def _load_css() -> str:
    return (Path(__file__).resolve().parent.parent / "assets" / "style.css").read_text(encoding="utf-8")


st.markdown(f"<style>{_load_css()}</style>", unsafe_allow_html=True)
st.components.v1.html(
    f"<script>window.parent.document.documentElement.setAttribute('data-theme', '{st.session_state.theme}');</script>",
    height=0,
)


# ----------------------------------------------------------------------------
# Backend calls
# ----------------------------------------------------------------------------
def _fetch_chats(user_id: str) -> list[dict]:
    try:
        resp = httpx.get(f"{BACKEND_URL}/chats/{user_id}", timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["chats"]
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the backend: {exc}")
        return []


def _rename_chat(chat_id: str, title: str) -> bool:
    try:
        resp = httpx.patch(f"{BACKEND_URL}/chat/{chat_id}", json={"title": title}, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        st.error(f"Could not rename conversation: {exc}")
        return False


def _delete_chat(chat_id: str) -> bool:
    try:
        resp = httpx.delete(f"{BACKEND_URL}/chat/{chat_id}", timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        st.error(f"Could not delete conversation: {exc}")
        return False


def _open_chat(chat_id: str, title: str) -> None:
    """Switch the shared session state to another conversation and jump back to the chat page."""
    st.session_state.session_id = chat_id
    st.session_state.chat_title = title
    st.session_state.history_loaded = False
    st.session_state.chat_history = []
    st.session_state.sources = []
    st.switch_page("streamlit_app.py")


@st.dialog("Delete conversation")
def _confirm_delete_dialog(chat_id: str, title: str) -> None:
    st.warning("Are you sure you want to delete this conversation? This action cannot be undone.")
    st.caption(f'"{title}"')
    cancel_col, delete_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with delete_col:
        if st.button("🗑️ Delete", use_container_width=True, type="primary"):
            if _delete_chat(chat_id):
                st.rerun()


with st.sidebar:
    st.markdown(
        '<div class="brand-row"><div class="brand-avatar">🐼</div>'
        '<div><div class="brand-name">PandexAI</div>'
        '<div class="brand-tagline">Your document, video &amp; web assistant</div></div></div>',
        unsafe_allow_html=True,
    )
    st.page_link("streamlit_app.py", label="💬 Chat")
    st.page_link("pages/1_History.py", label="📜 History")

st.markdown('<div class="welcome-title" style="font-size:1.9rem; text-align:left;">📜 Chat History</div>', unsafe_allow_html=True)
st.caption("Every conversation stored for this local user in MongoDB.")

header_col, new_col = st.columns([4, 1])
with new_col:
    if st.button("+ New Chat", use_container_width=True):
        import uuid

        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_title = "New chat"
        st.session_state.history_loaded = True
        st.session_state.chat_history = []
        st.session_state.sources = []
        st.switch_page("streamlit_app.py")

search = st.text_input("Search", placeholder="🔍 Search conversations by title...", label_visibility="collapsed")

chats = _fetch_chats(st.session_state.user_id)
if search:
    needle = search.lower()
    chats = [c for c in chats if needle in c["title"].lower()]


def _format_dt(iso_string: str) -> str:
    try:
        return datetime.fromisoformat(iso_string).strftime("%b %d, %Y · %H:%M")
    except ValueError:
        return iso_string


if not chats:
    st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
    st.caption("No conversations match your search." if search else "No conversations yet. Start a new chat to see your history here.")
else:
    st.caption(f"{len(chats)} conversation(s).")
    columns = st.columns(3)
    for index, chat in enumerate(chats):
        with columns[index % 3]:
            st.markdown(
                f'<div class="history-card">'
                f'<div class="history-card-title">💬 {html.escape(chat["title"])}</div>'
                f'<div class="history-card-meta">Created {_format_dt(chat["created_at"])}</div>'
                f'<div class="history-card-meta">Updated {_format_dt(chat["updated_at"])}</div>'
                f'<div class="history-card-meta">{chat["message_count"]} message(s)</div>'
                "</div>",
                unsafe_allow_html=True,
            )

            open_col, rename_col, delete_col = st.columns(3)
            with open_col:
                if st.button("Open", key=f"open_{chat['chat_id']}", use_container_width=True):
                    _open_chat(chat["chat_id"], chat["title"])
            with rename_col:
                with st.popover("✏️ Rename", use_container_width=True):
                    new_title = st.text_input(
                        "New title", value=chat["title"], key=f"rename_{chat['chat_id']}", label_visibility="collapsed"
                    )
                    if st.button("Save", key=f"save_{chat['chat_id']}", use_container_width=True):
                        if _rename_chat(chat["chat_id"], new_title):
                            st.rerun()
            with delete_col:
                if st.button("🗑️ Delete", key=f"del_{chat['chat_id']}", use_container_width=True):
                    _confirm_delete_dialog(chat["chat_id"], chat["title"])
