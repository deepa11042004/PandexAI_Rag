"""PandexAI - Chat History page: search, sort, filter, and page through every past conversation,
resume one (View), or permanently delete it.

Kept deliberately separate from the (already large) main chat script - this page only ever talks
to the backend's `/api/chat-history*` endpoints, never the RAG/chat ones. Card metadata (title,
timestamps, model, tokens, cost) is loaded cheaply via the list endpoint; a conversation's full
message history is only fetched when the user actually clicks View (which hands off to the main
chat page, reusing its existing history-loading + rendering code).
"""

from __future__ import annotations

import html
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.helpers import new_id  # noqa: E402

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HTTP_TIMEOUT = 10.0
PAGE_SIZE = 9

SORT_OPTIONS = {"Recently Updated": "updated", "Newest": "newest", "Oldest": "oldest"}

st.set_page_config(page_title="PandexAI - History", page_icon="🐼", layout="wide", initial_sidebar_state="expanded")

st.session_state.setdefault("theme", "dark")
st.session_state.setdefault("session_id", new_id())
st.session_state.setdefault("hist_items", [])
st.session_state.setdefault("hist_page", 0)
st.session_state.setdefault("hist_has_more", True)
st.session_state.setdefault("hist_total", 0)
st.session_state.setdefault("hist_available_models", [])
st.session_state.setdefault("hist_filters_sig", None)
st.session_state.setdefault("hist_bootstrapped", False)


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
def _fetch_page(page: int, filters: dict) -> dict | None:
    try:
        resp = httpx.get(
            f"{BACKEND_URL}/api/chat-history",
            params={**filters, "page": page, "page_size": PAGE_SIZE},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the backend: {exc}")
        return None


def _create_conversation() -> str | None:
    try:
        resp = httpx.post(f"{BACKEND_URL}/api/chat-history", timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["id"]
    except httpx.HTTPError as exc:
        st.error(f"Could not start a new conversation: {exc}")
        return None


def _update_pin(conversation_id: str, pinned: bool) -> bool:
    try:
        resp = httpx.put(
            f"{BACKEND_URL}/api/chat-history/{conversation_id}", json={"pinned": pinned}, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        st.error(f"Could not update conversation: {exc}")
        return False


def _delete_conversation(conversation_id: str) -> bool:
    try:
        resp = httpx.delete(f"{BACKEND_URL}/api/chat-history/{conversation_id}", timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        st.error(f"Could not delete conversation: {exc}")
        return False


def _reset_pagination() -> None:
    st.session_state.hist_items = []
    st.session_state.hist_page = 0
    st.session_state.hist_has_more = True


def _open_session(session_id: str) -> None:
    """Switch the shared session state to another conversation and jump back to the chat page.

    `history_loaded=False` forces the main page to re-fetch that session's chat/sources from the
    backend on arrival, rather than continuing to show whatever the previously active session had.
    """
    st.session_state.session_id = session_id
    st.session_state.history_loaded = False
    st.session_state.chat_history = []
    st.session_state.sources = []
    st.switch_page("streamlit_app.py", query_params={"sid": session_id})


@st.dialog("Delete conversation")
def _confirm_delete_dialog(conversation_id: str, title: str) -> None:
    st.warning("Are you sure you want to delete this conversation? This action cannot be undone.")
    st.caption(f'"{title}"')
    cancel_col, delete_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with delete_col:
        if st.button("🗑️ Delete", use_container_width=True, type="primary"):
            if _delete_conversation(conversation_id):
                _reset_pagination()
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
st.caption("Every conversation you've started, across all sessions on this backend.")

header_col, new_col = st.columns([4, 1])
with new_col:
    if st.button("+ New Conversation", use_container_width=True):
        new_conversation_id = _create_conversation()
        if new_conversation_id:
            _open_session(new_conversation_id)

# ----------------------------------------------------------------------------
# Search, sort & filters
# ----------------------------------------------------------------------------
search = st.text_input(
    "Search", placeholder="🔍 Search conversations by title or content...", label_visibility="collapsed"
)

sort_col, model_col, from_col, to_col = st.columns(4)
with sort_col:
    sort_label = st.selectbox("Sort by", list(SORT_OPTIONS.keys()))
with model_col:
    model_options = ["All models"] + st.session_state.hist_available_models
    model_label = st.selectbox("Model", model_options)
with from_col:
    date_from = st.date_input("Created after", value=None)
with to_col:
    date_to = st.date_input("Created before", value=None)

min_col, max_col = st.columns(2)
with min_col:
    min_tokens = st.number_input("Min total tokens", min_value=0, value=0, step=500)
with max_col:
    max_tokens = st.number_input("Max total tokens (0 = no limit)", min_value=0, value=0, step=500)

filters = {
    "search": search,
    "sort": SORT_OPTIONS[sort_label],
    "model": "" if model_label == "All models" else model_label,
    "date_from": date_from.isoformat() if date_from else "",
    "date_to": date_to.isoformat() if date_to else "",
    "min_tokens": min_tokens,
    "max_tokens": max_tokens,
}
filters_sig = tuple(filters.items())

if filters_sig != st.session_state.hist_filters_sig:
    st.session_state.hist_filters_sig = filters_sig
    _reset_pagination()

if not st.session_state.hist_items and st.session_state.hist_has_more:
    result = _fetch_page(1, filters)
    if result is not None:
        st.session_state.hist_items = result["items"]
        st.session_state.hist_page = 1
        st.session_state.hist_has_more = result["has_more"]
        st.session_state.hist_total = result["total"]
        st.session_state.hist_available_models = result["available_models"]
    if not st.session_state.hist_bootstrapped:
        # First visit to the page: the filter widgets above were drawn before this fetch
        # completed (e.g. the model dropdown had no options yet). One extra rerun lets them
        # redraw against the now-populated session state instead of waiting for a user click.
        st.session_state.hist_bootstrapped = True
        st.rerun()

items = st.session_state.hist_items
filters_active = any([search, filters["model"], filters["date_from"], filters["date_to"], min_tokens, max_tokens])


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def _format_dt(iso_string: str) -> str:
    try:
        return datetime.fromisoformat(iso_string).strftime("%b %d, %Y · %H:%M")
    except ValueError:
        return iso_string


def _format_duration(seconds: float | None) -> str:
    if not seconds:
        return ""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


if not items:
    st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
    if filters_active:
        st.caption("No conversations match your search/filters. Try adjusting them.")
    else:
        st.caption("No conversations yet. Start a new chat to see your history here.")
else:
    st.caption(f"Showing {len(items)} of {st.session_state.hist_total} conversation(s).")
    columns = st.columns(3)
    for index, conversation in enumerate(items):
        with columns[index % 3]:
            badge_label = conversation["model"] or conversation["provider"]
            badge_html = f'<span class="model-badge">{html.escape(badge_label)}</span>' if badge_label else ""
            pin_html = '<span class="pin-badge">📌 Pinned</span>' if conversation["pinned"] else ""
            duration_text = _format_duration(conversation["duration_seconds"])
            duration_html = f" · {duration_text}" if duration_text else ""

            st.markdown(
                f'<div class="history-card">'
                f'<div class="history-card-badges">{badge_html}{pin_html}</div>'
                f'<div class="history-card-title">💬 {html.escape(conversation["title"])}</div>'
                f'<div class="history-card-meta">Created {_format_dt(conversation["created_at"])}</div>'
                f'<div class="history-card-meta">Updated {_format_dt(conversation["updated_at"])}</div>'
                f'<div class="history-card-meta">{conversation["message_count"]} message(s){duration_html}</div>'
                '<div class="token-badge-row">'
                f'<span class="token-badge">Prompt {conversation["prompt_tokens"]:,}</span>'
                f'<span class="token-badge">Completion {conversation["completion_tokens"]:,}</span>'
                f'<span class="token-badge">Total {conversation["total_tokens"]:,}</span>'
                f'<span class="token-badge">Est. ${conversation["estimated_cost_usd"]:.4f}</span>'
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            pin_col, view_col, delete_col = st.columns(3)
            with pin_col:
                pin_icon = "📌" if conversation["pinned"] else "📍"
                if st.button(pin_icon, key=f"pin_{conversation['id']}", use_container_width=True, help="Pin/unpin"):
                    if _update_pin(conversation["id"], not conversation["pinned"]):
                        _reset_pagination()
                        st.rerun()
            with view_col:
                if st.button("View", key=f"open_{conversation['id']}", use_container_width=True):
                    _open_session(conversation["id"])
            with delete_col:
                if st.button("🗑️ Delete", key=f"del_{conversation['id']}", use_container_width=True):
                    _confirm_delete_dialog(conversation["id"], conversation["title"])

    if st.session_state.hist_has_more:
        st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)
        if st.button("Load more", use_container_width=True):
            more = _fetch_page(st.session_state.hist_page + 1, filters)
            if more is not None:
                st.session_state.hist_items.extend(more["items"])
                st.session_state.hist_page += 1
                st.session_state.hist_has_more = more["has_more"]
            st.rerun()
