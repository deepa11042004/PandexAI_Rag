"""PandexAI - Streamlit frontend.

A thin, premium UI that talks to the FastAPI backend (`backend/main.py`) over HTTP/SSE for
everything: ingestion, retrieval, and streamed chat generation. No LLM/vector-store logic lives
here - this file is purely presentation + HTTP calls.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
import re
from pathlib import Path

import httpx
import streamlit as st
from PIL import Image

from utils.export_chat import export_chat_to_json, export_chat_to_pdf, export_chat_to_text
from utils.helpers import format_file_size, new_id

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HTTP_TIMEOUT = 8.0  # status/GET calls should fail fast rather than freeze the page if the backend is down

SOURCE_ICONS = {
    "pdf": "📄", "docx": "📝", "txt": "📃", "csv": "📊", "xlsx": "📈",
    "pptx": "📽️", "image": "🖼️", "youtube": "▶️", "website": "🌐",
    "markdown": "Ⓜ️", "json": "🔧",
}

PROVIDER_LABELS = {
    "groq": "Groq (Llama)",
    "openrouter": "OpenRouter",
    "openai": "OpenAI (GPT)",
    "anthropic": "Claude (Anthropic)",
    "google": "Gemini (Google)",
}

# Drop a real mascot image here (any of these filenames) to replace the emoji placeholder on the
# welcome screen and in the sidebar header - picked up automatically, no code change needed.
MASCOT_CANDIDATES = ("mascot.png", "mascot.jpg", "mascot.jpeg", "mascot.webp", "panda-mascot.png")
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _find_mascot_image() -> Path | None:
    for name in MASCOT_CANDIDATES:
        candidate = ASSETS_DIR / name
        if candidate.exists():
            return candidate
    return None


def _png_data_uri(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


@st.cache_data(show_spinner=False)
def _mascot_hero_data_uri(path_str: str, mtime: float, max_width: int = 480) -> str:
    """Downscaled mascot image for the welcome-screen card.

    A phone-camera or AI-generated source photo can be several MB - inlining that as base64 on
    every single rerun (Streamlit reruns the whole script on every interaction) would noticeably
    slow the page down, which is exactly the "site loads slowly" complaint from earlier. Resizing
    once and caching by (path, mtime) keeps reruns cheap while still picking up a replaced file.
    """
    img = Image.open(path_str).convert("RGBA")
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, round(img.height * ratio)), Image.LANCZOS)
    return _png_data_uri(img)


@st.cache_data(show_spinner=False)
def _mascot_avatar_data_uri(path_str: str, mtime: float, size: int = 160) -> str:
    """Square, head-focused crop for the small circular sidebar avatar.

    Cropping the top square of the frame (rather than the dead-center square) keeps a portrait
    subject's face in frame for typical mascot-style photos, where the subject sits low in a
    taller/wider frame - a plain center crop would mostly show torso/background at avatar size.
    """
    img = Image.open(path_str).convert("RGBA")
    side = min(img.width, img.height)
    left = (img.width - side) // 2
    top_square = img.crop((left, 0, left + side, side))
    face_crop = top_square.crop((0, 0, side, round(side * 0.62)))
    face_square_side = min(face_crop.size)
    fw, fh = face_crop.size
    face_crop = face_crop.crop(((fw - face_square_side) // 2, 0, (fw - face_square_side) // 2 + face_square_side, face_square_side))
    face_crop = face_crop.resize((size, size), Image.LANCZOS)
    return _png_data_uri(face_crop)


st.set_page_config(page_title="PandexAI", page_icon="🐼", layout="wide", initial_sidebar_state="expanded")


# ----------------------------------------------------------------------------
# Backend HTTP client helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def _client() -> httpx.Client:
    """One pooled, keep-alive client for the process's lifetime.

    Opening a fresh httpx.Client per call (the previous behavior) pays a new TCP handshake for
    every single request - with 4-5 API calls per sidebar render, that overhead was the main
    source of perceived page slowness. Reusing one client keeps connections warm across reruns.
    """
    return httpx.Client(base_url=BACKEND_URL, timeout=HTTP_TIMEOUT)


def api_get(path: str) -> dict | None:
    try:
        resp = _client().get(path)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        st.session_state.backend_error = str(exc)
        return None


def api_post(path: str, json_body: dict | None = None, files=None, timeout: float | None = None) -> dict | list | None:
    resolved_timeout = timeout if timeout is not None else (120.0 if files else HTTP_TIMEOUT)
    try:
        resp = _client().post(path, json=json_body, files=files, timeout=resolved_timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        st.session_state.backend_error = str(exc)
        return None


def api_delete(path: str) -> dict | None:
    try:
        resp = _client().delete(path)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        st.session_state.backend_error = str(exc)
        return None


def stream_chat_events(session_id: str, message: str):
    """Yield parsed SSE event dicts from POST /chat as they arrive."""
    with _client().stream(
        "POST", "/chat", json={"session_id": session_id, "message": message}, timeout=120.0
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line and line.startswith("data: "):
                yield json.loads(line[len("data: ") :])


# ----------------------------------------------------------------------------
# Session bootstrap
# ----------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "session_id": new_id(),
        "theme": "dark",
        "backend_error": None,
        "chat_history": [],
        "sources": [],
        "history_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# Session identity lives purely in st.session_state, not the URL - a browser refresh gets a
# brand-new session_id (empty chat) rather than resurrecting whatever was last active. Jumping
# to a specific past conversation (Chat History page's "View") sets session_id/history_loaded
# directly before switching pages, so it doesn't depend on a `sid` query param round-trip either.
if not st.session_state.history_loaded:
    # Only mark this done once both calls actually succeed - flipping the flag unconditionally
    # meant a single transient failure (e.g. the backend still busy loading its embedding model
    # right after a restart) permanently stuck the sidebar on "Sources: 0" for the rest of the
    # browser session, even once the backend was back up and the data was there all along.
    history = api_get(f"/chat/{st.session_state.session_id}/history")
    sources = api_get(f"/sources/{st.session_state.session_id}")
    if history is not None and sources is not None:
        st.session_state.chat_history = history["messages"]
        st.session_state.sources = sources["sources"]
        st.session_state.history_loaded = True


def _typing_dots_html(label: str = "") -> str:
    """Animated dots (`.typing-dots` in style.css), same visual language for chat replies,
    file/image ingestion, and website/YouTube fetches."""
    suffix = f'<span class="typing-label">{html.escape(label)}</span>' if label else ""
    return f'<span class="typing-dots"><span></span><span></span><span></span></span>{suffix}'


def _typing_bubble_html(label: str = "") -> str:
    """Standalone bubble for ingestion panels (not reused inside the chat placeholder loop, which
    needs identical outer markup across updates - see the comment in handle_query)."""
    return f'<div class="chat-row ai"><div class="chat-bubble ai typing-bubble">{_typing_dots_html(label)}</div></div>'


def _load_css() -> str:
    return (Path(__file__).resolve().parent / "assets" / "style.css").read_text(encoding="utf-8")


st.markdown(f"<style>{_load_css()}</style>", unsafe_allow_html=True)
st.components.v1.html(
    f"<script>window.parent.document.documentElement.setAttribute('data-theme', '{st.session_state.theme}');</script>",
    height=0,
)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        mascot_path = _find_mascot_image()
        avatar_html = (
            f'<img src="{_mascot_avatar_data_uri(str(mascot_path), mascot_path.stat().st_mtime)}" '
            'style="width:38px;height:38px;border-radius:50%;object-fit:cover;flex-shrink:0;'
            'box-shadow:0 4px 14px #a8672f40;" />'
            if mascot_path
            else '<div class="brand-avatar">🐼</div>'
        )
        st.markdown(
            f'<div class="brand-row">{avatar_html}'
            '<div><div class="brand-name">PandexAI</div>'
            '<div class="brand-tagline">Your document, video &amp; web assistant</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.page_link("streamlit_app.py", label="💬 Chat")
        st.page_link("pages/1_History.py", label="📜 History")

        theme_col, clear_col = st.columns(2)
        with theme_col:
            if st.button("🌗 Theme", use_container_width=True):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
        with clear_col:
            if st.button("🗑️ Clear Chat", use_container_width=True, disabled=not st.session_state.chat_history):
                result = api_delete(f"/chat/{st.session_state.session_id}")
                if result is None:
                    st.error(f"Could not clear chat: {st.session_state.backend_error or 'backend unreachable'}")
                else:
                    st.session_state.chat_history = []
                    st.rerun()

        st.markdown('<div class="sidebar-section-title">Backend Connection</div>', unsafe_allow_html=True)
        health = api_get("/health")
        backend_up = bool(health)
        # Once we know the backend is down, avoid firing off several more doomed requests below
        # (settings, usage) that would each eat another timeout in sequence and stall the page.
        providers = api_get("/providers") or {} if backend_up else {}
        if health:
            configured = [name for name in ("groq", "openrouter", "openai", "anthropic", "google") if providers.get(name)]
            embeddings_ready = providers.get("embeddings_ready", True)
            if configured and embeddings_ready:
                st.markdown(
                    f'<span class="status-badge connected"><span class="status-dot"></span>'
                    f'Backend up · chat: {", ".join(configured)} · embeddings: '
                    f'{providers.get("embedding_provider", "local")}</span>',
                    unsafe_allow_html=True,
                )
            elif not embeddings_ready:
                st.markdown(
                    '<span class="status-badge disconnected"><span class="status-dot"></span>'
                    "Backend up, embeddings not ready</span>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY isn't set in backend/.env. "
                    "Either set that key, or set EMBEDDING_PROVIDER=local (free, no key) and restart."
                )
            else:
                st.markdown(
                    '<span class="status-badge disconnected"><span class="status-dot"></span>'
                    "Backend up, no chat LLM key set</span>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Set GROQ_API_KEY (free, https://console.groq.com/keys) in backend/.env, then restart the backend."
                )
        else:
            st.markdown(
                '<span class="status-badge disconnected"><span class="status-dot"></span>Backend unreachable</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Could not reach {BACKEND_URL}. Is `uvicorn main:app` running (from backend/)?")

        if not backend_up:
            return  # nothing below here works without the backend - stop making doomed requests

        with st.expander("⚙️ Settings"):
            provider_options = ["groq", "openrouter", "openai", "anthropic", "google"]
            current = api_get(f"/settings/{st.session_state.session_id}") or {}
            chosen_provider = st.selectbox(
                "LLM provider", provider_options,
                index=provider_options.index(current.get("llm_provider") or providers.get("default_provider", "groq")),
                format_func=lambda p: PROVIDER_LABELS.get(p, p),
            )
            chosen_model = st.text_input("Model override (optional)", value=current.get("chat_model") or "")
            top_k = st.slider("Rerank top-k (chunks used per answer)", 1, 10, current.get("rerank_top_k") or 8)
            mcp_enabled = st.toggle(
                "Enable MCP math server",
                value=current.get("mcp_math_enabled") if current.get("mcp_math_enabled") is not None else True,
                help="When off, math/calculation questions are answered from your documents like any other "
                "question instead of being routed to the MCP math server.",
            )
            if st.button("Save settings", use_container_width=True):
                api_post(
                    f"/settings/{st.session_state.session_id}",
                    {
                        "llm_provider": chosen_provider,
                        "chat_model": chosen_model or None,
                        "rerank_top_k": top_k,
                        "mcp_math_enabled": mcp_enabled,
                    },
                )
                st.toast("Settings saved.", icon="✅")

        render_sources_section()
        render_chat_history_panel()
        render_usage_panel()
        render_export_panel()


MAX_SOURCES_PER_CATEGORY = 5
_CATEGORY_LABELS = {"upload": "file uploads", "website": "websites", "youtube": "YouTube videos"}


def _source_category(source_type: str) -> str:
    return source_type if source_type in ("website", "youtube") else "upload"


def _category_count(category: str) -> int:
    return sum(1 for s in st.session_state.sources if _source_category(s["type"]) == category)


def _refresh_sources() -> None:
    """Re-fetch the source list from the backend rather than trusting the in-memory copy.

    `st.session_state.sources` is otherwise only ever set once (at page bootstrap) and then
    mutated locally on every add/delete - if a rerun ever gets interrupted between the backend
    call completing and that local mutation running (e.g. a page refresh mid-flight), the sidebar
    silently drifts out of sync with what's actually stored, and a since-added source looks
    invisible until you happen to resubmit it and get a confusing "duplicate" response.
    """
    sources = api_get(f"/sources/{st.session_state.session_id}")
    if sources is not None:
        st.session_state.sources = sources["sources"]


@st.dialog("Limit reached")
def _show_limit_dialog(category: str) -> None:
    st.warning(f"You can only add up to {MAX_SOURCES_PER_CATEGORY} {_CATEGORY_LABELS[category]} per session.")
    st.caption("Remove one from the Sources list below to add another.")
    if st.button("Got it", use_container_width=True):
        st.rerun()


def render_sources_section() -> None:
    st.markdown(
        f'<div class="sidebar-section-title">Sources <span class="source-type-badge">'
        f"{len(st.session_state.sources)}</span></div>",
        unsafe_allow_html=True,
    )

    upload_tab, website_tab, youtube_tab = st.tabs(["⬆️ Upload", "🌐 Website", "▶️ YouTube"])

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Drag & drop files here",
            type=["pdf", "docx", "txt", "csv", "xlsx", "pptx", "png", "jpg", "jpeg", "webp", "md", "json"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        upload_count = _category_count("upload")
        if upload_count >= MAX_SOURCES_PER_CATEGORY:
            st.caption(f"⚠️ Upload limit reached ({upload_count}/{MAX_SOURCES_PER_CATEGORY}).")
        if uploaded_files and st.button("⚡ Process files", use_container_width=True):
            if upload_count + len(uploaded_files) > MAX_SOURCES_PER_CATEGORY:
                _show_limit_dialog("upload")
            else:
                process_file_uploads(uploaded_files)
                st.rerun()

    with website_tab:
        url = st.text_input("Website URL", placeholder="https://example.com/article", label_visibility="collapsed")
        website_count = _category_count("website")
        if website_count >= MAX_SOURCES_PER_CATEGORY:
            st.caption(f"⚠️ Website limit reached ({website_count}/{MAX_SOURCES_PER_CATEGORY}).")
        if st.button("Add Website", use_container_width=True, disabled=not url):
            if website_count >= MAX_SOURCES_PER_CATEGORY:
                _show_limit_dialog("website")
            else:
                bubble = st.empty()
                bubble.markdown(_typing_bubble_html("Scraping & embedding..."), unsafe_allow_html=True)
                result = api_post(f"/sources/{st.session_state.session_id}/website", {"url": url}, timeout=60.0)
                bubble.empty()
                _handle_ingest_result(result)

    with youtube_tab:
        yt_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
        youtube_count = _category_count("youtube")
        if youtube_count >= MAX_SOURCES_PER_CATEGORY:
            st.caption(f"⚠️ YouTube limit reached ({youtube_count}/{MAX_SOURCES_PER_CATEGORY}).")
        if st.button("Add YouTube Video", use_container_width=True, disabled=not yt_url):
            if youtube_count >= MAX_SOURCES_PER_CATEGORY:
                _show_limit_dialog("youtube")
            else:
                bubble = st.empty()
                bubble.markdown(_typing_bubble_html("Fetching transcript & embedding..."), unsafe_allow_html=True)
                result = api_post(f"/sources/{st.session_state.session_id}/youtube", {"url": yt_url}, timeout=60.0)
                bubble.empty()
                _handle_ingest_result(result)

    if st.session_state.sources:
        for source in st.session_state.sources:
            icon = SOURCE_ICONS.get(source["type"], "📄")
            doc_col, del_col = st.columns([4, 1])
            with doc_col:
                st.markdown(
                    f'<div class="doc-card"><div><div class="doc-name">{icon} {html.escape(source["name"])}'
                    f'<span class="source-type-badge">{source["type"]}</span></div>'
                    f'<div class="doc-meta">{format_file_size(source["size_bytes"])} · {source["chunks"]} chunks</div>'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            with del_col:
                if st.button("✕", key=f"del_{source['id']}", help="Remove this source"):
                    api_delete(f"/sources/{st.session_state.session_id}/{source['id']}")
                    _refresh_sources()
                    st.rerun()

        if st.button("🗑️ Delete All Sources", use_container_width=True):
            api_delete(f"/sources/{st.session_state.session_id}")
            _refresh_sources()
            st.rerun()
    else:
        st.caption("No sources yet - upload a file or add a URL above.")


def process_file_uploads(uploaded_files: list) -> None:
    bubble = st.empty()
    bubble.markdown(_typing_bubble_html("Reading & embedding..."), unsafe_allow_html=True)
    files_payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploaded_files]
    results = api_post(f"/sources/{st.session_state.session_id}/upload", files=files_payload)
    bubble.empty()

    if not results:
        st.error(st.session_state.backend_error or "Upload failed.")
        return

    any_success = False
    for result in results:
        if result["success"]:
            any_success = True
        elif result["duplicate"]:
            st.info(result["message"])
        else:
            st.warning(result["message"])
    if any_success:
        _refresh_sources()
    st.toast("Upload complete.", icon="✅")


def _handle_ingest_result(result: dict | None) -> None:
    if not result:
        st.error(st.session_state.backend_error or "Request failed.")
        return
    if result["success"]:
        _refresh_sources()
        st.toast(result["message"], icon="✅")
        st.rerun()
    elif result["duplicate"]:
        st.info(result["message"])
    else:
        st.error(result["message"])


def render_chat_history_panel() -> None:
    st.markdown('<div class="sidebar-section-title">Chat History</div>', unsafe_allow_html=True)
    if st.button("📜 View All Conversations", use_container_width=True):
        st.switch_page("pages/1_History.py")


def render_usage_panel() -> None:
    usage = api_get(f"/usage/{st.session_state.session_id}") or {}
    if not usage.get("input_tokens") and not usage.get("output_tokens"):
        return  # nothing generated yet in this session - don't show an all-zero panel

    st.markdown('<div class="sidebar-section-title">Token Usage</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="stat-grid">'
        f'<div class="stat-tile"><div class="stat-value">{usage.get("input_tokens", 0):,}</div>'
        '<div class="stat-label">Input tokens</div></div>'
        f'<div class="stat-tile"><div class="stat-value">{usage.get("output_tokens", 0):,}</div>'
        '<div class="stat-label">Output tokens</div></div>'
        f'<div class="stat-tile"><div class="stat-value">${usage.get("estimated_cost_usd", 0):.4f}</div>'
        '<div class="stat-label">Est. cost</div></div>'
        f'<div class="stat-tile"><div class="stat-value">{len(st.session_state.sources)}</div>'
        '<div class="stat-label">Sources</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if usage.get("provider"):
        st.caption(f"Last answered by: {usage['provider']} / {usage['model']}")


@st.cache_data(show_spinner=False)
def _cached_pdf_export(chat_history: list[dict]) -> bytes | None:
    try:
        return export_chat_to_pdf(chat_history)
    except Exception:  # noqa: BLE001 - export is best-effort; never take down the whole page over it
        return None


@st.cache_data(show_spinner=False)
def _cached_text_export(chat_history: list[dict]) -> str:
    return export_chat_to_text(chat_history)


@st.cache_data(show_spinner=False)
def _cached_json_export(chat_history: list[dict]) -> str:
    return export_chat_to_json(chat_history)


def render_export_panel() -> None:
    if not st.session_state.chat_history:
        return
    st.markdown('<div class="sidebar-section-title">Export</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        pdf_bytes = _cached_pdf_export(st.session_state.chat_history)
        if pdf_bytes:
            st.download_button(
                "PDF", pdf_bytes, file_name="chat_export.pdf", mime="application/pdf", use_container_width=True,
            )
        else:
            st.button("PDF", disabled=True, use_container_width=True, help="PDF export failed for this conversation.")
    with e2:
        st.download_button(
            "TXT", _cached_text_export(st.session_state.chat_history),
            file_name="chat_export.txt", mime="text/plain", use_container_width=True,
        )
    with e3:
        st.download_button(
            "JSON", _cached_json_export(st.session_state.chat_history),
            file_name="chat_export.json", mime="application/json", use_container_width=True,
        )


# ----------------------------------------------------------------------------
# Chat rendering
# ----------------------------------------------------------------------------
_FENCED_CODE_RE = re.compile(r"```(?:\w*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")


def _render_inline_markdown(text: str) -> str:
    text = _INLINE_CODE_RE.sub(r"<code>\1</code>", text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    text = _LINK_RE.sub(r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)
    return text.replace("\n", "<br>")


def _render_markdown_lite(content: str) -> str:
    """Render a small, safe subset of markdown (fenced/inline code, bold, italic, links).

    Escapes the raw content *first*, then only ever introduces our own controlled tags on top of
    the escaped text - so this can never be used to inject arbitrary HTML from an uploaded
    document or an LLM response, unlike rendering markdown/HTML from the source directly.
    """
    escaped = html.escape(content)
    segments: list[str] = []
    last_end = 0
    for match in _FENCED_CODE_RE.finditer(escaped):
        segments.append(_render_inline_markdown(escaped[last_end : match.start()]))
        segments.append(f'<pre class="code-block"><code>{match.group(1)}</code></pre>')
        last_end = match.end()
    segments.append(_render_inline_markdown(escaped[last_end:]))
    return "".join(segments)


def render_message(role: str, content: str, citations: list[str] | None = None, timestamp: str = "") -> None:
    row_class = "user" if role == "user" else "ai"
    align = "flex-end" if role == "user" else "flex-start"
    safe_content = _render_markdown_lite(content)

    citation_html = ""
    if citations:
        pills = "".join(f'<span class="citation-pill">📎 {html.escape(c)}</span>' for c in citations)
        citation_html = f"<div>{pills}</div>"

    # Single line, no embedded newlines/indentation: a multi-line triple-quoted f-string here
    # previously left trailing indented "</div>" lines that Streamlit's markdown parser could
    # treat as a 4-space-indented Markdown code block instead of continuing the raw HTML block,
    # rendering literal "</div>" text under the bubble. Indentation only matters at line starts,
    # so a single line sidesteps the whole class of bug.
    st.markdown(
        f'<div class="chat-row {row_class}">'
        f'<div style="max-width:78%; display:flex; flex-direction:column; align-items:{align};">'
        f'<div class="chat-bubble {row_class}">{safe_content}</div>'
        f'<div class="chat-meta">{timestamp}</div>'
        f"{citation_html}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    st.markdown(
        '<div class="welcome-wrap">'
        '<div class="welcome-title">Hello, How can I assist you?</div>'
        '<div class="welcome-subtitle">Upload documents, add a website, or drop in a YouTube link '
        "in the Sources panel, then ask me anything about them - grounded strictly in what you "
        "provided, with citations.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    mascot_path = _find_mascot_image()
    mascot_visual = (
        f'<img src="{_mascot_hero_data_uri(str(mascot_path), mascot_path.stat().st_mtime)}" alt="PandexAI mascot" />'
        if mascot_path
        else '<div class="mascot-emoji">🐼📖</div>'
    )
    st.markdown(
        '<div class="mascot-wrap"><div class="mascot-card">'
        f"{mascot_visual}"
        '<div class="mascot-caption">Even <strong>PandexAI</strong> does its homework - '
        "upload yours and let's get reading.</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def handle_query(prompt: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": prompt, "citations": [], "timestamp": ""})
    render_message("user", prompt)

    # Keep the placeholder's HTML structure identical across every update (loading -> streaming) -
    # swapping between very different nested markup in one st.empty() can cause the browser to
    # render a stale/partial DOM patch. Only the text inside the bubble ever changes.
    def _render_bubble(placeholder_, text: str) -> None:
        placeholder_.markdown(
            f'<div class="chat-row ai"><div class="chat-bubble ai">{text}</div></div>',
            unsafe_allow_html=True,
        )

    placeholder = st.empty()
    _render_bubble(placeholder, _typing_dots_html())

    collected = ""
    citations: list[str] = []
    error_message: str | None = None
    try:
        for event in stream_chat_events(st.session_state.session_id, prompt):
            if event["type"] == "token":
                collected += event["data"]
                safe = html.escape(collected).replace("\n", "<br>")
                _render_bubble(placeholder, f"{safe}▌")
            elif event["type"] == "done":
                citations = event.get("citations", [])
            elif event["type"] == "error":
                error_message = event["message"]
    except httpx.HTTPError as exc:
        error_message = f"Could not reach the backend: {exc}"

    placeholder.empty()
    answer = error_message or collected or "No response received."
    render_message("assistant", answer, citations=citations)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "citations": citations, "timestamp": ""}
    )


# ----------------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------------
render_sidebar()

chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        render_welcome()
    else:
        for message in st.session_state.chat_history:
            render_message(
                message["role"], message["content"], message.get("citations"), message.get("timestamp", "")
            )

submission = st.chat_input("Ask a question about your documents... (or use the mic)", accept_audio=True)

prompt: str | None = None
if submission:
    if submission.audio is not None:
        with st.spinner("Transcribing your voice..."):
            transcription = api_post("/transcribe", files={"file": ("voice.wav", submission.audio.getvalue(), "audio/wav")})
        if transcription and transcription.get("text"):
            prompt = f"{submission.text} {transcription['text']}".strip() if submission.text else transcription["text"]
        else:
            st.error(f"Could not transcribe audio: {st.session_state.backend_error or 'unknown error'}")
    elif submission.text:
        prompt = submission.text

if prompt:
    with chat_container:
        handle_query(prompt)
    # render_sidebar() (and its Chat History panel) already ran earlier in this same script pass,
    # against the pre-question chat_history - without this rerun, the sidebar would only catch up
    # to the just-completed exchange on the user's *next* interaction, not immediately.
    st.rerun()
