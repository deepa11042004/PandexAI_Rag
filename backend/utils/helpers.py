"""Framework-agnostic helpers shared by the backend and frontend: text cleaning, token/cost
accounting, and small formatting utilities. Session/chat persistence lives in
`backend.session_manager` now that the app is split into a FastAPI backend + Streamlit frontend.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

import tiktoken

# Approximate USD price per 1K tokens (input, output) for supported models.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control/null characters from extracted document text."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens for a piece of text using tiktoken, falling back to a cl100k estimate."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text or ""))


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Estimate USD cost for a model call. Unknown/free-tier models (OpenRouter/Groq) cost $0."""
    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]


def format_file_size(num_bytes: int) -> str:
    """Human-readable file size, e.g. 1536 -> '1.5 KB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def now_str() -> str:
    return datetime.now().strftime("%H:%M")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def content_hash(data: bytes) -> str:
    """SHA-256 hash of raw content, used for duplicate-source detection."""
    import hashlib

    return hashlib.sha256(data).hexdigest()
