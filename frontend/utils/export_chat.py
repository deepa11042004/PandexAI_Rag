"""Export chat history to PDF, plain text, or JSON for download."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos

_SAFE_LATIN1 = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "--", "…": "...", "•": "-",
})

# Any unbroken run of 40+ non-space characters (long URLs, hashes, table artifacts pulled out of
# PDFs/resumes) has no wrap point for FPDF's multi_cell, which raises FPDFException ("not enough
# horizontal space to render a single character") instead of wrapping. Chunking it with soft spaces
# gives multi_cell somewhere to break.
_LONG_TOKEN_RE = re.compile(r"\S{40,}")


def _break_long_tokens(text: str, chunk: int = 40) -> str:
    return _LONG_TOKEN_RE.sub(lambda m: " ".join(m.group(0)[i : i + chunk] for i in range(0, len(m.group(0)), chunk)), text)


def _sanitize(text: str) -> str:
    """FPDF's core fonts are Latin-1 only; swap smart punctuation and drop anything else unsupported."""
    text = text.translate(_SAFE_LATIN1)
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return _break_long_tokens(text)


class _ChatPDF(FPDF):
    def header(self) -> None:
        # Deprecated `ln=True` leaves stale cursor/width state across an automatic page-break-
        # triggered header() call, which manifests later as multi_cell's "not enough horizontal
        # space" on unrelated content - explicit new_x/new_y avoids that entirely.
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, "Chat Export - RAG Document Assistant", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, _sanitize(datetime.now().strftime("%Y-%m-%d %H:%M")), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(4)


def export_chat_to_pdf(chat_history: list[dict[str, Any]]) -> bytes:
    """Render chat turns to a PDF and return raw bytes suitable for st.download_button."""
    pdf = _ChatPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for message in chat_history:
        role = "You" if message["role"] == "user" else "Assistant"
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(80, 40, 160) if role == "You" else pdf.set_text_color(20, 120, 100)
        # multi_cell's default new_x is XPos.RIGHT (not LMARGIN) in this fpdf2 version, which
        # strands the cursor at the right margin and makes the *next* multi_cell fail with
        # "not enough horizontal space to render a single character" - see header() above.
        pdf.multi_cell(0, 7, _sanitize(f"{role}:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(0, 6, _sanitize(message.get("content", "")), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        citations = message.get("citations") or []
        if citations:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(110, 110, 110)
            pdf.multi_cell(0, 5, _sanitize("Sources: " + ", ".join(citations)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(3)

    return bytes(pdf.output())


def export_chat_to_text(chat_history: list[dict[str, Any]]) -> str:
    lines = [f"Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M')}", "=" * 40, ""]
    for message in chat_history:
        role = "You" if message["role"] == "user" else "Assistant"
        lines.append(f"[{message.get('timestamp', '')}] {role}: {message.get('content', '')}")
        citations = message.get("citations") or []
        if citations:
            lines.append(f"  Sources: {', '.join(citations)}")
        lines.append("")
    return "\n".join(lines)


def export_chat_to_json(chat_history: list[dict[str, Any]]) -> str:
    return json.dumps(chat_history, indent=2, ensure_ascii=False)
