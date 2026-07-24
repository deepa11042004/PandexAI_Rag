"""OCR text extraction for images via Tesseract (lightweight: no PyTorch dependency).

Requires the Tesseract OCR *binary* to be installed on the system separately from the
`pytesseract` Python package (see README). If it's missing, we raise a clear, actionable
error instead of a cryptic one.
"""

from __future__ import annotations

import io

import pytesseract
from PIL import Image

from backend.config import get_settings

_configured = False


def _configure_tesseract() -> None:
    global _configured
    if _configured:
        return
    cmd = get_settings().tesseract_cmd
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    _configured = True


def parse_image(data: bytes) -> str:
    _configure_tesseract()
    try:
        image = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or not on PATH. Install it (see README) or set "
            "TESSERACT_CMD in your .env to its full executable path."
        ) from exc
