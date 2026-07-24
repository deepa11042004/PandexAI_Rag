"""Text extraction for plain text, Markdown, and JSON files."""

from __future__ import annotations

import json


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def parse_txt(data: bytes) -> str:
    return _decode(data)


def parse_markdown(data: bytes) -> str:
    # Kept as plain text; heading/list structure still carries semantic value for chunking.
    return _decode(data)


def _flatten_json(value, prefix: str = "") -> list[str]:
    """Flatten arbitrary JSON into readable 'path: value' lines so it embeds meaningfully."""
    lines: list[str] = []
    if isinstance(value, dict):
        for key, val in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json(val, path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            lines.extend(_flatten_json(item, f"{prefix}[{idx}]"))
    else:
        lines.append(f"{prefix}: {value}")
    return lines


def parse_json(data: bytes) -> str:
    parsed = json.loads(_decode(data))
    return "\n".join(_flatten_json(parsed))
