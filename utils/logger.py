"""Shared logging configuration used by both the backend and frontend processes."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def _configure_root(level: str) -> None:
    global _configured
    if _configured:
        return
    LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a module-scoped logger, configuring the root logger's handlers on first use."""
    _configure_root(level)
    return logging.getLogger(name)
