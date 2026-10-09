"""Centralised logging: console + rotating file handler."""

from __future__ import annotations

import logging
import logging.handlers

from it_assistant.config import Settings

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(settings: Settings, log_file: str = "app.log") -> None:
    """Configure root logging once per process (Streamlit reruns the script often)."""
    global _configured
    if _configured:
        return

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_dir / log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()  # stderr: never stdout (the MCP stdio channel)
    console_handler.setFormatter(formatter)

    root = logging.getLogger("it_assistant")
    root.setLevel(settings.log_level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.propagate = False

    # Keep third-party noise down.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)

    _configured = True
