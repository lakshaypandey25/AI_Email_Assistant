"""Structured logging with automatic PII redaction.

All logs go to **stderr** (never stdout) so the MCP stdio transport stays clean.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from email_assistant.core.pii import scrub_structure

_SKIP_KEYS = {"timestamp", "level", "logger"}


def redact_pii_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor that redacts PII in every string value."""
    return {k: (v if k in _SKIP_KEYS else scrub_structure(v)) for k, v in event_dict.items()}


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog + stdlib logging. Safe to call more than once."""
    logging.basicConfig(level=getattr(logging, level), stream=sys.stderr, format="%(message)s", force=True)
    for noisy in ("googleapiclient.discovery_cache", "httpx", "httpcore", "urllib3", "google_auth_httplib2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            redact_pii_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
