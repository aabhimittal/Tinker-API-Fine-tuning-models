"""Centralised logging setup (plain or structured JSON)."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure the root logger once. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    if json:
        try:
            from pythonjsonlogger import jsonlogger

            fmt = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        except ImportError:  # pragma: no cover - optional dependency
            fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    else:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
