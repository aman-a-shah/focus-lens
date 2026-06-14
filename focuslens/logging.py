"""Structured-ish logging setup.

Single helper so every entrypoint configures logging the same way. Kept dependency-free
(stdlib ``logging``); swap the formatter for JSON later if we ship logs anywhere.
"""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        datefmt=_DATEFMT,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
