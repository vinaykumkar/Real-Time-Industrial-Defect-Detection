"""Structured, rotating logging for the whole application.

Writes logs under logs/ (app.log) with rotation, and mirrors to console.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure the root 'defect' logger once; safe to call repeatedly."""
    global _CONFIGURED
    logger = logging.getLogger("defect")
    if _CONFIGURED:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOGS_DIR / "app.log", maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str = "defect") -> logging.Logger:
    return logging.getLogger(name)
