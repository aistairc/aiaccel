"""Lightweight logging utilities."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(log_level: str, output_dir: Path) -> Path:
    """Configure root logging and return the log file path."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"

    root = logging.getLogger()
    _clear_handlers(root.handlers)

    formatter = logging.Formatter(_LOG_FORMAT)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(level)

    logging.captureWarnings(True)
    return log_path


def _clear_handlers(handlers: Iterable[logging.Handler]) -> None:
    """Detach and close existing handlers to avoid duplicate output."""

    for handler in list(handlers):
        handler.close()
        logging.getLogger().removeHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module level logger."""

    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
