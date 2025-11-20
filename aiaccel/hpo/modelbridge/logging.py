"""Lightweight logging utilities."""

from __future__ import annotations

from collections.abc import Iterable
import logging
import os
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(log_level: str, output_dir: Path, *, reset_handlers: bool = True, console: bool = True) -> Path:
    """Configure root logging with rich console and file handlers."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"

    root = logging.getLogger()
    if reset_handlers:
        _clear_handlers(root.handlers)

    formatter = logging.Formatter(_LOG_FORMAT)
    _ensure_file_handler(root, log_path, formatter)

    if console and not os.environ.get("AIACCEL_LOG_SILENT"):
        _ensure_console_handler(root)

    root.setLevel(level)

    logging.captureWarnings(True)
    return log_path


def _clear_handlers(handlers: Iterable[logging.Handler]) -> None:
    """Detach and close existing handlers to avoid duplicate output."""

    for handler in list(handlers):
        handler.close()
        logging.getLogger().removeHandler(handler)


def _ensure_file_handler(logger: logging.Logger, path: Path, formatter: logging.Formatter) -> None:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(getattr(handler, "baseFilename", "")) == path:
            return
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _ensure_console_handler(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            return
    console_handler = RichHandler(
        console=Console(),
        rich_tracebacks=True,
        omit_repeated_times=False,
        show_time=False,
    )
    logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module level logger."""

    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
