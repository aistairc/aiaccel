"""Logging setup helpers for modelbridge and toolkit users."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(
    log_level: str,
    output_dir: Path,
    *,
    reset_handlers: bool = True,
    console: bool = True,
    file: bool = True,
    json_logs: bool = False,
) -> Path:
    """Configure root logging handlers for modelbridge.

    Args:
        log_level: Logging level name.
        output_dir: Output directory where log file is created.
        reset_handlers: Whether to clear existing root handlers first.
        console: Whether to enable console logging.
        file: Whether to enable file logging.
        json_logs: Whether to use JSON log format.

    Returns:
        Path: Log file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "aiaccel_modelbridge.log"

    file = file and not os.environ.get("AIACCEL_LOG_NO_FILE")
    if not console and not file:
        return log_path

    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()

    if reset_handlers:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)

    if json_logs:
        fmt_str = '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
    else:
        fmt_str = _LOG_FORMAT
    formatter = logging.Formatter(fmt_str)

    if file:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if console and not os.environ.get("AIACCEL_LOG_SILENT"):
        console_handler = RichHandler(
            console=Console(),
            rich_tracebacks=True,
            omit_repeated_times=False,
            show_time=False,
        )
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    root.setLevel(level)
    logging.captureWarnings(True)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name.

    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger(name)
