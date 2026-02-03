"""Utility functions for modelbridge (IO, Logging)."""

from __future__ import annotations

from typing import Any

from collections.abc import Iterable, Mapping
import csv
import hashlib
import json
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
    """Configure root logging with rich console and file handlers.

    Args:
        log_level (str): Logging level (e.g. "INFO", "DEBUG").
        output_dir (Path): Directory where the log file will be created.
        reset_handlers (bool): Whether to clear existing handlers.
        console (bool): Whether to enable console logging.
        file (bool): Whether to enable file logging.
        json_logs (bool): Whether to use JSON formatting for logs.

    Returns:
        Path: The path to the log file.
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
    """Return a module level logger.

    Args:
        name (str): Logger name.

    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger(name)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write rows to path as a CSV file.

    Args:
        path (Path): Destination file path.
        rows (Iterable[Mapping[str, Any]]): Data rows to write.

    Returns:
        Path: The written file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_iter = list(rows)
    if not rows_iter:
        path.write_text("", encoding="utf-8")
        return path

    header = sorted(rows_iter[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows_iter:
            writer.writerow({key: row.get(key, "") for key in header})
    return path


def write_json(path: Path, payload: Any) -> Path:
    """Serialise payload as JSON at path.

    Args:
        path (Path): Destination file path.
        payload (Any): Data to serialize.

    Returns:
        Path: The written file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read path into a list of dict rows.

    Args:
        path (Path): Source file path.

    Returns:
        list[dict[str, str]]: List of rows as dicts.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def read_json(path: Path) -> Any:
    """Load a JSON mapping from path.

    Args:
        path (Path): Source file path.

    Returns:
        Any: Parsed JSON data.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """Return the hex digest of path using the requested algorithm.

    Args:
        path (Path): File path to hash.
        algorithm (str): Hash algorithm name (default: "sha256").

    Returns:
        str: Hex digest string.
    """
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
