"""Generic IO helpers used by modelbridge steps."""

from __future__ import annotations

from typing import Any

from collections.abc import Iterable, Mapping
import csv
import hashlib
import json
from pathlib import Path


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write rows to a CSV file.

    Args:
        path: Destination CSV path.
        rows: Iterable of row mappings.

    Returns:
        Path: Written CSV path.
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


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into row dictionaries.

    Args:
        path: Source CSV path.

    Returns:
        list[dict[str, str]]: Parsed rows.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_json(path: Path, payload: Any) -> Path:
    """Serialize payload as JSON.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable object.

    Returns:
        Path: Written JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def read_json(path: Path) -> Any:
    """Load JSON from file path.

    Args:
        path: Source JSON path.

    Returns:
        Any: Parsed JSON object.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """Calculate file hash digest.

    Args:
        path: Target file path.
        algorithm: Hash algorithm name.

    Returns:
        str: Hex digest string.
    """
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
