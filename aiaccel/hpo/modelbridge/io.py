"""Small IO helpers used across the pipeline."""

from __future__ import annotations

from typing import Any

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path


def write_csv(path: Path, rows: Iterable[Mapping[str, float]]) -> Path:
    """Write ``rows`` to ``path`` as a CSV file and return the path."""

    import csv

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


def write_json(path: Path, payload: Mapping[str, object]) -> Path:
    """Serialise ``payload`` as JSON at ``path`` and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read ``path`` into a list of dict rows."""

    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping from ``path``."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON document at {path} must be a mapping")
    return data


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """Return the hex digest of ``path`` using the requested algorithm."""

    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["write_csv", "write_json", "read_csv", "read_json", "hash_file"]
