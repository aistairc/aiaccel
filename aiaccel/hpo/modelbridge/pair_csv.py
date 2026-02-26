"""Shared pair-CSV conversion helpers for collect/analyze stages."""

from __future__ import annotations

from typing import Any

from collections.abc import Sequence
import logging
from pathlib import Path

from .common import read_csv

PairRecord = tuple[int, dict[str, float], dict[str, float]]
_LOGGER = logging.getLogger(__name__)


def pairs_to_rows(pairs: Sequence[PairRecord]) -> list[dict[str, Any]]:
    """Convert pair tuples into CSV row dictionaries.

    This helper serializes pair records into ``run_id``/``macro_*``/``micro_*``
    columns for CSV writing.

    Args:
        pairs: Parsed pair records.

    Returns:
        list[dict[str, Any]]: CSV-compatible row payloads.

    Raises:
        ValueError: If pair payload cannot be converted to row mappings.
    """
    rows: list[dict[str, Any]] = []
    for run_id, macro, micro in pairs:
        row: dict[str, Any] = {"run_id": run_id}
        row.update({f"macro_{name}": value for name, value in macro.items()})
        row.update({f"micro_{name}": value for name, value in micro.items()})
        rows.append(row)
    return rows


def parse_pairs_csv(path: Path) -> list[PairRecord]:
    """Parse pair CSV into run/macro/micro tuples.

    This helper reads pair CSV rows, validates ``run_id``, and returns rows that
    include both macro and micro parameter payloads.

    Args:
        path: Source pair CSV path.

    Returns:
        list[PairRecord]: Parsed pair records.

    Raises:
        OSError: If the source CSV file cannot be read.
        ValueError: If numeric field conversion fails.
    """
    parsed: list[PairRecord] = []
    for row_index, row in enumerate(read_csv(path), start=1):
        try:
            run_id = int(row.get("run_id", "0"))
        except ValueError:
            _LOGGER.warning("Skipping pair row with invalid run_id (file=%s, row=%d)", path, row_index)
            continue
        macro: dict[str, float] = {}
        micro: dict[str, float] = {}
        for key, raw in row.items():
            if raw in {"", None}:
                continue
            if key.startswith("macro_"):
                macro[key.removeprefix("macro_")] = float(raw)
            elif key.startswith("micro_"):
                micro[key.removeprefix("micro_")] = float(raw)
        if macro and micro:
            parsed.append((run_id, macro, micro))
    return parsed
