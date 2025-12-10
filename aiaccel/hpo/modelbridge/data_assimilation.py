"""Data assimilation pipeline for external command execution."""

from __future__ import annotations

from typing import Any

import os
from pathlib import Path
import shlex
import subprocess

from .config import DataAssimilationConfig
from .io import hash_file, write_json
from .logging import configure_logging, get_logger


def run_data_assimilation(
    config: DataAssimilationConfig,
    *,
    dry_run: bool = False,
    quiet: bool = True,
    log_to_file: bool = False,
    json_logs: bool = False,
) -> dict[str, Any]:
    """Execute the data assimilation workflow via an external command."""

    if not config.enabled:
        if not quiet:
            print("Data assimilation is disabled in configuration.")
        return {"status": "disabled"}

    if dry_run:
        return {
            "status": "dry_run",
            "command": config.command,
            "cwd": str(config.cwd) if config.cwd else None,
            "env": config.env,
        }

    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    console_enabled = not quiet
    if log_to_file or console_enabled:
        configure_logging(
            "INFO",
            output_root,
            reset_handlers=True,
            console=console_enabled,
            file=log_to_file,
            json_logs=json_logs,
        )
    logger = get_logger(__name__)

    cmd_args = (
        shlex.split(config.command) if isinstance(config.command, str) else [str(c) for c in config.command]
    )
    cwd = Path(config.cwd).resolve() if config.cwd else Path.cwd()
    env = os.environ.copy()
    if config.env:
        env.update(config.env)

    logger.info("Starting data assimilation command: %s", cmd_args)
    logger.info("Working directory: %s", cwd)

    try:
        # Capture output to log files if logging is enabled, or pass through
        stdout_dest = subprocess.PIPE
        stderr_dest = subprocess.PIPE

        process = subprocess.Popen(
            cmd_args,
            cwd=cwd,
            env=env,
            stdout=stdout_dest,
            stderr=stderr_dest,
            text=True,
        )

        stdout_str, stderr_str = process.communicate()

        if stdout_str:
            logger.info("Command stdout:\n%s", stdout_str.strip())
        if stderr_str:
            logger.error("Command stderr:\n%s", stderr_str.strip())

        if process.returncode != 0:
            logger.error("Data assimilation command failed with return code %d", process.returncode)
            raise subprocess.CalledProcessError(process.returncode, cmd_args, output=stdout_str, stderr=stderr_str)

        logger.info("Data assimilation command completed successfully.")

    except Exception as exc:
        logger.exception("Failed to execute data assimilation command")
        raise RuntimeError(f"Data assimilation failed: {exc}") from exc

    # We don't know what the external command produced, so we just check for manifest
    # or summary if expected, but here we just return a success status.
    # The external command is responsible for writing its own artifacts.
    # However, we can look for a 'data_assimilation_summary.json' in output_root if it exists.
    summary_path = output_root / "data_assimilation_summary.json"
    summary: dict[str, Any] = {"status": "success", "return_code": 0}

    if summary_path.exists():
        try:
            import json
            with summary_path.open("r", encoding="utf-8") as f:
                summary["external_summary"] = json.load(f)
            _write_da_manifest(output_root, summary_path)
        except Exception as exc:
            logger.warning("Found summary file but failed to read it: %s", exc)

    return summary


def _write_da_manifest(output_root: Path, summary_path: Path) -> None:
    artifacts = []
    if summary_path.exists():
        artifacts.append(
            {"path": str(summary_path), "sha256": hash_file(summary_path), "size": summary_path.stat().st_size, "algorithm": "sha256"}
        )
    # The external command might have produced other files, but we can only track known ones
    # unless we scan the directory. For now, we only track the summary if it exists.
    write_json(output_root / "data_assimilation_manifest.json", {"artifacts": artifacts})


__all__ = ["run_data_assimilation"]
