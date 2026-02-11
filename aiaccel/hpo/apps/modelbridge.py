"""CLI adapter for modelbridge."""

from __future__ import annotations

from typing import Any

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path

from aiaccel.hpo.modelbridge import api
from aiaccel.hpo.modelbridge.config import BridgeConfig, generate_schema
from aiaccel.hpo.modelbridge.toolkit.logging import get_logger

STEP_COMMANDS = {
    "prepare-train": "prepare_train",
    "prepare-eval": "prepare_eval",
    "collect-train": "collect_train",
    "collect-eval": "collect_eval",
    "fit-regression": "fit_regression",
    "evaluate-model": "evaluate_model",
    "publish-summary": "publish_summary",
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional CLI argument sequence.

    Returns:
        argparse.Namespace: Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description="Run aiaccel modelbridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute modelbridge steps/profile")
    _add_common_args(run_parser, include_run_args=True)

    for command in STEP_COMMANDS:
        subparser = subparsers.add_parser(command, help=f"Execute {command}")
        _add_common_args(subparser, include_run_args=False)
        if command in {"prepare-train", "prepare-eval"}:
            subparser.add_argument("--emit-commands", action="store_true", help="Emit commands after prepare step")
            subparser.add_argument("--execution-target", choices=["local", "abci"], help="Execution target")
        if command == "collect-train":
            subparser.add_argument("--train-db-path", action="append", default=[], help="Explicit train DB path")
        if command == "collect-eval":
            subparser.add_argument("--eval-db-path", action="append", default=[], help="Explicit eval DB path")

    emit_parser = subparsers.add_parser("emit-commands", help="Emit optimize commands from plan")
    _add_common_args(emit_parser, include_run_args=False)
    emit_parser.add_argument("--role", choices=["train", "eval"], required=True, help="Target role")
    emit_parser.add_argument("--format", choices=["shell", "json"], required=True, help="Output format")
    emit_parser.add_argument("--execution-target", choices=["local", "abci"], help="Execution target override")

    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    _add_common_args(validate_parser, include_run_args=False, include_logging=False)
    validate_parser.add_argument("--print-config", action="store_true", help="Print resolved config JSON")

    subparsers.add_parser("schema", help="Print JSON schema for modelbridge config")
    return parser.parse_args(argv)


def _add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_run_args: bool,
    include_logging: bool = True,
) -> None:
    """Add shared CLI argument groups to a parser.

    Args:
        parser: Target parser to extend.
        include_run_args: Whether to include run-specific options.
        include_logging: Whether to include logging options.
    """
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--config", "-c", required=True, help="Path to modelbridge config")
    io_group.add_argument("--output_dir", "-o", help="Override output directory")

    options_group = parser.add_argument_group("Options")
    if include_run_args:
        options_group.add_argument("--steps", help="Comma-separated steps")
        options_group.add_argument("--profile", choices=["prepare", "analyze", "full"], help="Execution profile")
        options_group.add_argument("--train-db-path", action="append", default=[], help="Explicit train DB path")
        options_group.add_argument("--eval-db-path", action="append", default=[], help="Explicit eval DB path")
        options_group.add_argument(
            "--prepare-emit-commands",
            action="store_true",
            help="Emit commands during prepare profile execution",
        )
        options_group.add_argument(
            "--prepare-execution-target",
            choices=["local", "abci"],
            help="Execution target for prepare profile command emission",
        )

    options_group.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        help="Override configuration values (dot paths), example: --set bridge.seed=42",
    )
    if include_logging:
        options_group.add_argument("--json-log", action="store_true", help="Emit JSON logs")
        options_group.add_argument("--quiet", action="store_true", help="Disable console logs")
        options_group.add_argument("--no-log", action="store_true", help="Disable file logs")


def _parse_override_pairs(values: Sequence[str] | None) -> dict[str, object]:
    """Parse `--set key=value` pairs into nested mapping."""
    overrides: dict[str, object] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid override '{raw}'. Use key=value.")
        key, raw_value = raw.split("=", 1)
        _assign_override(overrides, key.split("."), _coerce_value(raw_value))
    return overrides


def _assign_override(payload: dict[str, object], path: list[str], value: object) -> None:
    """Assign a value to nested mapping path."""
    cursor: dict[str, object] = payload
    for index, segment in enumerate(path):
        if index == len(path) - 1:
            cursor[segment] = value
            return
        existing = cursor.get(segment)
        if existing is None:
            cursor[segment] = {}
            cursor = cursor[segment]  # type: ignore[assignment]
            continue
        if not isinstance(existing, dict):
            raise SystemExit(f"Override path '{'.'.join(path)}' conflicts with scalar value")
        cursor = existing


def _coerce_value(raw: str) -> object:
    """Coerce CLI override string to JSON/bool/string value."""
    try:
        return json.loads(raw)
    except Exception:
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return raw


def _ensure_bridge_overrides(overrides: dict[str, object]) -> dict[str, object]:
    """Ensure bridge-level override mapping exists and return it."""
    if "bridge" not in overrides or not isinstance(overrides["bridge"], dict):
        overrides["bridge"] = {}
    return overrides["bridge"]  # type: ignore[return-value]


def _ensure_execution_overrides(overrides: dict[str, object]) -> dict[str, object]:
    """Ensure bridge.execution override mapping exists and return it."""
    bridge = _ensure_bridge_overrides(overrides)
    if "execution" not in bridge or not isinstance(bridge["execution"], dict):
        bridge["execution"] = {}
    return bridge["execution"]  # type: ignore[return-value]


def _apply_cli_overrides(
    args: argparse.Namespace,
    overrides: dict[str, object],
    *,
    command: str,
) -> dict[str, object]:
    """Apply CLI options that map to configuration overrides."""
    if getattr(args, "output_dir", None):
        bridge = _ensure_bridge_overrides(overrides)
        bridge["output_dir"] = args.output_dir
    if getattr(args, "json_log", False):
        bridge = _ensure_bridge_overrides(overrides)
        bridge["json_log"] = True

    if command == "run":
        if getattr(args, "prepare_emit_commands", False):
            execution = _ensure_execution_overrides(overrides)
            execution["emit_on_prepare"] = True
        if getattr(args, "prepare_execution_target", None):
            execution = _ensure_execution_overrides(overrides)
            execution["target"] = args.prepare_execution_target

    if command in {"prepare-train", "prepare-eval"}:
        if getattr(args, "emit_commands", False):
            execution = _ensure_execution_overrides(overrides)
            execution["emit_on_prepare"] = True
        if getattr(args, "execution_target", None):
            execution = _ensure_execution_overrides(overrides)
            execution["target"] = args.execution_target

    return overrides


def _parse_steps(raw_steps: str | None) -> list[str] | None:
    """Split and normalize comma-separated step names."""
    if not raw_steps:
        return None
    return [step.strip() for step in raw_steps.split(",") if step.strip()]


def _apply_logging_flags(args: argparse.Namespace) -> None:
    """Apply logging-related environment flags from CLI options."""
    if getattr(args, "quiet", False):
        os.environ["AIACCEL_LOG_SILENT"] = "1"
    if getattr(args, "no_log", False):
        os.environ["AIACCEL_LOG_NO_FILE"] = "1"


def _parse_path_list(values: Sequence[str] | None) -> list[Path]:
    """Convert path argument list into resolved `Path` objects."""
    return [Path(value).expanduser().resolve() for value in values or []]


def _maybe_print_config(args: argparse.Namespace, config: BridgeConfig, command: str) -> bool:
    """Optionally print resolved config payload."""
    if getattr(args, "print_config", False):
        print(json.dumps(config.model_dump(mode="json"), indent=2, default=str))
        return command == "validate"
    return False


def _run_loaded_command(
    args: argparse.Namespace,
    command: str,
    bridge_config: BridgeConfig,
    logger: Any,
) -> None:
    """Dispatch an already-loaded command to modelbridge API calls."""
    if _maybe_print_config(args, bridge_config, command):
        return

    if command == "validate":
        logger.info("Configuration validated successfully.")
        return

    if command == "run":
        _run_profile_command(args, bridge_config)
        logger.info("Modelbridge run completed.")
        return

    if command in STEP_COMMANDS:
        step = _run_step_command(args, command, bridge_config)
        logger.info("Step completed: %s", step)
        return

    if command == "emit-commands":
        path = _run_emit_commands_command(args, bridge_config)
        logger.info("Commands written: %s", path)
        print(path)
        return

    raise SystemExit(f"Unsupported command: {command}")


def _run_profile_command(args: argparse.Namespace, bridge_config: BridgeConfig) -> None:
    """Run `modelbridge run` command."""
    steps = _parse_steps(args.steps)
    profile = args.profile
    if steps is not None and profile is not None:
        raise SystemExit("--steps and --profile are mutually exclusive")
    if getattr(args, "prepare_emit_commands", False) and profile != "prepare":
        raise SystemExit("--prepare-emit-commands requires --profile prepare")
    if getattr(args, "prepare_execution_target", None) and profile != "prepare":
        raise SystemExit("--prepare-execution-target requires --profile prepare")
    api.run(
        bridge_config,
        steps=steps,
        profile=profile,
        train_db_paths=_parse_path_list(args.train_db_path),
        eval_db_paths=_parse_path_list(args.eval_db_path),
    )


def _run_step_command(args: argparse.Namespace, command: str, bridge_config: BridgeConfig) -> str:
    """Run one step command and return executed step name."""
    step = STEP_COMMANDS[command]
    train_paths: list[Path] | None = None
    eval_paths: list[Path] | None = None
    if command == "collect-train":
        train_paths = _parse_path_list(getattr(args, "train_db_path", []))
    if command == "collect-eval":
        eval_paths = _parse_path_list(getattr(args, "eval_db_path", []))
    api.run(
        bridge_config,
        steps=[step],
        train_db_paths=train_paths,
        eval_db_paths=eval_paths,
    )
    return step


def _run_emit_commands_command(args: argparse.Namespace, bridge_config: BridgeConfig) -> Path:
    """Run emit-commands command and return output path."""
    return api.emit_commands_step(
        bridge_config,
        role=args.role,
        fmt=args.format,
        execution_target=args.execution_target,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run modelbridge CLI.

    Args:
        argv: Optional CLI argument sequence.

    Raises:
        SystemExit: When command execution fails.
    """
    args = _parse_args(argv)
    command = args.command
    logger = get_logger(__name__)

    try:
        if command == "schema":
            print(json.dumps(generate_schema(), indent=2, default=str))
            return

        config_path = Path(args.config).expanduser().resolve()
        cli_overrides = _parse_override_pairs(getattr(args, "overrides", None))
        cli_overrides = _apply_cli_overrides(args, cli_overrides, command=command)
        bridge_config = api.load_config(config_path, overrides=cli_overrides)
        _apply_logging_flags(args)
        _run_loaded_command(args, command, bridge_config, logger)
    except Exception as exc:
        logger.error("modelbridge failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
