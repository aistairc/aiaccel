"""CLI adapter for modelbridge."""

from __future__ import annotations

from typing import Any, cast

import argparse
from collections.abc import Callable, Sequence
import json
import logging
import os
from pathlib import Path

from aiaccel.hpo.modelbridge import api
from aiaccel.hpo.modelbridge.config import BridgeConfig, deep_merge_mappings, generate_schema

Handler = Callable[[argparse.Namespace, Any, BridgeConfig | None], None]
StepHandler = Callable[..., object]

STEP_COMMANDS: tuple[str, ...] = (
    "prepare-train",
    "prepare-eval",
    "hpo-train",
    "hpo-eval",
    "collect-train",
    "collect-eval",
    "fit-regression",
    "evaluate-model",
    "publish-summary",
)


def _resolve_step_handler(command: str) -> StepHandler:
    """Return step API callable for one CLI command."""
    handlers: dict[str, StepHandler] = {
        "prepare-train": api.prepare_train_step,
        "prepare-eval": api.prepare_eval_step,
        "hpo-train": api.hpo_train_step,
        "hpo-eval": api.hpo_eval_step,
        "collect-train": api.collect_train_step,
        "collect-eval": api.collect_eval_step,
        "fit-regression": api.fit_regression_step,
        "evaluate-model": api.evaluate_model_step,
        "publish-summary": api.publish_summary_step,
    }
    if command not in handlers:
        raise SystemExit(f"Unsupported step command: {command}")
    return handlers[command]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run aiaccel modelbridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in STEP_COMMANDS:
        subparser = subparsers.add_parser(command, help=f"Execute {command}")
        _add_common_args(subparser, include_logging=True)
        if command == "collect-train":
            subparser.add_argument("--train-db-path", action="append", default=[], help="Explicit train DB path")
        if command == "collect-eval":
            subparser.add_argument("--eval-db-path", action="append", default=[], help="Explicit eval DB path")
        subparser.set_defaults(handler=_handle_step)

    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    _add_common_args(validate_parser, include_logging=False)
    validate_parser.add_argument("--print-config", action="store_true", help="Print resolved config JSON")
    validate_parser.set_defaults(handler=_handle_validate)

    schema_parser = subparsers.add_parser("schema", help="Print JSON schema for modelbridge config")
    schema_parser.set_defaults(handler=_handle_schema, no_config=True)
    return parser.parse_args(argv)


def _add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_logging: bool = True,
) -> None:
    """Add shared argument groups."""
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--config", "-c", required=True, help="Path to modelbridge config")
    io_group.add_argument("--output_dir", "-o", help="Override output directory")

    options_group = parser.add_argument_group("Options")
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
    """Parse ``--set key=value`` pairs into nested mapping."""
    overrides: dict[str, object] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid override '{raw}'. Use key=value.")
        key, raw_value = raw.split("=", 1)
        _set_override(overrides, key.split("."), _coerce_value(raw_value))
    return overrides


def _set_override(payload: dict[str, object], path: list[str], value: object) -> None:
    """Set one nested override value."""
    cursor: dict[str, object] = payload
    for segment in path[:-1]:
        current = cursor.get(segment)
        if current is None:
            current = {}
            cursor[segment] = current
        if not isinstance(current, dict):
            raise SystemExit(f"Override path '{'.'.join(path)}' conflicts with scalar value")
        cursor = cast(dict[str, object], current)
    cursor[path[-1]] = value


def _coerce_value(raw: str) -> object:
    """Coerce CLI override string to JSON value when possible."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return raw


def _build_cli_override_patch(args: argparse.Namespace, *, command: str) -> dict[str, object]:
    """Build override mapping from CLI options."""
    bridge: dict[str, object] = {}
    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        bridge["output_dir"] = output_dir
    if bool(getattr(args, "json_log", False)):
        bridge["json_log"] = True
    _ = command
    return {"bridge": bridge} if bridge else {}


def _apply_logging_flags(args: argparse.Namespace) -> None:
    """Apply logging-related environment flags from CLI options."""
    if getattr(args, "quiet", False):
        os.environ["AIACCEL_LOG_SILENT"] = "1"
    if getattr(args, "no_log", False):
        os.environ["AIACCEL_LOG_NO_FILE"] = "1"


def _parse_path_list(values: Sequence[str] | None) -> list[Path]:
    """Convert path argument list into resolved ``Path`` objects."""
    return [Path(value).expanduser().resolve() for value in values or []]


def _handle_schema(_args: argparse.Namespace, _logger: Any, _config: BridgeConfig | None) -> None:
    """Handle schema command."""
    print(json.dumps(generate_schema(), indent=2, default=str))


def _handle_validate(args: argparse.Namespace, logger: Any, config: BridgeConfig | None) -> None:
    """Handle validate command."""
    config = _require_config(config, command="validate")
    if getattr(args, "print_config", False):
        print(json.dumps(config.model_dump(mode="json"), indent=2, default=str))
    logger.info("Configuration validated successfully.")


def _handle_step(args: argparse.Namespace, logger: Any, config: BridgeConfig | None) -> None:
    """Handle one explicit step command."""
    config = _require_config(config, command=str(args.command))
    command = str(args.command)
    handler = _resolve_step_handler(command)
    if args.command == "collect-train":
        handler(config, db_paths=_parse_path_list(getattr(args, "train_db_path", [])))
    elif args.command == "collect-eval":
        handler(config, db_paths=_parse_path_list(getattr(args, "eval_db_path", [])))
    else:
        handler(config)
    logger.info("Step completed: %s", command)


def main(argv: Sequence[str] | None = None) -> None:
    """Run modelbridge CLI entrypoint.

    This function parses CLI arguments, resolves configuration overrides, and
    dispatches the requested step command handler.

    Args:
        argv: Optional CLI argument sequence.

    Returns:
        None: This function exits after command handling.

    Raises:
        SystemExit: Raised with non-zero status when command handling fails.
    """
    args = _parse_args(argv)
    logger = logging.getLogger(__name__)

    try:
        _apply_logging_flags(args)
        handler = cast(Handler, args.handler)
        if getattr(args, "no_config", False):
            handler(args, logger, None)
            return

        config_path = Path(args.config).expanduser().resolve()
        cli_overrides = _parse_override_pairs(getattr(args, "overrides", None))
        cli_overrides = deep_merge_mappings(cli_overrides, _build_cli_override_patch(args, command=args.command))
        bridge_config = api.load_config(config_path, overrides=cli_overrides)
        handler(args, logger, bridge_config)
    except Exception as exc:
        logger.error("modelbridge failed: %s", exc)
        raise SystemExit(1) from exc


def _require_config(config: BridgeConfig | None, *, command: str) -> BridgeConfig:
    """Return loaded config or raise a command-specific error."""
    if config is None:
        raise SystemExit(f"{command} requires loaded config")
    return config


if __name__ == "__main__":
    main()
