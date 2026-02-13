"""CLI adapter for modelbridge."""

from __future__ import annotations

from typing import Any, cast

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path

from aiaccel.hpo.modelbridge import api
from aiaccel.hpo.modelbridge.config import BridgeConfig, ExecutionTarget, deep_merge_mappings, generate_schema
from aiaccel.hpo.modelbridge.pipeline import PIPELINE_PROFILES, STEP_SPECS

Handler = Callable[[argparse.Namespace, Any, BridgeConfig | None], None]


@dataclass(frozen=True)
class CliOverridePatch:
    """Typed CLI override patch mapped onto modelbridge config overrides.

    Args:
        output_dir: Optional output directory override.
        json_log: Whether JSON logging is enabled.
        emit_on_prepare: Whether prepare should emit command scripts.
        execution_target: Optional execution target override.
    """

    output_dir: str | None = None
    json_log: bool = False
    emit_on_prepare: bool = False
    execution_target: ExecutionTarget | None = None

    def to_overrides(self) -> dict[str, object]:
        """Render this patch into nested config override mapping.

        Returns:
            dict[str, object]: Nested override mapping payload.
        """
        bridge: dict[str, object] = {}
        if self.output_dir is not None:
            bridge["output_dir"] = self.output_dir
        if self.json_log:
            bridge["json_log"] = True

        execution: dict[str, object] = {}
        if self.emit_on_prepare:
            execution["emit_on_prepare"] = True
        if self.execution_target is not None:
            execution["target"] = self.execution_target
        if execution:
            bridge["execution"] = execution

        return {"bridge": bridge} if bridge else {}


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
    run_parser.set_defaults(handler=_handle_run)

    for step_name, cli_command, _profiles in STEP_SPECS:
        subparser = subparsers.add_parser(cli_command, help=f"Execute {cli_command}")
        _add_common_args(subparser, include_run_args=False)
        subparser.set_defaults(handler=_handle_step, step_name=step_name)
        if cli_command in {"prepare-train", "prepare-eval"}:
            subparser.add_argument("--emit-commands", action="store_true", help="Emit commands after prepare step")
            subparser.add_argument("--execution-target", choices=["local", "abci"], help="Execution target")
        if cli_command == "collect-train":
            subparser.add_argument("--train-db-path", action="append", default=[], help="Explicit train DB path")
        if cli_command == "collect-eval":
            subparser.add_argument("--eval-db-path", action="append", default=[], help="Explicit eval DB path")

    emit_parser = subparsers.add_parser("emit-commands", help="Emit optimize commands from plan")
    _add_common_args(emit_parser, include_run_args=False)
    emit_parser.add_argument("--role", choices=["train", "eval"], required=True, help="Target role")
    emit_parser.add_argument("--format", choices=["shell", "json"], required=True, help="Output format")
    emit_parser.add_argument("--execution-target", choices=["local", "abci"], help="Execution target override")
    emit_parser.set_defaults(handler=_handle_emit_commands)

    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    _add_common_args(validate_parser, include_run_args=False, include_logging=False)
    validate_parser.add_argument("--print-config", action="store_true", help="Print resolved config JSON")
    validate_parser.set_defaults(handler=_handle_validate)

    schema_parser = subparsers.add_parser("schema", help="Print JSON schema for modelbridge config")
    schema_parser.set_defaults(handler=_handle_schema, no_config=True)
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
        options_group.add_argument("--profile", choices=list(PIPELINE_PROFILES), help="Execution profile")
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
    """Parse ``--set key=value`` pairs into nested mapping."""
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


def _build_cli_override_patch(args: argparse.Namespace, *, command: str) -> CliOverridePatch:
    """Build typed override patch from CLI flags."""
    emit_on_prepare = False
    execution_target: ExecutionTarget | None = None

    if command == "run":
        emit_on_prepare = bool(getattr(args, "prepare_emit_commands", False))
        execution_target = cast(ExecutionTarget | None, getattr(args, "prepare_execution_target", None))
    elif command in {"prepare-train", "prepare-eval"}:
        emit_on_prepare = bool(getattr(args, "emit_commands", False))
        execution_target = cast(ExecutionTarget | None, getattr(args, "execution_target", None))

    return CliOverridePatch(
        output_dir=getattr(args, "output_dir", None),
        json_log=bool(getattr(args, "json_log", False)),
        emit_on_prepare=emit_on_prepare,
        execution_target=execution_target,
    )


def _apply_cli_override_patch(
    overrides: dict[str, object],
    patch: CliOverridePatch,
) -> dict[str, object]:
    """Apply typed CLI override patch over parsed ``--set`` overrides."""
    patch_mapping = patch.to_overrides()
    if not patch_mapping:
        return overrides
    return deep_merge_mappings(overrides, patch_mapping)


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
    """Convert path argument list into resolved ``Path`` objects."""
    return [Path(value).expanduser().resolve() for value in values or []]


def _handle_schema(_args: argparse.Namespace, _logger: Any, _config: BridgeConfig | None) -> None:
    """Handle schema command."""
    print(json.dumps(generate_schema(), indent=2, default=str))


def _handle_validate(args: argparse.Namespace, logger: Any, config: BridgeConfig | None) -> None:
    """Handle validate command."""
    if config is None:
        raise SystemExit("validate requires loaded config")
    if getattr(args, "print_config", False):
        print(json.dumps(config.model_dump(mode="json"), indent=2, default=str))
    logger.info("Configuration validated successfully.")


def _handle_run(args: argparse.Namespace, logger: Any, config: BridgeConfig | None) -> None:
    """Handle run command."""
    if config is None:
        raise SystemExit("run requires loaded config")

    steps = _parse_steps(args.steps)
    profile = args.profile
    if steps is not None and profile is not None:
        raise SystemExit("--steps and --profile are mutually exclusive")
    if getattr(args, "prepare_emit_commands", False) and profile != "prepare":
        raise SystemExit("--prepare-emit-commands requires --profile prepare")
    if getattr(args, "prepare_execution_target", None) and profile != "prepare":
        raise SystemExit("--prepare-execution-target requires --profile prepare")

    api.run(
        config,
        steps=steps,
        profile=profile,
        train_db_paths=_parse_path_list(args.train_db_path),
        eval_db_paths=_parse_path_list(args.eval_db_path),
    )
    logger.info("Modelbridge run completed.")


def _handle_step(args: argparse.Namespace, logger: Any, config: BridgeConfig | None) -> None:
    """Handle one explicit step command."""
    if config is None:
        raise SystemExit(f"{args.command} requires loaded config")

    step_name = args.step_name
    train_paths: list[Path] | None = None
    eval_paths: list[Path] | None = None
    if args.command == "collect-train":
        train_paths = _parse_path_list(getattr(args, "train_db_path", []))
    if args.command == "collect-eval":
        eval_paths = _parse_path_list(getattr(args, "eval_db_path", []))

    api.run(
        config,
        steps=[step_name],
        train_db_paths=train_paths,
        eval_db_paths=eval_paths,
    )
    logger.info("Step completed: %s", step_name)


def _handle_emit_commands(args: argparse.Namespace, logger: Any, config: BridgeConfig | None) -> None:
    """Handle emit-commands command."""
    if config is None:
        raise SystemExit("emit-commands requires loaded config")

    path = api.emit_commands_step(
        config,
        role=args.role,
        fmt=args.format,
        execution_target=args.execution_target,
    )
    logger.info("Commands written: %s", path)
    print(path)


def main(argv: Sequence[str] | None = None) -> None:
    """Run modelbridge CLI.

    Args:
        argv: Optional CLI argument sequence.

    Raises:
        SystemExit: When command execution fails.
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
        override_patch = _build_cli_override_patch(args, command=args.command)
        cli_overrides = _apply_cli_override_patch(cli_overrides, override_patch)
        bridge_config = api.load_config(config_path, overrides=cli_overrides)
        handler(args, logger, bridge_config)
    except Exception as exc:
        logger.error("modelbridge failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
