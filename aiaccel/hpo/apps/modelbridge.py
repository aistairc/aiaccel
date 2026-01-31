"""CLI adapter that routes to the modelbridge pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from omegaconf import OmegaConf

from aiaccel.config import load_config, resolve_inherit, setup_omegaconf
from aiaccel.hpo.modelbridge.config import BridgeConfig, generate_schema, load_bridge_config
from aiaccel.hpo.modelbridge.pipeline import run_pipeline
from aiaccel.hpo.modelbridge.utils import get_logger


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the aiaccel modelbridge pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Run Command ---
    run_parser = subparsers.add_parser("run", help="Execute the pipeline")

    # Input/Output Group
    io_group = run_parser.add_argument_group("Input/Output")
    io_group.add_argument("--config", "-c", required=True, help="Path to the bridge configuration YAML")
    io_group.add_argument("--output_dir", "-o", help="Output directory (overrides config if specified)")

    # Options Group
    opt_group = run_parser.add_argument_group("Options")
    opt_group.add_argument(
        "--steps",
        help="Comma-separated list of steps to execute (setup_train, setup_eval, regression, evaluate_model, summary, da)",
    )
    opt_group.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        help="Override configuration values (dot paths), e.g. --set bridge.seed=42",
    )
    opt_group.add_argument("--quiet", action="store_true", help="Suppress console logs")
    opt_group.add_argument("--no-log", action="store_true", help="Disable file logging")
    opt_group.add_argument("--json-log", action="store_true", help="Emit JSON structured logs")

    # --- Validate Command ---
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--config", "-c", required=True, help="Path to the bridge configuration YAML")
    validate_parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        help="Override configuration values (dot paths), e.g. --set bridge.log_level=DEBUG",
    )
    validate_parser.add_argument("--print-config", action="store_true", help="Print the resolved configuration")

    # --- Schema Command ---
    subparsers.add_parser("schema", help="Print the modelbridge configuration JSON schema")

    return parser.parse_args(argv)


def _load_bridge_config(path: Path, cli_overrides: Mapping[str, object] | None = None) -> BridgeConfig:
    """Load and validate a configuration file located at ``path``.

    Args:
        path (Path): Path to the configuration file.
        cli_overrides (Mapping[str, object] | None): CLI overrides to apply.

    Returns:
        BridgeConfig: The loaded configuration object.
    """

    setup_omegaconf()
    parent_ctx = {"config_path": str(path), "config_dir": str(path.parent)}
    conf = load_config(path, parent_ctx)
    conf = resolve_inherit(conf)
    container = OmegaConf.to_container(conf, resolve=True)
    if not isinstance(container, Mapping):
        raise ValueError("Bridge configuration must be a mapping")
    merged_overrides = parent_ctx if not cli_overrides else _merge_overrides(parent_ctx, cli_overrides)
    # The merged_overrides dict comes from strings, so keys are str.
    # We cast to satisfy the Mapping[str, Any] requirement.
    from typing import Any, cast

    return load_bridge_config(cast(Mapping[str, Any], container), overrides=cast(Mapping[str, Any], merged_overrides))


def _merge_overrides(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_overrides(merged[key], value)  # type: ignore
        else:
            merged[key] = value
    return merged


def _parse_override_pairs(values: Sequence[str] | None) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid override '{raw}'. Use key=value with dot-separated paths.")
        key, raw_value = raw.split("=", 1)
        _assign_override(overrides, key.split("."), _coerce_value(raw_value))
    return overrides


def _assign_override(payload: dict[str, object], path: list[str], value: object) -> None:
    cursor: dict[str, object] = payload
    for idx, segment in enumerate(path):
        if idx == len(path) - 1:
            cursor[segment] = value
            return
        existing = cursor.get(segment)
        if existing is None:
            cursor[segment] = {}
            cursor = cursor[segment]  # type: ignore
            continue
        if not isinstance(existing, dict):
            raise SystemExit(f"Override path '{'.'.join(path)}' conflicts with a scalar value")
        cursor = existing


def _coerce_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except Exception:
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return raw


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for ``aiaccel-hpo modelbridge``.

    Args:
        argv (Sequence[str] | None): Command line arguments. If None, uses sys.argv.
    """

    args = _parse_args(argv)
    command = args.command

    if command == "schema":
        print(json.dumps(generate_schema(), indent=2, default=str))
        return

    config_path = Path(args.config).expanduser().resolve()
    cli_overrides = _parse_override_pairs(getattr(args, "overrides", None))

    # Handle output_dir arg
    if getattr(args, "output_dir", None):
        if "bridge" not in cli_overrides or not isinstance(cli_overrides["bridge"], dict):
            cli_overrides["bridge"] = cli_overrides.get("bridge", {})
        cli_overrides["bridge"]["output_dir"] = args.output_dir  # type: ignore

    # Handle json-log flag by injecting into overrides
    if getattr(args, "json_log", False):
        if "bridge" not in cli_overrides or not isinstance(cli_overrides["bridge"], dict):
            cli_overrides["bridge"] = cli_overrides.get("bridge", {})
        cli_overrides["bridge"]["json_log"] = True  # type: ignore

    bridge_config = _load_bridge_config(config_path, cli_overrides)

    if getattr(args, "print_config", False):
        print(json.dumps(bridge_config.model_dump(mode="json"), indent=2, default=str))
        if command == "validate":
            return

    logger = get_logger(__name__)

    try:
        if command == "validate":
            logger.info("Configuration validated successfully")
            return

        if command == "run":
            if getattr(args, "quiet", False):
                import os

                os.environ["AIACCEL_LOG_SILENT"] = "1"

            steps = [s.strip() for s in args.steps.split(",")] if args.steps else None
            _ = run_pipeline(bridge_config, steps=steps)
            logger.info("Pipeline completed successfully.")

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()


__all__ = ["main"]
