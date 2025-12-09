"""CLI adapter that routes to the modelbridge pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path

from omegaconf import OmegaConf

from aiaccel.config import load_config, overwrite_omegaconf_dumper, resolve_inherit
from aiaccel.hpo.modelbridge.config import BridgeConfig, generate_schema, load_bridge_config
from aiaccel.hpo.modelbridge.data_assimilation import run_data_assimilation
from aiaccel.hpo.modelbridge.logging import get_logger
from aiaccel.hpo.modelbridge.runner import PHASE_ORDER, execute_pipeline, plan_pipeline, run_pipeline

PHASE_CHOICES = tuple(PHASE_ORDER) + ("full",)


def _build_common_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", "-c", required=True, help="Path to the bridge configuration YAML")
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        help="Override configuration values (dot paths), e.g. --set bridge.seed=42",
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=PHASE_CHOICES,
        dest="phases",
        help="phase(s): hpo/regress/evaluate/summary/full (order preserved). Repeatable.",
    )
    parser.add_argument("--scenario", action="append", dest="scenarios", help="Limit execution to scenario(s).")
    parser.add_argument("--role", choices=("train", "eval"), help="Only valid with --phase hpo.")
    parser.add_argument("--target", choices=("macro", "micro"), help="Only valid with --phase hpo.")
    parser.add_argument("--run-id", type=int, help="Zero-based run index for HPO.")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run the aiaccel modelbridge pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _build_common_parser(subparsers.add_parser("plan", help="Print the planned contexts and exit"))
    run_parser = _build_common_parser(subparsers.add_parser("run", help="Execute the pipeline"))
    run_parser.add_argument("--quiet", action="store_true", help="Suppress console logs")
    run_parser.add_argument("--no-log", action="store_true", help="Disable file logging")
    run_parser.add_argument("--json-log", action="store_true", help="Emit JSON structured logs")

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

    subparsers.add_parser("schema", help="Print the modelbridge configuration JSON schema")

    da_parser = subparsers.add_parser("data-assimilation", help="Run MAS-Bench data assimilation workflow")
    da_parser.add_argument("--config", "-c", required=True, help="Path to the bridge configuration YAML")
    da_parser.add_argument("--dry-run", action="store_true", help="Print planned phases without executing")
    da_parser.add_argument("--quiet", action="store_true", help="Suppress console logs")
    da_parser.add_argument("--no-log", action="store_true", help="Disable file logging")
    da_parser.add_argument("--json-log", action="store_true", help="Emit JSON structured logs")

    return parser.parse_args(argv)


def _load_bridge_config(path: Path, cli_overrides: Mapping[str, object] | None = None) -> BridgeConfig:
    """Load and validate a configuration file located at ``path``."""

    overwrite_omegaconf_dumper()
    parent_ctx = {"config_path": str(path), "config_dir": str(path.parent)}
    conf = load_config(path, parent_ctx)
    conf = resolve_inherit(conf)
    container = OmegaConf.to_container(conf, resolve=True)
    if not isinstance(container, Mapping):
        raise ValueError("Bridge configuration must be a mapping")
    merged_overrides = parent_ctx if not cli_overrides else _merge_overrides(parent_ctx, cli_overrides)
    return load_bridge_config(container, overrides=merged_overrides)


def _merge_overrides(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_overrides(merged[key], value)  # type: ignore[arg-type,index]
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
            cursor = cursor[segment]  # type: ignore[assignment]
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


def _normalize_phases(phases: Iterable[str] | None) -> Sequence[str] | None:
    if not phases:
        return None
    unique: list[str] = []
    for item in phases:
        if item == "full":
            return None
        if item not in unique:
            unique.append(item)
    return tuple(unique)


def _normalize_scenarios(values: Iterable[str] | None) -> Sequence[str] | None:
    if not values:
        return None
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for ``aiaccel-hpo modelbridge``."""

    args = _parse_args(argv)
    command = args.command
    if command == "schema":
        print(json.dumps(generate_schema(), indent=2, default=str))
        return
    if command == "data-assimilation":
        config_path = Path(args.config).expanduser().resolve()
        bridge_config = _load_bridge_config(config_path, _parse_override_pairs(getattr(args, "overrides", None)))
        if bridge_config.data_assimilation is None:
            raise SystemExit("data_assimilation section is required for this command")
        run_data_assimilation(
            bridge_config.data_assimilation,
            dry_run=bool(getattr(args, "dry_run", False)),
            quiet=bool(getattr(args, "quiet", False)),
            log_to_file=not bool(getattr(args, "no_log", False)),
            json_logs=bool(getattr(args, "json_log", False)),
        )
        return

    config_path = Path(args.config).expanduser().resolve()
    cli_overrides = _parse_override_pairs(getattr(args, "overrides", None))
    bridge_config = _load_bridge_config(config_path, cli_overrides)

    if getattr(args, "print_config", False):
        print(json.dumps(bridge_config.model_dump(mode="json"), indent=2, default=str))
        if command == "validate":
            return

    selected_phases = _normalize_phases(getattr(args, "phases", None))
    selected_scenarios = _normalize_scenarios(getattr(args, "scenarios", None))
    hpo_filter_requested = getattr(args, "role", None) is not None or getattr(args, "target", None) is not None or getattr(args, "run_id", None) is not None
    if hpo_filter_requested:
        selected_phases = selected_phases or ("hpo",)
        if selected_phases and "hpo" not in selected_phases:
            raise SystemExit("--role/--target/--run-id are only applicable when --phase hpo is set")
        if args.role is None or args.target is None:
            raise SystemExit("--phase hpo requires --role and --target to be specified")

    logger = get_logger(__name__)

    try:
        if command == "plan":
            plan = plan_pipeline(
                bridge_config,
                phases=selected_phases,
                scenarios=selected_scenarios,
                role=getattr(args, "role", None),
                target=getattr(args, "target", None),
                run_id=getattr(args, "run_id", None),
            )
            print(json.dumps(plan.serializable(), indent=2, default=str))
            return

        if command == "validate":
            logger.info("Configuration validated successfully")
            return

        plan = plan_pipeline(
            bridge_config,
            phases=selected_phases,
            scenarios=selected_scenarios,
            role=getattr(args, "role", None),
            target=getattr(args, "target", None),
            run_id=getattr(args, "run_id", None),
        )
        summary = execute_pipeline(
            plan,
            quiet=bool(getattr(args, "quiet", False)),
            log_to_file=not bool(getattr(args, "no_log", False)),
            json_logs=bool(getattr(args, "json_log", False)),
        )
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        logger.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Result:\n%s", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main"]
