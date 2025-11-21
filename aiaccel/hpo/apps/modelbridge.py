"""CLI adapter that routes to the modelbridge pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path

from omegaconf import OmegaConf

from aiaccel.config import load_config, overwrite_omegaconf_dumper, print_config, resolve_inherit
from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config
from aiaccel.hpo.modelbridge.logging import get_logger
from aiaccel.hpo.modelbridge.runner import PHASE_ORDER, execute_pipeline, plan_pipeline, run_pipeline

PHASE_CHOICES = tuple(PHASE_ORDER) + ("full",)


def _build_common_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", "-c", required=True, help="Path to the bridge configuration YAML")
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

    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--config", "-c", required=True, help="Path to the bridge configuration YAML")
    validate_parser.add_argument("--print-config", action="store_true", help="Print the resolved configuration")

    return parser.parse_args(argv)


def _load_bridge_config(path: Path) -> BridgeConfig:
    """Load and validate a configuration file located at ``path``."""

    overwrite_omegaconf_dumper()
    parent_ctx = {"config_path": str(path), "config_dir": str(path.parent)}
    conf = load_config(path, parent_ctx)
    conf = resolve_inherit(conf)
    container = OmegaConf.to_container(conf, resolve=True)
    if not isinstance(container, Mapping):
        raise ValueError("Bridge configuration must be a mapping")
    return load_bridge_config(container)


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
    config_path = Path(args.config).expanduser().resolve()
    bridge_config = _load_bridge_config(config_path)

    if getattr(args, "print_config", False):
        print_config(OmegaConf.load(config_path))
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
        )
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        logger.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Result:\n%s", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main"]
