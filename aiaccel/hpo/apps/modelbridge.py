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
from aiaccel.hpo.modelbridge.runner import PHASE_ORDER, run_pipeline

_DEPRECATED_PHASES = {
    "train-macro": "--phase hpo --role train --target macro",
    "train-micro": "--phase hpo --role train --target micro",
    "eval-macro": "--phase hpo --role eval --target macro",
    "eval-micro": "--phase hpo --role eval --target micro",
}

PHASE_CHOICES = tuple(PHASE_ORDER) + ("full",) + tuple(_DEPRECATED_PHASES)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return parsed CLI arguments for the modelbridge runner."""

    parser = argparse.ArgumentParser(description="Run the aiaccel modelbridge pipeline")
    parser.add_argument("--config", "-c", required=True, help="Path to the bridge configuration YAML")
    parser.add_argument(
        "--phase",
        action="append",
        choices=PHASE_CHOICES,
        dest="phases",
        help="Run phase(s): hpo/regress/evaluate/summary/full. Repeatable.",
    )
    parser.add_argument(
        "--role",
        choices=("train", "eval"),
        help="Role for HPO (required when --phase hpo is used).",
    )
    parser.add_argument(
        "--target",
        choices=("macro", "micro"),
        help="Target for HPO (required when --phase hpo is used).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help="Zero-based run index for HPO/eval. Defaults to all runs when omitted.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Restrict execution to the given scenario name. Repeatable.",
    )
    parser.add_argument("--print-config", action="store_true", help="Print the resolved configuration")
    parser.add_argument("--dry-run", action="store_true", help="Only validate the configuration")
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
    ordering = {name: index for index, name in enumerate(PHASE_ORDER)}
    unique.sort(key=lambda name: ordering[name])
    return tuple(unique)


def _normalize_scenarios(values: Iterable[str] | None) -> Sequence[str] | None:
    if not values:
        return None
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def _reject_deprecated_phases(phases: Sequence[str] | None) -> None:
    if not phases:
        return
    bad = [phase for phase in phases if phase in _DEPRECATED_PHASES]
    if bad:
        suggestion = _DEPRECATED_PHASES[bad[0]]
        raise SystemExit(f"Deprecated phase '{bad[0]}' is no longer supported. Use {suggestion}.")


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for ``aiaccel-hpo modelbridge``."""

    args = _parse_args(argv)
    _reject_deprecated_phases(args.phases)
    config_path = Path(args.config).expanduser().resolve()
    bridge_config = _load_bridge_config(config_path)

    if args.print_config:
        print_config(OmegaConf.load(config_path))

    logger = get_logger(__name__)

    if args.dry_run:
        logger.info("Configuration validated successfully")
        return

    selected_phases = _normalize_phases(args.phases)
    selected_scenarios = _normalize_scenarios(args.scenarios)

    hpo_filter_requested = args.role is not None or args.target is not None or args.run_id is not None
    if hpo_filter_requested:
        selected_phases = selected_phases or ("hpo",)
        if "hpo" not in selected_phases:
            raise SystemExit("--role/--target/--run-id are only applicable when --phase hpo is set")
        if args.role is None or args.target is None:
            raise SystemExit("--phase hpo requires --role and --target to be specified")

    try:
        summary = run_pipeline(
            bridge_config,
            phases=selected_phases,
            scenarios=selected_scenarios,
            role=args.role,
            target=args.target,
            run_id=args.run_id,
        )
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        logger.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Result:\n%s", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main"]
