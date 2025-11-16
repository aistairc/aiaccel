"""CLI adapter that routes to the modelbridge pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path

from omegaconf import OmegaConf

from aiaccel.config import load_config, overwrite_omegaconf_dumper, print_config, resolve_inherit
from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config
from aiaccel.hpo.modelbridge.exceptions import ModelBridgeError
from aiaccel.hpo.modelbridge.logging import get_logger
from aiaccel.hpo.modelbridge.runner import PHASE_ORDER, run_pipeline

PHASE_CHOICES = tuple(PHASE_ORDER) + ("full",)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return parsed CLI arguments for the modelbridge runner."""

    parser = argparse.ArgumentParser(description="Run the aiaccel modelbridge pipeline")
    parser.add_argument("--config", "-c", required=True, help="Path to the bridge configuration YAML")
    parser.add_argument(
        "--phase",
        action="append",
        choices=PHASE_CHOICES,
        dest="phases",
        help="Run only the specified phase (macro/micro/regress/summary/full). Repeatable.",
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
        raise ModelBridgeError("Bridge configuration must be a mapping")
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


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for ``aiaccel-hpo modelbridge``."""

    args = _parse_args(argv)
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

    try:
        summary = run_pipeline(bridge_config, phases=selected_phases, scenarios=selected_scenarios)
    except ModelBridgeError as exc:  # pragma: no cover - exercised via CLI tests
        logger.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Result:\n%s", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main"]
