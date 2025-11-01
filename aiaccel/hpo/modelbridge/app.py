"""CLI entry point for the simplified modelbridge pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf

from aiaccel.config import load_config, overwrite_omegaconf_dumper, print_config, resolve_inherit

from .config import BridgeConfig, load_bridge_config
from .exceptions import ModelBridgeError
from .logging import get_logger
from .runner import run_pipeline


def _parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments for the modelbridge runner."""

    parser = argparse.ArgumentParser(description="Run the aiaccel modelbridge pipeline")
    parser.add_argument("--config", "-c", required=True, help="Path to the bridge configuration YAML")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved configuration")
    parser.add_argument("--dry-run", action="store_true", help="Only validate the configuration")
    return parser.parse_args()


def _load_bridge_config(path: Path) -> BridgeConfig:
    """Load and validate a configuration file located at ``path``."""

    overwrite_omegaconf_dumper()
    parent_ctx = {"config_path": str(path), "config_dir": str(path.parent)}
    conf = load_config(path, parent_ctx)
    conf = resolve_inherit(conf)
    container = OmegaConf.to_container(conf, resolve=True)
    if not isinstance(container, Mapping):
        raise ModelBridgeError("Bridge configuration must be a mapping")
    return load_bridge_config(cast(Mapping[str, Any], container))


def main() -> None:
    """CLI entry point used by ``aiaccel-hpo modelbridge``."""

    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    bridge_config = _load_bridge_config(config_path)

    if args.print_config:
        print_config(OmegaConf.load(config_path))

    logger = get_logger(__name__)

    if args.dry_run:
        logger.info("Configuration validated successfully")
        return

    try:
        summary = run_pipeline(bridge_config)
    except ModelBridgeError as exc:
        logger.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Summary:\n%s", json.dumps(summary, indent=2, default=str))


__all__ = ["main"]
