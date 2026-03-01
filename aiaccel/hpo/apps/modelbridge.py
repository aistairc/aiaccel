"""CLI adapter for modelbridge tools."""

from __future__ import annotations

from typing import cast

import argparse
from collections.abc import Callable, Sequence
import logging
from pathlib import Path

from aiaccel.hpo.modelbridge import collect, evaluate, fit_model, prepare

StepHandler = Callable[[argparse.Namespace], int]

STEP_COMMANDS: tuple[str, ...] = (
    "prepare",
    "collect",
    "fit-model",
    "evaluate",
)


def _resolve_step_handler(command: str) -> StepHandler:
    """Resolve one CLI subcommand to a handler."""
    handlers: dict[str, StepHandler] = {
        "prepare": _handle_prepare,
        "collect": _handle_collect,
        "fit-model": _handle_fit_model,
        "evaluate": _handle_evaluate,
    }
    if command not in handlers:
        raise SystemExit(f"Unsupported modelbridge command: {command}")
    return handlers[command]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run aiaccel modelbridge tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Generate per-run HPO configs.")
    prepare_parser.add_argument("--config", required=True, help="Path to modelbridge config.yaml")
    prepare_parser.add_argument("--workspace", required=True, help="Workspace directory")
    prepare_parser.set_defaults(handler=_handle_prepare)

    collect_parser = subparsers.add_parser("collect", help="Collect best parameter pairs from Optuna DBs.")
    collect_parser.add_argument("--workspace", required=True, help="Workspace directory")
    collect_parser.add_argument("--phase", required=True, choices=["train", "test"], help="Collect phase")
    collect_parser.set_defaults(handler=_handle_collect)

    fit_parser = subparsers.add_parser("fit-model", help="Fit regression model from train_pairs.csv.")
    fit_parser.add_argument("--workspace", required=True, help="Workspace directory")
    fit_parser.set_defaults(handler=_handle_fit_model)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate fitted model on test_pairs.csv.")
    evaluate_parser.add_argument("--workspace", required=True, help="Workspace directory")
    evaluate_parser.set_defaults(handler=_handle_evaluate)

    return parser.parse_args(argv)


def _handle_prepare(args: argparse.Namespace) -> int:
    prepare.run_prepare(config_path=Path(args.config), workspace=Path(args.workspace))
    return 0


def _handle_collect(args: argparse.Namespace) -> int:
    collect.run_collect(workspace=Path(args.workspace), phase=str(args.phase))
    return 0


def _handle_fit_model(args: argparse.Namespace) -> int:
    fit_model.run_fit_model(workspace=Path(args.workspace))
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    evaluate.run_evaluate(workspace=Path(args.workspace))
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    """Run modelbridge CLI entrypoint."""
    args = _parse_args(argv)
    logger = logging.getLogger(__name__)
    handler = cast(StepHandler, getattr(args, "handler", _resolve_step_handler(str(args.command))))
    try:
        exit_code = int(handler(args))
    except Exception as exc:  # pragma: no cover - CLI boundary.
        logger.error("modelbridge failed: %s", exc)
        raise SystemExit(1) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
