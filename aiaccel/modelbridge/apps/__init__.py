# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import logging

from . import collect, evaluate, fit_model, prepare

__all__ = [
    "collect",
    "evaluate",
    "fit_model",
    "main",
    "prepare",
]

StepMain = Callable[[Sequence[str] | None], int]

# Each step module owns its argument parser; this adapter only selects the step, the same way
# the shared ``aiaccel.launcher`` does for the ``aiaccel-modelbridge`` console script.
STEP_MAINS: dict[str, StepMain] = {
    "prepare": prepare.main,
    "collect": collect.main,
    "fit-model": fit_model.main,
    "evaluate": evaluate.main,
}
STEP_COMMANDS: tuple[str, ...] = tuple(STEP_MAINS)


def main(argv: Sequence[str] | None = None) -> None:
    """Run one modelbridge step, as ``python -m aiaccel.modelbridge.apps <command> [args...]``.

    Args:
        argv: Optional command-line arguments. When omitted, uses ``sys.argv``.

    Raises:
        SystemExit: Always, with the step exit code, or ``1`` when the step raises an exception.
    """
    parser = argparse.ArgumentParser(description="Run aiaccel modelbridge tools.", add_help=False)
    parser.add_argument("command", choices=STEP_COMMANDS, help="The step to run.")
    args, step_argv = parser.parse_known_args(argv)
    try:
        exit_code = STEP_MAINS[args.command](step_argv)
    except Exception as exc:
        logging.getLogger(__name__).error("modelbridge %s failed: %s", args.command, exc)
        raise SystemExit(1) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
