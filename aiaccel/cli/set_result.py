from __future__ import annotations

import copy
import warnings
from argparse import ArgumentParser
from pathlib import Path

from aiaccel.storage.storage import Storage
from aiaccel.util.data_type import str_or_float_or_int

warnings.filterwarnings("ignore", category=RuntimeWarning)


def write_results_to_database(
    storage_file_path: str | Path,
    trial_id: int,
    objective: list[str | float | int] | None,
    returncode: int | None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> None:
    storage = Storage(storage_file_path)

    if objective is None:
        raise Exception("Could not get objective")
    storage.result.set_any_trial_objective(trial_id, objective)
    if returncode is not None:
        storage.returncode.set_any_trial_returncode(trial_id, returncode)
    if start_time is not None:
        storage.timestamp.set_any_trial_start_time(trial_id, start_time)
    if end_time is not None:
        storage.timestamp.set_any_trial_end_time(trial_id, end_time)


def main() -> None:
    """Writes the result of a trial to a file."""

    parser = ArgumentParser()
    parser.add_argument("--storage_file_path", type=str, required=True)
    parser.add_argument("--trial_id", type=int, required=True)
    parser.add_argument("--objective", nargs="+", type=str_or_float_or_int, default=None)
    parser.add_argument("--returncode", type=int, default=None)

    args = parser.parse_known_args()[0]

    unknown_args_list = parser.parse_known_args()[1]
    for unknown_arg in unknown_args_list:
        if unknown_arg.startswith("--"):
            parts = unknown_arg.split("=")
            name = parts[0].replace("--", "")
            parser.add_argument(f"--{name}", type=str_or_float_or_int)
    args = parser.parse_known_args()[0]

    xs = vars(copy.deepcopy(args))
    delete_keys = [
        "storage_file_path",
        "trial_id",
        "config",
        "objective",
        "returncode",
    ]

    for key in delete_keys:
        if key in xs.keys():
            del xs[key]

    write_results_to_database(
        storage_file_path=args.storage_file_path,
        trial_id=args.trial_id,
        objective=args.objective,
        returncode=args.returncode,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
