from __future__ import annotations

import copy
from argparse import ArgumentParser
from pathlib import Path

from aiaccel.experimental.mpi.config import load_config
from aiaccel.parameter import (
    CategoricalParameter,
    FloatParameter,
    HyperParameterConfiguration,
    IntParameter,
    OrdinalParameter,
)
from aiaccel.util.data_type import str_or_float_or_int
from aiaccel.util.filesystem import create_yaml


def main() -> None:
    """Writes the result of a trial to a file."""

    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--trial_id", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start_time", type=str, default="", required=True)
    parser.add_argument("--end_time", type=str, default="", required=True)
    parser.add_argument("--objective", nargs="+", type=str_or_float_or_int, default=None)
    parser.add_argument("--error", type=str, default="")
    parser.add_argument("--exitcode", type=int, default=None)

    args = parser.parse_known_args()[0]

    config_path = None
    config = None

    if args.config is not None:
        config_path = Path(args.config).resolve()
        config = load_config(config_path)

        parameters_config = HyperParameterConfiguration(config.optimize.parameters)

        for p in parameters_config.get_parameter_list():
            if isinstance(p, FloatParameter):
                parser.add_argument(f"--{p.name}", type=float)
            elif isinstance(p, IntParameter):
                parser.add_argument(f"--{p.name}", type=int)
            elif isinstance(p, CategoricalParameter):
                parser.add_argument(f"--{p.name}", type=str_or_float_or_int)
            elif isinstance(p, OrdinalParameter):
                parser.add_argument(f"--{p.name}", type=str_or_float_or_int)
            else:
                raise ValueError(f"Unknown parameter type: {p.type}")
        args = parser.parse_known_args()[0]
    else:
        unknown_args_list = parser.parse_known_args()[1]
        for unknown_arg in unknown_args_list:
            if unknown_arg.startswith("--"):
                name = unknown_arg.replace("--", "")
                parser.add_argument(f"--{name}", type=str_or_float_or_int)
        args = parser.parse_known_args()[0]

    xs = vars(copy.deepcopy(args))
    delete_keys = ["file", "trial_id", "config", "start_time", "end_time", "objective", "error", "exitcode"]

    for key in delete_keys:
        if key in xs.keys():
            del xs[key]

    contents = {
        "trial_id": args.trial_id,
        "result": args.objective,
        "paramerters": xs,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "exitcode": args.exitcode,
        "error": args.error,
    }

    if args.error == "":
        del contents["error"]

    print(contents)

    create_yaml(args.file, contents)


if __name__ == "__main__":  # pragma: no cover
    main()
