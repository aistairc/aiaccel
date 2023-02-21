from __future__ import annotations
import copy
from pathlib import Path
from aiaccel.config import Config
from aiaccel.parameter import load_parameter
from argparse import ArgumentParser
from aiaccel import dict_result
from aiaccel import extension_hp
from aiaccel.util.filesystem import create_yaml


def main():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--trial_id', type=int, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--start_time', type=str, default='', required=True)
    parser.add_argument('--end_time', type=str, default='', required=True)
    parser.add_argument('--objective', type=float, required=True)
    parser.add_argument('--error', type=str, default='')
    args = parser.parse_known_args()[0]

    config_path = None
    config = None

    if args.config is not None:
        config_path = Path(args.config).resolve()
        config = Config(config_path)
        parameters_config = load_parameter(config.hyperparameters.get())

        for p in parameters_config.get_parameter_list():
            if p.type.lower() == "float":
                parser.add_argument(f"--{p.name}", type=float)
            elif p.type.lower() == "int":
                parser.add_argument(f"--{p.name}", type=int)
            elif p.type.lower() == "choice":
                parser.add_argument(f"--{p.name}", type=str)
            elif p.type.lower() == "ordinal":
                parser.add_argument(f"--{p.name}", type=float)
            else:
                raise ValueError(f"Unknown parameter type: {p.type}")
        args = parser.parse_known_args()[0]
    else:
        unknown_args_list = parser.parse_known_args()[1]
        for unknown_arg in unknown_args_list:
            if unknown_arg.startswith("--"):
                name = unknown_arg.replace("--", "")
                parser.add_argument(f"--{name}", type=float)
        args = parser.parse_known_args()[0]

    xs = vars(copy.deepcopy(args))
    delete_keys = [
        "file",
        "trial_id",
        "config",
        "start_time",
        "end_time",
        "objective",
        "error"
    ]

    for key in delete_keys:
        if key in xs.keys():
            del xs[key]

    contents = {
        'trial_id': args.trial_id,
        'paramerters': xs,
        'result': args.objective,
        'error': args.error,
        'start_time': args.start_time,
        'end_time': args.end_time
    }

    create_yaml(args.file, contents)


if __name__ == "__main__":  # pragma: no cover
    main()
