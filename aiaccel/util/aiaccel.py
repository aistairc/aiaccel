from __future__ import annotations

import logging
import sys
import traceback
from argparse import ArgumentParser
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from aiaccel.common import datetime_format
from aiaccel.config import load_config
from aiaccel.parameter import (
    CategoricalParameter,
    FloatParameter,
    HyperParameterConfiguration,
    IntParameter,
    OrdinalParameter,
)
from aiaccel.util import cast_y
from aiaccel.util.data_type import str_or_float_or_int
from aiaccel.workspace import Workspace


class CommandLineArgs:
    """Command line arguments.

    Args:
        None

    Attributes:
        parser (ArgumentParser): Argument parser.
        args (Namespace): Namespace of arguments.
        trial_id (int | None): Trial ID.
        config_path (Path | None): Path to configuration file.
        config (dict[str, Any] | None): A dictionary of configuration.
        parameters_config (HyperParameterConfiguration | None):
            Hyper parameter configuration.
    """

    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self.parser.add_argument("--trial_id", type=int, required=False)
        self.parser.add_argument("--config", type=str, required=False)
        self.args = self.parser.parse_known_args()[0]
        self.trial_id = None
        self.config_path = None
        self.config = None

        if self.args.trial_id is not None:
            self.trial_id = self.args.trial_id
        if self.args.config is not None:
            self.config_path = Path(self.args.config).resolve()
            self.config = load_config(self.config_path)
            self.parameters_config = HyperParameterConfiguration(self.config.optimize.parameters)

            for p in self.parameters_config.get_parameter_list():
                if isinstance(p, FloatParameter):
                    self.parser.add_argument(f"--{p.name}", type=float)
                elif isinstance(p, IntParameter):
                    self.parser.add_argument(f"--{p.name}", type=int)
                elif isinstance(p, CategoricalParameter):
                    self.parser.add_argument(f"--{p.name}", type=str_or_float_or_int)
                elif isinstance(p, OrdinalParameter):
                    self.parser.add_argument(f"--{p.name}", type=str_or_float_or_int)
                else:
                    raise ValueError(f"Unknown parameter type: {p.type}")
            self.args = self.parser.parse_known_args()[0]
        else:
            unknown_args_list = self.parser.parse_known_args()[1]
            for unknown_arg in unknown_args_list:
                if unknown_arg.startswith("--"):
                    name = unknown_arg.replace("--", "")
                    self.parser.add_argument(f"--{name}", type=str_or_float_or_int)
            self.args = self.parser.parse_known_args()[0]

    def get_xs_from_args(self) -> dict[str, Any]:
        """Get a dictionary of parameters from command line arguments.

        Args:
            args (Namespace): Namespace of arguments.

        Returns:
            dict[str, Any]: A dictionary of parameters.
        """
        xs = vars(self.args)
        delete_keys = ["trial_id", "config"]
        for key in delete_keys:
            if key in xs.keys():
                del xs[key]

        return xs


class Run:
    """An Interface between user program or python function object.

    Args:
        config_path (str | Path | None, optional): A path to configration file.
            Defaults to None.

    Attributes:
        config_path (str | Path | None): A path to configration file.
        config (dict[str, Any] | None): A dictionary of configuration.
        workspace (Path | None): A path to workspace.
        args (CommandLineArgs): Command line arguments.

    Examples:
        *User program* ::

            import aiaccel

            run = aiaccel.Run()
            run.execute_and_report("execute user_program")

        Note that `execute user_program` is a command to execute a user
        program.
        See :doc:`../examples/wrapper_sample`.

        *Python function* ::

            import aiaccel

            def func(p: dict[str, Any]) -> float:
                # Write your operation to calculate objective value.

                return objective_y

            if __name__ == "__main__":
                run = aiaccel.Run()
                run.execute_and_report(func)
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config = None
        self.workspace = None
        self.args = CommandLineArgs()
        self.config_path = self.args.config_path or config_path
        self.config = self.args.config
        if self.config is not None:
            self.workspace = Workspace(self.config.generic.workspace)

    def execute(
        self,
        func: Callable[[dict[str, float | int | str]], float],
        xs: "dict[str, float | int | str]",
        y_data_type: "str | None",
    ) -> Any:
        """Executes the target function.

        Args:
            func (Callable[[dict[str, float | int | str]], float]):
                User-defined python function.
            trial_id (int): Trial ID.
            y_data_type (str | None): Name of data type of objective value.

        Returns:
            tuple[dict[str, float | int | str] | None, float | int | str | None, str]:
                A dictionary of parameters, a casted objective value, and error
                string.
        """
        if self.workspace is not None and self.args.trial_id is not None:
            set_logging_file_for_trial_id(self.workspace.path, self.args.trial_id)

        ys = None
        err = ""

        start_time = datetime.now().strftime(datetime_format)

        try:
            ys = cast_y(func(xs), y_data_type)
        except BaseException:
            err = str(traceback.format_exc())
            ys = None
        else:
            err = ""

        end_time = datetime.now().strftime(datetime_format)

        return xs, ys, err, start_time, end_time

    def execute_and_report(
        self, func: Callable[[dict[str, float | int | str]], float], y_data_type: str | None = None
    ) -> None:
        """Executes the target function and report the results.

        Args:
            func (Callable[[dict[str, float | int | str]], float]):
                User-defined python function.
            y_data_type (str | None, optional): Name of data type of
                objective value. Defaults to None.

        Examples:
         ::

            from aiaccel.util import aiaccel

            def func(p: dict[str, Any]) -> float:
                # Write your operation to calculate objective value.

                return objective_y

            if __name__ == "__main__":
                run = aiaccel.Run()
                run.execute_and_report(func)
        """

        xs = self.args.get_xs_from_args()
        ys: Any = None
        _, ys, err, _, _ = self.execute(func, xs, y_data_type)

        self.report(ys, err)

    def report(self, ys: Any, err: str) -> None:
        """Save the results to a text file.

        Args:
            ys (Any): Objective values.
            err (str): Error string.
        """

        if ys is not None:
            if isinstance(ys, str):
                ys = ys.replace(" ", "")
                ys = ys.split(",")
                for y in ys:
                    sys.stdout.write(f"{y}\n")
            elif isinstance(ys, (list, tuple)):
                for y in ys:
                    sys.stdout.write(f"{y}\n")
            else:
                sys.stdout.write(f"{ys}\n")
            sys.stdout.flush()
        if err != "":
            sys.stderr.write(f"{err}\n")
            sys.stdout.flush()
            exit(1)


def set_logging_file_for_trial_id(workspace: Path, trial_id: int) -> None:
    log_dir = workspace / "log"
    log_path = log_dir / f"job_{trial_id}.log"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    logging.basicConfig(filename=log_path, level=logging.DEBUG, force=True)
