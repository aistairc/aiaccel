from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aiaccel.config import Config
from aiaccel.parameter import load_parameter
from aiaccel.util.cast import cast_y
from aiaccel.util.time_tools import get_time_now


class CommandLineArgs:
    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self.parser.add_argument('--trial_id', type=int, required=False)
        self.parser.add_argument('--config', type=str, required=False)
        self.args = self.parser.parse_known_args()[0]
        self.trial_id = None
        self.config_path = None
        self.config = None

        if self.args.trial_id is not None:
            self.trial_id = self.args.trial_id
        if self.args.config is not None:
            self.config_path = Path(self.args.config).resolve()
            self.config = Config(self.config_path)
            self.parameters_config = load_parameter(self.config.hyperparameters.get())

            for p in self.parameters_config.get_parameter_list():
                if p.type.lower() == "float":
                    self.parser.add_argument(f"--{p.name}", type=float)
                elif p.type.lower() == "int":
                    self.parser.add_argument(f"--{p.name}", type=int)
                elif p.type.lower() == "categorical":
                    self.parser.add_argument(f"--{p.name}", type=str)
                elif p.type.lower() == "ordinal":
                    self.parser.add_argument(f"--{p.name}", type=float)
                else:
                    raise ValueError(f"Unknown parameter type: {p.type}")
            self.args = self.parser.parse_known_args()[0]
        else:
            unknown_args_list = self.parser.parse_known_args()[1]
            for unknown_arg in unknown_args_list:
                if unknown_arg.startswith("--"):
                    name = unknown_arg.replace("--", "")
                    self.parser.add_argument(f"--{name}", type=float)
            self.args = self.parser.parse_known_args()[0]

    def get_xs_from_args(self) -> dict[str, Any]:
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
        args (dict): A dictionary object which contains command line arguments
            given by aiaccel.
        trial_id (int): Trial Id.
        config_path (Path): A Path object which points to the
            configuration file.
        config (Config): A Config object.
        workspace (Path): A Path object which points to the workspace.
        logger (Logger): A Logger object.

    Examples:
        *User program* ::

            from aiaccel.util import aiaccel

            run = aiaccel.Run()
            run.execute_and_report("execute user_program")

        Note that `execute user_program` is a command to execute a user
        program.
        See :doc:`../examples/wrapper_sample`.

        *Python function* ::

            from aiaccel.util import aiaccel

            def func(p: dict[str, Any]) -> float:
                # Write your operation to calculate objective value.

                return objective_y

            if __name__ == "__main__":
                run = aiaccel.Run()
                run.execute_and_report(func)
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = None
        self.config = None
        self.workspace = None

        self.args = CommandLineArgs()
        self.config_path = self.args.config_path or config_path
        self.config = self.args.config
        if self.config is not None:
            self.workspace = Path(self.config.workspace.get()).resolve()

    def execute(
        self,
        func: Callable[[dict[str, float | int | str]], float],
        xs: 'dict[str, float | int | str]',
        y_data_type: 'str | None'
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
            set_logging_file_for_trial_id(self.workspace, self.args.trial_id)

        y = None
        err = ""

        start_time = get_time_now()

        try:
            y = cast_y(func(xs), y_data_type)
        except BaseException as e:
            err = str(e)
            y = None
        else:
            err = ""

        end_time = get_time_now()

        return xs, y, err, start_time, end_time

    def execute_and_report(
        self,
        func: Callable[[dict[str, float | int | str]], float],
        y_data_type: str | None = None
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
        y: Any = None
        _, y, err, _, _ = self.execute(func, xs, y_data_type)

        self.report(y, err)

    def report(self, y: Any, err: str) -> None:
        """Save the results to a text file.

        Args:
            y (Any): Objective value.
            err (str): Error string.
        """

        sys.stdout.write(f"{y}\n")
        if err != "":
            sys.stderr.write(f"{err}\n")


def set_logging_file_for_trial_id(workspace: Path, trial_id: int) -> None:
    log_dir = workspace / "log"
    log_path = log_dir / f"job_{trial_id}.log"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    logging.basicConfig(filename=log_path, level=logging.DEBUG, force=True)
