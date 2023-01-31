from __future__ import annotations
import subprocess
from argparse import ArgumentParser
from functools import singledispatchmethod
from typing import Any, Dict, Tuple, Union, Optional
from collections.abc import Callable
from pathlib import Path

from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.util.time_tools import get_time_now
from aiaccel.parameter import load_parameter


class CommandLineArgs:
    def __init__(self):
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
                elif p.type.lower() == "choice":
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

    def get_xs(self) -> dict:
        xs = vars(self.args)
        del xs["trial_id"]
        del xs["config"]

        return xs


class _Message:
    """

    Attributes:
        label (str)    : A something like message ID.
        outputs (list) : It will be output messages.
        delimiter (str): The received data will be divided by this symbol.

    Example:
        self.m = Message("test")
        self.m.out("hogehoge")
        STDOUT: test:hogehoge
        self.parse_result(stdout)
            -> return hogehoge
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.outputs = []
        self.delimiter = "@"

    def create_message(self, message: Any):
        """ Concatenates a label and a message.

        Args:
            message (Any): Content to be out to stdout.

        Returns:
            None
        """
        if type(message).__name__ != "list":
            mess = [message]
        else:
            mess = message
        tmp = self.delimiter.join(map(str, mess))
        self.outputs.append(f"{self.label}:{tmp}")

    def out(self, all=False) -> None:
        """ Output message to stdout.

        Args
            all (bool): If its value is true then print all self.output.
        """
        if all is True:
            for o in self.outputs:
                print(o)
        else:
            for o in self.outputs:
                if o.split(":")[1] != "":
                    print(o)

    def parse(self, raw_data: str) -> None:
        """

        Args:
            raw_data (str): It is assumed the format
            e.g "{label}:{message}".format(
                    label="HOGE",
                    message="hoge@hoge@hoge"
                )
        """
        raw_data = raw_data.split("\n")
        target_data = []
        for line in raw_data:
            label = line.split(":")[0]
            if label != self.label:
                continue
            target_data = line.split(":")[1].split(self.delimiter)
            break
        if len(target_data) == 0:
            target_data.append("")
        return target_data

    def clear(self):
        self.outputs = []


class Messages:
    def __init__(self, *labels: tuple) -> None:
        labels = list(labels)
        self.d = {}
        for label in labels:
            self.d[label] = _Message(label)

    def create_message(self, label: str, mess: str) -> None:
        """ Create a message with a any label.

        Args:
            label (str): Create a message with this label name.
            mess (str) : Message.
        """
        self.d[label].create_message(mess)

    def out(self, label):
        """ Display any labels message.

        Args:
            label (str): Displays a message with this label name.
        """
        self.d[label].out()

    def clear(self, label):
        self.d[label].clear()

    def parse(self, label, mess):
        """
        Args:
            label (str): Name of the label to be extracted.
            mess (str) : Raw messages.

        Note:
            The target data is only in the form of {label}:{message}.

        Examples:
            parse("hoge", "hoge:foo")
            -> return "foo"
        """
        return self.d[label].parse(mess)

    def get(self, label, index=-1):
        return self.d[label].outputs[index]


class WrapperInterface:
    """Interface between Wrapper and User function.

    Note:
        user function:
            output  "objective_y: y"
                    "objective_err: err"
        wrapper:
            input   "objective_y: y" -> y
                    "objective_err: err" -> err
    """

    def __init__(self):
        self.stdout = Messages(
            "objective_y",
            "objective_err"
        )

    def get_data(self, output: subprocess.CompletedProcess[bytes]) -> tuple:
        """For wrapper side, gets stdout and stderr of user program.

        Args:
            output (CompletedProcess[str]): A CompletedProcess instance which
            have attributes args, returncode, stdout and stderr.

        Returns:
            tuple(ys: str, err: str):
                ys (list): The return value of user program.
                err (list): The error message of user program.
        """
        ys = self.stdout.parse(
            "objective_y", output.stdout.decode("UTF-8")
        )
        if ys is None:
            ys = self.stdout.parse(
                "objective_y", output.stderr.decode("UTF-8")
            )

        err = self.stdout.parse(
            "objective_err", output.stdout.decode("UTF-8")
        )
        if err is None:
            err = self.stdout.parse(
                "objective_err", output.stderr.decode("UTF-8")
            )

        return (ys, err)

    def out(
        self, objective_y: Union[float, int, str] = None,
        objective_err: str = None
    ) -> None:
        """For user program side, outputs the objective value and error message
        generated in the user-defined function.

        Args:
            objective_y (Union[float, int, str], optional): Objective value
            returned from the user-defined function. Defaults to None.
            objective_err (str, optional): Error message. Defaults to None.
        """
        y = objective_y if objective_y is not None else float("nan")
        e = objective_err if objective_err is not None else ""
        self.stdout.create_message("objective_y", y)
        self.stdout.create_message("objective_err", e)

        self.stdout.out("objective_y")
        self.stdout.out("objective_err")

        self.stdout.clear("objective_y")
        self.stdout.clear("objective_err")


class Run:
    """An Interface between user program or python function and Storage object.

    Attributes:
        args (dict): A dictionary object which contains command line arguments
        given by aiaccel.
        trial_id (int): Trial Id.
        config_path (Path): A Path object which points to the configuration
        file.
        config (Config): A Config object.
        workspace (Path): A Path object which points to the workspace.
        storage (Storage): A Storage object.
        logger (Logger): A Logger object.
        com (WrapperInterface): A WrapperInterface object.

    Examples:
        *User program*

        ```
        from aiaccel.util import aiaccel

        run = aiaccel.Run()
        run.execute_and_report("execute user_program")
        ```
        Note that `execute user_program` is a command to execute a user
        program.
        See :doc:`docs/source/examples/wrapper_sample`.

        *Python function*

        ```
        from aiaccel.util import aiaccel

        def func(p: dict[str, Any]) -> float:
            # Write your operation to calculate objective value.

            return objective_y

        if __name__ == "__main__":
            run = aiaccel.Run()
            run.execute_and_report(func)
        ```

    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initial method for Run.

        Args:
            config_path (Optional[Union[str, Path]], optional): A path to
            configration file. Defaults to None.
        """

        self.config_path = None
        self.config = None
        self.workspace = None
        self.storage = None

        self.args = CommandLineArgs()
        self.config_path = self.args.config_path or config_path
        self.config = self.args.config
        if self.config is not None:
            self.workspace = Path(self.config.workspace.get()).resolve()
            self.storage = Storage(self.workspace)
        self.com = WrapperInterface()

    def generate_commands(
        self,
        command: str,
        xs: dict[str, Optional[Union[float, int, str]]]
    ) -> list[str]:
        """ Generate execution command of user program.

        Args:
            command (str): An Execution command to calculate objective value.
            xs (dict[str, Optional[Union[float, int, str]]]): A dictionary of
            parameters of which key is parameter name and value is parameter
            value.

        Returns:
            List: A list of execution command and options.
        """

        commands = command.split(" ")
        # commands.append(f"--config={str(config_path)}")
        # commands.append(f"--trial_id={trial_id}")

        for key in xs:
            name = key
            value = xs[key]
            if value is not None:
                # command = f"--{name} {value}"
                # commands.append(command)
                commands.append(f"--{name}")
                commands.append(f"{value}")

        return commands

    def get_any_trial_xs(self, trial_id: int) -> Optional[dict[str]]:
        """Gets a parameter list of specific trial ID from Storage object.

        Args:
            trial_id (int): Trial ID.

        Returns:
            Optional[dict]: A dictionary of parameters. None if the parameter
            specified by the given trial ID is not registered.
        """
        params = self.storage.hp.get_any_trial_params(trial_id=trial_id)
        if params is None:
            return

        xs = {}
        for param in params:
            xs[param.param_name] = param.param_value

        return xs

    def cast_y(
        self, y_value: Any, y_data_type: Optional[str]
    ) -> Union[float, int, str]:
        """Casts y to the appropriate data type.

        Args:
            y_value (Any): y value to be casted.
            y_data_type (Optional[str]): Name of data type of objective value.

        Returns:
            Union[float, int, str]: Casted y value.

        Raises:
            TypeError: Occurs when given `y_data_type` is other than `float`,
            `int`, or `str`.
        """
        if y_data_type is None:
            y = y_value
        elif y_data_type.lower() == 'float':
            y = float(y_value)
        elif y_data_type.lower() == 'int':
            y = int(float(y_value))
        elif y_data_type.lower() == 'str':
            y = str(y_value)
        else:
            TypeError(f'{y_data_type} cannot be specified')

        return y

    @singledispatchmethod
    def execute(
        self,
        func: Callable[[dict[str, Union[float, int, str]]], float],
        xs: Optional[Dict[str, Union[float, int, str]]],
        y_data_type: Optional[str]
    ) -> tuple[Optional[dict[str, Union[float, int, str]]],
               Optional[Union[float, int, str]],
               str]:
        """Executes the target function.

        Args:
            func (Callable[[dict[str, Union[float, int, str]]], float]):
            User-defined python function.
            trial_id (int): Trial ID.
            y_data_type (Optional[str]): Name of data type of objective value.

        Returns:
            tuple[Optional[dict[str, Union[float, int, str]]],
            Optional[Union[float, int, str]], str]: A dictionary of parameters,
            a casted objective value, and error string.
        """

        y = None
        err = ""

        start_time = get_time_now()

        try:
            y = self.cast_y(func(xs), y_data_type)
        except BaseException as e:
            err = str(e)
        finally:
            self.com.out(objective_y=y, objective_err=err)

        end_time = get_time_now()

        return xs, y, err, start_time, end_time

    @execute.register
    def _(
        self, command: str,
        xs: Optional[Dict[str, Union[float, int, str]]],
        y_data_type: Optional[str]
    ) -> Tuple[Optional[Dict[str, Union[float, int, str]]],
               Optional[Union[float, int, str]],
               str]:
        """ Executes the user program.

        Args:
            command (str): An Execution command to calculate objective value.
            trial_id (int): Trial ID.
            y_data_type (Optional[str]): Name of data type of objective value.

        Returns:
            tuple[Optional[dict[str, Union[float, int, str]]],
            Optional[Union[float, int, str]], str]: A dictionary of parameters,
            a casted objective value, and error string.
        """

        err = ""
        y = None

        start_time = get_time_now()

        # Make running command of user program
        if command == "":
            y = [float("nan")]
            return xs, y, err

        commands = self.generate_commands(command, xs)

        print(commands)

        output = subprocess.run(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        ys, err = self.com.get_data(output)
        if y_data_type is None:
            y = self.cast_y(ys[0], 'float')
        else:
            y = self.cast_y(ys[0], y_data_type)
        err = ("\n").join(err)

        end_time = get_time_now()

        return xs, y, err, start_time, end_time

    @singledispatchmethod
    def execute_and_report(
        self, func: Callable[[dict[str, Union[float, int, str]]], float],
        y_data_type: Optional[str] = None
    ) -> None:
        """Executes the target function and report the results.

        Args:
            func (Callable[[dict[str, Union[float, int, str]]], float]):
            User-defined python function.
            y_data_type (Optional[str], optional): Name of data type of
            objective value. Defaults to None.

        Examples:
            from aiaccel.util import aiaccel

            def func(p: dict[str, Any]) -> float:
                # Write your operation to calculate objective value.

                return objective_y

            if __name__ == "__main__":
                run = aiaccel.Run()
                run.execute_and_report(func)
        """
        trial_id = self.args.trial_id
        xs = self.args.get_xs()
        xs, y, err, start_time, end_time = self.execute(func, xs, y_data_type)

        self.report(trial_id, xs, y, err, start_time, end_time)

    @execute_and_report.register
    def _(self, command: str, y_data_type: Optional[str] = None):
        """Executes the user program.

        Args:
            command (str): An Execution command to calculate objective value.
            y_data_type (Optional[str], optional): Name of data type of
            objective value. Defaults to None.

        Examples:
            from aiaccel.util import aiaccel
            run = aiaccel.Run()
            run.execute_and_report("execute user_program")
        """
        trial_id = self.args.trial_id
        xs = self.args.get_xs()
        xs, y, err, start_time, end_time = self.execute(command, xs, y_data_type)

        self.report(trial_id, xs, y, err, start_time, end_time)

    def report(
        self, trial_id: int, xs: dict, y: any, err: str, start_time: str,
        end_time: str
    ) -> None:
        """Saves results in the Storage object.

        Args:
            trial_id (int): Trial ID.
            xs (dict): A dictionary of parameters.
            y (any): Objective value.
            err (str): Error string.
            start_time (str): Execution start time.
            end_time (str): Execution end time.
        """

        if self.storage is not None:
            self.storage.result.set_any_trial_objective(trial_id, y)
            self.storage.timestamp.set_any_trial_start_time(trial_id, start_time)
            self.storage.timestamp.set_any_trial_end_time(trial_id, end_time)
            if err != "":
                self.storage.error.set_any_trial_error(trial_id, err)
        else:
            print(
                {
                    'trial_id': trial_id,
                    'params': xs,
                    'objective': y,
                    'error': err,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
