import logging
import pathlib
import subprocess
from argparse import ArgumentParser
from functools import singledispatchmethod
from logging import StreamHandler, getLogger
from typing import Any, Union

from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.util.time_tools import get_time_now


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
    """ Interface of between Wrapper and User function.

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

    def get_data(self, output) -> tuple:
        """ For wrapper side.

        Args:
            raw_stdout(str): stdout(UTF-8)

        Returns:
            tuple(ys: str, err: str):
                ys (list) : The return value of user program.
                err (list): The error message of user program.
        """
        ys = self.stdout.parse("objective_y", output.stdout.decode("UTF-8"))
        if ys is None:
            ys = self.stdout.parse("objective_y", output.stderr.decode("UTF-8"))

        err = self.stdout.parse("objective_err", output.stdout.decode("UTF-8"))
        if err is None:
            err = self.stdout.parse("objective_err", output.stderr.decode("UTF-8"))

        return (ys, err)

    def out(self, objective_y=None, objective_err=None):
        """ For user program side.
        """
        y = objective_y if objective_y is not None else float("nan")
        e = objective_err if objective_err is not None else ""
        self.stdout.create_message("objective_y", y)
        self.stdout.create_message("objective_err", e)

        self.stdout.out("objective_y")
        self.stdout.out("objective_err")


class Run:
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument('--config', type=str)
        parser.add_argument('--workspace', type=str)
        parser.add_argument('--trial_id', type=str)
        parser.add_argument('--max_trial_number', type=str, required=False)
        parser.add_argument('--num_node', type=str, required=False)
        parser.add_argument('--goal', type=str, required=False)
        parser.add_argument('--name_length', type=str, required=False)

        args = parser.parse_known_args()[0]

        self.args = vars(args)
        self.trial_id = self.args["trial_id"]
        self.config_path = pathlib.Path(self.args["config"])
        self.config = None

        self.max_trial_number = self.args["max_trial_number"]
        self.num_node = self.args["num_node"]
        self.goal = self.args["goal"]
        self.name_length = self.args["name_length"]
        self.workspace = self.args["workspace"]

        if self.max_trial_number is None:
            if self.config is None:
                self.config = Config(self.config_path)
                self.max_trial_number = self.config.trial_number.get()

        if self.num_node is None:
            if self.config is None:
                self.config = Config(self.config_path)
                self.num_node = self.config.num_node.get()

        if self.goal is None:
            if self.config is None:
                self.config = Config(self.config_path)
                self.goal = self.config.goal.get()

        if self.name_length is None:
            if self.config is None:
                self.config = Config(self.config_path)
                self.name_length = self.config.name_length.get()

        if self.workspace is None:
            if self.config is None:
                self.config = Config(self.config_path)
                self.workspace = self.config.workspace.get()

        self.workspace = pathlib.Path(self.workspace).resolve()
        self.storage = Storage(self.workspace)

        # logger
        log_dir = self.workspace / "log"
        log_path = log_dir / f"job_{self.trial_id}.log"
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        logging.basicConfig(filename=log_path, level=logging.DEBUG)
        self.logger = getLogger(__name__)
        self.logger.addHandler(StreamHandler())

        self.com = WrapperInterface()


    def generate_commands(self, command: str, xs: list) -> list:
        """ Generate execution command of user program.

        Returns:
            list: execution command.
        """
        commands = command.split(" ")
        commands.append(f"--config={str(self.config_path)}")
        commands.append(f"--trial_id={self.trial_id}")

        for key in xs:
            name = key
            value = xs[key]
            if value is not None:
                command = f"--{name}={value}"
                commands.append(command)

        return commands

    def get_any_trial_xs(self, trial_id: int) -> dict:
        params = self.storage.hp.get_any_trial_params(trial_id=trial_id)
        if params is None:
            return

        xs = {}
        for param in params:
            cast = eval(param.param_type.lower())
            xs[param.param_name] = cast(param.param_value)

        return xs

    def cast_y(self, y_value: any, y_data_type: Union[None, str]):
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
    def execute(self, func: callable, trial_id: int, y_data_type: Union[None, str]) -> tuple:
        """ Execution the target function.

        Return:
            Objective value.
        """

        xs = self.get_any_trial_xs(trial_id)
        y = None
        err = ""

        try:
            y = self.cast_y(func(xs), y_data_type)
        except BaseException as e:
            err = str(e)
        finally:
            self.com.out(objective_y=y, objective_err=err)

        return xs, y, err

    @execute.register
    def _(self, command: str, trial_id: int, y_data_type: Union[None, str]) -> tuple:
        """ Execution the user program.

        Returns:
            ys (list): This is a list of the return values of the user program.
        """

        xs = self.get_any_trial_xs(trial_id)
        err = ""
        y = None

        # Make running command of user program
        if command == "":
            y = [float("nan")]
            return xs, y, err

        commands = self.generate_commands(command, xs)

        output = subprocess.run(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        ys, err = self.com.get_data(output)
        if y_data_type is None:
            y = self.cast_y(ys[0], 'float')
        else:
            y = self.cast_y(ys[0], y_data_type)
        err = ("\n").join(err)

        return xs, y, err

    @singledispatchmethod
    def execute_and_report(self, func: callable, y_data_type: Union[None, str] = None):
        """

        Examples:
            def obj(p)
                y = p["x1"]
                return y
            run = aiaccel.Run()
            run.execute_and_report(obj)
        """
        start_time = get_time_now()
        xs, y, err = self.execute(func, self.trial_id, y_data_type)
        end_time = get_time_now()

        self.report(self.trial_id, xs, y, err, start_time, end_time)

    @execute_and_report.register
    def _(self, command: str, y_data_type: Union[None, str] = None):
        """

        Examples:
            run = aiaccel.Run()
            p = run.parameters
            run.execute_and_report(f"echo {p['x1']}", False)
        """
        start_time = get_time_now()
        xs, y, err = self.execute(command, self.trial_id, y_data_type)
        end_time = get_time_now()

        self.report(self.trial_id, xs, y, err, start_time, end_time)

    def report(self, trial_id: int, xs: dict, y: any, err: str, start_time: str, end_time: str) -> None:
        """ Write the result in yaml format to the result directory.
        """
        self.logger.info(f"{trial_id}, {xs, y}, {err}, {start_time}, {end_time}")

        self.storage.result.set_any_trial_objective(trial_id, y)
        self.storage.timestamp.set_any_trial_start_time(trial_id, start_time)
        self.storage.timestamp.set_any_trial_end_time(trial_id, end_time)
        if err != "":
            self.storage.error.set_any_trial_error(trial_id, err)
