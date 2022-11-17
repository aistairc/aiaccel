import pathlib
import subprocess
from functools import singledispatchmethod
from typing import Any

import aiaccel
import numpy as np
import os
import pathlib
import subprocess
import threading

from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.util.time_tools import get_time_now

from argparse import ArgumentParser
from functools import singledispatchmethod
from logging import StreamHandler, getLogger
from typing import Union

from aiaccel.config import Config
from aiaccel.optimizer.create import create_optimizer
from aiaccel.storage.storage import Storage
from aiaccel.util.time_tools import get_time_now
from aiaccel.util.filesystem import create_yaml


logger = getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
logger.addHandler(StreamHandler())

parser = ArgumentParser()
parser.add_argument('--config', type=str, default="")
parser.add_argument('--trial_id', type=str, required=False)
parser.add_argument('--resume', type=int, default=None)
parser.add_argument('--clean', nargs='?', const=True, default=False)
args = parser.parse_known_args()[0]


SUPPORTED_TYPES = [
    int,
    float,
    str,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.float128,
    np.complex64,
    np.complex128,
    np.complex256,
    np.bool,
    np.unicode,
    np.object
]


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


def report(objective_y=None, objective_err=None):
    """ user side reporting function

    Examples:
        import from aiaccel.util import aiaccel

        result = 0.0
        opt.report(result)
    """
    WrapperInterface().out(objective_y, objective_err)


class Abstruct:
    """
        It is assumed to refer to the user program
    """

    def __init__(self) -> None:
        self.args = vars(args)
        self.trial_id = self.args["trial_id"]
        self.config_path = pathlib.Path(self.args["config"])
        self.config = Config(self.config_path)
        self.workspace = pathlib.Path(self.config.workspace.get()).resolve()
        self.storage = Storage(self.workspace)

        # logger
        log_dir = self.workspace / "log"
        self.log_path = log_dir / f"job_{self.trial_id}.log"
        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        self.com = WrapperInterface()
        self.max_trial_number = self.config.trial_number.get()
        self.num_node = self.config.num_node.get()

    def get_any_trial_xs(self, trial_id: int) -> dict:
        params = self.optimizer.storage.hp.get_any_trial_params(trial_id=trial_id)
        if params is None:
            return

        xs = {}
        for param in params:
            cast = eval(param.param_type.lower())
            xs[param.param_name] = cast(param.param_value)

        return xs

    def generate_commands(self) -> list:
        """ Generate execution command of user program.

        Returns:
            list: execution command.
        """
        raise NotImplementedError

    def execute(self, func: callable) -> None:
        """ Execution the target function.

        Return:
            Objective value.
        """

        raise NotImplementedError

    def report(
        self,
        trial_id: int,
        xs: dict,
        y: any,
        err: str,
        start_time: str,
        end_time: str
    ) -> None:
        """ Write the result in yaml format to the result directory.
        """
        logger.info(f"{trial_id}, {xs, y}, {err}, {start_time}, {end_time}")

        self.optimizer.storage.result.set_any_trial_objective(
            trial_id=trial_id,
            objective=y
        )
        self.optimizer.storage.timestamp.set_any_trial_start_time(
            trial_id=trial_id,
            start_time=start_time
        )
        self.optimizer.storage.timestamp.set_any_trial_end_time(
            trial_id=trial_id,
            end_time=end_time
        )
        if err != "":
            self.optimizer.storage.error.set_any_trial_error(
                trial_id=trial_id,
                error_message=err
            )

    def execute_and_report(self) -> None:
        """
        Examples:
            def obj(p)
                y = p["x1"]
                return y

            run = aiaccel.Run()
            run.execute_and_report(obj)
        """

        raise NotImplementedError


class Abci(Abstruct):
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

    @singledispatchmethod
    def execute(self, func: callable, trial_id: int) -> tuple:
        """ Execution the target function.

        Return:
            Objective value.
        """

        xs = self.get_any_trial_xs(trial_id)
        y = None
        err = ""

        try:
            y = func(xs)
        except BaseException as e:
            err = str(e)
        finally:
            self.com.out(objective_y=y, objective_err=err)

        return xs, y, err

    @execute.register
    def _(self, command: str, trial_id: int):
        """ Execution the user program.

        Returns:
            ys (list): This is a list of the return values of the user program.
        """

        xs = self.get_any_trial_xs(trial_id)
        err = ""
        y = None

        # Make running command of user program
        if command == "":
            self.set_error("Invalid execute command")
            y = [float("nan")]
            return xs, y, err

        commands = self.generate_commands(command, xs)

        output = subprocess.run(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        ys, err = self.com.get_data(output)
        y = float(ys[0])  # todo: do refactoring
        err = ("\n").join(err)

        return xs, y, err

    @singledispatchmethod
    def execute_and_report(self, func: callable):
        """
        Examples:
            def obj(p)
                y = p["x1"]
                return y

            run = aiaccel.Run()
            run.execute_and_report(obj)
        """
        start_time = get_time_now()
        xs, y, err = self.execute(func, self.trial_id)
        end_time = get_time_now()

        self.report(self.trial_id, xs, y, err, start_time, end_time)

    @execute_and_report.register
    def _(self, command: str):
        """
        Examples:
            run = aiaccel.Run()
            p = run.parameters
            run.execute_and_report(f"echo {p['x1']}", False)
        """
        start_time = get_time_now()
        xs, y, err = self.execute(command, self.trial_id)
        end_time = get_time_now()

        self.report(self.trial_id, xs, y, err, start_time, end_time)


class Local(Abstruct):
    def __init__(self):
        super().__init__()
        Optimizer = create_optimizer(self.args['config'])
        self.optimizer = Optimizer(self.args)

    @singledispatchmethod
    def call_func(self, func: callable, xs: dict):
        y = None
        err = ""

        try:
            y = func(xs)
        except Exception as e:
            err = e

        return y, err

    @call_func.register
    def _(self, commands: list):

        output = subprocess.run(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        ys, err = self.com.get_data(output)
        y = float(ys[0])  # todo: do refactoring
        err = ("\n").join(err)

        return y, err

    def generate_commands(self, command: str, xs: list) -> list:
        """ Generate execution command of user program.

        Returns:
            list: execution command.
        """
        commands = command.split(" ")
        commands.append(f"--config={str(self.config_path)}")

        for key in xs:
            name = key
            value = xs[key]
            if value is not None:
                command = f"--{name}={value}"
                commands.append(command)

        return commands

    @singledispatchmethod
    def execute(self, func: callable, trial_id: int) -> tuple:
        xs = self.get_any_trial_xs(trial_id)
        y, err = self.call_func(func, xs)

        return xs, y, err

    @execute.register
    def _(self, command: str, trial_id: int):
        xs = self.get_any_trial_xs(trial_id)
        commands = self.generate_commands(command, xs)
        y, err = self.call_func(commands)

        return xs, y, err

    def run_once(self, objective: Union[callable, str], trial_id: int):
        self.optimizer.storage.trial.set_any_trial_state(
            trial_id=trial_id,
            state='running'
        )

        start_time = get_time_now()
        xs, y, err = self.execute(objective, trial_id)
        end_time = get_time_now()

        self.optimizer.storage.trial.set_any_trial_state(
            trial_id=trial_id,
            state='finished'
        )

        self.report(trial_id, xs, y, err, start_time, end_time)
        self.create_result_file(trial_id, y, err)

    def execute_and_report(self, objective: Union[callable, str]):
        self.optimizer.pre_process()

        while self.optimizer.check_finished() is False:
            pool_size = self.optimizer.get_pool_size()
            hp_ready = self.optimizer.storage.get_num_ready()

            if (pool_size <= 0 or hp_ready >= self.num_node):
                continue

            for _ in range(pool_size):
                new_params = self.optimizer.generate_new_parameter()
                if new_params is not None and len(new_params) > 0:
                    self.optimizer.register_new_parameters(new_params)
                    self.optimizer.trial_id.increment()
                    self.optimizer._serialize(self.optimizer.trial_id.integer)
                else:
                    continue

            trial_ids = self.optimizer.storage.trial.get_any_state_list('ready')
            if trial_ids is None:
                continue

            for trial_id in trial_ids:
                if self.num_node > 1:
                    th = threading.Thread(target=self.run_once, args=(objective, trial_id))
                    th.start()
                else:
                    self.run_once(objective, trial_id)

    def create_result_file(self, trial_id: int, y: any, err: str) -> None:
        content = self.optimizer.storage.get_hp_dict(trial_id)
        content['result'] = y

        if err is not None:
            content['error'] = err

        result_file_path = self.workspace / aiaccel.dict_result / (str(trial_id) + '.hp')
        create_yaml(result_file_path, content)


class Run:
    def __new__(cls):
        config = Config(args.config)
        resource_type = config.resource_type.get()
        if resource_type.lower() == 'abci':
            logger.info("abci")
            return Abci()
        elif resource_type.lower() == 'local':
            logger.info("local")
            return Local()
