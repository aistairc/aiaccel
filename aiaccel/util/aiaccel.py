import pathlib
from functools import singledispatchmethod
from aiaccel.parameter import load_parameter
from aiaccel.util.time_tools import get_time_now
from aiaccel.wrapper_tools import save_result
import aiaccel
import argparse
import subprocess
import logging
from typing import Any
from aiaccel.config import Config


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
        self.outputs.append("{}:{}".format(self.label, tmp))

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
    """　Interface of between Wrapper and User function.

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


class Run:
    """
        It is assumed to refer to the user program
    """

    def __init__(self) -> None:

        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--index', type=str, required=False)
        parser.add_argument('-c', '--config', type=str, required=False)

        self.args = vars(parser.parse_known_args()[0])

        self.xs = {}
        self.ys = None
        self.err = ""

        self.index = self.args["index"]
        self.config = None
        if self.args["config"] is not None:
            self.config_path = pathlib.Path(self.args["config"])
            self.config = Config(self.config_path)

            # create paths
            self.workspace = pathlib.Path(self.config.workspace.get())
            self.dict_lock = self.workspace / aiaccel.dict_lock
            self.dict_alive = self.workspace / aiaccel.dict_alive

        parameters_config = load_parameter(self.config.hyperparameters.get())
        for p in parameters_config.get_parameter_list():
            type_func = str
            if p.type == "FLOAT":
                type_func = float
            elif p.type == "INT":
                type_func = int
            # TODO Fix
            # elif p.type == "ORDINAL":
            #     type_func = float
            parser.add_argument(f"--{p.name}", type=type_func)
        # reparse arguments and load parameters
        self.args = vars(parser.parse_args())
        for p in parameters_config.get_parameter_list():
            self.xs[p.name] = self.args[p.name]

        # logger
        log_dir = self.workspace / "log"
        self.log_path = log_dir / f"job_{self.index}.log"
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.DEBUG
        )

        # t variables
        self.start_time = None
        self.end_time = None

        self.com = WrapperInterface()

    @property
    def hashname(self) -> str:
        """ Get tha hashname of this trial.

        Returns:
            index (str): hashname of this trial.
        """
        return self.index

    @property
    def parameters(self) -> dict:
        """ Get parameters dictionary.

        Returns
            xs (dict): Parameters dictionary.
        """
        return self.xs

    @property
    def objective(self) -> Any:
        """ Get the objective value.

        Returns
            y (Any): Objective value.
        """
        return self.ys

    @property
    def error(self):
        """ Get the error message from user program.

        Returns:
            str: error message.
        """
        return self.err

    def exist_error(self) -> bool:
        """ Return True if exist error else False.

        Returns:
            bool:   True : There is an error.
                    False: There is no error.
        """
        if self.err is None:
            return False
        if self.err != "":
            return True
        return False

    def trial_stop(self) -> None:
        """ Enforce an error to stop this trial.
        """
        if self.exist_error():
            pass
        else:
            self.set_error("Faital error")
        self.report(float('nan'))

    def set_error(self, mess: str) -> None:
        """ Set any error message.
        """
        self.err = mess

    def _generate_commands(self, command: str, auto_args) -> list:
        """ Generate execution command of user program.

        Returns:
            list: execution command.
        """
        commands = command.split(" ")
        if not auto_args:
            return commands

        commands.append("--config={}".format(str(self.config_path)))
        commands.append("--index={}".format(self.hashname))

        for key in self.xs:
            name = key
            value = self.xs[key]
            if value is not None:
                command = f"--{name}={value}"
                commands.append(command)

        return commands

    @singledispatchmethod
    def execute(self, func: callable):
        """ Execution the target function.

        Return:
            Objective value.
        """
        self.start_time = get_time_now()

        try:
            self.ys = func(self.xs)
        except BaseException as e:
            self.err = str(e)
        finally:
            self.end_time = get_time_now()
            self.com.out(objective_y=self.ys, objective_err=self.err)

        # stdout

        return self.ys

    @execute.register
    def _(self, command: str, auto_args: bool = True):
        """ Execution the user program.

        Returns:
            ys (list): This is a list of the return values of the user program.
        """

        # Make running command of user program
        if command == "":
            self.set_error("Invalid execute command")
            logging.debug("execute(err): {}".format(self.err))
            self.ys = [float("nan")]
            return self.ys

        commands = self._generate_commands(command, auto_args)
        logging.debug("command: {}".format(commands))

        self.start_time = get_time_now()
        logging.debug("start time: {}".format(self.start_time))

        output = subprocess.run(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ys, err = self.com.get_data(output)
        self.ys = float(ys[0])  # todo: do refactoring
        self.err = ("\n").join(err)
        logging.debug("execute(out): {}".format(self.ys))
        logging.debug("execute(err): {}".format(self.err))

        self.end_time = get_time_now()
        logging.debug("end time: {}".format(self.end_time))

        return self.ys

    def report(self, y):
        """ Write the result in yaml format to the result directory.

        Args:
            y (Union): Objective value. (return values of the user program.)
        """
        if (
            self.args["index"] is None or
            self.args["config"] is None
        ):
            return

        if self.args["config"] is not None:
            err = self.err
            if not (
                type(y) == int or
                type(y) == float or
                type(y) == str
            ):
                y = float("nan")
                err = f"user function returns invalid type value, {type(y)}({y})."

            save_result(
                self.workspace,
                self.dict_lock,
                self.index,
                y,
                self.start_time,
                self.end_time,
                err
            )

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
        self.report(self.execute(func))

    @execute_and_report.register
    def _(self, command: str, auto_args: bool = True):
        """
        Examples:
            run = aiaccel.Run()
            p = run.parameters
            run.execute_and_report(f"echo {p['x1']}", False)
        """
        self.report(self.execute(command, auto_args))
