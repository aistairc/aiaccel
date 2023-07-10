import logging

from colorama import Fore, init

init(autoreset=True)


class ColoredHandler(logging.StreamHandler):  # type: ignore
    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == logging.DEBUG:
            self.stream.write(f"{Fore.BLUE}{self.format(record)}\n")
        elif record.levelno == logging.INFO:
            self.stream.write(f"{Fore.GREEN}{self.format(record)}\n")
        elif record.levelno == logging.WARNING:
            self.stream.write(f"{Fore.YELLOW}{self.format(record)}\n")
        elif record.levelno == logging.ERROR:
            self.stream.write(f"{Fore.RED}{self.format(record)}\n")
        elif record.levelno == logging.CRITICAL:
            self.stream.write(f"{Fore.MAGENTA}{self.format(record)}\n")
        else:  # default color
            self.stream.write(f"{self.format(record)}\n")


def str_to_logging_level(s: str) -> int:
    """Convert from a input string to a logging level.

    Args:
        s (str): A string that specify a logging level.

    Returns:
        int: It returns a logging level.

    Raises:
        ValueError: Causes when an invalid argument s is given.
    """
    if "DEBUG" in s:
        return logging.DEBUG
    elif "INFO" in s:
        return logging.INFO
    elif "WARNING" in s:
        return logging.WARNING
    elif "WARN" in s:
        return logging.WARNING
    elif "ERROR" in s:
        return logging.ERROR
    elif "CRITICAL" in s:
        return logging.CRITICAL
    else:
        raise ValueError(f"Invalid logging level: {s}, {type(s)}")
