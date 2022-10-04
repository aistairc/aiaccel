# import logging


# def str_to_logging_level(s: str) -> int:
#     """Convert from a input string to a logging level.

#     Args:
#         s(str): A string that specify a logging level.

#     Returns:
#         (int): It returns a logging level.

#     Raises:
#         ValueError: Causes when an invalid argument s is given.
#     """
#     if 'DEBUG' in s:
#         return logging.DEBUG
#     elif 'INFO' in s:
#         return logging.INFO
#     elif 'WARNING' in s:
#         return logging.WARNING
#     elif 'WARN' in s:
#         return logging.WARNING
#     elif 'ERROR' in s:
#         return logging.ERROR
#     elif 'CRITICAL' in s:
#         return logging.CRITICAL
#     else:
#         raise ValueError(f'Invalid logging level: {s}, {type(s)}')


import logging
from pathlib import Path


class Logger:
    def __init__(self, logger_name: str, logfile_path: Path):
        self.logger_name = logger_name
        self.logfile_path = logfile_path

    def create_logger(
        self,
        file_level: str,
        stream_level: str,
        module_type: str
    ) -> None:
        """Set a default logger options.

        Args:
            logger_name (str): A name of a logger.
            logfile (Path): A path to a log file.
            file_level (str): A logging level for a log file output.
            stream_level (str): A logging level for a stream output.
            module_type (str): A module type of a caller.

        Returns:
            None
        """
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(self.logfile_path, mode='w')
        self.fh_formatter = (
            '%(asctime)s %(levelname)-8s %(filename)-12s line '
            '%(lineno)-4s %(message)s'
        )
        self.fh_formatter = logging.Formatter(self.fh_formatter)
        self.fh.setFormatter(self.fh_formatter)
        self.fh.setLevel(self.str_to_logging_level(file_level))

        self.ch = logging.StreamHandler()
        self.ch_formatter = (f'{module_type} %(levelname)-8s %(message)s')
        self.ch_formatter = logging.Formatter(self.ch_formatter)
        self.ch.setFormatter(self.ch_formatter)
        self.ch.setLevel(self.str_to_logging_level(stream_level))

        logger.addHandler(self.fh)
        logger.addHandler(self.ch)
        return logger

    def delete(self, logger):
        logger.removeHandler(self.fh)
        logger.removeHandler(self.ch)

    def get_file_handler(self):
        return self.fh

    def get_stream_handler(self):
        return self.ch

    def str_to_logging_level(self, level: str) -> int:
        """Convert from a input string to a logging level.

        Args:
            s(str): A string that specify a logging level.

        Returns:
            (int): It returns a logging level.

        Raises:
            ValueError: Causes when an invalid argument s is given.
        """
        if level.lower() == 'debug':
            return logging.DEBUG
        elif level.lower() == 'info':
            return logging.INFO
        elif level.lower() == 'warning':
            return logging.WARNING
        elif level.lower() == 'warn':
            return logging.WARNING
        elif level.lower() == 'error':
            return logging.ERROR
        elif level.lower() == 'critical':
            return logging.CRITICAL
        else:
            raise ValueError(f'Invalid logging level: {level}, {type(level)}')
