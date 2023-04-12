import logging


def str_to_logging_level(s: str) -> int:
    """Convert from a input string to a logging level.

    Args:
        s (str): A string that specify a logging level.

    Returns:
        int: It returns a logging level.

    Raises:
        ValueError: Causes when an invalid argument s is given.
    """
    if 'DEBUG' in s:
        return logging.DEBUG
    elif 'INFO' in s:
        return logging.INFO
    elif 'WARNING' in s:
        return logging.WARNING
    elif 'WARN' in s:
        return logging.WARNING
    elif 'ERROR' in s:
        return logging.ERROR
    elif 'CRITICAL' in s:
        return logging.CRITICAL
    else:
        raise ValueError(f'Invalid logging level: {s}, {type(s)}')
