import datetime


def get_now_str() -> str:
    """
    Get the current time as a string.

    Args:
        None

    Returns:
        str: A string in the format of yyyymmdd-hhMMss-nnn, where nnn represents milliseconds.
    """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]


def get_timestamp(now_str: str) -> float:
    """
    Get the timestamp from the string representation of a datetime.

    Args:
        now_str (str): The string representation of the datetime.

    Returns:
        float: The timestamp value.
    """
    dt = datetime.datetime.strptime(now_str + "000", "%Y%m%d-%H%M%S-%f")
    return dt.timestamp()
