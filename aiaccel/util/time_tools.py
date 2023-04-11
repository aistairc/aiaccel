import datetime


def get_time_delta(seconds: int) -> datetime.timedelta:
    """Get a delta time from specified seconds.

    Args:
        seconds (int): Seconds.

    Returns:
        datetime.timedelta: A timedelta object from specified seconds.
    """
    return datetime.timedelta(seconds=seconds)


def get_time_now_object() -> datetime.datetime:
    """Get a now datetime object.

    Returns:
        datetime.datetime: A now datetime object.
    """
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(
        hours=9)))


def get_time_now() -> str:
    """Get a now time string.

    Returns:
        str: A now time string with format '%m/%d/%Y %H:%M:%S'.
    """
    d = get_time_now_object()
    return d.strftime('%m/%d/%Y %H:%M:%S')


def get_time_string_from_object(d: datetime.datetime) -> str:
    """Get a time string from specified datetime object.

    Args:
        d (datetime.datetime): A datetime object.

    Returns:
        str: A time string from specified datetime object.
    """
    return d.strftime('%m/%d/%Y %H:%M:%S')


def get_datetime_from_string(string: str) -> datetime.datetime:
    """Get a datetime object from specified string.

    Args:
        string (str): A time string with format '%m/%d/%Y %H:%M:%S'.

    Returns:
        datetime.datetime: A datetime object from specified string.
    """
    return datetime.datetime.strptime(string, '%m/%d/%Y %H:%M:%S')
