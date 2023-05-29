import datetime


def format_datetime_to_str(d: datetime.datetime) -> str:
    """Get a time string from specified datetime object.

    Args:
        d (datetime.datetime): A datetime object.

    Returns:
        str: A time string from specified datetime object.
    """
    return d.strftime("%m/%d/%Y %H:%M:%S")
