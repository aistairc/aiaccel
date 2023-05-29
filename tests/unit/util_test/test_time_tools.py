from datetime import datetime

from aiaccel.util import format_datetime_to_str


def test_format_datetime_to_str() -> None:
    now_object = datetime.now()
    now_string = now_object.strftime("%m/%d/%Y %H:%M:%S")
    assert format_datetime_to_str(now_object) == now_string
