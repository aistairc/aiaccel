from aiaccel.util.time_tools import get_time_delta, get_time_now_object,\
    get_time_now, get_time_string_from_object, get_datetime_from_string
import datetime


def test_time():
    delta_10_sec = get_time_delta(10)
    now = get_time_now_object()
    t = get_datetime_from_string('10/20/2020 00:00:00')
    t += delta_10_sec
    assert type(delta_10_sec) is datetime.timedelta
    assert type(now) is datetime.datetime
    assert type(get_time_now()) is str
    assert get_time_string_from_object(t) == '10/20/2020 00:00:10'
