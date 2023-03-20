import logging

from aiaccel.util.logger import str_to_logging_level


def test_str_to_logging_level():
    assert str_to_logging_level("DEBUG") == logging.DEBUG
    assert str_to_logging_level("INFO") == logging.INFO
    assert str_to_logging_level("WARN") == logging.WARNING
    assert str_to_logging_level("WARNING") == logging.WARNING
    assert str_to_logging_level("ERROR") == logging.ERROR
    assert str_to_logging_level("CRITICAL") == logging.CRITICAL

    try:
        str_to_logging_level("invalid")
        assert False
    except ValueError:
        assert True
