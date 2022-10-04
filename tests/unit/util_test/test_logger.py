from aiaccel.util.logger import Logger
import logging


def test_str_to_logging_level():

    _logger = Logger(
        logger_name='root.optimizer',
        logfile_path=''
    )
    assert _logger.str_to_logging_level('DEBUG') == logging.DEBUG
    assert _logger.str_to_logging_level('INFO') == logging.INFO
    assert _logger.str_to_logging_level('WARN') == logging.WARNING
    assert _logger.str_to_logging_level('WARNING') == logging.WARNING
    assert _logger.str_to_logging_level('ERROR') == logging.ERROR
    assert _logger.str_to_logging_level('CRITICAL') == logging.CRITICAL

    try:
        _logger.str_to_logging_level('invalid')
        assert False
    except ValueError:
        assert True
