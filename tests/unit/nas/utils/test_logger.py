import logging
from logging import FileHandler, Formatter, Logger, LogRecord
from pathlib import Path

import pytest

from aiaccel.nas.utils.logger import (
    _architecture_search_result_filter,
    _create_logger,
    _result_filter,
    _retrain_result_filter,
    _supernet_train_result_filter,
    create_architecture_search_logger,
    create_architecture_search_report,
    create_logger,
    create_retrain_logger,
    create_retrain_report,
    create_supernet_train_logger,
    create_supernet_train_report,
)


def test_create_supernet_train_report():
    report = create_supernet_train_report(1, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert report == {
        "result": "",
        "epoch": 1,
        "elapsed_time": 2.0,
        "train_loss": 3.0,
        "top_1_train_acc": 4.0,
        "top_5_train_acc": 5.0,
        "learning_rate": 6.0,
    }

    report = create_supernet_train_report()
    assert report == {
        "result": "",
        "epoch": None,
        "elapsed_time": None,
        "train_loss": None,
        "top_1_train_acc": None,
        "top_5_train_acc": None,
        "learning_rate": None,
    }


def test_create_architecture_search_report():
    report = create_architecture_search_report(1, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert report == {
        "result": "",
        "epoch": 1,
        "elapsed_time": 2.0,
        "valid_loss": 3.0,
        "top_1_valid_acc": 4.0,
        "top_5_valid_acc": 5.0,
        "convergence": 6.0,
    }

    report = create_architecture_search_report()
    assert report == {
        "result": "",
        "epoch": None,
        "elapsed_time": None,
        "valid_loss": None,
        "top_1_valid_acc": None,
        "top_5_valid_acc": None,
        "convergence": None,
    }


def test_create_retrain_report():
    report = create_retrain_report(1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    assert report == {
        "result": "",
        "epoch": 1,
        "elapsed_time": 2.0,
        "train_loss": 3.0,
        "top_1_train_acc": 4.0,
        "top_5_train_acc": 5.0,
        "test_loss": 6.0,
        "top_1_test_acc": 7.0,
        "top_5_test_acc": 8.0,
        "convergence": 9.0,
        "learning_rate": 10.0,
    }

    report = create_retrain_report()
    assert report == {
        "result": "",
        "epoch": None,
        "elapsed_time": None,
        "train_loss": None,
        "top_1_train_acc": None,
        "top_5_train_acc": None,
        "test_loss": None,
        "top_1_test_acc": None,
        "top_5_test_acc": None,
        "convergence": None,
        "learning_rate": None,
    }


def test__create_logger(caplog):
    logger = _create_logger("test", "DEBUG", Path("/tmp/test_create_logger.log"), lambda record: True, "%(message)s")
    logger.info("Test created logger")

    assert len(caplog.records) == 1

    record = next(iter(caplog.records))
    assert record.message == "Test created logger"
    assert record.levelno == logging.INFO
    assert record.module == "test_logger"


def test_create_supernet_train_logger(mocker):
    logger_mock = mocker.Mock(spec=Logger)
    mocker.patch("aiaccel.nas.utils.logger._create_logger", return_value=logger_mock)
    mocker.patch("pathlib.Path.cwd", return_value="/path/to/workdir")

    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("pathlib.Path.mkdir")

    logger = create_supernet_train_logger(None, "test")
    assert logger == logger_mock
    assert logger.propagate

    logger = create_supernet_train_logger("/path/to/workdir", "test")
    assert logger == logger_mock
    assert logger.propagate


def test_create_architecture_search_logger(mocker):
    logger_mock = mocker.Mock(spec=Logger)
    mocker.patch("aiaccel.nas.utils.logger._create_logger", return_value=logger_mock)
    mocker.patch("pathlib.Path.cwd", return_value="/path/to/workdir")

    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("pathlib.Path.mkdir")

    logger = create_architecture_search_logger(None, "test")
    assert logger == logger_mock
    assert logger.propagate

    logger = create_architecture_search_logger("/path/to/workdir", "test")
    assert logger == logger_mock
    assert logger.propagate


def test_create_retrain_logger(mocker):
    logger_mock = mocker.Mock(spec=Logger)
    mocker.patch("aiaccel.nas.utils.logger._create_logger", return_value=logger_mock)
    mocker.patch("pathlib.Path.cwd", return_value="/path/to/workdir")

    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("pathlib.Path.mkdir")

    logger = create_retrain_logger(None, "test")
    assert logger == logger_mock
    assert logger.propagate

    logger = create_retrain_logger("/path/to/workdir", "test")
    assert logger == logger_mock
    assert logger.propagate


def test_create_logger(mocker):
    logger_mock = mocker.Mock(spec=Logger)
    mocker.patch("aiaccel.nas.utils.logger._create_logger", return_value=logger_mock)
    mocker.patch("pathlib.Path.cwd", return_value="/path/to/workdir")

    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("pathlib.Path.mkdir")

    logger = create_logger(None, "test")
    assert logger == logger_mock

    logger = create_logger("/path/to/workdir", "test")
    assert logger == logger_mock


def test_supernet_train_result_filter(mocker):
    record_mock = mocker.Mock(spec=LogRecord)
    record_mock.name = "root.search.supernet_train_result"
    record_mock.epoch = 1
    record_mock.elapsed_time = 2.0
    record_mock.train_loss = 0.1
    record_mock.top_1_train_acc = 0.9
    record_mock.top_5_train_acc = 0.8
    record_mock.learning_rate = 0.01

    record_mock.result = "test"
    assert _supernet_train_result_filter(record_mock)
    assert record_mock.result == "1,2.0,0.1,0.9,0.8,0.01"

    delattr(record_mock, "result")
    assert not _supernet_train_result_filter(record_mock)


def test_retrain_result_filter(mocker):
    # Mock the LogRecord class
    record_mock = mocker.Mock(spec=LogRecord)
    record_mock.name = "root.retrain.retrain_result"
    record_mock.epoch = 1
    record_mock.elapsed_time = 2.0
    record_mock.train_loss = 0.1
    record_mock.top_1_train_acc = 0.9
    record_mock.top_5_train_acc = 0.8
    record_mock.test_loss = 0.2
    record_mock.top_1_test_acc = 0.7
    record_mock.top_5_test_acc = 0.6
    record_mock.convergence = 0.5
    record_mock.learning_rate = 0.01

    # Test the _retrain_result_filter function with a record that has a "result" attribute
    record_mock.result = "test"
    assert _retrain_result_filter(record_mock)
    assert record_mock.result == "1,2.0,0.1,0.9,0.8,0.2,0.7,0.6,0.5,0.01"

    # Test the _retrain_result_filter function with a record that does not have a "result" attribute
    delattr(record_mock, "result")
    assert not _retrain_result_filter(record_mock)


def test_result_filter(mocker):
    record_mock = mocker.Mock(spec=LogRecord)
    record_mock.epoch = 1
    record_mock.elapsed_time = 2.0
    record_mock.train_loss = 0.1
    record_mock.top_1_train_acc = 0.9
    record_mock.top_5_train_acc = 0.8
    record_mock.valid_loss = 0.2
    record_mock.top_1_valid_acc = 0.7
    record_mock.top_5_valid_acc = 0.6
    record_mock.test_loss = 0.3
    record_mock.top_1_test_acc = 0.5
    record_mock.top_5_test_acc = 0.4
    record_mock.convergence = 0.3
    record_mock.learning_rate = 0.01

    for name in [
        "root.search.supernet_train_result",
        "root.search.architecture_search_result",
        "root.retrain.retrain_result",
    ]:
        record_mock.name = name
        record_mock.result = "test"
        assert _result_filter(record_mock)

    record_mock.name = "unknown"
    record_mock.result = "test"
    assert _result_filter(record_mock)
    assert record_mock.levelname == "WARNING"
    assert record_mock.levelno == 30

    delattr(record_mock, "result")
    assert _result_filter(record_mock)
