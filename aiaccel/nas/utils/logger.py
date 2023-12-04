from __future__ import annotations

import os
from collections.abc import Callable
from logging import FileHandler, Formatter, Logger, LogRecord, getLogger
from pathlib import Path
from typing import Any


def create_supernet_train_report(
    epoch: int | None = None,
    elapsed_time: float | None = None,
    train_loss: float | None = None,
    top_1_train_acc: float | None = None,
    top_5_train_acc: float | None = None,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    report = {
        "result": "",
        "epoch": epoch,
        "elapsed_time": elapsed_time,
        "train_loss": train_loss,
        "top_1_train_acc": top_1_train_acc,
        "top_5_train_acc": top_5_train_acc,
        "learning_rate": learning_rate,
    }
    return report


def create_architecture_search_report(
    epoch: int | None = None,
    elapsed_time: float | None = None,
    valid_loss: float | None = None,
    top_1_valid_acc: float | None = None,
    top_5_valid_acc: float | None = None,
    convergence: float | None = None,
) -> dict[str, Any]:
    report = {
        "result": "",
        "epoch": epoch,
        "elapsed_time": elapsed_time,
        "valid_loss": valid_loss,
        "top_1_valid_acc": top_1_valid_acc,
        "top_5_valid_acc": top_5_valid_acc,
        "convergence": convergence,
    }
    return report


def create_retrain_report(
    epoch: int | None = None,
    elapsed_time: float | None = None,
    train_loss: float | None = None,
    top_1_train_acc: float | None = None,
    top_5_train_acc: float | None = None,
    test_loss: float | None = None,
    top_1_test_acc: float | None = None,
    top_5_test_acc: float | None = None,
    convergence: float | None = None,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    report = {
        "result": "",
        "epoch": epoch,
        "elapsed_time": elapsed_time,
        "train_loss": train_loss,
        "top_1_train_acc": top_1_train_acc,
        "top_5_train_acc": top_5_train_acc,
        "test_loss": test_loss,
        "top_1_test_acc": top_1_test_acc,
        "top_5_test_acc": top_5_test_acc,
        "convergence": convergence,
        "learning_rate": learning_rate,
    }
    return report


def _create_logger(
    name: str,
    level: str | int,
    filepath: Path | str,
    filter: Callable[[LogRecord], bool],
    format: str,
) -> Logger:
    logger = getLogger(name)
    logger.setLevel(level)
    handler = FileHandler(filepath)
    handler.addFilter(filter)
    handler.setFormatter(Formatter(format))
    for hdr in logger.handlers[:]:
        logger.removeHandler(hdr)
        hdr.close()
    logger.addHandler(handler)
    return logger


def create_supernet_train_logger(
    workdir: str | Path | None = None, name: str = "root.search.supernet_train_result"
) -> Logger:
    workdir = Path(workdir if workdir else os.getcwd()).resolve()
    if not workdir.exists():
        workdir.mkdir(parents=True)
    logger = _create_logger(
        name,
        "DEBUG",
        workdir / "supernet_train.csv",
        _supernet_train_result_filter,
        "%(result)s",
    )
    logger.propagate = False
    logger.info(
        "",
        extra=create_supernet_train_report(
            epoch="epoch",
            elapsed_time="elapsed_time",
            train_loss="train_loss",
            top_1_train_acc="top_1_train_acc",
            top_5_train_acc="top_5_train_acc",
            learning_rate="learning_rate",
        ),
    )
    logger.propagate = True
    return logger


def create_architecture_search_logger(
    workdir: str | Path | None = None,
    name: str = "root.search.architecture_search_result",
) -> Logger:
    workdir = Path(workdir if workdir else os.getcwd()).resolve()
    if not workdir.exists():
        workdir.mkdir(parents=True)
    logger = _create_logger(
        name,
        "DEBUG",
        workdir / "architecture_search.csv",
        _architecture_search_result_filter,
        "%(result)s",
    )
    logger.propagate = False
    logger.info(
        "",
        extra=create_architecture_search_report(
            epoch="epoch",
            elapsed_time="elapsed_time",
            valid_loss="valid_loss",
            top_1_valid_acc="top_1_valid_acc",
            top_5_valid_acc="top_5_valid_acc",
            convergence="convergence",
        ),
    )
    logger.propagate = True
    return logger


def create_retrain_logger(workdir: str | Path | None = None, name="root.retrain.retrain_result") -> Logger:
    workdir = Path(workdir if workdir else os.getcwd()).resolve()
    if not workdir.exists():
        workdir.mkdir(parents=True)
    logger = _create_logger(
        name,
        "DEBUG",
        workdir / "retrain.csv",
        _retrain_result_filter,
        "%(result)s",
    )
    logger.propagate = False
    logger.info(
        "",
        extra=create_retrain_report(
            epoch="epoch",
            elapsed_time="elapsed_time",
            train_loss="train_loss",
            top_1_train_acc="top_1_train_acc",
            top_5_train_acc="top_5_train_acc",
            test_loss="test_loss",
            top_1_test_acc="top_1_test_acc",
            top_5_test_acc="top_5_test_acc",
            convergence="convergence",
            learning_rate="learning_rate",
        ),
    )
    logger.propagate = True
    return logger


def create_logger(workdir: str | Path | None = None, name: str = "root") -> Logger:
    workdir = Path(workdir if workdir else os.getcwd()).resolve()
    if not workdir.exists():
        workdir.mkdir(parents=True)
    logger = _create_logger(
        name,
        "DEBUG",
        workdir / f"{name.split('.')[-1]}.log",
        _result_filter,
        "%(asctime)s %(levelname)-7s %(name)s:%(filename)s:%(lineno)s %(message)s",
    )
    return logger


def _supernet_train_result_filter(record: LogRecord) -> bool:
    if hasattr(record, "result") and record.name == "root.search.supernet_train_result":
        result = ",".join(
            map(
                str,
                [
                    record.epoch,
                    record.elapsed_time,
                    record.train_loss,
                    record.top_1_train_acc,
                    record.top_5_train_acc,
                    record.learning_rate,
                ],
            )
        )
        record.result = result
        return True
    return False


def _architecture_search_result_filter(record: LogRecord) -> bool:
    if hasattr(record, "result") and record.name == "root.search.architecture_search_result":
        result = ",".join(
            map(
                str,
                [
                    record.epoch,
                    record.elapsed_time,
                    record.valid_loss,
                    record.top_1_valid_acc,
                    record.top_5_valid_acc,
                    record.convergence,
                ],
            )
        )
        record.result = result
        return True
    return False


def _retrain_result_filter(record: LogRecord) -> bool:
    if hasattr(record, "result") and record.name == "root.retrain.retrain_result":
        result = ",".join(
            map(
                str,
                [
                    record.epoch,
                    record.elapsed_time,
                    record.train_loss,
                    record.top_1_train_acc,
                    record.top_5_train_acc,
                    record.test_loss,
                    record.top_1_test_acc,
                    record.top_5_test_acc,
                    record.convergence,
                    record.learning_rate,
                ],
            )
        )
        record.result = result
        return True
    return False


def _result_filter(record: LogRecord) -> bool:
    if hasattr(record, "result"):
        if record.name == "root.search.supernet_train_result":
            msg = (
                f"epoch: {record.epoch:8d}, elapsed time: {record.elapsed_time:17.6f}, "
                f" train loss: {record.train_loss:10.6f},  top-1 train acc.: {record.top_1_train_acc:10.6f}, "
                f" top-5 train acc.: {record.top_5_train_acc:10.6f}, learning rate: {record.learning_rate:10.6f}"
            )
        elif record.name == "root.search.architecture_search_result":
            msg = (
                f"epoch: {record.epoch:8d}, elapsed time: {record.elapsed_time:17.6f}, "
                f"valid. loss: {record.valid_loss:10.6f}, top-1 valid. acc.: {record.top_1_valid_acc:10.6f}, "
                f"top-5 valid. acc.: {record.top_5_valid_acc:10.6f}, convergence: {record.convergence:10.6f}"
            )
        elif record.name == "root.retrain.retrain_result":
            msg = (
                f"epoch: {record.epoch:8d}, elapsed time: {record.elapsed_time:17.6f}, "
                f"train loss: {record.train_loss:10.6f}, top-1 train acc.: {record.top_1_train_acc:10.6f}, "
                f"top-5 train acc.: {record.top_5_train_acc:10.6f}, test loss: {record.test_loss:10.6f}, "
                f"top-1 test acc.: {record.top_1_test_acc:10.6f}, top-5 test acc.: {record.top_5_test_acc:10.6f}, "
                f"convergence: {record.convergence:10.6f}, learning rate: {record.learning_rate:10.6f}"
            )
        else:
            record.levelname = "WARNING"
            record.levelno = 30
            msg = (
                f"Unknown logger: {record.name}, "
                f"epoch: {getattr(record, 'epoch', None):8d}, "
                f"elapsed time: {getattr(record, 'elapsed_time', None):17.6f}, "
                f" train loss: {getattr(record, 'train_loss', None):10.6f}, "
                f" top-1 train acc.: {getattr(record, 'train_acc1', None):10.6f}, "
                f" top-5 train acc.: {getattr(record, 'train_acc5', None):10.6f}, "
                f"valid. loss: {getattr(record, 'valid_loss', None):10.6f}, "
                f"top-1 valid. acc.: {getattr(record, 'valid_acc1', None):10.6f}, "
                f"top-5 valid. acc.: {getattr(record, 'valid_acc5', None):10.6f}, "
                f"  test loss: {getattr(record, 'test_loss', None):10.6f}, "
                f"  top-1 test acc.: {getattr(record, 'test_acc1', None):10.6f}, "
                f"  top-5 test acc.: {getattr(record, 'test_acc5', None):10.6f}, "
                f"  convergence: {getattr(record, 'convergence', None):10.6f}, "
                f"learning rate: {getattr(record, 'learning_rate', None):10.6f}"
            )
        record.msg = msg
    return True
