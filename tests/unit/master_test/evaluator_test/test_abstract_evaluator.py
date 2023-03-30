from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path

import pytest

from aiaccel.command_line_options import CommandLineOptions
from aiaccel.common import dict_hp_finished
from aiaccel.common import dict_result
from aiaccel.common import extension_hp
from aiaccel.common import file_final_result
from aiaccel.master import AbstractEvaluator
from tests.base_test import BaseTest


class TestAbstractEvaluator(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_options(self) -> Generator[None, None, None]:
        self.options = CommandLineOptions(
            config=str(self.config_json),
            resume=None,
            clean=False,
            process_name="test"
        )
        self.evaluator = AbstractEvaluator(self.options)
        yield
        self.options = None
        self.evaluator = None

    def test_init(self) -> None:
        assert self.evaluator.hp_result is None

    def test_evaluate(self) -> None:
        assert self.evaluator.evaluate() is None

    def test_print(
        self,
        monkeypatch: pytest.MonkeyPatch,
        setup_hp_finished: Callable[[int], None],
        work_dir: Path
    ) -> None:
        setup_hp_finished(1)
        with monkeypatch.context() as m:
            m.setattr(self.evaluator, "hp_result", work_dir.joinpath(dict_hp_finished, f'001.{extension_hp}'))
            assert self.evaluator.print() is None

    def test_save(
        self,
        monkeypatch: pytest.MonkeyPatch,
        setup_hp_finished: Callable[[int], None],
        work_dir: Path
    ) -> None:
        setup_hp_finished(1)
        with monkeypatch.context() as m:
            m.setattr(self.evaluator, "hp_result", work_dir.joinpath(dict_hp_finished, f'001.{extension_hp}'))
            self.evaluator.save()
            assert work_dir.joinpath(dict_result, file_final_result).exists()
