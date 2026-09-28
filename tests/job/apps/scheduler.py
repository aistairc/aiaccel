# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from aiaccel.job.apps import SchedulerJobApp


class DummySchedulerJobApp(SchedulerJobApp):
    def __init__(self) -> None:
        # JobApp.__init__() は argparse を実行するので呼ばない
        self.status_filename_list = []

    def build_job_script(self) -> str:
        return "dummy"

    def build_submit_command(self) -> tuple[str, str]:
        return ("dummy-submit", "")

    def prepare_array_job_context(self) -> None:
        pass


@pytest.mark.parametrize(
    ("status_values", "expected"),
    [
        pytest.param(["0"], True, id="single-success"),
        pytest.param(["1"], False, id="single-failure"),
        pytest.param(["0", "0"], True, id="array-success"),
        pytest.param(["0", "1"], False, id="array-failure"),
        pytest.param(["0", None], False, id="array-missing"),
    ],
)
def test_is_skip_job_submission_status(
    tmp_path: Path,
    status_values: list[str | None],
    expected: bool,
) -> None:
    app = DummySchedulerJobApp()

    job_script = "#! /bin/bash\necho test\n"
    job_filename = tmp_path / "test.sh"
    job_filename.write_text(job_script)

    status_filename_list = []
    for i, status_value in enumerate(status_values):
        status_filename = tmp_path / f"test.{i}.out"
        status_filename_list.append(status_filename)

        if status_value is not None:
            status_filename.write_text(status_value)

    assert (
        app._is_skip_job_submission(
            job_filename,
            job_script,
            status_filename_list,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("job_file_content", "expected"),
    [
        pytest.param("#! /bin/bash\necho test\n", True, id="same-script"),
        pytest.param("#! /bin/bash\necho changed\n", False, id="changed-script"),
        pytest.param(None, False, id="missing-script"),
    ],
)
def test_is_skip_job_submission_job_script(
    tmp_path: Path,
    job_file_content: str | None,
    expected: bool,
) -> None:
    app = DummySchedulerJobApp()

    job_script = "#! /bin/bash\necho test\n"
    job_filename = tmp_path / "test.sh"

    if job_file_content is not None:
        job_filename.write_text(job_file_content)

    status_filename = tmp_path / "test.out"
    status_filename.write_text("0")

    assert (
        app._is_skip_job_submission(
            job_filename,
            job_script,
            [status_filename],
        )
        is expected
    )
