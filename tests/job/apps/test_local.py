# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import pytest

from tests.job.apps.base_job_tests import BaseJobTests


class TestLocal(BaseJobTests):
    use_config_default = False

    # Skip test_cpu_cancel_job in local
    def skip_cpu_cancel_job(self) -> None:
        pytest.skip("Scheduler cancellation is not supported")
