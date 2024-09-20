import subprocess

from typing import Any


def qstat_xml(txt_data_path: str = "tests/job/qstat_dat.txt") -> Any:
    with open(txt_data_path) as f:
        p = subprocess.CompletedProcess(
            [],
            returncode=0,
            stdout=f.read().encode(),
            stderr=b"",
        )
    return p
