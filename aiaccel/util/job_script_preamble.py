from __future__ import annotations

from pathlib import Path


def create_job_script_preamble(job_script_preamble: str, job_script_preamble_path: Path | str | None) -> str:
    """Create a job script preamble.
    args:
        job_script_preamble_path (Path | str | None): A path to a job script preamble file.
        content (str): A content of a job script preamble. It is a written in a config file.

    returns:
        str: A job script preamble.
    """
    content = job_script_preamble
    if isinstance(content, str) and len(content) > 0:
        return content
    if job_script_preamble_path is not None and len(str(job_script_preamble_path)) > 0:
        job_script_preamble_path = Path(job_script_preamble_path).resolve()
        if Path(job_script_preamble_path).exists():
            with open(job_script_preamble_path, "r") as f:
                content = f.read()
            return content
        else:
            raise FileNotFoundError(f"File not found: {job_script_preamble_path}")
    return content
