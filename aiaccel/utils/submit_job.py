from pathlib import Path
import subprocess
import time


def submit_job(
    command: str,
    submission_command: str,
    working_directory: Path | str,
    script_name: str = "job.sh",
    done_filename: Path | None = None,
    polling_time: float = 5.0,
) -> None:
    """
    Submit a job using a specified command and optionally wait for its completion.

    Parameters:
        command (str):
            The command to be executed, saved into a job script file.

        submission_command (str):
            The template command used to submit the job, containing '{job_filename}' as a placeholder.

        working_directory (Path | str):
            Directory where the job script will be created and executed.

        script_name (str, optional):
            Filename for the job script. Defaults to "job.sh".

        done_filename (Path | None, optional):
            Path to a file whose existence indicates the job is complete.
            If provided, the function will block until this file exists.
            Defaults to None.

        polling_time (float, optional):
            Interval (in seconds) between checks for the completion file. Defaults to 5.0 seconds.
    """

    if isinstance(working_directory, str):
        working_directory = Path(working_directory)

    working_directory.mkdir(parents=True, exist_ok=True)

    job_filename = working_directory / script_name
    with open(job_filename, "w") as f:
        f.write(command)

    subprocess.run(submission_command.format(job_filename=job_filename), shell=True)

    if done_filename is not None:
        while True:
            if done_filename.exists():
                break

            time.sleep(polling_time)
