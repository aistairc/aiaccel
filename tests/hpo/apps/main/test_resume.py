from collections.abc import Generator
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
from unittest.mock import patch
import uuid

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory with test files and clean up afterwards"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        original_dir = os.getcwd()
        os.chdir(tmp_dir)

        source_dir = Path(__file__).parent
        test_files = ["config.yaml", "objective_for_test.py", "execute_optuna_for_test.py"]

        for file_name in test_files:
            source_file = source_dir / file_name
            target_file = temp_path / file_name
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                os.chmod(target_file, 0o755)
                print(f"\n=== Content of {file_name} ===")
                print((target_file).read_text())
                print("=" * 40)
            else:
                pytest.skip(f"Required test file {file_name} not found in {source_dir}")

        yield temp_path
        os.chdir(original_dir)


def get_trial_count(db_path: Path, study_name: str) -> int:
    """Get the number of trials from the SQLite database for a specific study

    Args:
        db_path (Path): Path to the SQLite database
        study_name (str): Name of the study to query

    Returns:
        int: Number of trials for the specified study
    """
    if not db_path.exists():
        raise AssertionError("Database file does not exist")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) FROM trials
        WHERE study_id = (SELECT study_id FROM studies WHERE study_name = ?)
    """,
        (study_name,),
    )
    count = cursor.fetchone()[0]
    conn.close()
    if count is None:
        return 0
    return int(count)


def get_trial_values(db_path: Path, study_name: str) -> list[float]:
    """Get the values of all completed trials for a specific study

    Args:
        db_path (Path): Path to the SQLite database
        study_name (str): Name of the study to query

    Returns:
        list[float]: List of trial values in order of trial number
    """
    if not db_path.exists():
        raise AssertionError("Database file does not exist")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT trial_values.value
        FROM trials
        JOIN studies ON trials.study_id = studies.study_id
        JOIN trial_values ON trials.trial_id = trial_values.trial_id
        WHERE studies.study_name = ?
            AND trials.state = 'COMPLETE'
        ORDER BY trials.number
    """,
        (study_name,),
    )
    # Fetch all values from the query result

    values = [row[0] for row in cursor.fetchall()]
    conn.close()
    return values


def modify_config(config_path: Path, study_name: str, n_trials: int, db_name: str) -> Path:
    """Modify config file with new study name and number of trials"""
    with open(config_path) as f:
        content = f.read()
    content = content.replace("study_name: my_study", f"study_name: {study_name}")
    content = content.replace("n_trials: 30", f"n_trials: {n_trials}")

    if db_name:
        content = content.replace("url: sqlite:///aiaccel_storage.db", f"url: sqlite:///{db_name}")
    new_config_path = config_path.parent / f"config_{study_name}.yaml"
    with open(new_config_path, "w") as f:
        f.write(content)
    print(f"\nnew_config_path: {new_config_path}")
    return new_config_path


def test_optimization_consistency(temp_dir: Path) -> None:
    """Test that split execution (resumable + resume) gives same results as normal execution."""
    from aiaccel.hpo.apps.optimize import main

    # Use different database files for normal and split execution
    normal_db = "normal_storage.db"
    split_db = "split_storage.db"

    # Normal execution with 30 trials
    study_name_normal = f"test_study_{uuid.uuid4().hex[:8]}"
    normal_config = modify_config(temp_dir / "config.yaml", study_name_normal, 30, normal_db)

    with patch(
        "sys.argv",
        [
            "optimize.py",
            "--config",
            str(normal_config),
            "--command",
            "python -m aiaccel.jobs.cli.local gpu jobs/{job_name}.log "
            + "-- bash -c 'python objective_for_test.py --x1={x1} --x2={x2} > {out_filename}'",
        ],
    ):
        main()

    normal_results = get_trial_values(temp_dir / normal_db, study_name_normal)
    assert len(normal_results) == 30, "Normal execution should have 30 trials"
    normal_best = min(normal_results)

    trial_count = get_trial_count(temp_dir / normal_db, study_name_normal)
    assert trial_count == 30

    # Split execution with 15 + 15 trials
    study_name_split = f"test_study_{uuid.uuid4().hex[:8]}"
    split_config = modify_config(temp_dir / "config.yaml", study_name_split, 15, split_db)

    # First 15 trials
    with patch(
        "sys.argv",
        [
            "optimize.py",
            "--config",
            str(split_config),
            "--resumable",
            "--command",
            "python -m aiaccel.jobs.cli.local gpu jobs/{job_name}.log "
            + "-- bash -c 'python objective_for_test.py --x1={x1} --x2={x2} > {out_filename}'",
        ],
    ):
        main()

    trial_count = get_trial_count(temp_dir / split_db, study_name_split)

    assert trial_count == 15

    # Second 15 trials
    with patch(
        "sys.argv",
        [
            "optimize.py",
            "--config",
            str(split_config),
            "--resume",
            "--command",
            "python -m aiaccel.jobs.cli.local gpu jobs/{job_name}.log "
            + "-- bash -c 'python objective_for_test.py --x1={x1} --x2={x2} > {out_filename}'",
        ],
    ):
        main()

    trial_count = get_trial_count(temp_dir / split_db, study_name_split)
    assert trial_count == 30

    # optuna resume
    subprocess.run(["python", "execute_optuna_for_test.py"], check=True)
    # get best value
    normal_result = get_trial_values(temp_dir / "test_normal.db", "test_study_normal")
    normal_expected_best = min(normal_result)
    resume_result = get_trial_values(temp_dir / "test_resume.db", "test_study_resume")
    resume_expected_best = min(resume_result)

    # Compare results
    resume_results = get_trial_values(temp_dir / split_db, study_name_split)
    resume_best = min(resume_results)
    assert len(resume_results) == 30, "Split execution should have 30 trials"
    ## optuna's best value VS aiaccel's best value (normal execution)
    assert (
        abs(normal_expected_best - normal_best) < 1e-6
    ), f"Best values differ: normal={normal_best}, optuna={normal_expected_best}"
    ## optuna's best value VS aiaccel's best value (resume execution)
    assert (
        abs(resume_expected_best - resume_best) < 1e-6
    ), f"Best values differ: resume={resume_best}, optuna={resume_expected_best}"


def test_normal_execution(temp_dir: Path) -> None:
    """Test normal execution without resume functionality"""
    from aiaccel.hpo.apps.optimize import main

    db_name = "normal_test.db"
    study_name = f"test_study_{uuid.uuid4().hex[:8]}"
    config_path = modify_config(temp_dir / "config.yaml", study_name, 30, db_name)

    with patch(
        "sys.argv",
        [
            "optimize.py",
            "--config",
            str(config_path),
            "--command",
            "python -m aiaccel.jobs.cli.local gpu jobs/{job_name}.log "
            + "-- bash -c 'python objective_for_test.py --x1={x1} --x2={x2} > {out_filename}'",
        ],
    ):
        main()

    trial_count = get_trial_count(temp_dir / db_name, study_name)
    assert trial_count == 30


def test_resumable_execution(temp_dir: Path) -> None:
    """Test execution with `--resumable`
    This should run only the first half of the trials and save the state to a database file.
    """
    from aiaccel.hpo.apps.optimize import main

    db_name = "resumable_test.db"
    study_name = f"test_study_{uuid.uuid4().hex[:8]}"
    config_path = modify_config(temp_dir / "config.yaml", study_name, 15, db_name)

    with patch(
        "sys.argv",
        [
            "optimize.py",
            "--config",
            str(config_path),
            "--resumable",
            "--command",
            "python -m aiaccel.jobs.cli.local gpu jobs/{job_name}.log "
            + "-- bash -c 'python objective_for_test.py --x1={x1} --x2={x2} > {out_filename}'",
        ],
    ):
        main()

    db_path = temp_dir / db_name
    assert db_path.exists(), "Database file was not created"
    trial_count = get_trial_count(db_path, study_name)
    assert trial_count == 15


def test_resume_execution(temp_dir: Path) -> None:
    """Test the resume functionality of the optimization process.

    This test verifies that the optimization process can be resumed correctly
    from a saved state. It performs the following steps:
    1. Runs the optimization process with the `--resumable` flag and checks
       that the correct number of trials are completed.
    2. Resumes the optimization process with the `--resume` flag and checks
       that the remaining trials are completed correctly.

    Args:
        temp_dir (pathlib.Path): Temporary directory for test files.

    Raises:
        AssertionError: If the number of trials after each run does not match
                        the expected values.
    """

    from aiaccel.hpo.apps.optimize import main

    db_name = "resume_test.db"
    study_name = f"test_study_{uuid.uuid4().hex[:8]}"
    config_path = modify_config(temp_dir / "config.yaml", study_name, 15, db_name)

    with patch(
        "sys.argv",
        [
            "optimize.py",
            "--config",
            str(config_path),
            "--resumable",
            "--command",
            "python -m aiaccel.jobs.cli.local gpu jobs/{job_name}.log "
            + "-- bash -c 'python objective_for_test.py --x1={x1} --x2={x2} > {out_filename}'",
        ],
    ):
        main()

    db_path = temp_dir / db_name
    trial_count = get_trial_count(db_path, study_name)
    assert trial_count == 15

    with patch(
        "sys.argv",
        [
            "optimize.py",
            "--config",
            str(config_path),
            "--resume",
            "--command",
            "python -m aiaccel.jobs.cli.local gpu jobs/{job_name}.log "
            + "-- bash -c 'python objective_for_test.py --x1={x1} --x2={x2} > {out_filename}'",
        ],
    ):
        main()

    trial_count = get_trial_count(db_path, study_name)
    assert trial_count == 30
