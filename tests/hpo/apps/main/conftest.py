from typing import Literal, TypeAlias

from collections.abc import Callable, Generator
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import pytest

# Define type aliases for the database utility functions
TrialCountFunc: TypeAlias = Callable[[Path, str], int]
TrialValuesFunc: TypeAlias = Callable[[Path, str], list[float]]
DBUtils: TypeAlias = dict[Literal["get_trial_count", "get_trial_values"], TrialCountFunc | TrialValuesFunc]
ConfigModFunc: TypeAlias = Callable[[Path, str, int, str], Path]


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with test files and clean up afterwards"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        original_dir = os.getcwd()
        os.chdir(tmp_dir)

        source_dir = Path(__file__).parent
        test_files = ["config.yaml", "objective.sh", "objective_for_test.py"]

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

        os.chmod(temp_path / "objective.sh", 0o755)
        yield temp_path
        os.chdir(original_dir)


@pytest.fixture
def db_utils() -> DBUtils:
    """Fixture providing database utility functions"""

    def get_trial_count(db_path: Path, study_name: str) -> int:
        """Get the number of trials from the SQLite database for a specific study"""
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
        return int(count) if count is not None else 0

    def get_trial_values(db_path: Path, study_name: str) -> list[float]:
        """Get the values of all completed trials for a specific study"""
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
        values = [row[0] for row in cursor.fetchall()]
        conn.close()
        return values

    return {"get_trial_count": get_trial_count, "get_trial_values": get_trial_values}


@pytest.fixture
def config_modifier() -> ConfigModFunc:
    """Fixture providing configuration file modification functionality"""

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
        return new_config_path

    return modify_config
