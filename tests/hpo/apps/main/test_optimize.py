from collections.abc import Generator
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from unittest.mock import patch
import uuid

import pytest


class TestOptimize:

    @pytest.fixture
    def temp_dir(self) -> Generator[Path]:
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

    def get_trial_count(self, db_path: Path, study_name: str) -> int:
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

    def get_trial_values(self, db_path: Path, study_name: str) -> list[float]:
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

    def modify_config(self, config_path: Path, study_name: str, n_trials: int, db_name: str) -> Path:
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


class TestOptimizeNormalExecution(TestOptimize):

    def test_normal_execution(self, temp_dir: Path) -> None:
        """Test normal execution without resume functionality"""
        from aiaccel.hpo.apps.optimize import main

        db_name = "normal_test.db"
        study_name = f"test_study_{uuid.uuid4().hex[:8]}"
        config_path = self.modify_config(temp_dir / "config.yaml", study_name, 30, db_name)

        with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(config_path)]):
            main()

        trial_count = self.get_trial_count(temp_dir / db_name, study_name)
        assert trial_count == 30
