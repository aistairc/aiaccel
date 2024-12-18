from pathlib import Path
from unittest.mock import patch
import uuid

from test_optimize import TestOptimize


class TestResume(TestOptimize):

    def test_optimization_consistency(self, temp_dir: Path) -> None:
        """Test that split execution (resumable + resume) gives same results as normal execution.

        Test steps:
        1. Run 30 trials in normal mode:    python optimize.py objective.sh --config config.yaml
        2. Run 15 trials in resumable mode: python optimize.py objective.sh --config config.yaml --resumable
        3. Run 15 trials in resume mode:    python optimize.py objective.sh --config config.yaml --resume

        Assertions:
        - Both executions should complete 30 trials
        - The best values from both executions should be nearly identical (within 1e-6)
        """
        from aiaccel.hpo.apps.optimize import main

        # Use different database files for normal and split execution
        normal_db = "normal_storage.db"
        split_db = "split_storage.db"

        # Normal
        study_name_normal = f"test_study_{uuid.uuid4().hex[:8]}"
        normal_config = self.modify_config(temp_dir / "config.yaml", study_name_normal, 30, normal_db)

        with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(normal_config)]):
            main()

        normal_results = self.get_trial_values(temp_dir / normal_db, study_name_normal)
        assert len(normal_results) == 30, "Normal execution should have 30 trials"
        normal_best = min(normal_results)

        trial_count = self.get_trial_count(temp_dir / normal_db, study_name_normal)
        assert trial_count == 30

        # Split
        study_name_split = f"test_study_{uuid.uuid4().hex[:8]}"
        split_config = self.modify_config(temp_dir / "config.yaml", study_name_split, 15, split_db)

        # First 15 trials
        with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(split_config), "--resumable"]):
            main()

        trial_count = self.get_trial_count(temp_dir / split_db, study_name_split)

        assert trial_count == 15

        # Second 15 trials
        with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(split_config), "--resume"]):
            main()

        trial_count = self.get_trial_count(temp_dir / split_db, study_name_split)
        assert trial_count == 30

        # Compare results
        split_results = self.get_trial_values(temp_dir / split_db, study_name_split)
        split_best = min(split_results)
        assert len(split_results) == 30, "Split execution should have 30 trials"
        assert abs(normal_best - split_best) < 1e-6, f"Best values differ: normal={normal_best}, split={split_best}"

    def test_resumable_execution(self, temp_dir: Path) -> None:
        """Test execution with `--resumable`
        This should run only the first half of the trials and save the state to a database file.
        """
        from aiaccel.hpo.apps.optimize import main

        db_name = "resumable_test.db"
        study_name = f"test_study_{uuid.uuid4().hex[:8]}"
        config_path = self.modify_config(temp_dir / "config.yaml", study_name, 15, db_name)

        with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(config_path), "--resumable"]):
            main()

        db_path = temp_dir / db_name
        assert db_path.exists(), "Database file was not created"
        trial_count = self.get_trial_count(db_path, study_name)
        assert trial_count == 15

    def test_resume_execution(self, temp_dir: Path) -> None:
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
        config_path = self.modify_config(temp_dir / "config.yaml", study_name, 15, db_name)

        with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(config_path), "--resumable"]):
            main()

        db_path = temp_dir / db_name
        trial_count = self.get_trial_count(db_path, study_name)
        assert trial_count == 15

        with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(config_path), "--resume"]):
            main()

        trial_count = self.get_trial_count(db_path, study_name)
        assert trial_count == 30
