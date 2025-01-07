from typing import cast

from pathlib import Path
from unittest.mock import patch
import uuid

from conftest import ConfigModFunc, DBUtils, TrialValuesFunc


def test_optimization_consistency(
    temp_dir: Path,
    db_utils: DBUtils,
    config_modifier: ConfigModFunc
) -> None:
    """Test that split execution (resumable + resume) gives same results as normal execution."""
    from aiaccel.hpo.apps.optimize import main

    # Use different database files for normal and split execution
    normal_db = "normal_storage.db"
    split_db = "split_storage.db"

    # Normal execution
    study_name_normal = f"test_study_{uuid.uuid4().hex[:8]}"
    normal_config = config_modifier(temp_dir / "config.yaml", study_name_normal, 30, normal_db)

    with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(normal_config)]):
        main()

    get_trial_values = cast(TrialValuesFunc, db_utils["get_trial_values"])
    normal_results = get_trial_values(temp_dir / normal_db, study_name_normal)
    assert len(normal_results) == 30, "Normal execution should have 30 trials"
    normal_best = min(normal_results)

    trial_count = db_utils["get_trial_count"](temp_dir / normal_db, study_name_normal)
    assert trial_count == 30

    # Split execution
    study_name_split = f"test_study_{uuid.uuid4().hex[:8]}"
    split_config = config_modifier(temp_dir / "config.yaml", study_name_split, 15, split_db)

    # First 15 trials
    with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(split_config), "--resumable"]):
        main()

    trial_count = db_utils["get_trial_count"](temp_dir / split_db, study_name_split)
    assert trial_count == 15

    # Second 15 trials
    with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(split_config), "--resume"]):
        main()

    trial_count = db_utils["get_trial_count"](temp_dir / split_db, study_name_split)
    assert trial_count == 30

    # Compare results
    split_results = get_trial_values(temp_dir / split_db, study_name_split)
    split_best = min(split_results)
    assert len(split_results) == 30, "Split execution should have 30 trials"
    assert abs(normal_best - split_best) < 1e-6, f"Best values differ: normal={normal_best}, split={split_best}"


def test_resumable_execution(
    temp_dir: Path,
    db_utils: DBUtils,
    config_modifier: ConfigModFunc
) -> None:
    """Test execution with `--resumable`"""
    from aiaccel.hpo.apps.optimize import main

    db_name = "resumable_test.db"
    study_name = f"test_study_{uuid.uuid4().hex[:8]}"
    config_path = config_modifier(temp_dir / "config.yaml", study_name, 15, db_name)

    with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(config_path), "--resumable"]):
        main()

    db_path = temp_dir / db_name
    assert db_path.exists(), "Database file was not created"
    trial_count = db_utils["get_trial_count"](db_path, study_name)
    assert trial_count == 15


def test_resume_execution(
    temp_dir: Path,
    db_utils: DBUtils,
    config_modifier: ConfigModFunc
) -> None:
    """Test the resume functionality of the optimization process."""
    from aiaccel.hpo.apps.optimize import main

    db_name = "resume_test.db"
    study_name = f"test_study_{uuid.uuid4().hex[:8]}"
    config_path = config_modifier(temp_dir / "config.yaml", study_name, 15, db_name)

    # First run with --resumable
    with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(config_path), "--resumable"]):
        main()

    db_path = temp_dir / db_name
    trial_count = db_utils["get_trial_count"](db_path, study_name)
    assert trial_count == 15

    # Second run with --resume
    with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(config_path), "--resume"]):
        main()

    trial_count = db_utils["get_trial_count"](db_path, study_name)
    assert trial_count == 30
