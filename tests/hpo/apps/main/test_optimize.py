from pathlib import Path
from unittest.mock import patch
import uuid

from conftest import ConfigModFunc, DBUtils


def test_normal_execution(
    temp_dir: Path,
    db_utils: DBUtils,
    config_modifier: ConfigModFunc
) -> None:
    """Test normal execution without resume functionality"""
    from aiaccel.hpo.apps.optimize import main

    db_name = "normal_test.db"
    study_name = f"test_study_{uuid.uuid4().hex[:8]}"
    config_path = config_modifier(temp_dir / "config.yaml", study_name, 30, db_name)

    with patch("sys.argv", ["optimize.py", "objective.sh", "--config", str(config_path)]):
        main()

    trial_count = db_utils["get_trial_count"](temp_dir / db_name, study_name)
    assert trial_count == 30
