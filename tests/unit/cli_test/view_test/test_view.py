
from pathlib import Path

from aiaccel.cli.view import Viewer
from aiaccel.config import load_config
from aiaccel.storage import Storage
from aiaccel.workspace import Workspace


def test_view(clean_work_dir, work_dir, create_tmp_config):
    clean_work_dir()
    workspace = Workspace(str(work_dir))
    if workspace.path.exists():
        workspace.clean()
    workspace.create()

    config_path = Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)
    config = load_config(config_path)

    workspace = Workspace(config.generic.workspace)
    storage = Storage(workspace.storage_file_path)

    trial_id = 5
    objective = 42.0
    start_time = "00:00"
    end_time = "10:00"
    jobstate = "test_state"
    state = "ready"

    storage.trial.set_any_trial_state(trial_id=trial_id, state=state)
    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)
    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)
    storage.jobstate.set_any_trial_jobstate(trial_id=trial_id, state=jobstate)
    storage.result.set_any_trial_objective(trial_id=trial_id, objective=objective)

    trial_id = 6
    objective = 45.0
    start_time = "00:00"
    end_time = "10:00"
    jobstate = "test_state"
    state = "ready"
    error = "hogehoge"

    storage.trial.set_any_trial_state(trial_id=trial_id, state=state)
    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)
    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)
    storage.jobstate.set_any_trial_jobstate(trial_id=trial_id, state=jobstate)
    storage.result.set_any_trial_objective(trial_id=trial_id, objective=objective)
    storage.error.set_any_trial_error(trial_id=trial_id, error_message=error)

    trial_id = 7
    objective = 50.0
    start_time = "00:00"
    end_time = "10:00"
    jobstate = "test_state"
    state = "ready"
    error = "foo"

    storage.trial.set_any_trial_state(trial_id=trial_id, state=state)
    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)
    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)
    storage.jobstate.set_any_trial_jobstate(trial_id=trial_id, state=jobstate)
    storage.result.set_any_trial_objective(trial_id=trial_id, objective=objective)
    storage.error.set_any_trial_error(trial_id=trial_id, error_message=error)

    config = load_config(config_path)
    viewer = Viewer(config)
    assert viewer.view() is None
