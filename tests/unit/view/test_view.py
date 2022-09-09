from aiaccel.storage.storage import Storage
from aiaccel.cli.view import Viewer
import pathlib
from functools import wraps
from aiaccel.workspace import Workspace
from tests.unit.view.base import t_base
from tests.unit.view.base import db_path
from tests.unit.view.base import ws
from tests.unit.view.base import config_path
from aiaccel.config import Config


@t_base()
def test_view():
    storage = Storage(ws.path)

    trial_id = 5
    objective = 42.0
    start_time = "00:00"
    end_time = "10:00"
    jobstate = "test_state"
    state = "ready"

    storage.trial.set_any_trial_state(
        trial_id=trial_id,
        state=state
    )

    storage.timestamp.set_any_trial_start_time(
        trial_id=trial_id,
        start_time=start_time
    )

    storage.timestamp.set_any_trial_end_time(
        trial_id=trial_id,
        end_time=end_time
    )

    storage.jobstate.set_any_trial_jobstate(
        trial_id=trial_id,
        state=jobstate
    )

    storage.result.set_any_trial_objective(
        trial_id=trial_id,
        objective=objective
    )

    config = Config(config_path)

    options = {
        'config': config_path,
        'resume': None,
        'clean': False,
        'fs': False,
        'process_name': 'master'
    }

    viewer = Viewer(config, options)
    assert viewer.view() is None
