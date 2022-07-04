from aiaccel.storage.storage import Storage
from base import t_base
from base import db_path
from base import ws
from base import config_path


# set_any_trial_param
@t_base()
def test_set_any_trial_jobstate():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    trial_id = 0
    state = "test_state"

    assert storage.jobstate.set_any_trial_jobstate(
        trial_id=trial_id,
        state=state
    ) is None


# get_any_trial_jobstate
@t_base()
def test_get_any_trial_jobstate():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    trial_id = 0
    state = "test_state"

    storage.jobstate.set_any_trial_jobstate(
        trial_id=trial_id,
        state=state
    )

    s = storage.jobstate.get_any_trial_jobstate(trial_id)
    assert s == state


# is_failure
@t_base()
def test_is_failure():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)
    test_states = [
        'RunnerFailure',
        'HpRunningFailure',
        'JobFailure',
        'HpFinishedFailure',
        'HpExpireFailure',
        'HpExpiredFailure',
        'KillFailure',
        'HpCancelFailure',
        'Success'
    ]
    ids = range(len(test_states))

    for i in range(len(test_states)):
        storage.jobstate.set_any_trial_jobstate(
            trial_id=ids[i],
            state=test_states[i]
        )

    for i in range(len(test_states)):
        if "Failure" in test_states[i]:
            assert storage.jobstate.is_failure(ids[i]) is True
        else:
            assert storage.jobstate.is_failure(ids[i]) is False
