from aiaccel.storage.storage import Storage
from base import db_path, t_base, ws


# set_any_trial_param
@t_base()
def test_set_any_trial_jobstate():
    storage = Storage(ws.path)

    trial_id = 0
    state = "test_state"

    assert storage.jobstate.set_any_trial_jobstate(
        trial_id=trial_id,
        state=state
    ) is None

    # update
    assert storage.jobstate.set_any_trial_jobstate(
        trial_id=trial_id,
        state="hogehoge"
    ) is None


# get_any_trial_jobstate
@t_base()
def test_get_any_trial_jobstate():
    storage = Storage(ws.path)

    trial_id = 0
    state = "test_state"

    storage.jobstate.set_any_trial_jobstate(
        trial_id=trial_id,
        state=state
    )

    s = storage.jobstate.get_any_trial_jobstate(trial_id)
    assert s == state

    # update
    storage.jobstate.set_any_trial_jobstate(
        trial_id=trial_id,
        state="hogehoge"
    )
    s = storage.jobstate.get_any_trial_jobstate(trial_id)
    assert s == "hogehoge"


# set_any_trial_jobstates
@t_base()
def test_set_any_trial_jobstates():
    storage = Storage(ws.path)

    states = [
        {'trial_id': 0, 'jobstate': 'stata_0'},
        {'trial_id': 1, 'jobstate': 'stata_1'},
        {'trial_id': 2, 'jobstate': 'stata_2'},
        {'trial_id': 3, 'jobstate': 'stata_3'},
        {'trial_id': 4, 'jobstate': 'stata_4'}
    ]

    assert storage.jobstate.set_any_trial_jobstates(states) is None
    assert storage.jobstate.get_any_trial_jobstate(0) == 'stata_0'
    assert storage.jobstate.get_any_trial_jobstate(1) == 'stata_1'
    assert storage.jobstate.get_any_trial_jobstate(2) == 'stata_2'
    assert storage.jobstate.get_any_trial_jobstate(3) == 'stata_3'
    assert storage.jobstate.get_any_trial_jobstate(4) == 'stata_4'

# get_all_trial_jobstate
@t_base()
def test_get_all_trial_jobstate():
    storage = Storage(ws.path)
    print(storage.jobstate.get_all_trial_jobstate())
    assert storage.jobstate.get_all_trial_jobstate() == [{'trial_id': None, 'jobstate': None}]

    states = [
        {'trial_id': 0, 'jobstate': 'stata_0'},
        {'trial_id': 1, 'jobstate': 'stata_1'},
        {'trial_id': 2, 'jobstate': 'stata_2'},
        {'trial_id': 3, 'jobstate': 'stata_3'},
        {'trial_id': 4, 'jobstate': 'stata_4'}
    ]
    
    storage.jobstate.set_any_trial_jobstates(states)

    assert storage.jobstate.get_all_trial_jobstate() == states

# is_failure
@t_base()
def test_is_failure():
    storage = Storage(ws.path)
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

    assert storage.jobstate.is_failure(len(ids) + 1) is False

# delete_any_trial_jobstate
@t_base()
def test_delete_any_trial_jobstate():
    storage = Storage(ws.path)

    trial_id = 0
    state = "test_state_0"
    storage.jobstate.set_any_trial_jobstate(
        trial_id=trial_id,
        state=state
    )

    trial_id = 1
    state = "test_state_1"
    storage.jobstate.set_any_trial_jobstate(
        trial_id=trial_id,
        state=state
    )

    trial_id = 2
    state = "test_state_2"
    storage.jobstate.set_any_trial_jobstate(
        trial_id=trial_id,
        state=state
    )

    assert storage.jobstate.get_any_trial_jobstate(trial_id=0) is not None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=1) is not None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=2) is not None

    assert storage.jobstate.delete_any_trial_jobstate(trial_id=0) is None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=0) is None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=1) is not None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=2) is not None

    assert storage.jobstate.delete_any_trial_jobstate(trial_id=1) is None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=0) is None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=1) is None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=2) is not None

    assert storage.jobstate.delete_any_trial_jobstate(trial_id=2) is None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=0) is None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=1) is None
    assert storage.jobstate.get_any_trial_jobstate(trial_id=2) is None
