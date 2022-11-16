from aiaccel.storage.storage import Storage
from aiaccel.storage.output import AbciOutput
from base import db_path, t_base, ws, init
from undecorated import undecorated
from sqlalchemy.exc import SQLAlchemyError

import pytest


# set_any_trial_abci_output
@t_base()
def test_set_any_trial_abci_output():

    output = AbciOutput(db_path)

    trial_id = 0
    message = "hoge"

    assert output.set_any_trial_abci_output(
        trial_id=trial_id,
        message=message
    ) is None

    # update
    assert output.set_any_trial_abci_output(
        trial_id=trial_id,
        message="update"
    ) is None


# set_any_trial_abci_output exception
@t_base()
def test_set_any_trial_abci_output_exception():

    output = AbciOutput(db_path)

    trial_id = 0
    message = "hoge"

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_trial_abci_output = undecorated(output.set_any_trial_abci_output)
        set_any_trial_abci_output(
            output,
            trial_id=trial_id,
            message=message
        )


# get_any_trial_abci_output
@t_base()
def test_get_any_trial_abci_output():

    output = AbciOutput(db_path)

    trial_id = 0
    message = "hoge"

    output.set_any_trial_abci_output(
        trial_id=trial_id,
        message=message
    )

    assert output.get_any_trial_abci_output(trial_id=trial_id) == message

    # update
    output.set_any_trial_abci_output(
        trial_id=trial_id,
        message="update"
    )
    assert output.get_any_trial_abci_output(trial_id=trial_id) == "update"

    # nodata
    assert output.get_any_trial_abci_output(trial_id=1) is None
