from aiaccel.abci.abci_batch import create_abci_batch_file
from aiaccel.wrapper_tools import create_runner_command


def test_create_abci_batch_file(
    clean_work_dir,
    get_one_parameter,
    load_test_config,
    data_dir,
    work_dir
):
    config = load_test_config()
    dict_lock = work_dir.joinpath('lock')
    batch_file = work_dir.joinpath('runner', 'run_test.sh')
    commands = create_runner_command(
        config.job_command.get(),
        get_one_parameter(),
        'test',
        'config.json'
    )
    wrapper_file = data_dir.joinpath(config.job_script_preamble.get())
    create_abci_batch_file(batch_file, wrapper_file, commands, dict_lock)
    assert work_dir.joinpath('runner/run_test.sh').exists()
