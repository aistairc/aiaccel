from pathlib import Path
from os import environ


def get_root():
    return Path(environ['GITHUB_WORKSPACE'])/'mpi_work'


def get_rank_log():
    return get_root()/'work/experimental/mpi/rank_log'


def test_logf():
    s = (get_root()/'logf').read_text()
    assert 'Scheduler INFO     1/1, finished, ready: 0, running: 0' in s
    assert 'value : 40.076' in s

def test_rank_log_0_csv():
    s = (get_rank_log()/'0.csv').read_text()
    assert ',"prepare: rank=0 tag=0",' in s
    assert ',"submit start: recv: tag=1 trial_id=0 list=[1,' in s


def test_rank_log_1_csv():
    s = (get_rank_log()/'1.csv').read_text()
    assert ',"_func_sub(): tag=1 command=[' in s
    assert ',"_func_sub(): debug: line=40.076' in s
