from pathlib import Path
import numpy as np
from aiaccel.storage.variable import Serializer


def test_serialize():
    file_name = Path('/tmp/storage.db')
    var = Serializer(file_name=file_name)
    var.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    var.d['hoge'].set(trial_id=1, value=1.1)
    var.d['hoge'].set(trial_id=2, value=2.2)
    var.d['hoge'].set(trial_id=3, value=3.3)

    assert var.d['hoge'].get(trial_id=1) == 1.1
    assert var.d['hoge'].get(trial_id=2) == 2.2
    assert var.d['hoge'].get(trial_id=3) == 3.3

    var.d['hoge'].set(trial_id=3, value=4.4)
    assert var.d['hoge'].get(trial_id=3) == 4.4

    var.register(process_name='optimizer', labels=['random_state'])
    rs = np.random.get_state()
    var.d['hoge'].set(trial_id=1, value=rs)
    var.d['hoge'].set(trial_id=1, value=rs)


    d = var.d['hoge'].get(trial_id=1)
    for i in range(len(d)):
        if type(d[i]) is np.ndarray:
            assert all(d[i] == rs[i])
        else:
            assert d[i] == rs[i]
