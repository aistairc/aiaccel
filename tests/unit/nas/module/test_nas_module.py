import pytest

from aiaccel.nas.module.nas_module import NASModule


class MockNASModule(NASModule):
    def __init__(self, name):
        super().__init__(name)


def test_nasmodule_methods():
    MockNASModule.__abstractmethods__ = frozenset()
    module = MockNASModule(name="Mock")

    assert module.name == "Mock"
    assert module.active_op is None
    assert module.forward(X=None) is None

    with pytest.raises(NotImplementedError):
        module.build(config=None)

    with pytest.raises(NotImplementedError):
        module.select_active_op(structure_info=None)

    with pytest.raises(NotImplementedError):
        module.enumerate_categorical_variables()

    with pytest.raises(NotImplementedError):
        module.fix_arc()

    with pytest.raises(NotImplementedError):
        module.print_active_op(log_dir=None)

    with pytest.raises(NotImplementedError):
        module.get_param_num_list()
