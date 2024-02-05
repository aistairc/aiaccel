import pytest

from aiaccel.nas.data_module.nas_dataloader import NAS1shotDataLoader


class MockNAS1shotDataLoader(NAS1shotDataLoader):
    pass


def test_nas1shotdataloader_methods():
    MockNAS1shotDataLoader.__abstractmethods__ = frozenset()
    loader = MockNAS1shotDataLoader()

    with pytest.raises(NotImplementedError):
        loader.get_supernet_train_dataloader()

    with pytest.raises(NotImplementedError):
        loader.get_architecture_search_dataloader()

    with pytest.raises(NotImplementedError):
        loader.get_num_supernet_train_data()

    with pytest.raises(NotImplementedError):
        loader.get_num_architecture_search_data()

    with pytest.raises(NotImplementedError):
        loader.get_dims()

    with pytest.raises(NotImplementedError):
        loader.get_num_classes()
