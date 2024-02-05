import pytest
from torch.utils.data.sampler import SubsetRandomSampler

from aiaccel.nas.data_module.subset_random_sampling_dataloader import SubsetRandomSamplingDataLoader


def test_create_sampler(mocker):
    SubsetRandomSamplingDataLoader.__abstractmethods__ = frozenset()
    loader = SubsetRandomSamplingDataLoader()
    loader._supernet_train_data_indices = [0, 1, 2]
    loader._architecture_search_data_indices = [3, 4, 5]

    loader._create_sampler()
    assert isinstance(loader._sampler_supernet_train, SubsetRandomSampler)
    assert isinstance(loader._sampler_architecture_search, SubsetRandomSampler)
