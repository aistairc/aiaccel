from torch.utils.data import Dataset

from pytest_mock.plugin import MockerFixture

from aiaccel.torch.datasets.scatter_dataset import scatter_dataset


def test_scatter_dataset(mocker: MockerFixture) -> None:
    class DummyDataset(Dataset[int]):
        def __len__(self) -> int:
            return 16

        def __getitem__(self, index: int) -> int:
            return index

    orig_dataset = DummyDataset()

    mocker.patch("torch.distributed.get_world_size", return_value=4)
    indices = []
    for rank in range(4):
        mocker.patch("torch.distributed.get_rank", return_value=rank)
        dataset = scatter_dataset(orig_dataset)

        assert len(dataset) == len(orig_dataset) // 4
        indices += list(dataset.indices)

    assert sorted(indices) == list(range(16))
