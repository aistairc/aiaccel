from collections import defaultdict

from torch.utils.data import Dataset

from aiaccel.torch.datasets.cached_dataset import CachedDataset


def test_cached_dataset() -> None:
    class DummyDataset(Dataset[int]):
        def __init__(self) -> None:
            self.counter = defaultdict[int, int](lambda: 0)

        def __getitem__(self, index: int) -> int:
            self.counter[index] += 1
            return index

    orig_dataset = DummyDataset()
    dataset = CachedDataset(orig_dataset)

    for _ in range(2):
        for ii in range(5):
            assert dataset[ii] == ii

    assert all(count == 1 for count in orig_dataset.counter.values())
