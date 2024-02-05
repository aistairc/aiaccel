import pytest

from aiaccel.nas.mnas_structure_info import MnasNetStructureInfo


def test_mnasnet_structure_info():
    keys = ["a", "b", "c"]
    structure_info = MnasNetStructureInfo(keys)

    for key in keys:
        assert structure_info(key) == 0

    values = [1, 2, 3]
    structure_info.update_values(values)

    for key, value in zip(keys, values):
        assert structure_info(key) == value

    max_values = [10, 20, 30]
    structure_info.update_random_values(max_values)

    for key, max_value in zip(keys, max_values):
        value = structure_info(key)
        assert 0 <= value < max_value
