import random
from collections import OrderedDict


class MnasNetStructureInfo:
    def __init__(self, keys):
        self.structure_info_dic = OrderedDict()

        for key in keys:
            self.structure_info_dic[key] = 0

    def __call__(self, key):
        return self.structure_info_dic[key]

    def update_values(self, values):
        for key, value in zip(self.structure_info_dic.keys(), values):
            self.structure_info_dic[key] = value

    def update_random_values(self, max_values):
        for key, max_value in zip(self.structure_info_dic.keys(), max_values):
            value = random.randint(0, max_value - 1)
            self.structure_info_dic[key] = value
