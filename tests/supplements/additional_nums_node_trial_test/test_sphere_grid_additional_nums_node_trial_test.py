from tests.supplements.additional_nums_node_trial_test import AdditionalNumsNodeTrialTest


class TestSphereGrid(AdditionalNumsNodeTrialTest):
    @classmethod
    def setup_class(cls):
        cls.search_algorithm = 'grid'
        cls.python_program = 'user.py'
