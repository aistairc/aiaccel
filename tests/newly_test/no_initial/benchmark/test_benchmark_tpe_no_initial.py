

from tests.newly_test.no_initial.no_initial_test import NoInitialTest


class TestBenchmarkTpeNoInitial(NoInitialTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "tpe"
