

from tests.newly_test.no_initial.no_initial_test import NoInitialTest


class TestBenchmarkNelderMeadNoInitial(NoInitialTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "nelder-mead"
