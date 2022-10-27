

from tests.newly_test.no_initial_test.integration_test import NoInitialIntegrationTest


class TestBenchmarkNelderMeadNoInitial(NoInitialIntegrationTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "nelder-mead"
