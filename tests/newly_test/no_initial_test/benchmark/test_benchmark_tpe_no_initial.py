

from tests.newly_test.no_initial_test.integration_test import NoInitialIntegrationTest


class TestBenchmarkTpeNoInitial(NoInitialIntegrationTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "tpe"
