from tests.integration.integration_test import IntegrationTest


class TestSphereSobol(IntegrationTest):
    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "sobol_int"
