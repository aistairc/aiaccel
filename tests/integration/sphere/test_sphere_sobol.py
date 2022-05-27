from tests.integration.integration_test import IntegrationTest
import aiaccel


class TestSphereSobol(IntegrationTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = aiaccel.search_algorithm_sobol
