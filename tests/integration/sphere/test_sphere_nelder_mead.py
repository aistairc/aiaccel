from tests.integration.integration_test import IntegrationTest
import aiaccel


class TestSphereNelderMead(IntegrationTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "nelder-mead"

