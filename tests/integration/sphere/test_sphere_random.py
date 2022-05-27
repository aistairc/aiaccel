from tests.integration.integration_test import IntegrationTest
import aiaccel


class TestSphereRandom(IntegrationTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = aiaccel.search_algorithm_random
