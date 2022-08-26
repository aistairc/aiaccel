from tests.integration.integration_test import IntegrationTest
import aiaccel


class TestSphereGrid(IntegrationTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = 'grid'
