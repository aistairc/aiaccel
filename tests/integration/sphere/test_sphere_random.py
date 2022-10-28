import aiaccel

from tests.integration.integration_test import IntegrationTest


class TestSphereRandom(IntegrationTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = 'random'
