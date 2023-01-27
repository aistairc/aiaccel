import aiaccel

from tests.integration.integration_test import IntegrationTest


class TestSphereMOTPE(IntegrationTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = 'motpe'
