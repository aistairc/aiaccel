import aiaccel

from tests.resumption.resumption_test import ResumptionTest


class TestSphereSobolResumption(ResumptionTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "sobol"
