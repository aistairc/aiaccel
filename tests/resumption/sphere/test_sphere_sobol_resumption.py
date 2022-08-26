from tests.resumption.resumption_test import ResumptionTest
import aiaccel


class TestSphereSobolResumption(ResumptionTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = 'sobol'
