from tests.resumption.resumption_test import ResumptionTest
import aiaccel


class TestSphereTpeResumption(ResumptionTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = aiaccel.search_algorithm_tpe