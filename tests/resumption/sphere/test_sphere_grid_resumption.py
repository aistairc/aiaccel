from tests.resumption.resumption_test import ResumptionTest
import aiaccel


class TestSphereGridResumption(ResumptionTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "grid"
