import aiaccel
from tests.resumption.resumption_test import ResumptionTest


class TestSphereMOTpeResumption(ResumptionTest):
    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "motpe"
