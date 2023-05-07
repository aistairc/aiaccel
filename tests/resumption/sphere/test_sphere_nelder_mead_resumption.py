from tests.resumption.resumption_test import ResumptionTest


class TestSphereNelderMeadResumption(ResumptionTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "nelder_mead"
