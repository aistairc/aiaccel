# Comparison with trial number 20, where benchmark results generally converge.

from tests.supplements.additional_resumption_test import AdditionalResumptionTest


class TestBenchmarkNelderMeadResumption(AdditionalResumptionTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "nelder-mead"
