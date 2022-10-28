# Comparison with trial number 20, where benchmark results generally converge.

from tests.newly_test.additional_resumption.additional_resumption_test import AdditionalResumptionTest


class TestBenchmarkNelderMeadResumption(AdditionalResumptionTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "nelder-mead"
