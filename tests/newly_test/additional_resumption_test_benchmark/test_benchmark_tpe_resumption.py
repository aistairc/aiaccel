# Comparison with trial number 40, where benchmark results generally converge.

from tests.newly_test.additional_resumption_test import AdditionalResumptionTest


class TestBenchmarkTpeResumption(AdditionalResumptionTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "tpe"
