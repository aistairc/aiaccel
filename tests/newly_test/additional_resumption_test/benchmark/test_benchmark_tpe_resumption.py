# 概ねbenchmarkの結果が収束する trial number 40 で比較

from tests.newly_test.additional_resumption_test.resumption_test import AdditionalResumptionTest


class TestBenchmarkTpeResumption(AdditionalResumptionTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "tpe"
