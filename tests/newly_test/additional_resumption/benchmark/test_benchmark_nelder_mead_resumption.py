# 概ねbenchmarkの結果が収束する trial number 20 で比較

from tests.newly_test.additional_resumption.additional_resumption_test import AdditionalResumptionTest


class TestBenchmarkNelderMeadResumption(AdditionalResumptionTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "nelder-mead"
