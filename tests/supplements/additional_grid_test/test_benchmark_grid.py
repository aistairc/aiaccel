
from tests.supplements.additional_grid_test import AdditionalGridTest


class TestBenchmarkGrid(AdditionalGridTest):

    @classmethod
    def setup_class(cls):
        cls.search_algorithm = "grid"
