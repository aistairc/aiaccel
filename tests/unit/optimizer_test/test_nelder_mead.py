import copy
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from aiaccel.optimizer import NelderMead
from aiaccel.parameter import HyperParameterConfiguration
from aiaccel.storage import Storage
from aiaccel.workspace import Workspace
from aiaccel.optimizer.value import Value
from aiaccel.optimizer._nelder_mead import Simplex, Vertex
from tests.base_test import BaseTest


class TestVertex(unittest.TestCase):
    def setUp(self):
        self.xs = np.array([1.0, 2.0, 3.0])
        self.value = 10.0
        self.vertex = Vertex(self.xs, self.value)

    def test_generate_random_name(self):
        name = self.vertex.generate_random_name()
        self.assertEqual(len(name), 10)

    def test_coordinates(self):
        self.assertTrue(np.array_equal(self.vertex.coordinates, self.xs))

    def test_set_value(self):
        new_value = 20.0
        self.vertex.set_value(new_value)
        self.assertEqual(self.vertex.value, new_value)

    def test_set_id(self):
        new_id = "new_id"
        self.vertex.set_id(new_id)
        self.assertEqual(self.vertex.id, new_id)

    def test_set_new_id(self):
        old_id = self.vertex.id
        self.vertex.set_new_id()
        self.assertNotEqual(self.vertex.id, old_id)

    def test_set_xs(self):
        new_xs = np.array([4.0, 5.0, 6.0])
        self.vertex.set_xs(new_xs)
        self.assertTrue(np.array_equal(self.vertex.xs, new_xs))

    def test_update(self):
        new_xs = np.array([4.0, 5.0, 6.0])
        new_value = 20.0
        self.vertex.update(new_xs, new_value)
        self.assertTrue(np.array_equal(self.vertex.xs, new_xs))
        self.assertEqual(self.vertex.value, new_value)

    def test_add(self):
        other = Vertex(np.array([1.0, 1.0, 1.0]))
        new_vertex = self.vertex + other
        expected_xs = np.array([2.0, 3.0, 4.0])
        self.assertTrue(np.array_equal(new_vertex.xs, expected_xs))

    def test_subtract(self):
        other = Vertex(np.array([1.0, 1.0, 1.0]))
        new_vertex = self.vertex - other
        expected_xs = np.array([0.0, 1.0, 2.0])
        self.assertTrue(np.array_equal(new_vertex.xs, expected_xs))

    def test_multiply(self):
        factor = 2.0
        new_vertex = self.vertex * factor
        expected_xs = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.array_equal(new_vertex.xs, expected_xs))

    def test_equal(self):
        other = Vertex(np.array([1.0, 2.0, 3.0]), self.value)
        self.assertTrue(self.vertex == other)
        self.assertFalse(self.vertex == 5.0)

    def test_not_equal(self):
        other = Vertex(np.array([1.0, 2.0, 3.0]), self.value)
        self.assertFalse(self.vertex != other)
        self.assertTrue(self.vertex != 5.0)

    def test_less_than(self):
        other = Vertex(np.array([1.0, 2.0, 3.0]), self.value + 1.0)
        self.assertTrue(self.vertex < other)
        self.assertTrue(self.vertex < 15.0)
        other = Vertex(np.array([1.0, 2.0, 3.0]), self.value - 1.0)
        self.assertFalse(self.vertex < other)
        self.assertFalse(self.vertex <other.value)

    def test_less_than_or_equal(self):
        other = Vertex(np.array([1.0, 2.0, 3.0]), self.value + 1.0)
        self.assertTrue(self.vertex <= other)
        self.assertTrue(self.vertex <= 10.0)
        self.assertTrue(self.vertex <= self.value)
        other = Vertex(np.array([1.0, 2.0, 3.0]), self.value - 1.0)
        self.assertFalse(self.vertex <= other)
        self.assertFalse(self.vertex <= other.value)

    def test_greater_than(self):
        other = Vertex(np.array([1.0, 2.0, 3.0]), self.value - 1.0)
        self.assertTrue(self.vertex > other)
        self.assertTrue(self.vertex > self.value - 1.0)
        other = Vertex(np.array([1.0, 2.0, 3.0]), self.value + 1.0)
        self.assertFalse(self.vertex > other)
        self.assertFalse(self.vertex > other.value)

    def test_greater_than_or_equal(self):
        other = Vertex(np.array([1.0, 2.0, 3.0]), self.value - 1.0)
        self.assertTrue(self.vertex >= other)
        self.assertTrue(self.vertex >= self.value)
        other = Vertex(np.array([1.0, 2.0, 3.0]), self.value + 1.0)
        self.assertFalse(self.vertex >= other)
        self.assertFalse(self.vertex >= other.value)


def test_simplex():
    # Test initialization
    simplex_coordinates = np.array([[1, 2], [3, 4], [5, 6]])
    simplex = Simplex(simplex_coordinates)
    assert simplex.n_dim == 2
    assert len(simplex.vertices) == 3
    assert isinstance(simplex.centroid, Vertex)

    # Test get_simplex_coordinates
    assert np.array_equal(simplex.get_simplex_coordinates(), simplex_coordinates)

    # Test set_value
    vertex_id = simplex.vertices[0].id
    assert simplex.set_value(vertex_id, 10)
    assert simplex.vertices[0].value == 10
    assert not simplex.set_value("invalid_id", 20)

    # Test order_by
    simplex.vertices[0].set_value(5)
    simplex.vertices[1].set_value(3)
    simplex.vertices[2].set_value(7)
    simplex.order_by()
    assert simplex.vertices[0].value == 3
    assert simplex.vertices[1].value == 5
    assert simplex.vertices[2].value == 7

    # Test calc_centroid
    simplex.calc_centroid()
    assert np.array_equal(simplex.centroid.xs, np.array([2, 3]))

    # Test reflect
    simplex.coef = {"r": 1}
    xr = simplex.reflect()
    assert np.array_equal(xr.xs, np.array([-1, 0]))

    # Test expand
    simplex.coef = {"e": 2}
    xe = simplex.expand()
    assert np.array_equal(xe.xs, np.array([-4, -3]))

    # Test inside_contract
    simplex.coef = {"ic": 0.5}
    xic = simplex.inside_contract()
    assert np.array_equal(xic.xs, np.array([0.5, 1.5]))

    # Test outside_contract
    simplex.coef = {"oc": 0.5}
    xoc = simplex.outside_contract()
    assert np.array_equal(xoc.xs, np.array([0.5, 1.5]))

    # Test shrink
    simplex.coef = {"s": 0.5}
    shrunk_vertices = simplex.shrink()
    assert np.array_equal(shrunk_vertices[0].xs, np.array([3., 4.]))
    assert np.array_equal(shrunk_vertices[1].xs, np.array([2., 3.]))
    assert np.array_equal(shrunk_vertices[2].xs, np.array([4., 5.]))



@pytest.fixture
def nelder_mead():
    initial_parameters = np.array([[1, 2], [3, 4], [5, 6]])
    return NelderMead(initial_parameters)


def test_get_n_waits(nelder_mead):
    assert nelder_mead.get_n_waits() == 3


def test_get_n_dim(nelder_mead):
    assert nelder_mead.get_n_dim() == 2


def test_get_state(nelder_mead):
    assert nelder_mead.get_state() == "initialize"


def test_change_state(nelder_mead):
    nelder_mead.change_state("reflect")
    assert nelder_mead.get_state() == "reflect"


def test_set_value(nelder_mead):
    id = nelder_mead.simplex.vertices[0].id
    nelder_mead.set_value(id, 10)
    assert nelder_mead.simplex.vertices[0].value == 10


def test_initialize(nelder_mead):
    vertices = nelder_mead.initialize()
    assert len(vertices) == 3
    assert vertices[0].coordinates.tolist() == [1, 2]



def test_expand(nelder_mead):
    vertex = nelder_mead.expand()
    assert vertex.coordinates.tolist() == [-10.0, -12.0]


def test_inside_contract(nelder_mead):
    vertex = nelder_mead.inside_contract()
    assert vertex.coordinates.tolist() == [2.5, 3.0]


def test_outside_contract(nelder_mead):
    vertex = nelder_mead.outside_contract()
    assert vertex.coordinates.tolist() == [-2.5, -3.0]


def test_shrink(nelder_mead):
    vertices = nelder_mead.shrink()
    assert len(vertices) == 3
    assert vertices[0].coordinates.tolist() == [1, 2]


def test_search(nelder_mead):
    vertices = nelder_mead.search()
    assert len(vertices) == 3
    assert vertices[0].coordinates.tolist() == [1, 2]
