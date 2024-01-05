import unittest

import numpy as np
import optuna

from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadSampler, NelderMeadState, Simplex, Vertex


class TestVertex(unittest.TestCase):
    def setUp(self):
        self.xs = np.array([1.0, 2.0, 3.0])
        self.value = 10.0
        self.vertex = Vertex(self.xs, self.value)

    def test_coordinates(self):
        self.assertTrue(np.array_equal(self.vertex.coordinates, self.xs))

    def test_set_value(self):
        new_value = 20.0
        self.vertex.set_value(new_value)
        self.assertEqual(self.vertex.value, new_value)

    def test_update(self):
        new_xs = np.array([4.0, 5.0, 6.0])
        new_value = 20.0
        self.vertex.update(new_xs, new_value)
        self.assertTrue(np.array_equal(self.vertex.xs, new_xs))
        self.assertEqual(self.vertex.value, new_value)

    def test_add(self):
        # add Vertex
        other = Vertex(np.array([1.0, 1.0, 1.0]))
        new_vertex = self.vertex + other
        expected_xs = np.array([2.0, 3.0, 4.0])
        self.assertTrue(np.array_equal(new_vertex.xs, expected_xs))

        # add np.array
        other = np.array([1.0, 1.0, 1.0])
        new_vertex = self.vertex + other
        expected_xs = np.array([2.0, 3.0, 4.0])
        self.assertTrue(np.array_equal(new_vertex.xs, expected_xs))

        # add string (TypeError)
        other = "test"
        with self.assertRaises(TypeError):
            new_vertex = self.vertex + other

    def test_subtract(self):
        # add Vertex
        other = Vertex(np.array([1.0, 1.0, 1.0]))
        new_vertex = self.vertex - other
        expected_xs = np.array([0.0, 1.0, 2.0])
        self.assertTrue(np.array_equal(new_vertex.xs, expected_xs))

        # add np.array
        other = np.array([1.0, 1.0, 1.0])
        new_vertex = self.vertex - other
        expected_xs = np.array([0.0, 1.0, 2.0])
        self.assertTrue(np.array_equal(new_vertex.xs, expected_xs))

        # add string (TypeError)
        other = "test"
        with self.assertRaises(TypeError):
            new_vertex = self.vertex - other

    def test_multiply(self):
        factor = 2.0
        new_vertex = self.vertex * factor
        expected_xs = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.array_equal(new_vertex.xs, expected_xs))

    def test_equal(self):
        # True
        other = Vertex(np.array([1.0, 2.0, 3.0]), 10.0)
        self.assertTrue(self.vertex == other)
        self.assertTrue(self.vertex == 10.0)

        # False
        other = Vertex(np.array([1.0, 2.0, 3.0]), 5.0)
        self.assertFalse(self.vertex == other)
        self.assertFalse(self.vertex == 5.0)

    def test_not_equal(self):
        # True
        other = Vertex(np.array([1.0, 2.0, 3.0]), 5.0)
        self.assertTrue(self.vertex != other)
        self.assertTrue(self.vertex != 5.0)

        # False
        other = Vertex(np.array([1.0, 2.0, 3.0]), 10.0)
        self.assertFalse(self.vertex != other)
        self.assertFalse(self.vertex != 10.0)

    def test_less_than(self):
        # True
        other = Vertex(np.array([1.0, 2.0, 3.0]), 15.0)
        self.assertTrue(self.vertex < other)
        self.assertTrue(self.vertex < 15.0)

        # False (eq)
        other = Vertex(np.array([1.0, 2.0, 3.0]), 10.0)
        self.assertFalse(self.vertex < other)
        self.assertFalse(self.vertex < 10.0)

        # False
        other = Vertex(np.array([1.0, 2.0, 3.0]), 5.0)
        self.assertFalse(self.vertex < other)
        self.assertFalse(self.vertex < 5.0)

        # TypeError
        other = "test"
        with self.assertRaises(TypeError):
            self.vertex < other

    def test_less_than_or_equal(self):
        # True
        other = Vertex(np.array([1.0, 2.0, 3.0]), 15.0)
        self.assertTrue(self.vertex <= other)
        self.assertTrue(self.vertex <= 15.0)

        # True (eq)
        other = Vertex(np.array([1.0, 2.0, 3.0]), 10.0)
        self.assertTrue(self.vertex <= other)
        self.assertTrue(self.vertex <= 10.0)

        # False
        other = Vertex(np.array([1.0, 2.0, 3.0]), 5.0)
        self.assertFalse(self.vertex <= other)
        self.assertFalse(self.vertex <= 5.0)

        # TypeError
        other = "test"
        with self.assertRaises(TypeError):
            self.vertex <= other

    def test_greater_than(self):
        # True
        other = Vertex(np.array([1.0, 2.0, 3.0]), 5.0)
        self.assertTrue(self.vertex > other)
        self.assertTrue(self.vertex > 5.0)

        # False (eq)
        other = Vertex(np.array([1.0, 2.0, 3.0]), 10.0)
        self.assertFalse(self.vertex > other)
        self.assertFalse(self.vertex > 10.0)

        # False
        other = Vertex(np.array([1.0, 2.0, 3.0]), 15.0)
        self.assertFalse(self.vertex > other)
        self.assertFalse(self.vertex > 15.0)

        # TypeError
        other = "test"
        with self.assertRaises(TypeError):
            self.vertex > other

    def test_greater_than_or_equal(self):
        # True
        other = Vertex(np.array([1.0, 2.0, 3.0]), 5.0)
        self.assertTrue(self.vertex >= other)
        self.assertTrue(self.vertex >= 5.0)

        # True (eq)
        other = Vertex(np.array([1.0, 2.0, 3.0]), 10.0)
        self.assertTrue(self.vertex >= other)
        self.assertTrue(self.vertex >= 10.0)

        # False
        other = Vertex(np.array([1.0, 2.0, 3.0]), 15.0)
        self.assertFalse(self.vertex >= other)
        self.assertFalse(self.vertex >= 15.0)

        # TypeError
        other = "test"
        with self.assertRaises(TypeError):
            self.vertex >= other


class TestSimplex(unittest.TestCase):
    def setUp(self):
        # simplex_coordinates = np.array([[1, 2], [3, 4], [5, 6]])
        self.simplex = Simplex()
        self.vertices = [
            Vertex(np.array([1.0, 2.0])),
            Vertex(np.array([3.0, 4.0]))
        ]

    def test_add_vertices(self):
        self.simplex.add_vertices(self.vertices[0])
        self.assertEqual(self.simplex.vertices, [Vertex(np.array([1.0, 2.0]))])

        self.simplex.add_vertices(self.vertices[1])
        self.assertEqual(self.simplex.vertices, [Vertex(np.array([1.0, 2.0])), Vertex(np.array([3.0, 4.0]))])

    def test_num_of_vertices(self):
        self.simplex.add_vertices(self.vertices[0])
        self.assertEqual(self.simplex.num_of_vertices(), 1)

        self.simplex.add_vertices(self.vertices[1])
        self.assertEqual(self.simplex.num_of_vertices(), 2)

    def test_get_simplex_coordinates(self):
        self.simplex.add_vertices(self.vertices[0])
        self.assertTrue(np.array_equal(self.simplex.get_simplex_coordinates(), np.array([[1.0, 2.0]])))

        self.simplex.add_vertices(self.vertices[1])
        self.assertTrue(np.array_equal(self.simplex.get_simplex_coordinates(), np.array([[1.0, 2.0], [3.0, 4.0]])))


class TestSimplexOperation(unittest.TestCase):
    def setUp(self):
        self.simplex = Simplex()
        self.vertices = [
            Vertex(np.array([1.0, 2.0]), 5),
            Vertex(np.array([3.0, 4.0]), 3),
            Vertex(np.array([5.0, 6.0]), 7)
        ]
        for vertex in self.vertices:
            self.simplex.add_vertices(vertex)

    def test_order_by(self):
        self.simplex.order_by()

        self.assertEqual(self.simplex.vertices[0], Vertex(np.array([3.0, 4.0]), 3))
        self.assertEqual(self.simplex.vertices[1], Vertex(np.array([1.0, 2.0]), 5))
        self.assertEqual(self.simplex.vertices[2], Vertex(np.array([5.0, 6.0]), 7))

    def test_calc_centroid(self):
        self.simplex.calc_centroid()
        self.assertTrue(np.array_equal(self.simplex.centroid.xs, np.array([2, 3])))

    def test_reflect(self):
        self.simplex.calc_centroid()
        xr = self.simplex.reflect()
        self.assertTrue(np.array_equal(xr.xs, np.array([-1.0, 0.0])))

    def test_expand(self):
        self.simplex.calc_centroid()
        xe = self.simplex.expand()
        self.assertTrue(np.array_equal(xe.xs, np.array([-4.0, -3.0])))

    def test_inside_contract(self):
        self.simplex.calc_centroid()
        xic = self.simplex.inside_contract()
        self.assertTrue(np.array_equal(xic.xs, np.array([3.5, 4.5])))

    def test_outside_contract(self):
        self.simplex.calc_centroid()
        xoc = self.simplex.outside_contract()
        self.assertTrue(np.array_equal(xoc.xs, np.array([0.5, 1.5])))

    def test_shrink(self):
        self.simplex.calc_centroid()
        xsh = self.simplex.shrink()
        self.assertTrue(np.array_equal(xsh[0].xs, np.array([3.0, 4.0])))
        self.assertTrue(np.array_equal(xsh[1].xs, np.array([2.0, 3.0])))
        self.assertTrue(np.array_equal(xsh[2].xs, np.array([4.0, 5.0])))


class TestNelderMeadState(unittest.TestCase):
    def setUp(self):
        self.state = NelderMeadState()

    def test_get_state(self):
        self.assertEqual(self.state.get_state(), "initial")

    def test_reflect(self):
        self.state.reflect()
        self.assertEqual(self.state.get_state(), "reflect")

    def test_expand(self):
        self.state.expand()
        self.assertEqual(self.state.get_state(), "expand")

    def test_inside_contract(self):
        self.state.inside_contract()
        self.assertEqual(self.state.get_state(), "inside_contract")

    def test_outside_contract(self):
        self.state.outside_contract()
        self.assertEqual(self.state.get_state(), "outside_contract")

    def test_shrink(self):
        self.state.shrink()
        self.assertEqual(self.state.get_state(), "shrink")


class TestNelderMeadSamplerOperation(unittest.TestCase):
    def setUp(self):
        search_space = {"x": [-5, 5], "y": [-5, 5]}
        self.sampler = NelderMeadSampler(search_space=search_space, seed=42)
        vertices = [
            Vertex(np.array([1.0, 2.0]), 5),
            Vertex(np.array([3.0, 4.0]), 3),
            Vertex(np.array([5.0, 6.0]), 7)
        ]
        for vertex in vertices:
            self.sampler.simplex.add_vertices(vertex)
        self.sampler.simplex.calc_centroid()

    def test_after_initialize(self):
        self.sampler.after_initialize()

        self.assertEqual(self.sampler.state.get_state(), "reflect")

    def test_reflect(self):
        xr = self.sampler.reflect()

        self.assertTrue(np.array_equal(xr.xs, np.array([-1.0, 0.0])))

    def test_after_reflect_to_reflect(self):
        # self.simplex.vertices[0] <= self.store.r < self.simplex.vertices[-2]
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(4.0)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([-1.0, 0.0])))
        self.assertEqual(vertex.value, 4.0)

    def test_after_reflect_to_expand(self):
        # self.store.r < self.simplex.vertices[0]
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(2.0)

        self.assertEqual(self.sampler.state.get_state(), "expand")

    def test_after_reflect_to_outside_contract(self):
        # self.simplex.vertices[-2] <= self.store.r < self.simplex.vertices[-1]
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(6.0)

        self.assertEqual(self.sampler.state.get_state(), "outside_contract")

    def test_after_reflect_to_inside_contract(self):
        # self.simplex.vertices[-1] <= self.store.r
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(8.0)

        self.assertEqual(self.sampler.state.get_state(), "inside_contract")

    def test_expand(self):
        xe = self.sampler.expand()

        self.assertTrue(np.array_equal(xe.xs, np.array([-4.0, -3.0])))

    def test_after_expand_less_than_r(self):
        # self.store.e < self.store.r:
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(2.0)
        self.sampler.store.e = self.sampler.expand()
        self.sampler.after_expand(1.0)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([-4.0, -3.0])))
        self.assertEqual(vertex.value, 1.0)

    def test_after_expand_more_than_r(self):
        # else (self.store.r <= self.store.e):
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(2.0)
        self.sampler.store.e = self.sampler.expand()
        self.sampler.after_expand(3.0)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([-1.0, 0.0])))
        self.assertEqual(vertex.value, 2.0)

    def test_inside_contract(self):
        xic = self.sampler.inside_contract()

        self.assertTrue(np.array_equal(xic.xs, np.array([3.5, 4.5])))

    def test_after_inside_contract_to_reflect(self):
        # self.store.ic < self.simplex.vertices[-1]
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(8.0)
        self.sampler.store.ic = self.sampler.inside_contract()
        self.sampler.after_inside_contract(6.0)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([3.5, 4.5])))
        self.assertEqual(vertex.value, 6.0)

    def test_after_inside_contract_to_shrink(self):
        # else (self.simplex.vertices[-1] <= self.store.ic)
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(8.0)
        self.sampler.store.ic = self.sampler.inside_contract()
        self.sampler.after_inside_contract(8.0)

        self.assertEqual(self.sampler.state.get_state(), "shrink")

    def test_outside_contract(self):
        xoc = self.sampler.outside_contract()
        self.assertTrue(np.array_equal(xoc.xs, np.array([0.5, 1.5])))

    def test_after_outside_contract_to_reflect(self):
        # self.store.oc <= self.store.r
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(6.0)
        self.sampler.store.oc = self.sampler.outside_contract()
        self.sampler.after_outside_contract(5.0)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([0.5, 1.5])))
        self.assertEqual(vertex.value, 5.0)

    def test_after_outside_contract_to_shrink(self):
        # else (self.store.r < self.store.oc)
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(6.0)
        self.sampler.store.oc = self.sampler.outside_contract()
        self.sampler.after_outside_contract(7.0)

        self.assertEqual(self.sampler.state.get_state(), "shrink")

    def test_shrink(self):
        xsh = self.sampler.shrink()
        self.assertTrue(np.array_equal(xsh[0].xs, np.array([3.0, 4.0])))
        self.assertTrue(np.array_equal(xsh[1].xs, np.array([2.0, 3.0])))
        self.assertTrue(np.array_equal(xsh[2].xs, np.array([4.0, 5.0])))
        vertex = self.sampler.simplex.vertices[0]
        self.assertTrue(np.array_equal(vertex.xs, np.array([3.0, 4.0])))
        self.assertEqual(vertex.value, 3.0)

    def test_aftter_shrink(self):
        self.sampler.after_shrink()

        self.assertEqual(self.sampler.state.get_state(), "reflect")

    def test_is_within_range(self):
        # True
        coordinates = np.array([3.0, 4.0])
        self.assertTrue(self.sampler.is_within_range(coordinates))

        # False
        coordinates = np.array([-6.0, 4.0])
        self.assertFalse(self.sampler.is_within_range(coordinates))
        coordinates = np.array([6.0, 4.0])
        self.assertFalse(self.sampler.is_within_range(coordinates))
        coordinates = np.array([3.0, -6.0])
        self.assertFalse(self.sampler.is_within_range(coordinates))
        coordinates = np.array([3.0, 6.0])
        self.assertFalse(self.sampler.is_within_range(coordinates))

    def test_get_next_coordinates_reflect(self):
        self.sampler.state.reflect()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, np.array([-1.0, 0.0])))

    def test_get_next_coordinates_expand(self):
        self.sampler.state.expand()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, np.array([-4.0, -3.0])))

    def test_get_next_coordinates_inside_contract(self):
        self.sampler.state.inside_contract()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, np.array([3.5, 4.5])))

    def test_get_next_coordinates_outside_contract(self):
        self.sampler.state.outside_contract()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, np.array([0.5, 1.5])))

    def test_get_next_coordinates_shrink(self):
        self.sampler.state.shrink()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, np.array([2.0, 3.0])))
        self.assertTrue(np.array_equal(self.sampler.xs[0], np.array([2.0, 3.0])))
        self.assertTrue(np.array_equal(self.sampler.xs[1], np.array([4.0, 5.0])))

    def test_get_next_coordinates_reflect_out_of_range(self):
        self.sampler._search_space["x"] = [0.0, 5.0]
        self.sampler.state.reflect()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, np.array([3.5, 4.5])))

    def test_set_objective_initial(self):
        self.sampler.simplex = Simplex()

        self.sampler.set_objective(np.array([1.0, 2.0]), 5.0)
        self.assertEqual(self.sampler.state.get_state(), "initial")
        vertex = self.sampler.simplex.vertices[0]
        self.assertTrue(np.array_equal(vertex.xs, np.array([1.0, 2.0])))
        self.assertEqual(vertex.value, 5.0)

        self.sampler.set_objective(np.array([3.0, 4.0]), 3.0)
        self.assertEqual(self.sampler.state.get_state(), "initial")
        vertex = self.sampler.simplex.vertices[1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([3.0, 4.0])))
        self.assertEqual(vertex.value, 3.0)

        self.sampler.set_objective(np.array([5.0, 6.0]), 7.0)
        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[2]
        self.assertTrue(np.array_equal(vertex.xs, np.array([5.0, 6.0])))
        self.assertEqual(vertex.value, 7.0)

    def test_set_objective_reflect(self):
        self.sampler.state.reflect()
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.set_objective(np.array([-1.0, 0.0]), 4.0)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([-1.0, 0.0])))
        self.assertEqual(vertex.value, 4.0)

    def test_set_objective_expand(self):
        self.sampler.state.expand()
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(2.0)
        self.sampler.store.e = self.sampler.expand()
        self.sampler.set_objective(np.array([-4.0, -3.0]), 1.0)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([-4.0, -3.0])))
        self.assertEqual(vertex.value, 1.0)

    def test_set_objective_inside_contract(self):
        self.sampler.state.inside_contract()
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(8.0)
        self.sampler.store.ic = self.sampler.inside_contract()
        self.sampler.set_objective(np.array([3.5, 4.5]), 6.0)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([3.5, 4.5])))
        self.assertEqual(vertex.value, 6.0)

    def test_set_objective_outside_contract(self):
        self.sampler.state.outside_contract()
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(6.0)
        self.sampler.store.oc = self.sampler.outside_contract()
        self.sampler.set_objective(np.array([0.5, 1.5]), 5.0)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([0.5, 1.5])))
        self.assertEqual(vertex.value, 5.0)

    def test_set_objective_shrink(self):
        self.sampler.state.shrink()
        self.sampler.xs = [v.coordinates for v in self.sampler.shrink()[1:]]

        self.sampler.set_objective(np.array([2.0, 3.0]), 4.0)
        self.assertEqual(self.sampler.state.get_state(), "shrink")
        vertex = self.sampler.simplex.vertices[1]
        self.assertTrue(np.array_equal(vertex.xs, np.array([2.0, 3.0])))
        self.assertEqual(vertex.value, 4.0)

        self.sampler.set_objective(np.array([4.0, 5.0]), 6.0)
        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[2]
        self.assertTrue(np.array_equal(vertex.xs, np.array([4.0, 5.0])))
        self.assertEqual(vertex.value, 6.0)
