import unittest

import numpy as np

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
        self.simplex = Simplex()
        self.vertices = [
            Vertex(np.array([1.0, 2.0])),
            Vertex(np.array([3.0, 4.0]))
        ]

    def test_add_vertices(self):
        self.simplex.add_vertices(self.vertices[0])
        self.assertEqual(self.simplex.vertices, self.vertices[:1])

        self.simplex.add_vertices(self.vertices[1])
        self.assertEqual(self.simplex.vertices, self.vertices)

    def test_num_of_vertices(self):
        self.simplex.add_vertices(self.vertices[0])
        self.assertEqual(self.simplex.num_of_vertices(), 1)

        self.simplex.add_vertices(self.vertices[1])
        self.assertEqual(self.simplex.num_of_vertices(), 2)

    def test_get_simplex_coordinates(self):
        coordinates = np.array([v.coordinates for v in self.vertices])
        self.simplex.add_vertices(self.vertices[0])
        self.assertTrue(np.array_equal(self.simplex.get_simplex_coordinates(), coordinates[:1]))

        self.simplex.add_vertices(self.vertices[1])
        self.assertTrue(np.array_equal(self.simplex.get_simplex_coordinates(), coordinates))


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

        self.assertEqual(self.simplex.vertices[0], self.vertices[1])
        self.assertEqual(self.simplex.vertices[1], self.vertices[0])
        self.assertEqual(self.simplex.vertices[2], self.vertices[2])

    def test_calc_centroid(self):
        centroid_xs = np.array([2, 3])

        self.simplex.calc_centroid()
        self.assertTrue(np.array_equal(self.simplex.centroid.xs, centroid_xs))

    def test_reflect(self):
        reflect_xs = np.array([-1.0, 0.0])

        self.simplex.calc_centroid()
        xr = self.simplex.reflect()
        self.assertTrue(np.array_equal(xr.xs, reflect_xs))

    def test_expand(self):
        expand_xs = np.array([-4.0, -3.0])

        self.simplex.calc_centroid()
        xe = self.simplex.expand()
        self.assertTrue(np.array_equal(xe.xs, expand_xs))

    def test_inside_contract(self):
        inside_contract_xs = np.array([3.5, 4.5])

        self.simplex.calc_centroid()
        xic = self.simplex.inside_contract()
        self.assertTrue(np.array_equal(xic.xs, inside_contract_xs))

    def test_outside_contract(self):
        outside_contract_xs = np.array([0.5, 1.5])

        self.simplex.calc_centroid()
        xoc = self.simplex.outside_contract()
        self.assertTrue(np.array_equal(xoc.xs, outside_contract_xs))

    def test_shrink(self):
        shrink_xs = [
            np.array([3.0, 4.0]),
            np.array([2.0, 3.0]),
            np.array([4.0, 5.0])
        ]

        self.simplex.calc_centroid()
        xsh = self.simplex.shrink()
        self.assertTrue(np.array_equal(xsh[0].xs, shrink_xs[0]))
        self.assertTrue(np.array_equal(xsh[1].xs, shrink_xs[1]))
        self.assertTrue(np.array_equal(xsh[2].xs, shrink_xs[2]))


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
        reflect_xs = np.array([-1.0, 0.0])
        xr = self.sampler.reflect()

        self.assertTrue(np.array_equal(xr.xs, reflect_xs))

    def test_after_reflect_to_reflect(self):
        # self.simplex.vertices[0] <= self.store.r < self.simplex.vertices[-2]
        reflect_xs = np.array([-1.0, 0.0])
        objective = 4.0

        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(objective)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, reflect_xs))
        self.assertEqual(vertex.value, objective)

    def test_after_reflect_to_expand(self):
        # self.store.r < self.simplex.vertices[0]
        objective = 2.0

        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(objective)

        self.assertEqual(self.sampler.state.get_state(), "expand")

    def test_after_reflect_to_outside_contract(self):
        # self.simplex.vertices[-2] <= self.store.r < self.simplex.vertices[-1]
        objective = 6.0

        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(objective)

        self.assertEqual(self.sampler.state.get_state(), "outside_contract")

    def test_after_reflect_to_inside_contract(self):
        # self.simplex.vertices[-1] <= self.store.r
        objective = 8.0

        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(objective)

        self.assertEqual(self.sampler.state.get_state(), "inside_contract")

    def test_expand(self):
        expand_xs = np.array([-4.0, -3.0])

        xe = self.sampler.expand()

        self.assertTrue(np.array_equal(xe.xs, expand_xs))

    def test_after_expand_less_than_r(self):
        # self.store.e < self.store.r:
        reflect_value = 2.0
        expand_value = 1.0
        expand_xs = np.array([-4.0, -3.0])

        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(reflect_value)
        self.sampler.store.e = self.sampler.expand()
        self.sampler.after_expand(expand_value)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, expand_xs))
        self.assertEqual(vertex.value, expand_value)

    def test_after_expand_more_than_r(self):
        # else (self.store.r <= self.store.e):
        reflect_value = 2.0
        expand_value = 3.0
        reflect_xs = np.array([-1.0, 0.0])

        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(reflect_value)
        self.sampler.store.e = self.sampler.expand()
        self.sampler.after_expand(expand_value)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, reflect_xs))
        self.assertEqual(vertex.value, reflect_value)

    def test_inside_contract(self):
        inside_contract_xs = np.array([3.5, 4.5])

        xic = self.sampler.inside_contract()

        self.assertTrue(np.array_equal(xic.xs, inside_contract_xs))

    def test_after_inside_contract_to_reflect(self):
        # self.store.ic < self.simplex.vertices[-1]
        reflect_value = 8.0
        inside_contract_value = 6.0
        inside_contract_xs = np.array([3.5, 4.5])

        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(reflect_value)
        self.sampler.store.ic = self.sampler.inside_contract()
        self.sampler.after_inside_contract(inside_contract_value)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, inside_contract_xs))
        self.assertEqual(vertex.value, inside_contract_value)

    def test_after_inside_contract_to_shrink(self):
        # else (self.simplex.vertices[-1] <= self.store.ic)
        reflect_value = 8.0
        inside_contract_value = 8.0

        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(reflect_value)
        self.sampler.store.ic = self.sampler.inside_contract()
        self.sampler.after_inside_contract(inside_contract_value)

        self.assertEqual(self.sampler.state.get_state(), "shrink")

    def test_outside_contract(self):
        outside_contract_xs = np.array([0.5, 1.5])

        xoc = self.sampler.outside_contract()
        self.assertTrue(np.array_equal(xoc.xs, outside_contract_xs))

    def test_after_outside_contract_to_reflect(self):
        # self.store.oc <= self.store.r
        reflect_value = 6.0
        outside_contract_value = 5.0
        outside_contract_xs = np.array([0.5, 1.5])

        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(reflect_value)
        self.sampler.store.oc = self.sampler.outside_contract()
        self.sampler.after_outside_contract(outside_contract_value)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, outside_contract_xs))
        self.assertEqual(vertex.value, outside_contract_value)

    def test_after_outside_contract_to_shrink(self):
        # else (self.store.r < self.store.oc)
        reflect_value = 6.0
        outside_contract_value = 7.0

        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(reflect_value)
        self.sampler.store.oc = self.sampler.outside_contract()
        self.sampler.after_outside_contract(outside_contract_value)

        self.assertEqual(self.sampler.state.get_state(), "shrink")

    def test_shrink(self):
        shrink_xs = [
            np.array([3.0, 4.0]),
            np.array([2.0, 3.0]),
            np.array([4.0, 5.0])
        ]
        min_value = self.sampler.simplex.vertices[0].value

        xsh = self.sampler.shrink()
        self.assertTrue(np.array_equal(xsh[0].xs, shrink_xs[0]))
        self.assertTrue(np.array_equal(xsh[1].xs, shrink_xs[1]))
        self.assertTrue(np.array_equal(xsh[2].xs, shrink_xs[2]))
        vertex = self.sampler.simplex.vertices[0]
        self.assertTrue(np.array_equal(vertex.xs, shrink_xs[0]))
        self.assertEqual(vertex.value, min_value)

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
        reflect_xs = np.array([-1.0, 0.0])

        self.sampler.state.reflect()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, reflect_xs))

    def test_get_next_coordinates_expand(self):
        expand_xs = np.array([-4.0, -3.0])

        self.sampler.state.expand()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, expand_xs))

    def test_get_next_coordinates_inside_contract(self):
        inside_contract_xs = np.array([3.5, 4.5])

        self.sampler.state.inside_contract()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, inside_contract_xs))

    def test_get_next_coordinates_outside_contract(self):
        outside_contract_xs = np.array([0.5, 1.5])

        self.sampler.state.outside_contract()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, outside_contract_xs))

    def test_get_next_coordinates_shrink(self):
        shrink_xs = [
            np.array([2.0, 3.0]),
            np.array([4.0, 5.0])
        ]

        self.sampler.state.shrink()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, shrink_xs[0]))
        self.assertTrue(np.array_equal(self.sampler.xs[0], shrink_xs[0]))
        self.assertTrue(np.array_equal(self.sampler.xs[1], shrink_xs[1]))

    def test_get_next_coordinates_reflect_out_of_range(self):
        expand_xs = np.array([3.5, 4.5])

        self.sampler._search_space["x"] = [0.0, 5.0]
        self.sampler.state.reflect()
        self.sampler.get_next_coordinates()
        self.assertTrue(np.array_equal(self.sampler.x, expand_xs))

    def test_set_objective_initial(self):
        self.sampler.simplex = Simplex()
        initial_xs = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0])
        ]
        initial_value = [5.0, 3.0, 7.0]

        self.sampler.set_objective(initial_xs[0], initial_value[0])
        self.assertEqual(self.sampler.state.get_state(), "initial")
        vertex = self.sampler.simplex.vertices[0]
        self.assertTrue(np.array_equal(vertex.xs, initial_xs[0]))
        self.assertEqual(vertex.value, initial_value[0])

        self.sampler.set_objective(initial_xs[1], initial_value[1])
        self.assertEqual(self.sampler.state.get_state(), "initial")
        vertex = self.sampler.simplex.vertices[1]
        self.assertTrue(np.array_equal(vertex.xs, initial_xs[1]))
        self.assertEqual(vertex.value, initial_value[1])

        self.sampler.set_objective(initial_xs[2], initial_value[2])
        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[2]
        self.assertTrue(np.array_equal(vertex.xs, initial_xs[2]))
        self.assertEqual(vertex.value, initial_value[2])

    def test_set_objective_reflect(self):
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 4.0

        self.sampler.state.reflect()
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.set_objective(reflect_xs, reflect_value)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, reflect_xs))
        self.assertEqual(vertex.value, reflect_value)

    def test_set_objective_expand(self):
        reflect_value = 2.0
        expand_value = 1.0
        expand_xs = np.array([-4.0, -3.0])

        self.sampler.state.expand()
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(reflect_value)
        self.sampler.store.e = self.sampler.expand()
        self.sampler.set_objective(expand_xs, expand_value)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, expand_xs))
        self.assertEqual(vertex.value, expand_value)

    def test_set_objective_inside_contract(self):
        reflect_value = 8.0
        inside_contract_value = 6.0
        inside_contract_xs = np.array([3.5, 4.5])

        self.sampler.state.inside_contract()
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(reflect_value)
        self.sampler.store.ic = self.sampler.inside_contract()
        self.sampler.set_objective(inside_contract_xs, inside_contract_value)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, inside_contract_xs))
        self.assertEqual(vertex.value, inside_contract_value)

    def test_set_objective_outside_contract(self):
        reflect_value = 6.0
        outside_contract_value = 5.0
        outside_contract_xs = np.array([0.5, 1.5])

        self.sampler.state.outside_contract()
        self.sampler.store.r = self.sampler.reflect()
        self.sampler.after_reflect(reflect_value)
        self.sampler.store.oc = self.sampler.outside_contract()
        self.sampler.set_objective(outside_contract_xs, outside_contract_value)

        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[-1]
        self.assertTrue(np.array_equal(vertex.xs, outside_contract_xs))
        self.assertEqual(vertex.value, outside_contract_value)

    def test_set_objective_shrink(self):
        shrink_xs = [
            np.array([2.0, 3.0]),
            np.array([4.0, 5.0])
        ]
        shrink_value = [4.0, 6.0]
        self.sampler.state.shrink()
        self.sampler.xs = [v.coordinates for v in self.sampler.shrink()[1:]]

        self.sampler.set_objective(shrink_xs[0], shrink_value[0])
        self.assertEqual(self.sampler.state.get_state(), "shrink")
        vertex = self.sampler.simplex.vertices[1]
        self.assertTrue(np.array_equal(vertex.xs, shrink_xs[0]))
        self.assertEqual(vertex.value, shrink_value[0])

        self.sampler.set_objective(shrink_xs[1], shrink_value[1])
        self.assertEqual(self.sampler.state.get_state(), "reflect")
        vertex = self.sampler.simplex.vertices[2]
        self.assertTrue(np.array_equal(vertex.xs, shrink_xs[1]))
        self.assertEqual(vertex.value, shrink_value[1])
