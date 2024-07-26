import numpy as np
import numpy.typing as npt


def generate_initial_simplex(
    dim: int,
    edge: float = 0.5,
    centroid: float = 0.5,
    rng: np.random.RandomState | None = None,
) -> npt.NDArray[np.float64]:
    """
    Generate an initial simplex with a regular shape.
    """

    assert 0.0 <= centroid <= 1.0, "The centroid must be exists in the unit hypercube. "

    assert 0.0 < edge <= max(centroid, 1 - centroid), f"Maximum edge length is {max(centroid, 1 - centroid)}"

    # Our implementation normalizes the search space to unit hypercube [0, 1]^n.
    bdrys = np.array([[0, 1] for _ in range(dim)])

    # Generate regular simplex.
    initial_simplex = np.zeros((dim + 1, dim))
    b = 0.0
    for i in range(dim):
        c = np.sqrt(1 - b)
        initial_simplex[i][i] = c
        r = ((-1 / dim) - b) / c
        for j in range(i + 1, dim + 1):
            initial_simplex[j][i] = r
        b = b + r**2

    # Rotate the generated initial simplex.
    v = rng.random((dim, dim)) if rng is not None else np.random.random((dim, dim))

    for i in range(dim):
        for j in range(0, i):
            v[i] = v[i] - np.dot(v[i], v[j]) * v[j]
        v[i] = v[i] / (np.sum(v[i] ** 2) ** 0.5)
    for i in range(dim + 1):
        initial_simplex[i] = np.dot(initial_simplex[i], v)

    #  Scale up or down and move the generated initial simplex.
    for i in range(dim + 1):
        initial_simplex[i] = edge * initial_simplex[i]
    matrix_centroid = initial_simplex.mean(axis=0)
    initial_simplex = initial_simplex + (centroid - matrix_centroid)

    # Check the condition of the generated initial simplex.
    if check_initial_simplex(initial_simplex, bdrys):
        generate_initial_simplex(dim, edge, centroid)
    y = np.array(initial_simplex)

    return y


def check_initial_simplex(initial_simplex: npt.NDArray[np.float64], bdrys: npt.NDArray[np.float64]) -> bool:
    """
    Check whether there is at least one vertex of the generated simplex in the search space.
    """
    dim = len(initial_simplex) - 1
    return not (dim + 1 > sum([out_of_boundary(vertex, bdrys) for vertex in initial_simplex]))


def out_of_boundary(y: npt.NDArray[np.float64], bdrys: npt.NDArray[np.float64]) -> bool:
    """
    Check whether the input vertex is in the search space.
    """
    for yi, b in zip(y, bdrys, strict=False):
        if float(b[0]) <= float(yi) <= float(b[1]):
            pass
        else:
            return True
    return False
