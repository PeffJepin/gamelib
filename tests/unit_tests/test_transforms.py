import pytest
import numpy as np

from gamelib.transforms import Mat3
from gamelib.transforms import Mat4
from gamelib.transforms import Transform

from ..conftest import assert_all_equal


@pytest.mark.parametrize(
    "matrix, expected",
    # fmt: off
    (
        (Mat3.rotate_about_x(90), (1, -1, 1)),
        (Mat3.rotate_about_axis((1, 0, 0), 90), (1, -1, 1)),
        (Mat3.rotate_about_y(90), (1, 1, -1)),
        (Mat3.rotate_about_axis((0, 1, 0), 90), (1, 1, -1)),
        (Mat3.rotate_about_z(90), (-1, 1, 1)),
        (Mat3.rotate_about_axis((0, 0, 1), 90), (-1, 1, 1)),
        (Mat3.rotate_about_axis((-1, 1, 0), 180), (-1, -1, -1)),
    ),
    # fmt: on
)
def test_3d_rotation_matrices(matrix, expected):
    input_vector = np.array((1, 1, 1))
    transformed_vector = matrix.dot(input_vector)

    assert_all_equal(expected, transformed_vector)


def test_mat4():
    vertex = np.array((1, 1, 1, 1))
    transformed = Mat4.rotate_about_axis((1, 0, 0), 90).T.dot(vertex)

    assert_all_equal(transformed, (1, -1, 1))

@pytest.mark.parametrize(
    "transform, expected",
    # fmt: off
    (
        (Transform(pos=(2, 3, 4)), (3, 4, 5)),
        (Transform(scale=(2, 3, 10)), (2, 3, 10)),
        (Transform(axis=(1, 0, 0), theta=90), (1, -1, 1)),
        (Transform(pos=(3, 3, 5), axis=(1, 0, 0), theta=90), (4, 2, 6)),
        (Transform(pos=(3, 3, 5), scale=(2, 2, 2)), (5, 5, 7)),
        (Transform(pos=(3, 3, 5), scale=(2, 2, 2), axis=(1, 0, 0), theta=90), (5, 1, 7)),
        (Transform(scale=(2, 2, 2), axis=(1, 0, 0), theta=90), (2, -2, 2)),
    )
    # fmt: on
)
def test_transform(transform, expected):
    vertex = np.array((1, 1, 1))
    transform.apply(vertex)

    assert_all_equal(expected, vertex)

    # also test that fiddling with the attributes gets the expected
    # results against these inputs
    vertex = np.array((1, 1, 1))
    blank_transform = Transform()
    blank_transform.pos = transform.pos
    blank_transform.scale = transform.scale
    blank_transform.theta = transform.theta
    blank_transform.axis = transform.axis
    blank_transform.apply(vertex)

    assert_all_equal(expected, vertex)
