import pytest
import numpy as np

from gamelib.transforms import Mat3

from ..conftest import assert_all_equal


@pytest.mark.parametrize(
    "vector, matrix, expected",
    (
            ((1, 1, 1), Mat3.rotate_about_x(90), (1, -1, 1)),
            ((1, 1, 1), Mat3.rotate_about_axis((1, 0, 0), 90), (1, -1, 1)),
            ((1, 1, 1), Mat3.rotate_about_y(90), (1, 1, -1)),
            ((1, 1, 1), Mat3.rotate_about_axis((0, 1, 0), 90), (1, 1, -1)),
            ((1, 1, 1), Mat3.rotate_about_z(90), (-1, 1, 1)),
            ((1, 1, 1), Mat3.rotate_about_axis((0, 0, 1), 90), (-1, 1, 1)),
            ((1, 1, 1), Mat3.rotate_about_axis((-1, 1, 0), 180), (-1, -1, -1)),
    ),
)
def test_3d_rotations(vector, matrix, expected):
    vec = np.array(vector)
    rotated = matrix.dot(vec)

    assert_all_equal(expected, rotated)
