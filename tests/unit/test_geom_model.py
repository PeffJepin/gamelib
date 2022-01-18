import pytest

from gamelib.geometry import base
from gamelib.geometry import cube

from ..conftest import assert_approx


@pytest.fixture
def cube_model():
    """A cube centered on the origin with side lengths of 2. This means the
    corners will fall on (-1, -1, -1), (-1, -1, 1), etc..."""

    return cube.Cube(scale=2)


@pytest.mark.parametrize(
    "anchor_location, expected_translation",
    (
        ((0, 0, 0), (1, 1, 1)),
        ((0.5, 0.5, 0.5), (0, 0, 0)),
        ((1, 1, 1), (-1, -1, -1)),
    ),
)
def test_anchoring_a_model(cube_model, anchor_location, expected_translation):
    expected_vertices = cube_model.vertices.copy()
    expected_vertices += expected_translation

    cube_model.anchor(anchor_location)

    assert_approx(expected_vertices, cube_model.vertices)


@pytest.mark.parametrize(
    "model, expected_min, expected_max",
    (
        (cube.Cube(scale=1), (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
        (cube.Cube(scale=2), (-1, -1, -1), (1, 1, 1)),
        (cube.Cube(scale=2.5), (-1.25, -1.25, -1.25), (1.25, 1.25, 1.25)),
    ),
)
def test_boundaries(model, expected_min, expected_max):
    assert_approx(model.v_min, expected_min)
    assert_approx(model.v_max, expected_max)


def test_boundaries_after_anchoring():
    model = cube.Cube()
    model.anchor((0, 0, 0))

    assert_approx(model.v_min, (0, 0, 0))
    assert_approx(model.v_max, (1, 1, 1))


def test_anchor_on_init():
    model = cube.Cube(anchor=(0, 0, 0))

    assert_approx(model.v_min, (0, 0, 0))
    assert_approx(model.v_max, (1, 1, 1))
