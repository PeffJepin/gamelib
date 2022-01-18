import pytest
import numpy as np

from gamelib.geometry import transforms
from ..conftest import assert_approx


@pytest.fixture(autouse=True, scope="function")
def cleanup():
    transforms.TransformComponent.clear()


@pytest.fixture(
    params=[
        ((1, 2, 3), (1, 1, 1), (1, 1, 1), 0),
        ((1, 2, 3), (1, 2, 3), (1, 1, 1), 0),
        ((0, 0, 0), (1, 2, 3), (1, 1, 1), 0),
        ((1, 2, 3), (1, 2, 3), (1, 2, 3), 45),
        ((1, 2, 3), (1, 1, 1), (1, 2, 3), 45),
        ((0, 0, 0), (1, 1, 1), (1, 2, 3), 45),
    ]
)
def transform_inputs(request):
    pos, scale, axis, theta = request.param
    return pos, scale, axis, theta


def test_transform_base_case(transform_inputs):
    # assert that TransformComponent behaves in the same manner as Transform.
    regular = transforms.Transform(*transform_inputs)
    component = transforms.TransformComponent(*transform_inputs)

    vertex1 = np.array((1, 1, 1))
    vertex2 = np.array((1, 1, 1))
    regular.apply(vertex1)
    component.apply(vertex2)

    assert_approx(vertex1, vertex2)
