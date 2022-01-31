import pytest
import numpy as np

import gamelib
from gamelib import ecs
from gamelib.geometry import transforms
from ..conftest import assert_approx


@pytest.fixture(autouse=True, scope="function")
def cleanup():
    ecs.Entity.clear()


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
    component = ecs.Transform.create(*transform_inputs)

    vertex1 = np.array((1, 1, 1))
    vertex2 = np.array((1, 1, 1))
    regular.apply(vertex1)
    component.apply(vertex2)

    assert_approx(vertex1, vertex2)


class VectorComponent(ecs.Component):
    pos: gamelib.Vec3


class TestVectorComponentIntegration:
    def test_creation(self):
        v1 = gamelib.Vec3(1, 2, 3)
        v2 = gamelib.Vec3(3, 2, 1)
        c1 = VectorComponent(pos=v1)
        c2 = VectorComponent(pos=tuple(v2))

        assert c1.pos == v1
        assert c2.pos == v2
        assert isinstance(c1.pos, gamelib.Vec3)
        assert isinstance(c2.pos, gamelib.Vec3)

        expected = np.stack((v1, v2))
        assert np.allclose(expected, VectorComponent.pos)

    def test_destroy(self):
        v1 = gamelib.Vec3(1, 2, 3)
        v2 = gamelib.Vec3(3, 2, 1)
        c1 = VectorComponent(pos=v1)
        c2 = VectorComponent(pos=v2)

        VectorComponent.destroy(c1)

        assert np.allclose(c2.pos, VectorComponent.pos)

    def test_instance_ops(self):
        v = gamelib.Vec3(1, 2, 3)
        c = VectorComponent(pos=v)

        c.pos += (3, 2, 1)
        v += (3, 2, 1)
        assert VectorComponent.get(c.id).pos == v

        c.pos *= 3
        v *= 3
        assert VectorComponent.get(c.id).pos == v

        c.pos -= gamelib.Vec3(1, 2, 3)
        v -= gamelib.Vec3(1, 2, 3)
        assert VectorComponent.get(c.id).pos == v

    def test_array_ops(self):
        v1 = gamelib.Vec3(1, 2, 3)
        v2 = gamelib.Vec3(3, 2, 1)
        VectorComponent(pos=v1)
        VectorComponent(pos=v2)

        dv = (1, 2, 3)
        VectorComponent.pos += dv
        v1 += dv
        v2 += dv
        assert np.allclose(VectorComponent.pos, np.stack((v1, v2)))

        dv = gamelib.Vec3(2, 2, 3)
        VectorComponent.pos *= dv
        v1 *= dv
        v2 *= dv
        assert np.allclose(VectorComponent.pos, np.stack((v1, v2)))

        dv = 2
        VectorComponent.pos /= dv
        v1 /= dv
        v2 /= dv
        assert np.allclose(VectorComponent.pos, np.stack((v1, v2)))
