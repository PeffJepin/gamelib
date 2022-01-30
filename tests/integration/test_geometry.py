import pytest
import numpy as np
import gamelib

from gamelib import rendering
from gamelib import geometry

from ..conftest import assert_approx

WIDTH = 1920
HEIGHT = 1080
ASPECT = WIDTH / HEIGHT


@pytest.fixture(autouse=True, scope="module")
def stub_gamelib_window_stats():
    gamelib.get_width = lambda: WIDTH
    gamelib.get_height = lambda: HEIGHT
    gamelib.get_aspect_ratio = lambda: ASPECT


def test_screen_to_ray():
    camera = rendering.PerspectiveCamera(
        position=(0, -1, 0), direction=(0, 1, 0), fov_y=90, near=1
    )

    ray = camera.screen_to_ray(WIDTH // 2, HEIGHT)

    expected = geometry.Ray(origin=(0, -1, 0), direction=(0, 1, 1))
    assert_approx(expected.origin, ray.origin)
    assert_approx(expected.direction, ray.direction)


def test_applying_a_transform_to_a_model():
    vertices = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], gamelib.gl.vec3)
    normals = np.array([(0, 1, 2), (3, 4, 5), (2, 3, 4)], gamelib.gl.vec3)
    indices = np.array([(0, 1, 2)], gamelib.gl.uvec3)
    model = geometry.Model(vertices.copy(), indices, normals.copy())
    transform = geometry.Transform((0, 1, 2), (2, 3, 4), (1, 2, 3), 90)

    for v3 in vertices:
        transform.apply(v3)
    for n3 in normals:
        transform.apply(n3, normal=True)
    transform.apply(model)

    assert np.allclose(vertices, model.vertices)
    assert np.allclose(normals, model.normals)
    assert np.all(indices == model.triangles)
