import pytest
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
        pos=(0, -1, 0), dir=(0, 1, 0), fovy=90, near=1
    )

    ray = camera.screen_to_ray(WIDTH // 2, HEIGHT)

    expected = geometry.Ray(origin=(0, -1, 0), direction=(0, 1, 1))
    assert_approx(expected.origin, ray.origin)
    assert_approx(expected.direction, ray.direction)
