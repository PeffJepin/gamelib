import pytest

from gamelib.rendering.camera import PerspectiveCamera
from gamelib.rendering.camera import get_primary_view
from gamelib.rendering.camera import get_primary_proj
from tests.conftest import assert_approx


ASPECT = 16 / 9


@pytest.fixture(autouse=True, scope="module")
def stub_aspect_ratio():
    import gamelib

    normal_implementation = gamelib.get_aspect_ratio
    gamelib.get_aspect_ratio = lambda: ASPECT
    yield
    gamelib.get_aspect_ratio = normal_implementation


@pytest.fixture
def camera():
    return PerspectiveCamera((0, 0, 0), (1, 1, 1))


class TestPerspectiveCamera:
    def test_init_direction(self):
        cam = PerspectiveCamera((0, 0, 0), direction=(1, 0, 0))
        assert cam.direction == (1, 0, 0)

    def test_init_position(self):
        cam = PerspectiveCamera((1, 2, 3), (1, 1, 1))
        assert cam.position == (1.0, 2.0, 3)

    @pytest.mark.parametrize(
        "attr, value, expected_to_change",
        (
            ("position", (1, 2, 3), "view"),
            ("direction", (1, 2, 3), "view"),
            ("near", 3, "proj"),
            ("far", 19, "proj"),
            ("fov_y", 100, "proj"),
        ),
    )
    def test_changing_properties(
        self, camera, attr, value, expected_to_change
    ):
        proj_bytes = camera.proj_matrix.tobytes()
        view_bytes = camera.view_matrix.tobytes()

        setattr(camera, attr, value)

        if expected_to_change == "view":
            assert camera.view_matrix.tobytes() != view_bytes
            assert camera.proj_matrix.tobytes() == proj_bytes
        if expected_to_change == "proj":
            assert camera.view_matrix.tobytes() == view_bytes
            assert camera.proj_matrix.tobytes() != proj_bytes

    def test_relative_directions(self, camera):
        camera.position = (0, 0, 0)
        camera.direction = (1, 0, 0)

        assert camera.right == (0, -1, 0)
        assert camera.left == (0, 1, 0)

    def test_up_is_not_relative(self, camera):
        camera.position = (0, 0, 10)
        camera.target = (1, 1, 0)
        assert camera.up == (0, 0, 1)

    def test_move(self, camera):
        camera.position = (1, 2, 3)
        proj_bytes = camera.proj_matrix.tobytes()
        view_bytes = camera.view_matrix.tobytes()

        camera.move((3, 2, 1))

        assert camera.proj_matrix.tobytes() == proj_bytes
        assert camera.view_matrix.tobytes() != view_bytes
        assert camera.position == (4, 4, 4)

    def test_rotate(self, camera):
        camera.direction = (0, 1, 0)
        proj_bytes = camera.proj_matrix.tobytes()
        view_bytes = camera.view_matrix.tobytes()

        camera.rotate(axis=(0, 0, 1), theta=90)

        assert camera.proj_matrix.tobytes() == proj_bytes
        assert camera.view_matrix.tobytes() != view_bytes
        assert_approx(camera.direction, (-1, 0, 0))

    def test_near_width(self):
        camera = PerspectiveCamera((0, 0, 0), (1, 0, 0), near=1)
        camera.fov_y = 90

        assert camera.near_plane_width == pytest.approx(2 * ASPECT)

    def test_near_height(self):
        camera = PerspectiveCamera((0, 0, 0), (1, 0, 0), near=2)
        camera.fov_y = 90

        assert camera.near_plane_height == pytest.approx(4)

    def test_near_size(self):
        camera = PerspectiveCamera((0, 0, 0), (1, 0, 0), near=2)
        camera.fov_y = 90

        assert_approx(camera.near_plane_size, (4 * ASPECT, 4))

    def test_get_primary_view(self):
        c1 = PerspectiveCamera((12, 123, 1234), (1, 2, 3))
        c2 = PerspectiveCamera((12, 321, 1234), (1, 2, 3))

        c1.set_primary()
        c2.set_primary()

        assert_approx(c2.view_matrix, get_primary_view())

    def test_get_primary_proj(self):
        c1 = PerspectiveCamera((12, 123, 1234), (1, 2, 3))
        c2 = PerspectiveCamera((12, 321, 1234), (1, 2, 3))

        c1.set_primary()
        c2.set_primary()

        assert_approx(c2.proj_matrix, get_primary_proj())
