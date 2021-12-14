import pytest

from gamelib.rendering.camera import PerspectiveCamera
from tests.conftest import assert_all_equal


@pytest.fixture(autouse=True, scope="module")
def stub_aspect_ratio():
    import gamelib

    normal_implementation = gamelib.aspect_ratio
    gamelib.aspect_ratio = lambda: 16 / 9
    yield
    gamelib.aspect_ratio = normal_implementation


@pytest.fixture
def camera():
    return PerspectiveCamera((0, 0, 0), (1, 1, 1))


class TestPerspectiveCamera:
    def test_init_direction(self):
        cam = PerspectiveCamera((0, 0, 0), dir=(1, 0, 0))
        assert all(cam.direction == (1, 0, 0))

    def test_init_position(self):
        cam = PerspectiveCamera((1, 2, 3), (1, 1, 1))
        assert all(cam.pos == (1.0, 2.0, 3))

    @pytest.mark.parametrize(
        "attr, value, expected_to_change",
        (
            ("pos", (1, 2, 3), "view"),
            ("direction", (1, 2, 3), "view"),
            ("near", 3, "proj"),
            ("far", 19, "proj"),
            ("fov_y", 100, "proj"),
            ("fov_x", 95, "proj"),
        ),
    )
    def test_changing_properties(
        self, camera, attr, value, expected_to_change
    ):
        proj_bytes = camera.proj.tobytes()
        view_bytes = camera.view.tobytes()

        setattr(camera, attr, value)

        if expected_to_change == "view":
            assert camera.view.tobytes() != view_bytes
            assert camera.proj.tobytes() == proj_bytes
        if expected_to_change == "proj":
            assert camera.view.tobytes() == view_bytes
            assert camera.proj.tobytes() != proj_bytes

    def test_relative_directions(self, camera):
        camera.pos = (0, 0, 0)
        camera.direction = (1, 0, 0)

        assert all(camera.right == (0, -1, 0))
        assert all(camera.left == (0, 1, 0))

    def test_up_is_not_relative(self, camera):
        camera.pos = (0, 0, 10)
        camera.target = (1, 1, 0)
        assert all(camera.up == (0, 0, 1))

    def test_move(self, camera):
        camera.pos = (1, 2, 3)
        proj_bytes = camera.proj.tobytes()
        view_bytes = camera.view.tobytes()

        camera.move((3, 2, 1))

        assert camera.proj.tobytes() == proj_bytes
        assert camera.view.tobytes() != view_bytes
        assert all(camera.pos == (4, 4, 4))

    def test_rotate(self, camera):
        camera.direction = (0, 1, 0)
        proj_bytes = camera.proj.tobytes()
        view_bytes = camera.view.tobytes()

        camera.rotate(axis=(0, 0, 1), theta=90)

        assert camera.proj.tobytes() == proj_bytes
        assert camera.view.tobytes() != view_bytes
        assert_all_equal(camera.direction, (-1, 0, 0))
