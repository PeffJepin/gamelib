import numpy as np

import gamelib

from . import gl
from ..input import InputSchema
from ..geometry import normalize, Mat3, Mat4


class BaseCamera:
    def __init__(
        self,
        pos,
        up,
        dir,
        near,
        far,
    ):
        # target/direction are conflicting, one or the other should be
        # given, and one must be given

        self._near = near
        self._far = far
        self._pos = np.asarray(pos, gl.vec3)
        self._up = np.asarray(up, gl.vec3)
        self._dir = np.asarray(dir, gl.vec3)
        normalize(self._dir)
        normalize(self._up)

        self.view = np.empty(1, gl.mat4)
        self.proj = np.empty(1, gl.mat4)
        self._update_view()
        self._update_proj()

    @property
    def _aspect_ratio(self):
        return gamelib.aspect_ratio()

    @property
    def near(self):
        return self._near

    @near.setter
    def near(self, value):
        self._near = value
        self._update_proj()

    @property
    def far(self):
        return self._far

    @far.setter
    def far(self, value):
        self._far = value
        self._update_proj()

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos[:] = value
        self._update_view()

    @property
    def direction(self):
        return self._dir.copy()

    @direction.setter
    def direction(self, value):
        self._dir[:] = value
        normalize(self._dir)
        self._update_view()

    @property
    def up(self):
        return self._up.copy()

    @up.setter
    def up(self, value):
        self._up[:] = value
        normalize(self._up)
        self._update_view()

    @property
    def right(self):
        return np.cross(self.direction, self.up)

    @property
    def left(self):
        return -self.right

    def move(self, translation):
        translation = np.asarray(translation, self._pos.dtype)
        self._pos += translation
        self._update_view()

    def _update_view(self):
        raise NotImplementedError("Should be implemented in subclasses.")

    def _update_proj(self):
        raise NotImplementedError("Should be implemented in subclasses.")


class PerspectiveCamera(BaseCamera):
    def __init__(
        self,
        pos,
        dir,
        fovy=60,
        near=1,
        far=1000,
        up=(0, 0, 1),
        _controller=None,
    ):
        # target/direction are conflicting, one or the other should be
        # given, and one must be given

        self._fov_y = fovy
        super().__init__(pos, up, dir, near, far)
        self._controller = _controller or _FreePerspectiveController(self)

    @property
    def fov_y(self):
        return self._fov_y

    @fov_y.setter
    def fov_y(self, value):
        self._fov_y = value
        self._update_proj()

    @property
    def fov_x(self):
        return self.fov_y * self._aspect_ratio

    @fov_x.setter
    def fov_x(self, value):
        self.fov_y = value / self._aspect_ratio

    def rotate(self, axis=None, theta=None, matrix=None):
        if matrix is None:
            matrix = Mat3.rotate_about_axis(axis, theta)
        self.direction = matrix.dot(self.direction)

    def _update_view(self):
        self.view[:] = Mat4.look_at_transform(
            self.pos, self.pos + self.direction, self.up
        )

    def _update_proj(self):
        proj = Mat4.perspective_transform(
            self.fov_y, self._aspect_ratio, self.near, self.far
        )
        self.proj[:] = proj


class OrthogonalCamera(BaseCamera):
    def __init__(
        self, px_per_unit, pos=(0, 0, 10), up=(0, 1, 0), dir=(0, 0, -1)
    ):
        self._left = 0
        self._right = 1
        self._bottom = 0
        self._top = 1
        super().__init__(pos, up, dir, near=1, far=20)
        self.px_per_unit = px_per_unit
        self._controller = _FreeOrthogonalController(self)

    @property
    def px_per_unit(self):
        return self._px_per_unit

    @px_per_unit.setter
    def px_per_unit(self, value):
        self._px_per_unit = value
        width = gamelib.get_width() / value / 2
        height = gamelib.get_height() / value / 2
        self._left = -width
        self._right = width
        self._top = height
        self._bottom = -height
        self._update_proj()

    def _update_proj(self):
        self.proj[:] = Mat4.orthogonal_transform(
            self._left,
            self._right,
            self._bottom,
            self._top,
            self._near,
            self._far,
        )

    def _update_view(self):
        self.view[:] = Mat4.look_at_transform(
            self.pos, self.pos + self.direction, self.up
        )

    def rotate(self, theta):
        self.up = self.up.dot(Mat3.rotate_about_z(theta))
        self._update_view()


class _FreePerspectiveController:
    def __init__(self, camera: PerspectiveCamera, speed=35):
        self.camera = camera
        self.speed = speed
        self._schema = InputSchema(
            # need a input handler decorator for this use case
            # where one function handles the base event
            ("a", "is_pressed", self._pan_camera),
            ("a", "is_pressed", "shift", self._pan_camera),
            ("s", "is_pressed", self._pan_camera),
            ("s", "is_pressed", "shift", self._pan_camera),
            ("d", "is_pressed", self._pan_camera),
            ("d", "is_pressed", "shift", self._pan_camera),
            ("w", "is_pressed", self._pan_camera),
            ("w", "is_pressed", "shift", self._pan_camera),
            ("left", "is_pressed", self._pan_camera),
            ("left", "is_pressed", "shift", self._pan_camera),
            ("right", "is_pressed", self._pan_camera),
            ("right", "is_pressed", "shift", self._pan_camera),
            ("up", "is_pressed", self._pan_camera),
            ("up", "is_pressed", "shift", self._pan_camera),
            ("down", "is_pressed", self._pan_camera),
            ("drag", self._rotate_camera),
            ("scroll", self._z_scroll_camera),
        )

    def _pan_camera(self, event):
        if event.key in ("a", "left"):
            vec = self.camera.left
        elif event.key in ("s", "down"):
            vec = -self.camera.direction
        elif event.key in ("d", "right"):
            vec = self.camera.right
        elif event.key in ("w", "up"):
            vec = self.camera.direction
        vec[2] = 0
        norm = normalize(vec)
        mult = 2 if event.modifiers.shift else 1
        translation = norm * mult * self.speed * event.dt
        self.camera.move(translation)

    def _rotate_camera(self, event):
        if event.dx != 0:
            theta = -event.dx / gamelib.get_width() * self.camera.fov_x
            self.camera.rotate(matrix=Mat3.rotate_about_z(theta))
        if event.dy != 0:
            axis = self.camera.right
            theta = -event.dy / gamelib.get_width() * self.camera.fov_y
            self.camera.rotate(matrix=Mat3.rotate_about_axis(axis, theta))

    def _z_scroll_camera(self, event):
        speed = 1
        translation = -np.array((0, 0, event.dy)) * speed
        self.camera.move(translation)


class _FreeOrthogonalController:
    def __init__(self, camera: OrthogonalCamera, speed=35):
        self.camera = camera
        self.speed = speed
        self._schema = InputSchema(
            # need a input handler decorator for this use case
            # where one function handles the base event
            ("a", "is_pressed", self._pan_camera),
            ("a", "is_pressed", "shift", self._pan_camera),
            ("s", "is_pressed", self._pan_camera),
            ("s", "is_pressed", "shift", self._pan_camera),
            ("d", "is_pressed", self._pan_camera),
            ("d", "is_pressed", "shift", self._pan_camera),
            ("w", "is_pressed", self._pan_camera),
            ("w", "is_pressed", "shift", self._pan_camera),
            ("left", "is_pressed", self._pan_camera),
            ("left", "is_pressed", "shift", self._pan_camera),
            ("right", "is_pressed", self._pan_camera),
            ("right", "is_pressed", "shift", self._pan_camera),
            ("up", "is_pressed", self._pan_camera),
            ("up", "is_pressed", "shift", self._pan_camera),
            ("down", "is_pressed", self._pan_camera),
            ("drag", self._rotate_camera),
            ("scroll", self._z_scroll_camera),
        )

    def _pan_camera(self, event):
        if event.key in ("a", "left"):
            vec = self.camera.left
        elif event.key in ("s", "down"):
            vec = -self.camera.up
        elif event.key in ("d", "right"):
            vec = self.camera.right
        elif event.key in ("w", "up"):
            vec = self.camera.up
        vec[2] = 0
        norm = normalize(vec)
        mult = 2 if event.modifiers.shift else 1
        translation = norm * mult * self.speed * event.dt
        self.camera.move(translation)

    def _rotate_camera(self, event):
        if event.dx != 0:
            theta = -event.dx / gamelib.get_width() * 90
            self.camera.rotate(theta)

    def _z_scroll_camera(self, event):
        scale = 1.05
        if event.dy > 0:
            mult = scale
        elif event.dy < 0:
            mult = 1 / scale
        else:
            mult = 0
        self.camera.px_per_unit *= mult
