import math
import numpy as np

import gamelib

from gamelib import gl
from gamelib.core import input
from gamelib.geometry import transforms
from gamelib.geometry import collisions


class BaseCamera:
    """A base class for cameras. Subclasses must implement _update_view and
    _update_proj for updating the view and projection matrices."""

    def __init__(self, pos, up, dir, near, far, controller=None):
        """Initialize a camera. When a subclass calls super().__init__()
        _update_proj and _update_view will be called, so be sure the self
        object has the required attributes bound.

        Parameters
        ----------
        pos : Sequence
            xyz coordinate where the camera is located in world space.
        up : Sequence
            xyz up vector for world space.
        dir : Sequence
            xyz vector indicate the direction the camera is looking.
        near : float
            Distance to the near clipping plane.
        far : float
            Distance to the far clipping plane.
        controller : type
            A class marked with input handlers. Will be initialized if
            given and passed `self` as an __init__ argument and can then
            be toggled with the enable_controller/disable_controller methods.
        """

        self._view = np.empty(1, gl.mat4)
        self._proj = np.empty(1, gl.mat4)
        self._pos = np.asarray(pos, gl.vec3)
        self._up = transforms.normalize(np.asarray(up, gl.vec3))
        self._dir = transforms.normalize(np.asarray(dir, gl.vec3))
        self._near = near
        self._far = far
        if controller:
            self._controller = controller(self)
            self.enable_controller()
        else:
            self._controller = None
        self._update_view()
        self._update_proj()

    @property
    def _aspect_ratio(self):
        """Aspect ratio for the camera. Defaults implementation uses the
        windows aspect ratio.

        Returns
        -------
        float
        """

        return gamelib.get_aspect_ratio()

    @property
    def near(self):
        """Distance to the near clipping plane.

        Returns
        -------
        float
        """

        return self._near

    @near.setter
    def near(self, value):
        """Set the near clipping plane and update the projection matrix.

        Parameters
        ----------
        value : float
        """

        self._near = value
        self._update_proj()

    @property
    def far(self):
        """Get the distance to the far clipping plane.

        Returns
        -------
        float
        """

        return self._far

    @far.setter
    def far(self, value):
        """Sets distance to the far clipping plane and updates the
        projection matrix.

        Parameters
        ----------
        value : float
        """

        self._far = value
        self._update_proj()

    @property
    def pos(self):
        """Gets a copy of the camera position.

        Returns
        -------
        np.ndarray
        """

        return self._pos.copy()

    @pos.setter
    def pos(self, value):
        """Sets the camera position and updates the view matrix.

        Parameters
        ----------
        value : Sequence
            xyz coordinate
        """

        self._pos[:] = value
        self._update_view()

    @property
    def direction(self):
        """Gets the direction the camera is facing.

        Returns
        -------
        np.ndarray:
            xyz vector
        """

        return self._dir.copy()

    @direction.setter
    def direction(self, value):
        """Sets and transforms.normalizes the camera direction then updates the view
        matrix.

        Parameters
        ----------
        value : Sequence
            xyz vector for the camera to look towards.
        """

        self._dir[:] = value
        transforms.normalize(self._dir)
        self._update_view()

    @property
    def up(self):
        """Gets a copy of the up vector.

        Returns
        -------
        np.ndarray:
            xyz vector
        """

        return self._up.copy()

    @up.setter
    def up(self, value):
        """Sets and normalizes the up vector and updates the view matrix.

        Parameters
        ----------
        value : Sequence
            xyz vector
        """

        self._up[:] = value
        transforms.normalize(self._up)
        self._update_view()

    @property
    def down(self):
        """Gets the down vector.

        Returns
        -------
        np.ndarray:
            xyz vector
        """

        return -self.up

    @property
    def right(self):
        """Gets the right vector.

        Returns
        -------
        np.ndarray:
            xyz vector
        """

        return np.cross(self.direction, self.up)

    @property
    def left(self):
        """Gets the left vector.

        Returns
        -------
        np.ndarray
            xyz vector
        """

        return -self.right

    @property
    def view_matrix(self):
        """Gets the current view matrix.

        Returns
        -------
        np.ndarray:
            4x4 view matrix
        """

        return self._view

    @view_matrix.setter
    def view_matrix(self, mat4):
        """Sets the view matrix. The matrix is updated in place.

        Parameters
        ----------
        mat4 : array-like
            Most likely a matrix from geometry.transforms.Mat4 namespace.
        """

        self._view[:] = mat4

    @property
    def projection_matrix(self):
        """Gets the projection matrix.

        Returns
        -------
        np.ndarray:
            4x4 projection matrix
        """

        return self._proj

    @projection_matrix.setter
    def projection_matrix(self, mat4):
        """Updates the projection matrix in place.

        Parameters
        ----------
        mat4 : array-like
            Most likely a matrix from geometry.transforms.Mat4 namespace.
        """

        self._proj[:] = mat4

    def move(self, translation):
        """Offset current position by given translation.

        Parameters
        ----------
        translation : Sequence
            xyz translation vector
        """

        translation = np.asarray(translation, self._pos.dtype)
        self._pos += translation
        self._update_view()

    def enable_controller(self):
        """Enable an attached controller to start handling input events."""

        if self._controller:
            input.enable_handlers(self._controller)

    def disable_controller(self):
        """Stop handling input events with the attached controller."""

        if self._controller:
            input.disable_handlers(self._controller)

    def _update_view(self):
        """Set the view_matrix property based on current camera state."""

        raise NotImplementedError("Should be implemented in subclasses.")

    def _update_proj(self):
        """Set the projection_matrix property based on current camera state."""

        raise NotImplementedError("Should be implemented in subclasses.")


class PerspectiveCamera(BaseCamera):
    """Simple camera for rendering a 3d scene with a perspective projection."""

    def __init__(
        self,
        pos,
        dir,
        fovy=60,
        near=1,
        far=1000,
        up=(0, 0, 1),
        controller=None,
    ):
        """Initialize the camera.

        Parameters
        ----------
        pos : Sequence
            xyz position in world space.
        dir : Sequence
            xyz vector the camera is facing - applied locally at the camera.
        fovy : float
            y field of view, given in degrees.
        near : float
            Distance to the near clipping plane.
        far : float
            Distance to the far clipping plane.
        up : Sequence
            xyz vector pointing to up in world space.
        controller : type | bool
            Can be `True` to activate a default controller. See BaseCamera for
            more details otherwise.
        """

        self._fov_y = fovy
        if controller is True:
            controller = _FreePerspectiveController
        super().__init__(pos, up, dir, near, far, controller)

    @property
    def fov_y(self):
        """Get the y field of view.

        Returns
        -------
        float
        """

        return self._fov_y

    @fov_y.setter
    def fov_y(self, value):
        """Sets the y field of view and updates projection matrix.

        Parameters
        ----------
        value : float
            Field of view given in degrees.
        """

        self._fov_y = value
        self._update_proj()

    @property
    def near_plane_width(self):
        return self.near_plane_height * self._aspect_ratio

    @property
    def near_plane_height(self):
        return 2 * math.tan(math.radians(self.fov_y / 2)) * self.near

    @property
    def near_plane_size(self):
        return self.near_plane_width, self.near_plane_height

    def rotate(self, axis=None, theta=None):
        """Rotate the cameras direction vector.

        Parameters
        ----------
        axis : Sequence, optional
            xyz vector representing the axis of rotation.
        theta : float
            The rotation angle given in degrees.
            (Right handed coordinate system)
        """

        matrix = transforms.Mat3.rotate_about_axis(axis, theta)
        self.direction = matrix.dot(self.direction)

    def screen_to_ray(self, x, y):
        vec_to_near_plane = transforms.normalize(self.direction) * self.near
        near_plane_center = self.pos + vec_to_near_plane
        near_w, near_h = self.near_plane_size
        dx = (-0.5 + (x / gamelib.get_width())) * near_w
        dy = (-0.5 + (y / gamelib.get_height())) * near_h
        n_right = transforms.normalize(self.right)
        n_up = -transforms.normalize(np.cross(self.direction, n_right))
        x_y_to_near_plane = near_plane_center + n_up * dy + n_right * dx
        return collisions.Ray(self.pos, x_y_to_near_plane - self.pos)

    def cursor_to_ray(self):
        return self.screen_to_ray(*gamelib.get_cursor())

    def _update_view(self):
        self.view_matrix = transforms.Mat4.look_at_transform(
            self.pos, self.pos + self.direction, self.up
        )

    def _update_proj(self):
        self.projection_matrix = transforms.Mat4.perspective_transform(
            self.fov_y, self._aspect_ratio, self.near, self.far
        )


class OrthogonalCamera(BaseCamera):
    """Simple implementation of a camera with an orthogonal projection."""

    def __init__(
        self,
        px_per_unit,
        pos=(0, 0, 10),
        up=(0, 1, 0),
        dir=(0, 0, -1),
        controller=None,
    ):
        """Initialize the camera.

        Parameters
        ----------
        px_per_unit : float
            Ratio of screen pixels per world unit. This effectively controls
            the "zoom" of the camera.
        pos : Sequence
            xyz camera position in world space.
        up : Sequence
            xyz vector pointing up in world space.
        dir : Sequence
            xyz vector the camera is looking.
        controller : type | bool
            Can be `True` to activate a default controller. See BaseCamera for
            more details otherwise.
        """

        self._left = 0
        self._right = 1
        self._bottom = 0
        self._top = 1
        if controller is True:
            controller = _FreeOrthogonalController
        super().__init__(pos, up, dir, 1, 20, controller)
        self.px_per_unit = px_per_unit

    @property
    def px_per_unit(self):
        """Get the current px/unit value.

        Returns
        -------
        float
        """

        return self._px_per_unit

    @px_per_unit.setter
    def px_per_unit(self, value):
        """Sets the px/unit ratio and updates the projection matrix
        accordingly.

        Parameters
        ----------
        value : float
        """

        self._px_per_unit = value
        width = gamelib.get_width() / value / 2
        height = gamelib.get_height() / value / 2
        self._left = -width
        self._right = width
        self._top = height
        self._bottom = -height
        self._update_proj()

    def rotate(self, theta):
        """Rotates about the axis the camera is facing.

        Parameters
        ----------
        theta : float
            Angle of rotation given in degrees.
        """

        self.up = transforms.Mat3.rotate_about_axis(self.direction, theta).dot(
            self.up
        )
        self._update_view()

    def _update_proj(self):
        self.projection_matrix = transforms.Mat4.orthogonal_transform(
            self._left,
            self._right,
            self._bottom,
            self._top,
            self._near,
            self._far,
        )

    def _update_view(self):
        self.view_matrix = transforms.Mat4.look_at_transform(
            self.pos, self.pos + self.direction, self.up
        )


class _FreePerspectiveController:
    """Simple default controller."""

    def __init__(self, camera: PerspectiveCamera, speed=35):
        self.camera = camera
        self.speed = speed

    @input.KeyIsPressed.handler(iter("asdw"))
    def _pan_camera(self, event):
        if event.key == "a":
            vector = self.camera.left
        elif event.key == "s":
            vector = -self.camera.direction
        elif event.key == "d":
            vector = self.camera.right
        elif event.key == "w":
            vector = self.camera.direction

        # don't handle z axis movement here
        vector[2] = 0
        transforms.normalize(vector)

        # move fast with shift being held
        multiplier = 2 if event.modifiers.shift else 1

        translation = vector * multiplier * self.speed * event.dt
        self.camera.move(translation)

    @input.MouseDrag.handler
    def _rotate_camera(self, event):
        if event.dx != 0:
            # z rotation for left/right
            theta = (
                event.dx
                / gamelib.get_width()
                * self.camera.fov_y
                * self.camera._aspect_ratio
            )
            self.camera.rotate((0, 0, 1), theta)
        if event.dy != 0:
            # rotate along the left/right axis for up/down
            axis = self.camera.right
            theta = event.dy / gamelib.get_width() * self.camera.fov_y
            self.camera.rotate(axis, theta)

    @input.MouseScroll.handler
    def _z_scroll_camera(self, event):
        scroll_rate = 1
        translation = -np.array((0, 0, event.dy)) * scroll_rate
        self.camera.move(translation)


class _FreeOrthogonalController:
    """Simple default controller."""

    def __init__(self, camera: OrthogonalCamera, speed=35):
        self.camera = camera
        self.speed = speed

    @input.KeyIsPressed.handler(iter("asdw"))
    def _pan_camera(self, event):
        if event.key == "a":
            vector = self.camera.left
        elif event.key == "s":
            vector = self.camera.down
        elif event.key == "d":
            vector = self.camera.right
        elif event.key == "w":
            vector = self.camera.up

        # eliminate z axis motion. scroll wheel handles this
        vector[2] = 0
        transforms.normalize(vector)

        # move fast when shift is held
        multiplier = 2 if event.modifiers.shift else 1

        translation = vector * multiplier * self.speed * event.dt
        self.camera.move(translation)

    @input.MouseDrag.handler
    def _rotate_camera(self, event):
        if event.dx != 0:
            theta = event.dx / gamelib.get_width() * 90
            self.camera.rotate(theta)

    @input.MouseScroll.handler
    def _z_scroll_camera(self, event):
        scale = 1.05
        if event.dy > 0:
            mult = scale
        elif event.dy < 0:
            mult = 1 / scale
        else:
            mult = 0
        self.camera.px_per_unit *= mult
