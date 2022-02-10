import math
import numpy as np

import gamelib

from gamelib.core import input
from gamelib.core import gl
from gamelib.geometry import transforms
from gamelib.geometry import collisions


_primary_camera: "BaseCamera" = None


def get_primary_view():
    if _primary_camera is None:
        return transforms.Mat4.identity()
    else:
        return _primary_camera.view_matrix


def get_primary_proj():
    if _primary_camera is None:
        return transforms.Mat4.identity()
    else:
        return _primary_camera.proj_matrix


class BaseCamera:
    """A base class for cameras. Subclasses must implement _update_view and
    _update_proj for updating the view and projection matrices."""

    def __init__(self, position, up, direction, near, far, controller=None):
        """Initialize a camera. When a subclass calls super().__init__()
        _update_proj and _update_view will be called, so be sure the self
        object has the required attributes bound.

        Parameters
        ----------
        position : gamelib.Vec3 | Iterable
            xyz coordinate where the camera is located in world space.
        up : gamelib.Vec3 | Iterable
            xyz up vector for world space.
        direction : gamelib.Vec3 | Iterable
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

        self._initialized = False
        self._view = np.empty(1, gl.mat4)
        self._proj = np.empty(1, gl.mat4)
        self.position = position
        self.up = up
        self.direction = direction
        self.near = near
        self.far = far
        if controller:
            self._controller = controller(self)
            self.enable_controller()
        else:
            self._controller = None
        self._initialized = True
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
        if self._initialized:
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
        if self._initialized:
            self._update_proj()

    @property
    def position(self):
        """Gets a copy of the camera position.

        Returns
        -------
        gamelib.Vec3
        """

        return self._pos.copy()

    @position.setter
    def position(self, value):
        """Sets the camera position and updates the view matrix.

        Parameters
        ----------
        value : gamelib.Vec3 | Iterable
        """

        if isinstance(value, gamelib.Vec3):
            self._pos = value
        elif isinstance(value, np.ndarray):
            self._pos = value.view(gamelib.Vec3)
        else:
            self._pos = gamelib.Vec3(*value)
        if self._initialized:
            self._update_view()

    @property
    def direction(self):
        """Gets the direction the camera is facing.

        Returns
        -------
        gamelib.Vec3
        """

        return self._dir.copy()

    @direction.setter
    def direction(self, value):
        """Sets and transforms.normalizes the camera direction then updates the view
        matrix.

        Parameters
        ----------
        value : gamelib.Vec3 | Iterable
        """

        if isinstance(value, gamelib.Vec3):
            self._dir = value.normalize()
        elif isinstance(value, np.ndarray):
            self._dir = value.view(gamelib.Vec3).normalize()
        else:
            self._dir = gamelib.Vec3(*value).normalize()
        if self._initialized:
            self._update_view()

    @property
    def up(self):
        """Gets a copy of the up vector.

        Returns
        -------
        gamelib.Vec3
        """

        return self._up.copy()

    @up.setter
    def up(self, value):
        """Sets and normalizes the up vector and updates the view matrix.

        Parameters
        ----------
        value : gamelib.Vec3 | Iterable
        """

        if isinstance(value, gamelib.Vec3):
            self._up = value.normalize()
        elif isinstance(value, np.ndarray):
            self._up = value.view(gamelib.Vec3).normalize()
        else:
            self._up = gamelib.Vec3(*value).normalize()
        if self._initialized:
            self._update_view()

    @property
    def down(self):
        """Gets the down vector.

        Returns
        -------
        gamelib.Vec3
        """

        return -self.up

    @property
    def right(self):
        """Gets the right vector.

        Returns
        -------
        gamelib.Vec3
        """

        return self.direction.cross(self.up)

    @property
    def left(self):
        """Gets the left vector.

        Returns
        -------
        gamelib.Vec3
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
    def proj_matrix(self):
        """Gets the projection matrix.

        Returns
        -------
        np.ndarray:
            4x4 projection matrix
        """

        return self._proj

    @proj_matrix.setter
    def proj_matrix(self, mat4):
        """Updates the projection matrix in place.

        Parameters
        ----------
        mat4 : array-like
            Most likely a matrix from geometry.transforms.Mat4 namespace.
        """

        self._proj[:] = mat4

    def set_primary(self):
        """Sets this as the primary camera. Other parts of the application that
        need a camera, but aren't explicilty given one will fallback to this
        one."""

        global _primary_camera
        _primary_camera = self

    def move(self, translation):
        """Offset current position by given translation.

        Parameters
        ----------
        translation : gamelib.Vec3 | Iterable
            xyz translation vector
        """

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
        position,
        direction,
        up=(0, 0, 1),
        fov_y=60,
        near=1,
        far=1000,
        controller=None,
    ):
        """Initialize the camera.

        Parameters
        ----------
        position : gamelib.Vec3 | Iterable
            xyz position in world space.
        direction : gamelib.Vec3 | Iterable
            xyz vector the camera is facing - applied locally at the camera.
        up : gamelib.Vec3 | Iterable
            xyz vector pointing to up in world space.
        fov_y : float
            y field of view, given in degrees.
        near : float
            Distance to the near clipping plane.
        far : float
            Distance to the far clipping plane.
        controller : type | bool
            Can be `True` to activate a default controller. See BaseCamera for
            more details otherwise.
        """

        self._fov_y = fov_y
        if controller is True:
            controller = _FreePerspectiveController
        super().__init__(position, up, direction, near, far, controller)

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
        """Get the near plane width in world units.

        Returns
        -------
        float
        """

        return self.near_plane_height * self._aspect_ratio

    @property
    def near_plane_height(self):
        """Get the near plane height in world units.

        Returns
        -------
        float
        """

        return 2 * math.tan(math.radians(self.fov_y / 2)) * self.near

    @property
    def near_plane_size(self):
        """Gets the near width and height in world units.

        Returns
        -------
        tuple[float, float]
        """

        return self.near_plane_width, self.near_plane_height

    def rotate(self, axis, theta):
        """Rotate the cameras direction vector.

        Parameters
        ----------
        axis : Vec3 | Iterable
            xyz vector representing the axis of rotation.
        theta : float
            The rotation angle given in degrees.
            (Right handed coordinate system)
        """

        matrix = transforms.Mat3.rotate_about_axis(axis, theta)
        self.direction = matrix.dot(self.direction)

    def screen_to_ray(self, x, y):
        """Convert screen space into a ray fired from the camera. Assumes a
        coordinate space for the screen where (0, 0) is the bottom left.

        Parameters
        ----------
        x : int
        y : int

        Returns
        -------
        collisions.Ray
        """

        vec_to_near_plane = transforms.normalize(self.direction) * self.near
        near_plane_center = self.position + vec_to_near_plane
        near_w, near_h = self.near_plane_size
        dx = (-0.5 + (x / gamelib.get_width())) * near_w
        dy = (-0.5 + (y / gamelib.get_height())) * near_h
        n_right = transforms.normalize(self.right)
        n_up = -transforms.normalize(np.cross(self.direction, n_right))
        x_y_to_near_plane = near_plane_center + n_up * dy + n_right * dx
        return collisions.Ray(self.position, x_y_to_near_plane - self.position)

    def cursor_to_ray(self):
        """Convenience method. Returns a ray from camera position towards the
        current cursor position.

        Returns
        -------
        collisions.Ray
        """

        return self.screen_to_ray(*gamelib.get_cursor())

    def _update_view(self):
        self.view_matrix = transforms.Mat4.look_at_transform(
            self.position, self.position + self.direction, self.up
        )

    def _update_proj(self):
        self.proj_matrix = transforms.Mat4.perspective_transform(
            self.fov_y, self._aspect_ratio, self.near, self.far
        )


class OrthogonalCamera(BaseCamera):
    """Simple implementation of a camera with an orthogonal projection."""

    def __init__(
        self,
        px_per_unit,
        position=(0, 0, 10),
        up=(0, 1, 0),
        direction=(0, 0, -1),
        controller=None,
    ):
        """Initialize the camera.

        Parameters
        ----------
        px_per_unit : float
            Ratio of screen pixels per world unit. This effectively controls
            the "zoom" of the camera.
        position : gamelib.Vec3 | Iterable
            xyz camera position in world space.
        up : gamelib.Vec3 | Iterable
            xyz vector pointing up in world space.
        direction : gamelib.Vec3 | Iterable
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
        super().__init__(position, up, direction, 1, 20, controller)
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
        self.proj_matrix = transforms.Mat4.orthogonal_transform(
            self._left,
            self._right,
            self._bottom,
            self._top,
            self._near,
            self._far,
        )

    def _update_view(self):
        self.view_matrix = transforms.Mat4.look_at_transform(
            self.position, self.position + self.direction, self.up
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
        vector.normalize()

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
        vector.normalize()

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
