from typing import Sequence

import numpy as np

from .rendering import gl


def _radians(theta):
    """Transform theta from degrees to radians."""

    return theta * np.pi / 180


def normalize(vector):
    """Normalize the given vector in place.

    Parameters
    ----------
    vector : np.ndarray

    Returns
    -------
    np.ndarray:
        returns vector for convenience.
    """

    magnitude = np.sqrt(np.sum(vector ** 2))
    if magnitude == 0:
        return vector
    vector /= magnitude
    return vector


class Mat3:
    """
    Namespace for 3x3 transformation matrices.

    Notes
    -----
    https://mathworld.wolfram.com/RotationMatrix.html
    https://mathworld.wolfram.com/RodriguesRotationFormula.html
    """

    @staticmethod
    def rotate_about_x(theta, dtype=gl.mat3):
        """Create a 3x3 rotation matrix.

        Parameters
        ----------
        theta : float
            Rotation angle given in degrees. (Right hand coordinate system)
        dtype : np.dtype | str
            np compatible dtype for return matrix

        Returns
        -------
        np.ndarray
        """

        theta = _radians(theta)
        return np.array(
            (
                (1, 0, 0),
                (0, np.cos(theta), np.sin(theta)),
                (0, -np.sin(theta), np.cos(theta)),
            ),
            dtype,
        ).T

    @staticmethod
    def rotate_about_y(theta, dtype=gl.mat3):
        """Create a 3x3 rotation matrix.

        Parameters
        ----------
        theta : float
            Rotation angle in degrees. (Right hand coordinate system)
        dtype : np.dtype | str
            np compatible dtype for matrix

        Returns
        -------
        np.ndarray
        """

        theta = _radians(theta)
        return np.array(
            (
                (np.cos(theta), 0, -np.sin(theta)),
                (0, 1, 0),
                (np.sin(theta), 0, np.cos(theta)),
            ),
            dtype,
        ).T

    @staticmethod
    def rotate_about_z(theta, dtype=gl.mat3):
        """Create a 3x3 rotation matrix.

        Parameters
        ----------
        theta : float
            Rotation angle in degrees. (Right hand coordinate system)
        dtype : np.dtype | str
            np compatible dtype for matrix

        Returns
        -------
        np.ndarray
        """

        theta = _radians(theta)
        return np.array(
            (
                (np.cos(theta), np.sin(theta), 0),
                (-np.sin(theta), np.cos(theta), 0),
                (0, 0, 1),
            ),
            dtype,
        ).T

    @staticmethod
    def rotate_about_axis(axis, theta, dtype=gl.mat3):
        """Create a 3x3 rotation matrix.

        Parameters
        ----------
        axis : Sequence
            3 component vector, the axis about which the rotation will occur.
        theta : float
            Rotation angle in degrees. (Right hand coordinate system)
        dtype : np.dtype | str
            np compatible dtype for matrix

        Returns
        -------
        np.ndarray
        """

        theta = _radians(theta)
        axis = np.asarray(axis, "f4")
        normalize(axis)

        cos = np.cos(theta)
        sin = np.sin(theta)
        k = 1 - cos
        x, y, z = axis

        return np.array(
            (
                (cos + x * x * k, x * y * k - z * sin, y * sin + x * z * k),
                (z * sin + x * y * k, cos + y * y * k, -x * sin + y * z * k),
                (-y * sin + x * z * k, x * sin + y * z * k, cos + z * z * k),
            ),
            dtype,
        )


class Mat4:
    """Namespace for 4x4 transformation matrices."""

    @staticmethod
    def look_at_transform(eye, look_at, up, dtype=gl.mat4):
        """Transform vertices as if viewed from eye, towards look_at.

        Parameters
        ----------
        eye : Sequence
            The xyz position of the "eye" or viewing a scene of vertices.
        look_at : Sequence
            The xyz position that the eye should look at.
        up : Sequence
            The xyz vector that points up.
        dtype : np.dtype | str
            Numpy compatible dtype for the return matrix.

        Returns
        -------
        np.ndarray

        Notes
        -----
        https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml
        """

        eye = np.asarray(eye, gl.float)
        look_at = np.asarray(look_at, gl.float)
        up = np.asarray(up, gl.float)

        forward = normalize(look_at - eye)
        right = normalize(np.cross(forward, up))
        up = normalize(np.cross(right, forward))

        return np.array(
            (
                (*right, -np.dot(eye, right)),
                (*up, -np.dot(eye, up)),
                (*-forward, np.dot(eye, forward)),
                (0, 0, 0, 1),
            ),
            dtype,
        ).T

    @staticmethod
    def perspective_transform(fovy, aspect, near, far, dtype=gl.mat4):
        """Create a 4x4 perspective projection matrix.

        Parameters
        ----------
        fovy : float
            Y direction field of view given in degrees.
        aspect : float
            Camera aspect ratio.
        near : float
            Distance to the near clipping plane.
        far : float
            Distance to the far clipping plane.
        dtype : np.dtype | str
            Numpy compatible dtype for the return matrix.

        Returns
        -------
        np.ndarray

        Notes
        -----
        https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
        """

        theta = fovy * np.pi / 360
        f = np.cos(theta) / np.sin(theta)
        a = near + far
        b = near - far
        c = near * far

        return np.array(
            (
                (f / aspect, 0, 0, 0),
                (0, f, 0, 0),
                (0, 0, a / b, 2 * c / b),
                (0, 0, -1, 0),
            ),
            dtype,
        ).T

    @staticmethod
    def orthogonal_transform(
            left, right, bottom, top, near, far, dtype=gl.mat4
    ):
        """Create a 4x4 orthogonal projection matrix.

        Parameters
        ----------
        left : float
            Left bounds of the projection.
        right : float
            Right bounds of the projection.
        bottom : float
            Bottom bounds of the projection.
        top : float
            Top bounds of the projection.
        near : float
            Distance to the near clipping plane.
        far : float
            Distance to the far clipping plane.
        dtype : np.ndarray | str
            Any Numpy compatible dtype for the return matrix.

        Returns
        -------
        np.ndarray

        Notes
        -----
        https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml
        """

        a = 2 / (right - left)
        b = 2 / (top - bottom)
        c = -2 / (far - near)
        x = (right + left) / (right - left)
        y = (top + bottom) / (top - bottom)
        z = (far + near) / (far - near)

        return np.array(
            ((a, 0, 0, x), (0, b, 0, y), (0, 0, c, z), (0, 0, 0, 1)), dtype
        ).T
