from typing import Sequence

import numpy as np

from gamelib import gl


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
    def rotate_about_x(theta, dtype=gl.float):
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
        # fmt: off
        return np.array((
            (1, 0, 0),
            (0, np.cos(theta), np.sin(theta)),
            (0, -np.sin(theta), np.cos(theta))),
            dtype,
        ).T
        # fmt: on

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
        # fmt: off
        return np.array((
            (np.cos(theta), 0, -np.sin(theta)),
            (0, 1, 0),
            (np.sin(theta), 0, np.cos(theta))),
            dtype,
        ).T
        # fmt: on

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
        # fmt: off
        return np.array((
            (np.cos(theta), np.sin(theta), 0),
            (-np.sin(theta), np.cos(theta), 0),
            (0, 0, 1)),
            dtype,
        ).T
        # fmt: on

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

        # fmt: off
        return np.array((
            (cos + x * x * k, x * y * k - z * sin, y * sin + x * z * k),
            (z * sin + x * y * k, cos + y * y * k, -x * sin + y * z * k),
            (-y * sin + x * z * k, x * sin + y * z * k, cos + z * z * k)),
            dtype,
        )
        # fmt: on


class Mat4:
    """Namespace for 4x4 transformation matrices. Note that these matrices
    are in column major are transposed to meet OpenGL expectations, use the
    Transform.apply method to transform numpy vectors in python."""

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

        # fmt: off
        return np.array((
            (*right, -np.dot(eye, right)),
            (*up, -np.dot(eye, up)),
            (*-forward, np.dot(eye, forward)),
            (0, 0, 0, 1)),
            dtype,
        ).T
        # fmt: on

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

        # fmt: off
        return np.array((
            (f / aspect, 0, 0, 0),
            (0, f, 0, 0),
            (0, 0, a / b, 2 * c / b),
            (0, 0, -1, 0)),
            dtype,
        ).T
        # fmt: on

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

        # fmt: off
        return np.array((
            (a, 0, 0, x), 
            (0, b, 0, y), 
            (0, 0, c, z), 
            (0, 0, 0, 1)), 
            dtype
        ).T
        # fmt: on

    @staticmethod
    def rotate_about_x(theta, dtype=gl.float):
        """4x4 rotation matrix about positive x axis.

        Parameters
        ----------
        theta : float
            Angle in degrees.
        dtype : Any, optional

        Returns
        -------
        np.ndarray
        """
        mat = np.identity(4, gl.float)
        mat[0:3, 0:3] = Mat3.rotate_about_x(theta, dtype)
        return mat.T

    @staticmethod
    def rotate_about_y(theta, dtype=gl.float):
        """"4x4 rotation matrix about the positive y axis.

        Parameters
        ----------
        theta : float
            Angle in degrees.
        dtype : Any, optional

        Returns
        -------
        np.ndarray
        """

        mat = np.identity(4, dtype)
        mat[0:3, 0:3] = Mat3.rotate_about_y(theta, dtype)
        return mat.T

    @staticmethod
    def rotate_about_z(theta, dtype=gl.float):
        """4x4 rotation matrix about the positive z axis.

        Parameters
        ----------
        theta : float
            Angle in degrees.
        dtype : Any, optional

        Returns
        -------
        np.ndarray
        """

        mat4 = np.identity(4, dtype)
        mat4[0:3, 0:3] = Mat3.rotate_about_z(theta, dtype)
        return mat4.T

    @staticmethod
    def rotate_about_axis(axis, theta, dtype=gl.float):
        """4x4 rotation matrix about an arbitrary 3 dimensional axis.

        Parameters
        ----------
        axis : Sequence
            A vector describing the rotation axis.
        theta : float
            Angle in degrees.
        dtype : Any, optional

        Returns
        -------
        np.ndarray
        """

        mat4 = np.identity(4, dtype)
        mat4[0:3, 0:3] = Mat3.rotate_about_axis(axis, theta, dtype)
        return mat4.T

    @staticmethod
    def scale(scale, dtype=gl.float):
        """4x4 scaling transformation matrix.

        Parameters
        ----------
        scale : Sequence
            Scale for each axis.
        dtype : Any, optional

        Returns
        -------
        np.ndarray
        """

        x, y, z = scale
        # fmt: off
        return np.array((
            (x, 0, 0, 0), 
            (0, y, 0, 0), 
            (0, 0, z, 0), 
            (0, 0, 0, 1)), 
            dtype
        ).T
        # fmt: on

    @staticmethod
    def translation(translation_vector, dtype=gl.float):
        """4x4 translating transformation matrix.

        Parameters
        ----------
        translation_vector : Sequence
        dtype : Any

        Returns
        -------
        np.ndarray
        """

        x, y, z = translation_vector
        # fmt: off
        return np.array((
            (1, 0, 0, x), 
            (0, 1, 0, y), 
            (0, 0, 1, z), 
            (0, 0, 0, 1)),
            dtype
        ).T
        # fmt: on


class Transform:
    """Combines translation, scale, and rotation matrices together into a
    single transformation matrix. The Mat4 matrices are transposed for
    OpenGL, this class has an apply method to apply those matrices to a
    numpy ndarray."""

    def __init__(self, pos=(0, 0, 0), scale=(1, 1, 1), axis=(0, 0, 1), theta=0):
        """Initialize the transform.

        Parameters
        ----------
        pos : Sequence
            xyz translation vector.
        scale : Sequence
            xyz scaling vector.
        axis : Sequence
            xyz rotation axis
        theta : float
            Rotation angle in degrees.
        """

        self._pos = pos
        self._scale = scale
        self._axis = axis
        self._theta = theta
        self._matrix = np.empty((4, 4), gl.float)
        self._update_matrix()

    def __repr__(self):
        return f"<Transform(pos={self.pos}, scale={self.scale}, axis={self.axis}, theta={self.theta})>"

    @property
    def pos(self):
        """Gets the current translation vector.

        Returns
        -------
        Sequence: (x, y, z)
        """

        return self._pos

    @pos.setter
    def pos(self, translation):
        """Sets the translation vector.

        Parameters
        ----------
        translation : Sequence, (x, y, z)
        """

        self._pos = translation
        self._update_matrix()

    @property
    def scale(self):
        """Gets the scale vector.

        Returns
        -------
        Sequence: (x, y, z)
        """

        return self._scale

    @scale.setter
    def scale(self, scale_vector):
        """Sets the scale vector and updates the matrix.

        Parameters
        ----------
        scale_vector : Sequence, (x, y, z)
        """

        self._scale = scale_vector
        self._update_matrix()

    @property
    def axis(self):
        """Gets the vector describing the rotation axis.

        Returns
        -------
        Sequence: (x, y, z)
        """

        return self._axis

    @axis.setter
    def axis(self, rotation_axis):
        """Set the rotation axis and update the matrix.
        Parameters
        ----------
        rotation_axis : Sequence, (x, y, z)
        """

        self._axis = rotation_axis
        self._update_matrix()

    @property
    def theta(self):
        """Gets the current rotation angle.

        Returns
        -------
        float: (degrees)
        """

        return self._theta

    @theta.setter
    def theta(self, degrees):
        """Set the rotation angle and update the matrix.

        Parameters
        ----------
        degrees : float
        """

        self._theta = degrees
        self._update_matrix()

    @property
    def matrix(self):
        """Gets the current transformation matrix. This is updated whenever
        one of the transformation attributes are changed.

        Returns
        -------
        np.ndarray:
            4x4 translation matrix transposed for OpenGL
        """

        return self._matrix

    def apply(self, vertex):
        """Apply a Transform to a particular vertex.

        Parameters
        ----------
        vertex : np.ndarray
            Length 3 or 4 supported.

        Returns
        -------
        np.ndarray:
            Returns the input vertex, having been transformed.
        """

        matrix = self._get_transpose()
        dtype = vertex.dtype
        if len(vertex) == 3:
            tmp = np.zeros(4, dtype)
            tmp[0:3] = vertex
            tmp[3] = 1
            transformed = matrix.dot(tmp)[:3]
            if np.issubdtype(dtype, np.integer):
                transformed = np.rint(transformed)
            vertex[:] = transformed
            return vertex
        elif len(vertex) == 4:
            transformed = matrix.dot(vertex)
            if np.issubdtype(dtype, np.integer):
                transformed = np.rint(transformed)
            vertex[:] = transformed
            return vertex
        else:
            raise ValueError(
                f"Expected vertex of length 3/4, instead got {len(vertex)}."
            )

    def _update_matrix(self):
        """Updates the OpenGL matrix."""

        self._matrix[:] = Mat4.translation(self.pos).dot(
            Mat4.rotate_about_axis(self.axis, self.theta).dot(
                Mat4.scale(self.scale)
            )
        )

    def _get_transpose(self):
        """Constructs a transposed matrix for use against numpy ndarrays."""

        return Mat4.translation(self.pos).T.dot(
                Mat4.rotate_about_axis(self.axis, self.theta).T.dot(
                    Mat4.scale(self.scale).T
            )
        )
