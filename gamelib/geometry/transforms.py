from gamelib import Vec3

import numpy as np

from gamelib.core import gl
from gamelib.geometry import base


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
        Returns vector for convenience.
    """

    length, *shape = vector.shape
    if shape:
        magnitudes = np.sqrt(np.sum(vector * vector, axis=1))
        not0 = magnitudes != 0
        copy = vector.copy().T
        copy[not0] /= magnitudes[not0]
        vector[:] = copy.T
        return vector

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
    def identity():
        return np.identity(3, gl.float)

    @staticmethod
    def rotate_about_x(theta, dtype=gl.float):
        """Create a 3x3 rotation matrix about the positive x-axis.

        Parameters
        ----------
        theta : float
            Rotation angle given in degrees. (Right hand coordinate system)
        dtype : np.dtype | str
            Numpy compatible dtype for the returned matrix

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
        """Create a 3x3 rotation matrix about the positive y-axis.

        Parameters
        ----------
        theta : float
            Rotation angle in degrees. (Right hand coordinate system)
        dtype : np.dtype | str
            Numpy compatible dtype for the returned matrix.

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
        """Create a 3x3 rotation matrix about the positive z-axis.

        Parameters
        ----------
        theta : float
            Rotation angle in degrees. (Right hand coordinate system)
        dtype : np.dtype | str
            Numpy compatible dtype for the returned matrix.

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
            XYZ vector about which the rotation should occur.
        theta : float
            Rotation angle in degrees. (Right hand coordinate system)
        dtype : np.dtype | str
            Numpy compatible dtype for the returned matrix.

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
    are transposed for OpenGL. Use the transform.apply method to transform
    numpy vectors."""

    @staticmethod
    def identity():
        return np.identity(4, gl.float)

    @staticmethod
    def look_at_transform(eye, look_at, up, dtype=gl.mat4):
        """Transform vertices as if viewed from eye, towards look_at.

        Parameters
        ----------
        eye : Sequence
            XYZ position of the "eye" for viewing a scene of vertices.
        look_at : Sequence
            XYZ position that the eye should be looking at.
        up : Sequence
            XYZ vector describing world space 'up' direction.
        dtype : np.dtype | str
            Numpy compatible dtype for the returned matrix.

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
            Numpy compatible dtype for the returned matrix.

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
        """4x4 rotation matrix about the positive x axis.

        Parameters
        ----------
        theta : float
            Angle measured in degrees.
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
        """4x4 rotation matrix about the positive y axis.

        Parameters
        ----------
        theta : float
            Angle measured in degrees.
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
            Angle measured in degrees.
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
            XYZ vector describing the rotation axis.
        theta : float
            Angle measured in degrees.
        dtype : Any, optional

        Returns
        -------
        np.ndarray
        """

        mat4 = np.identity(4, dtype)
        mat4[0:3, 0:3] = Mat3.rotate_about_axis(axis, theta, dtype)
        return mat4.T

    @staticmethod
    def scale(scale_vector, dtype=gl.float):
        """4x4 scaling transformation matrix.

        Parameters
        ----------
        scale_vector : Sequence
            Scale for each axis.
        dtype : Any, optional

        Returns
        -------
        np.ndarray
        """

        x, y, z = scale_vector
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

    @classmethod
    def model_transform(
        cls, translation=(0, 0, 0), scale=(1, 1, 1), axis=(0, 0, 1), theta=0.0
    ):
        scale = cls.scale(scale)
        rotation = cls.rotate_about_axis(axis, theta)
        translation = cls.translation(translation)
        return scale.dot(rotation).dot(translation)


class Transform:
    """Combines translation, scale, and rotation matrices together into a
    single transformation matrix. The Mat4 matrices are transposed for
    OpenGL, this class has an apply method to apply those matrices to a
    Numpy ndarray."""

    def __init__(
        self, pos=(0, 0, 0), scale=(1, 1, 1), axis=(0, 0, 1), theta=0
    ):
        """Initialize the transform.

        Parameters
        ----------
        pos : Sequence
            XYZ translation vector.
        scale : Sequence
            XYZ scaling vector.
        axis : Sequence
            XYZ rotation axis
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
    def _inverse_matrix(self):
        return np.linalg.inv(self.matrix.T)

    @property
    def pos(self):
        """Gets the current translation vector.

        Returns
        -------
        Sequence
        """

        return self._pos

    @pos.setter
    def pos(self, translation):
        """Sets the translation vector and updated the matrix.

        Parameters
        ----------
        translation : Sequence
        """

        self._pos = translation
        self._update_matrix()

    @property
    def scale(self):
        """Gets the scale vector.

        Returns
        -------
        Sequence
        """

        return self._scale

    @scale.setter
    def scale(self, scale_vector):
        """Sets the scale vector and updates the matrix.

        Parameters
        ----------
        scale_vector : Sequence
        """

        self._scale = scale_vector
        self._update_matrix()

    @property
    def axis(self):
        """Gets the vector describing the rotation axis.

        Returns
        -------
        Sequence
        """

        return self._axis

    @axis.setter
    def axis(self, rotation_axis):
        """Set the rotation axis and update the matrix.

        Parameters
        ----------
        rotation_axis : Sequence
        """

        self._axis = rotation_axis
        self._update_matrix()

    @property
    def theta(self):
        """Gets the current rotation angle measured in degrees.

        Returns
        -------
        float
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

    def apply(self, target, *, normal=False):
        """Apply a Transform to a particular vertex.

        Parameters
        ----------
        target : np.ndarray | base.Model
            Length 3 or 4 supported.

        Returns
        -------
        np.ndarray | None:
            Returns the input vertex, having been transformed.
            Returns None if transform target is a Model.
        """

        return apply_transform(self.matrix.T, target, normal)

    def apply_inverse(self, target, *, normal=False):

        return apply_transform(self._inverse_matrix, target, normal)

    def _update_matrix(self):
        """Updates the OpenGL matrix."""

        self._matrix[:] = Mat4.model_transform(
            self.pos, self.scale, self.axis, self.theta
        )


def apply_transform(matrix, target, normal=False):
    if isinstance(target, base.Model):
        _transform_model(matrix, target)
    else:
        return _transform_vertex(matrix, target, normal)


def _transform_model(matrix, model):
    vcopy = model.vertices.copy()
    vpad = np.ones((len(vcopy), 1), vcopy.dtype)
    transformed = np.hstack((vcopy, vpad)).dot(matrix.T)
    model.vertices[:] = np.delete(transformed, 3, 1)

    if model.normals is not None:
        ncopy = model.normals.copy()
        npad = np.zeros((len(ncopy), 1), ncopy.dtype)
        transformed = np.hstack((ncopy, npad)).dot(matrix.T)
        model.normals[:] = np.delete(transformed, 3, 1)
        normalize(model.normals)

    model.recalculate_boundaries()


def _transform_vertex(matrix, vertex, normal):
    """Expecting a Mat4 matrix and a vec3/vec4 vertex."""

    if isinstance(vertex, Vec3):
        arr = np.asarray(tuple(vertex), float)
        transformed = _transform_vertex(matrix, arr, normal)
        return Vec3(*transformed)

    dtype = vertex.dtype
    if len(vertex) == 3:
        len4_temp = np.zeros(4, dtype)
        len4_temp[0:3] = vertex
        len4_temp[3] = 0 if normal else 1
        transformed = matrix.dot(len4_temp)[:3]
        if np.issubdtype(dtype, np.integer):
            transformed = np.rint(transformed)
        if normal:
            transformed = normalize(transformed)
        vertex[:] = transformed
        return vertex
    elif len(vertex) == 4:
        transformed = matrix.dot(vertex)
        if np.issubdtype(dtype, np.integer):
            transformed = np.rint(transformed)
        if normal:
            transformed = normalize(transformed)
        vertex[:] = transformed
        return vertex
    else:
        raise ValueError(
            f"Expected vertex of length 3/4, instead got {len(vertex)}."
        )
