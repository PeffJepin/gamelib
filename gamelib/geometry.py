import numpy as np


class GridMesh:
    def __init__(self, lod=1, scale=1):
        x = y = np.linspace(0, scale, lod + 1)
        xv, yv = np.meshgrid(x, y)
        self.vertices = np.empty(xv.size * 3, float)
        self.vertices[0::3] = xv.flatten()
        self.vertices[1::3] = yv.flatten()
        self.vertices[2::3] = 0

        num_quads = lod * lod
        order = np.array((0, lod + 1, lod + 2, 0, lod + 2, 1))
        self.indices = np.empty(order.size * num_quads)
        ptr = 0
        for x in range(lod):
            for y in range(lod):
                index = (y * (lod + 1)) + x
                self.indices[ptr : ptr + order.size] = order + index
                ptr += order.size


def _cotangent(x):
    # x should be in radians
    return np.cos(x) / np.sin(x)


def _radians(theta):
    return theta * np.pi / 180


def normalize(vector: np.ndarray):
    vector /= np.sqrt(np.sum(vector ** 2))
    return vector


class Mat3:
    # https://mathworld.wolfram.com/RotationMatrix.html
    @staticmethod
    def rotate_about_x(theta, dtype="f4"):
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
    def rotate_about_y(theta, dtype="f4"):
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
    def rotate_about_z(theta, dtype="f4"):
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
    def rotate_about_axis(axis, theta, dtype="f4"):
        # https://mathworld.wolfram.com/RodriguesRotationFormula.html

        theta = _radians(theta)
        axis = np.asarray(axis, dtype)
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
    @staticmethod
    def look_at_transform(eye, look_at, up, dtype="f4"):
        # https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml

        eye = np.asarray(eye, "f4")
        look_at = np.asarray(look_at, "f4")
        up = np.asarray(up, "f4")

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
    def perspective_transform(fovy, aspect, near, far, dtype="f4"):
        # https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml

        f = _cotangent(fovy * np.pi / 360)
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
    def orthogonal_transform(left, right, bottom, top, near, far, dtype="f4"):
        # https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml

        a = 2 / (right - left)
        b = 2 / (top - bottom)
        c = -2 / (far - near)
        x = (right + left) / (right - left)
        y = (top + bottom) / (top - bottom)
        z = (far + near) / (far - near)

        return np.array(
            ((a, 0, 0, x), (0, b, 0, y), (0, 0, c, z), (0, 0, 0, 1)), dtype
        ).T