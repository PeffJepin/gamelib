from gamelib import Vec3
import numpy as np


class Model:
    """A minimal representation of some 3d geometry. API subject to change."""

    def __init__(self, vertices, indices, normals=None, anchor=None):
        vertices = np.asarray(vertices)
        length, *shape = vertices.shape
        if not shape:
            vertices = vertices.reshape(-1, 3)

        indices = np.asarray(indices)
        length, *shape = indices.shape
        if not shape:
            indices = indices.reshape(-1, 3)

        self.vertices = vertices
        self.indices = indices
        self.normals = normals
        self.v_min, self.v_max = self._calculate_bounding_box()
        if anchor is not None:
            self.anchor(anchor)

    @property
    def triangles(self):
        return self.vertices[self.indices]

    def anchor(self, relative_anchor):
        """Anchors the model based on its bounding box.

        Parameters
        ----------
        relative_anchor : Vec3 | Iterable
            (0, 0, 0) would translate the model such that the minimum point of
                the bounding box would be positioned on the origin.
            (1, 1, 1) would translate the model such that the maximum point of
                the bounding box would be positioned on the origin.
        """

        assert all(0.0 <= val <= 1.0 for val in relative_anchor)

        if not isinstance(relative_anchor, Vec3):
            relative_anchor = Vec3(relative_anchor)

        shape = self.v_max - self.v_min
        anchor_point = self.v_min + shape * relative_anchor
        diff = 0 - anchor_point
        self.vertices += diff
        self.v_min += diff
        self.v_max += diff

    def recalculate_boundaries(self):
        self.v_min, self.v_max = self._calculate_bounding_box()

    def _calculate_bounding_box(self):
        min_x = np.min(self.vertices[:, 0])
        min_y = np.min(self.vertices[:, 1])
        min_z = np.min(self.vertices[:, 2])

        max_x = np.max(self.vertices[:, 0])
        max_y = np.max(self.vertices[:, 1])
        max_z = np.max(self.vertices[:, 2])

        return Vec3(min_x, min_y, min_z), Vec3(max_x, max_y, max_z)
