import numpy as np


class Model:
    def __init__(self, vertices, triangles, normals=None, anchor=None):
        self.vertices = vertices
        self.triangles = triangles
        self.normals = normals
        self.v_min, self.v_max = self._calculate_bounding_box()
        if anchor is not None:
            self.anchor(anchor)

    def anchor(self, anchor_location):
        ax, ay, az = anchor_location
        assert all(0.0 <= val <= 1.0 for val in anchor_location)
        box_size = tuple(
            max_val - min_val
            for max_val, min_val in zip(self.v_max, self.v_min)
        )
        anchor_point = tuple(
            min_val + (size_val * anchor_val)
            for min_val, size_val, anchor_val in zip(
                self.v_min, box_size, anchor_location
            )
        )
        diff = tuple(0 - val for val in anchor_point)
        self.vertices += diff
        self.v_max = tuple(cur + dif for cur, dif in zip(self.v_max, diff))
        self.v_min = tuple(cur + dif for cur, dif in zip(self.v_min, diff))

    def _calculate_bounding_box(self):
        min_x = np.min(self.vertices[:, 0])
        min_y = np.min(self.vertices[:, 1])
        min_z = np.min(self.vertices[:, 2])

        max_x = np.max(self.vertices[:, 0])
        max_y = np.max(self.vertices[:, 1])
        max_z = np.max(self.vertices[:, 2])

        return (min_x, min_y, min_z), (max_x, max_y, max_z)
