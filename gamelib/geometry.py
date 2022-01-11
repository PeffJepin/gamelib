import dataclasses

from typing import Optional
import numpy as np
from gamelib import gl
from gamelib import _obj


@dataclasses.dataclass
class Geometry:
    vertices: np.ndarray
    normals: Optional[np.ndarray]
    triangles: np.ndarray
    

def parse_file(path):
    if path.name.endswith(".obj"):
        return _obj.parse(path)
    raise ValueError(f"File format for {path=} not supported.")


class GridMesh:
    def __init__(self, lod=1, scale=1):
        """Creates a quad subdivided `lod` number of times and scaled up
        based on given scale. Z = 0 resulting in a plane along the x-y axis.

        Parameters
        ----------
        lod : int
            How many times to subdivide the plane.
        scale : float
            How much space the place occupies. Scale 1000 == 1000x1000
        """

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


class Cube:
    """A very simple cube for testing transforms."""

    def __init__(self):
        # fmt: off
        self.vertices = np.array([
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),  # z=0 quad
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],  # z=1 quad
            gl.vec3,
        )
        self.vertices -= 0.5
        self.indices = np.array([
            0, 2, 1, 0, 3, 2,  # -z face
            4, 7, 6, 4, 6, 5,  # +z face
            3, 7, 4, 3, 4, 0,  # -x face
            1, 5, 6, 1, 6, 2,  # +x face
            0, 4, 5, 0, 5, 1,  # -y face
            2, 6, 7, 2, 7, 3,  # +y face
        ])
        # fmt: on
