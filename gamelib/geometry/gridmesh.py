import numpy as np

from gamelib.core import gl
from gamelib.geometry import base


class GridMesh(base.Model):
    def __init__(self, lod=1, scale=1, **kwargs):
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
        vertices = np.empty(xv.size * 3, gl.float)
        vertices[0::3] = xv.flatten()
        vertices[1::3] = yv.flatten()
        vertices[2::3] = 0

        num_quads = lod * lod
        order = np.array((0, lod + 1, lod + 2, 0, lod + 2, 1))
        triangles = np.empty(order.size * num_quads, gl.uint)
        ptr = 0
        for x in range(lod):
            for y in range(lod):
                index = (y * (lod + 1)) + x
                triangles[ptr : ptr + order.size] = order + index
                ptr += order.size

        vertices = gl.coerce_array(vertices, gl.vec3)
        triangles = gl.coerce_array(triangles, gl.uvec3)
        super().__init__(vertices, triangles, **kwargs)
