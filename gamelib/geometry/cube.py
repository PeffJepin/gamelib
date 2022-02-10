import numpy as np

from gamelib.core import gl
from gamelib.geometry import base


class Cube(base.Model):
    """A very simple cube for testing transforms."""

    def __init__(self, scale=1, **kwargs):
        # fmt: off
        vertices = np.array([
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),  # z=0 quad
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],  # z=1 quad
            gl.vec3,
        )
        vertices -= 0.5
        vertices *= scale
        triangles = np.array([
            0, 2, 1, 0, 3, 2,  # -z face
            4, 7, 6, 4, 6, 5,  # +z face
            3, 7, 4, 3, 4, 0,  # -x face
            1, 5, 6, 1, 6, 2,  # +x face
            0, 4, 5, 0, 5, 1,  # -y face
            2, 6, 7, 2, 7, 3,  # +y face
        ])
        # fmt: on:
        super().__init__(vertices, triangles, **kwargs)
