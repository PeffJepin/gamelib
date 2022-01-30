import numpy as np

from gamelib import gl
from gamelib.ecs import base
from gamelib.geometry import transforms


class Transform(base.Component):
    """Like Transform but implemented as an ecs Component. See Transform if
    more documentation is needed."""

    _pos: gl.vec3
    _scale: gl.vec3
    _axis: gl.vec3
    _theta: gl.float
    model_matrix: gl.mat4

    def __init__(
            self, pos=(0, 0, 0), scale=(1, 1, 1), axis=(0, 0, 1), theta=0
    ):
        super().__init__()
        self._pos = pos
        self._scale = scale
        self._axis = axis
        self._update_matrix()
        self._theta = theta

    @property
    def _inverse_matrix(self):
        return np.linalg.inv(self.model_matrix.T)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self._update_matrix()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        self._update_matrix()

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, value):
        self._axis = value
        self._update_matrix()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        self._update_matrix()

    def apply(self, target, *, normal=False):
        return transforms.apply_transform(self.model_matrix.T, target, normal)

    def apply_inverse(self, target, *, normal=False):
        return transforms.apply_transform(self._inverse_matrix, target, normal)

    def _update_matrix(self):
        self.model_matrix = transforms.Mat4.model_transform(
            self.pos, self.scale, self.axis, self.theta
        )
