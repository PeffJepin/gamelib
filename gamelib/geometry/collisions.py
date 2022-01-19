import math

import numpy as np

from gamelib import gl


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.dir = direction
        self.inv_dir = self._inverse(direction)

    def intersects_aabb(self, bmin, bmax):
        tx1 = (bmin[0] - self.origin[0]) * self.inv_dir[0]
        tx2 = (bmax[0] - self.origin[0]) * self.inv_dir[0]
        tmin = min(tx1, tx2)
        tmax = max(tx1, tx2)

        ty1 = (bmin[1] - self.origin[1]) * self.inv_dir[1]
        ty2 = (bmax[1] - self.origin[1]) * self.inv_dir[1]
        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))

        tz1 = (bmin[2] - self.origin[2]) * self.inv_dir[2]
        tz2 = (bmax[2] - self.origin[2]) * self.inv_dir[2]
        tmin = max(tmin, min(tz1, tz2))
        tmax = min(tmax, max(tz1, tz2))

        return tmax >= tmin

    def _inverse(self, direction):
        inv = [0, 0, 0]
        for i, comp in enumerate(direction):
            try:
                inv[i] = 1 / comp
            except ZeroDivisionError:
                sign = -1 if comp < 0 else 1
                inv[i] = sign * math.inf
        return tuple(inv)


def ray_triangle_intersections(triangles, origin, dir):
    if not isinstance(origin, np.ndarray):
        origin = np.array(tuple(origin), gl.vec3)
    if not isinstance(dir, np.ndarray):
        dir = np.array(tuple(dir), gl.vec3)
    dir_mag = np.sqrt(np.sum(dir ** 2))

    # vertices
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    # edges
    e1 = v1 - v0
    e2 = v2 - v0

    # Möller–Trumbore intersection algorithm
    EPSILON = 0.0000001
    h = np.cross(dir, e2[:])
    a = np.sum(e1 * h, axis=1)
    f = 1 / a
    s = origin - v0
    u = f * np.sum(s * h, axis=1)
    q = np.cross(s, e1)
    v = f * np.sum(dir * q[:], axis=1)
    t = f * np.sum(e2 * q, axis=1)
    misses = (
        ((a < EPSILON) & (a > -EPSILON))
        | ((u < 0) | (u > 1))
        | ((v < 0) | (u + v > 1))
        | (t <= EPSILON)
    )

    result = t * dir_mag
    result[misses] = -1
    return result
