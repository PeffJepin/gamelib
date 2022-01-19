import math

import pytest
import numpy as np

from gamelib.geometry import collisions
from gamelib import gl


@pytest.mark.parametrize(
    "box_min, box_max, ray_origin, ray_dir, intersects",
    (
        ((0, 0, 0), (1, 1, 1), (-1, -1, -1), (1, 1, 1), True),
        ((0, 0, 0), (1, 1, 1), (0.5, 0.5, 0.5), (123, 321, 1000), True),
        ((0, 0, 0), (1, 1, 1), (1.5, 1.5, 1.5), (123, 321, 1000), False),
        ((0, 0, 0), (1, 1, 1), (-0.5, 0.5, 0.5), (1, 2, 0), False),
    ),
)
def test_ray_aabb_intersection(
    box_min, box_max, ray_origin, ray_dir, intersects
):
    ray = collisions.Ray(ray_origin, ray_dir)

    assert ray.intersects_aabb(box_min, box_max) is intersects


@pytest.mark.parametrize(
    "ray_origin, ray_dir, intersects_tri1, intersects_tri2",
    (
        ((0.5, -1, 0.1), (0, 1, 0), True, False),
        ((-0.5, -1, 0.1), (0, 1, 0), False, True),
        ((-10.5, -1, 0.1), (0, 1, 0), False, False),
    ),
)
def test_ray_triangles_intersection(
    ray_origin, ray_dir, intersects_tri1, intersects_tri2
):
    tri1 = np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1)], gl.vec3)
    tri2 = np.array([(0, 0, 0), (-1, 0, 0), (-1, 0, 1)], gl.vec3)
    triangles = np.stack((tri1, tri2), 0)

    result = collisions.ray_triangle_intersections(
        triangles=triangles, origin=ray_origin, dir=ray_dir
    )

    if intersects_tri1:
        assert result[0] != -1
    else:
        assert result[0] == -1

    if intersects_tri2:
        assert result[1] != -1
    else:
        assert result[1] == -1


@pytest.mark.parametrize(
    "ray_origin, ray_dir, distance",
    (
        ((1, -1, 1), (0, 1, 0), 1),
        ((1, -0.5, 1), (0, 1, 0), 0.5),
        ((1, -1, 2), (0, 1, -1), math.sqrt(2)),
    ),
)
def test_ray_triangles_intersection_distance(ray_origin, ray_dir, distance):
    triangle = np.array([(0, 0, 0), (2, 0, 0), (1, 0, 2)], gl.vec3)
    triangle = triangle.reshape(1, 3, 3)

    result = collisions.ray_triangle_intersections(
        triangle, ray_origin, ray_dir
    )

    assert result[0] == pytest.approx(distance)
