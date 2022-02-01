import math

import pytest
import numpy as np

from gamelib.geometry import collisions
from gamelib.geometry import transforms
from gamelib.geometry import base
from gamelib.geometry import gridmesh
from gamelib import gl

from ..conftest import assert_approx


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
    triangle = triangle.reshape((1, 3, 3))

    result = collisions.ray_triangle_intersections(
        triangle, ray_origin, ray_dir
    )

    assert result[0] == pytest.approx(distance)


@pytest.mark.parametrize(
    "triangle, intersects",
    (
        # fmt: off
        # edge intersections
        (np.array([(4, 2, 4), (2, 4, 4), (2, 2, 4)], gl.vec3), True),
        (np.array([(2, 4, 4), (4, 6, 4), (2, 6, 4)], gl.vec3), True),
        (np.array([(4, 6, 4), (6, 4, 4), (6, 6, 4)], gl.vec3), True),
        (np.array([(6, 4, 4), (4, 2, 4), (6, 2, 4)], gl.vec3), True),

        (np.array([(4, 4, 2), (2, 4, 4), (2, 4, 2)], gl.vec3), True),
        (np.array([(2, 4, 4), (4, 4, 6), (2, 4, 6)], gl.vec3), True),
        (np.array([(4, 4, 6), (6, 4, 4), (6, 4, 6)], gl.vec3), True),
        (np.array([(6, 4, 4), (4, 4, 2), (6, 4, 2)], gl.vec3), True),

        (np.array([(4, 4, 2), (4, 2, 4), (4, 2, 2)], gl.vec3), True),
        (np.array([(4, 2, 4), (4, 4, 6), (4, 2, 6)], gl.vec3), True),
        (np.array([(4, 4, 6), (4, 6, 4), (4, 6, 6)], gl.vec3), True),
        (np.array([(4, 6, 4), (4, 4, 2), (4, 6, 2)], gl.vec3), True),

        # vertex / face intersections
        (np.array([(2, 2, 4), (4, 4, 4), (6, 2, 4)], gl.vec3), True),
        (np.array([(2, 2, 4), (4, 4, 4), (2, 6, 4)], gl.vec3), True),
        (np.array([(2, 6, 4), (4, 4, 4), (6, 6, 4)], gl.vec3), True),
        (np.array([(6, 6, 4), (4, 4, 4), (6, 2, 4)], gl.vec3), True),
        (np.array([(2, 4, 6), (4, 4, 4), (6, 4, 6)], gl.vec3), True),
        (np.array([(2, 4, 2), (4, 4, 4), (6, 4, 2)], gl.vec3), True),

        # triangle face intersection
        (np.array([(0.9, 4.9, 2.9), (4.9, 0.9, 2.9), (4, 4, 7)], gl.vec3), True),

        # some near misses
        (np.array([(2.9, 2.9, 2.9), (2, 2, 2), (1, 1, 1)], gl.vec3), False),
        (np.array([(5.1, 5.1, 5.1), (6, 6, 6), (7, 7, 7)], gl.vec3), False),
        # fmt: on
    ),
)
def test_aabb_triangle_intersections(triangle, intersects):
    aabb = collisions.AABB((3, 3, 3), (5, 5, 5))
    always_intersects = np.array([(4, 4, 4), (3, 3, 3), (5, 5, 5)], gl.vec3)
    never_intersects = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2)], gl.vec3)

    triangles = np.stack((always_intersects, never_intersects, triangle), 0)
    expected = np.array([True, False, intersects], bool)

    actual = collisions.aabb_triangle_intersections(aabb, triangles)
    assert np.all(expected == actual)


def test_zone_does_not_grow_to_accomidate_clipped_triangles():
    # fmt: off
    vertices = np.array([
        # clips proposed area
        (-1.4, -1.4, -1.4), (-1.2, -1.2, -1.2), (-1, -1, -1),
        (1.4, 1.4, 1.4), (1.2, 1.2, 1.2), (1, 1, 1),
        # outside proposed area
        (-2.4, -2.4, -2.4), (-2.2, -2.2, -2.2), (-2, -2, -2),
        (2.4, 2.4, 2.4), (2.2, 2.2, 2.2), (2, 2, 2),
    ], gl.vec3)
    indices = np.array([
        # clipping triangles
        (0, 1, 2), (3, 4, 5),
        # outside triangles
        (6, 7, 8), (9, 10, 11)
    ], gl.uvec3)
    # fmt: on

    start_min, start_max = (-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)
    aabb = collisions.AABB(start_min, start_max)
    clipped = collisions.BVH_Helper.clamp_aabb(aabb, vertices, indices)
    clipped_indices = clipped

    assert_approx(aabb.min, start_min)
    assert_approx(aabb.max, start_max)
    assert np.all(clipped_indices == indices[:2])


def test_zone_does_shrink_when_not_clipping_triangles():
    vertices = np.array(
        [
            (-1.5, -1, -1),
            (-1, -1, -1),
            (0, 0, 0),
            (1.2, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
        ],
        gl.vec3,
    )
    indices = np.array([(0, 1, 2), (3, 4, 5)], gl.uvec3)
    aabb = collisions.AABB((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))
    clipped = collisions.BVH_Helper.clamp_aabb(aabb, vertices, indices)
    clipped_indices = clipped

    assert_approx((-1.5, -1, -1), aabb.min)
    assert_approx((1.2, 1, 1), aabb.max)
    assert np.all(clipped_indices == indices[:2])


class TestBVH:
    def test_creating_from_geometry(self):
        # fmt: off
        vertices = np.array([
            # -x, -y, -z
            (-2.5, -2.5, -2.5), (-1.5, -1.5, -1.5), (-2.0, -2.0, -2.0), (-1.5, -2.5, -2.5),
            # +x, -y, -z
            (2.5, -2.5, -2.5),  (1.5, -1.5, -1.5),  (2.0, -2.0, -2.0),  (1.5, -2.5, -2.5),
            # -x, +y, -z
            (-2.5, 2.5, -2.5),  (-1.5, 1.5, -1.5),  (-2.0, 2.0, -2.0),  (-1.5, 2.5, -2.5),
            # -x, -y, +z
            (-2.5, -2.5, 2.5),  (-1.5, -1.5, 1.5),  (-2.0, -2.0, 2.0),  (-1.5, -2.5, 2.5),
            # +x, +y, -z
            (2.5, 2.5, -2.5),   (1.5, 1.5, -1.5),   (2.0, 2.0, -2.0),   (1.5, 2.5, -2.5),
            # +x, +y, +z
            (2.5, 2.5, 2.5),    (1.5, 1.5, 1.5),    (2.0, 2.0, 2.0),    (1.5, 2.5, 2.5),
            # +x, -y, +z
            (2.5, -2.5, 2.5),   (1.5, -1.5, 1.5),   (2.0, -2.0, 2.0),   (1.5, -2.5, 2.5),
            # -x, +y, +z
            (-2.5, 2.5, 2.5),   (-1.5, 1.5, 1.5),   (-2.0, 2.0, 2.0),   (-1.5, 2.5, 2.5),
        ], dtype=gl.vec3)
        triangle_indices = np.array([
            # -x, -y, -z
            (0, 1, 2),    (0, 1, 3),    (3, 2, 1),    (0, 2, 3),
            # +x, -y, -z
            (4, 5, 6),    (4, 5, 7),    (7, 6, 5),    (4, 6, 7),
            # -x, +y, -z
            (8, 9, 10),   (8, 9, 11),   (11, 10, 9),  (8, 10, 11),
            # -x, -y, +z
            (12, 13, 14), (12, 13, 15), (15, 14, 13), (12, 14, 15),
            # +x, +y, -z
            (16, 17, 18), (16, 17, 19), (19, 18, 17), (16, 18, 19),
            # +x, +y, +z
            (20, 21, 22), (20, 21, 23), (23, 22, 21), (20, 22, 23),
            # +x, -y, +z
            (24, 25, 26), (24, 25, 27), (27, 26, 25), (24, 26, 27),
            # -x, +y, +z
            (28, 29, 30), (28, 29, 31), (31, 30, 29), (28, 30, 31),
        ], dtype=gl.uvec3)

        model = base.Model(vertices, triangle_indices)
        bvh = collisions.BVH.create_tree(model, target_density=4)

        expected_leaves = [
            collisions.BVH(collisions.AABB((-2.5, -2.5, -2.5), (-1.5, -1.5, -1.5)),indices=triangle_indices[0:4]),
            collisions.BVH(collisions.AABB((1.5, -2.5, -2.5), (2.5, -1.5, -1.5)),  indices=triangle_indices[4:8]),
            collisions.BVH(collisions.AABB((-2.5, 1.5, -2.5), (-1.5, 2.5, -1.5)),  indices=triangle_indices[8:12]),
            collisions.BVH(collisions.AABB((-2.5, -2.5, 1.5), (-1.5, -1.5, 2.5)),  indices=triangle_indices[12:16]),
            collisions.BVH(collisions.AABB((1.5, 1.5, -2.5),  (2.5, 2.5, -1.5)),   indices=triangle_indices[16:20]),
            collisions.BVH(collisions.AABB((1.5, 1.5, 1.5),   (2.5, 2.5, 2.5)),    indices=triangle_indices[20:24]),
            collisions.BVH(collisions.AABB((1.5, -2.5, 1.5),  (2.5, -1.5, 2.5)),   indices=triangle_indices[24:28]),
            collisions.BVH(collisions.AABB((-2.5, 1.5, 1.5),  (-1.5, 2.5, 2.5)),   indices=triangle_indices[28:32]),
        ]
        for node in bvh:
            if node.indices is not None:
                for i, exp in enumerate(expected_leaves.copy()):
                    if node.aabb == exp.aabb and np.all(node.indices == exp.indices):
                        expected_leaves.pop(i)
                        break

        # should have found all the expected leaves
        assert len(expected_leaves) == 0
        # fmt: on

    def test_triangles_leaf_node(self):
        vertices = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], gl.vec3)
        indices = np.array([(0, 1, 2)], gl.uvec3)
        node = collisions.BVH(
            collisions.AABB((0, 0, 0), (1, 1, 1)),
            vertices=vertices,
            indices=indices,
        )

        assert np.all(vertices[indices] == node.triangles)

    def test_triangles_not_leaf(self):
        node = collisions.BVH(collisions.AABB((0, 0, 0), (1, 1, 1)))

        assert node.triangles is None

    def test_number_of_triangles(self):
        vertices = gl.coerce_array(np.arange(180), gl.vec3)
        indices = gl.coerce_array(np.arange(60), gl.uvec3)
        model = base.Model(vertices, indices)
        root = collisions.BVH.create_tree(model, target_density=16)

        assert root.ntris == 20

    def test_aabb_leaf_vectors(self):
        vertices = gl.coerce_array(np.arange(180), gl.vec3)
        indices = gl.coerce_array(np.arange(60), gl.uvec3)
        model = base.Model(vertices, indices)
        root = collisions.BVH.create_tree(model, target_density=4)

        bmin = root.leaf_bmin_vectors
        bmax = root.leaf_bmax_vectors

        for node in root:
            if node.indices is not None:
                assert node in root.leaves
                expected = True
            else:
                assert node not in root.leaves
                expected = False

            equal_min = np.all(np.isclose(node.aabb.min, bmin), axis=1)
            equal_max = np.all(np.isclose(node.aabb.max, bmax), axis=1)
            assert np.any(equal_min & equal_max) == expected

    def test_bvh_is_cached_per_model(self):
        vertices = gl.coerce_array(np.arange(180), gl.vec3)
        indices = gl.coerce_array(np.arange(60), gl.uvec3)
        model = base.Model(vertices, indices)

        root1 = collisions.BVH.create_tree(model, target_density=16)
        root2 = collisions.BVH.create_tree(model, target_density=16)
        root3 = collisions.BVH.create_tree(model, target_density=24)

        assert root1 is root2
        assert root1 is not root3


class TestAABB:
    def test_getting_the_center(self):
        aabb = collisions.AABB((0, 0, 0), (2, 4, 8))

        assert aabb.center == (1, 2, 4)

    def test_setting_the_center(self):
        aabb = collisions.AABB((0, 0, 0), (2, 4, 6))
        aabb.center = (0, 0, 0)

        assert aabb.min == (-1, -2, -3)
        assert aabb.max == (1, 2, 3)

    def test_equality(self):
        assert collisions.AABB((0, 0, 0), (1, 1, 1)) == collisions.AABB(
            (0, 0, 0), (1, 1, 1)
        )
        assert not collisions.AABB((0, 1, 0), (1, 1, 1)) == collisions.AABB(
            (0, 0, 0), (1, 1, 1)
        )

    def test_shape(self):
        assert collisions.AABB((3, 3, 3), (5, 5, 7)).shape == (2, 2, 4)


class TestRay:
    @pytest.mark.parametrize(
        "aabb, ray, intersects",
        (
            # fmt: off
            (collisions.AABB((0, 0, 0), (1, 1, 1)), collisions.Ray((-1, -1, -1), (1, 1, 1)), True),
            (collisions.AABB((0, 0, 0), (1, 1, 1)), collisions.Ray((0.5, 0.5, 0.5), (123, 321, 1000)), True),
            (collisions.AABB((0, 0, 0), (1, 1, 1)), collisions.Ray((1.5, 1.5, 1.5), (123, 321, 1000)), False),
            (collisions.AABB((0, 0, 0), (1, 1, 1)), collisions.Ray((-0.5, 0.5, 0.5), (1, 2, 0)), False),
            # fmt: on
        ),
    )
    def test_aabb_intersections(self, aabb, ray, intersects):
        assert ray.collides_aabb(aabb) == intersects

    def test_aabb_batch_intersections(self):
        bmin = np.array([(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)])
        bmax = np.array([(1, 1, 1), (1.5, 1.5, 1.5), (2, 2, 2)])
        ray = collisions.Ray((0.75, -1, 0.75), (0, 1, 0))

        expected = (True, True, False)
        assert np.all(ray.collides_aabb(bmin=bmin, bmax=bmax) == expected)

    def test_bvh_intersection(self):
        model = gridmesh.GridMesh(lod=20, scale=5)
        bvh = collisions.BVH.create_tree(model, target_density=16)
        ray = collisions.Ray((2, 2, 5), (-2, -3, -10))

        assert ray.collides_bvh(bvh) > 0

    def test_bvh_distance(self):
        model = gridmesh.GridMesh(lod=20, scale=5)
        bvh = collisions.BVH.create_tree(model, target_density=16)
        ray = collisions.Ray((2, 2, 5), (0, 0, -10))

        assert ray.collides_bvh(bvh) == pytest.approx(5)

    def test_bvh_gets_nearest_hit(self):
        # fmt: off
        vertices = np.array([
            (0, 0, 0), (2, 4, 0), (4, 0, 0),
            (0, 0, 1), (2, 4, 1), (4, 0, 1),
            (0, 0, -2.5), (2, 4, -2.5), (4, 0, -2.5),
        ], gl.vec3)
        # fmt: on
        indices = np.array([(0, 1, 2), (3, 4, 5), (6, 7, 8)], gl.uvec3)
        model = base.Model(vertices, indices)
        bvh = collisions.BVH.create_tree(model, target_density=1)
        ray = collisions.Ray((2, 2, 2), (0, 0, -1))

        assert ray.collides_bvh(bvh) == pytest.approx(1)

    def test_bvh_miss(self):
        # fmt: off
        vertices = np.array([
            (0, 0, 0), (2, 4, 0), (4, 0, 0),
            (0, 0, 1), (2, 4, 1), (4, 0, 1),
            (0, 0, -2.5), (2, 4, -2.5), (4, 0, -2.5),
        ], gl.vec3)
        # fmt: on
        indices = np.array([(0, 1, 2), (3, 4, 5), (6, 7, 8)], gl.uvec3)
        model = base.Model(vertices, indices)
        bvh = collisions.BVH.create_tree(model, target_density=1)
        ray = collisions.Ray((100, 100, -1), (2, 2, -1))

        assert ray.collides_bvh(bvh) is False

    def test_transforming_a_ray_into_object_space(self):
        transform = transforms.Transform(
            (10, 20, 30), (4, 4, 8), (0, 0, 1), 90
        )
        ray = collisions.Ray((10, 0, 30), (0, 1, 0))

        ray.to_object_space(transform)

        assert_approx(ray.origin, (-5, 0, 0))
        assert_approx(ray.direction, (1, 0, 0))

    def test_transforming_multiple_times(self):
        transform1 = transforms.Transform(
            (10, 20, 30), (4, 4, 8), (0, 0, 1), 90
        )
        transform2 = transforms.Transform(
            (10, 40, 30), (4, 4, 8), (0, 0, 1), 90
        )
        ray = collisions.Ray((10, 0, 30), (0, 1, 0))

        ray.to_object_space(transform1)
        ray.to_object_space(transform2)

        # the second transform should still have access to the original ray
        # data and transform according to that.
        assert_approx(ray.origin, (-10, 0, 0))
        assert_approx(ray.direction, (1, 0, 0))

    def test_resetting_transform(self):
        transform = transforms.Transform(
            (10, 20, 30), (4, 4, 8), (0, 0, 1), 90
        )
        ray = collisions.Ray((10, 0, 30), (0, 1, 0))

        ray.to_object_space(transform)
        ray.reset_transform()

        assert_approx(ray.origin, (10, 0, 30))
        assert_approx(ray.direction, (0, 1, 0))
