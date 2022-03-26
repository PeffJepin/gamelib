import pytest
import numpy as np

from gamelib.geometry import collisions
from gamelib.geometry import base
from gamelib.geometry import gridmesh
from gamelib.core import gl

from ..conftest import assert_approx


@pytest.fixture
def ray_from_000_to_111():
    yield collisions.Ray(origin=(0, 0, 0), direction=(1, 1, 1))


def triangle(p1: tuple, p2: tuple, p3: tuple) -> np.ndarray:
    # composes points into a np array representing the triangle
    return np.array([p1, p2, p3], gl.vec3)


def triangles(*tris: np.ndarray) -> np.ndarray:
    # stacks triangles from the `triangle` function into a single array
    return np.stack(tris, 0)


class TestRayTringleIntersections:
    def test_all_should_intersect(self, ray_from_000_to_111):
        tris = triangles(
            triangle((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            triangle((2, 0, 0), (0, 2, 0), (0, 0, 2))
        )

        intersections = ray_from_000_to_111.intersects_triangles(tris)

        assert np.all(intersections != -1)

    def test_all_should_not_intersect(self, ray_from_000_to_111):
        tris = triangles(
            triangle((0, 0, 1), (1, 0, 0), (1, 0, 1)),
            triangle((0, 0, 2), (2, 0, 0), (2, 0, 2))
        )

        intersections = ray_from_000_to_111.intersects_triangles(tris)

        assert np.all(intersections == -1)

    def test_some_should_intersect(self, ray_from_000_to_111):
        tris = triangles(
            triangle((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            triangle((0, 0, 2), (2, 0, 0), (2, 0, 2))
        )

        intersections = ray_from_000_to_111.intersects_triangles(tris)

        assert intersections[0] != -1
        assert intersections[1] == -1

    def test_intersection_distances(self):
        ray = collisions.Ray(origin=(0.1, 0.1, 1), direction=(0, 0, -1))

        # the distance for each intersection should be the z difference
        tris = triangles(
            triangle((1, 0, 0), (0, 1, 0), (0, 0, 0)),
            triangle((1, 0, -1), (0, 1, -1), (0, 0, -1)),
        )

        intersections = ray.intersects_triangles(tris)

        assert intersections[0] == pytest.approx(1)
        assert intersections[1] == pytest.approx(2)


@pytest.mark.parametrize(
    "tri, intersects",
    (
        # fmt: off
        # edge intersections
        (triangle((4, 2, 4), (2, 4, 4), (2, 2, 4)), True),
        (triangle((2, 4, 4), (4, 6, 4), (2, 6, 4)), True),
        (triangle((4, 6, 4), (6, 4, 4), (6, 6, 4)), True),
        (triangle((6, 4, 4), (4, 2, 4), (6, 2, 4)), True),

        (triangle((4, 4, 2), (2, 4, 4), (2, 4, 2)), True),
        (triangle((2, 4, 4), (4, 4, 6), (2, 4, 6)), True),
        (triangle((4, 4, 6), (6, 4, 4), (6, 4, 6)), True),
        (triangle((6, 4, 4), (4, 4, 2), (6, 4, 2)), True),

        (triangle((4, 4, 2), (4, 2, 4), (4, 2, 2)), True),
        (triangle((4, 2, 4), (4, 4, 6), (4, 2, 6)), True),
        (triangle((4, 4, 6), (4, 6, 4), (4, 6, 6)), True),
        (triangle((4, 6, 4), (4, 4, 2), (4, 6, 2)), True),

        # vertex / face intersections
        (triangle((2, 2, 4), (4, 4, 4), (6, 2, 4)), True),
        (triangle((2, 2, 4), (4, 4, 4), (2, 6, 4)), True),
        (triangle((2, 6, 4), (4, 4, 4), (6, 6, 4)), True),
        (triangle((6, 6, 4), (4, 4, 4), (6, 2, 4)), True),
        (triangle((2, 4, 6), (4, 4, 4), (6, 4, 6)), True),
        (triangle((2, 4, 2), (4, 4, 4), (6, 4, 2)), True),

        # tri face intersection
        (triangle((0.9, 4.9, 2.9), (4.9, 0.9, 2.9), (4, 4, 7)), True),

        # some near misses
        (triangle((2.9, 2.9, 2.9), (2, 2, 2), (1, 1, 1)), False),
        (triangle((5.1, 5.1, 5.1), (6, 6, 6), (7, 7, 7)), False),
        # fmt: on
    ),
)
def test_aabb_tri_intersections(tri, intersects):
    aabb = collisions.AABB((3, 3, 3), (5, 5, 5))
    tris = triangles(tri)

    result = collisions.aabb_triangle_intersections(aabb, tris)

    assert result[0] == intersects


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
    # TODO: I'm not sure how I feel about some of these tests. On the one hand
    # they are terribly hard to read and reason about. On the other hand, they 
    # do assert the correct behavior. I think that my strategy on testing this 
    # should be revisited when the collision implementations are inevitably 
    # rewritten in C. As written they at least allow me to tinker with the
    # current implementation without worrying about unknowingly breaking it.

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
    def test_calculating_the_center_from_the_bounds(self):
        aabb = collisions.AABB((0, 0, 0), (2, 4, 8))

        assert aabb.center == (1, 2, 4)

    def test_setting_the_center_to_reposition_the_bounds(self):
        aabb = collisions.AABB((0, 0, 0), (2, 4, 6))
        aabb.center = (0, 0, 0)

        assert aabb.min == (-1, -2, -3)
        assert aabb.max == (1, 2, 3)

    def test_equality_comparison(self):
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
