import dataclasses
import weakref

from typing import Tuple
from typing import Iterable

import numpy as np

from gamelib.core import gl
from gamelib import Vec3


_bvh_cache = dict()


class Ray:
    """Used to check for ray collisions."""

    MAX_DISTANCE = np.finfo(gl.float).max

    def __init__(self, origin, direction):
        """Set initial values and cache the inverse direction.

        Parameters
        ----------
        origin : Vec3 | Iterable[float]
        direction : Vec3 | Iterable[float]
        """

        if not isinstance(origin, Vec3):
            origin = Vec3(*origin)
        if not isinstance(direction, Vec3):
            direction = Vec3(*direction)
        self._origin = origin
        self._direction = direction.normalize()
        self._inv = self._direction.inverse()
        self._transformed_origin = None
        self._transformed_direction = None
        self._transformed_inverse = None

    def __repr__(self):
        return f"<Ray(origin={self.origin!r}, direction={self.direction!r})>"

    @property
    def origin(self):
        """Get the origin point.

        Returns
        -------
        Vec3
        """

        return (
            self._transformed_origin
            if self._transformed_origin is not None
            else self._origin
        )

    @property
    def direction(self):
        """Get the direction vector.

        Returns
        -------
        Vec3
        """

        return (
            self._transformed_direction
            if self._transformed_direction is not None
            else self._direction
        )

    @property
    def inverse(self):
        """Get the inverse. Any 0 component will be set to math.inf.

        Returns
        -------
        Vec3
        """

        return (
            self._transformed_inverse
            if self._transformed_inverse is not None
            else self._inv
        )

    def to_object_space(self, transform):
        """Transform the origin and direction into an objects space given it's
        transform.

        Parameters
        ----------
        transform : transforms.Transform
        """

        origin = Vec3(*self._origin)
        direction = Vec3(*self._direction)

        self._transformed_origin = transform.apply_inverse(origin)
        self._transformed_direction = transform.apply_inverse(
            direction, normal=True
        ).normalize()
        self._transformed_inverse = self._transformed_direction.inverse()

    def reset_transform(self):
        """Reset the vector to it's world space coordinates."""

        self._transformed_origin = None
        self._transformed_direction = None
        self._transformed_inverse = None

    def collides_aabb(self, aabb=None, bmin=None, bmax=None):
        """Check if this ray intersects with a bounding box with slabs method.

        Parameters
        ----------
        aabb : AABB, optional
            aabb can be provided to check a single box
        bmin : np.ndarray, optional
            bmin and bmax arrays can be given to check batches of aabbs
        bmax : np.ndarray, optional
            bmin and bmax arrays can be given to check batches of aabbs

        Returns
        -------
        bool | np.ndarray:
            A single bool or vector of bools, depending on what args are given.
        """

        if bmin is not None:
            axis = 1
            mn = bmin
            mx = bmax
        else:
            axis = 0
            mn = aabb.min
            mx = aabb.max

        tmin = (mn[:] - self.origin) * self.inverse
        tmax = (mx[:] - self.origin) * self.inverse

        max_min = np.max(np.minimum(tmin, tmax), axis=axis)
        min_max = np.min(np.maximum(tmin, tmax), axis=axis)

        return min_max >= max_min

    def collides_bvh(self, bvh, *, _exit_early=False):
        """Check if this ray collides with the geometry described by a given
        BVH tree.

        Parameters
        ----------
        bvh : BVH
        _exit_early : bool, optional
            for most situations the default method of checking collisions
            will probably be faster because it will check all the leaf node
            aabb intersections in a batch with numpy, though this
            implementation is working, in most cases it's probably not
            recommended.

        Returns
        -------
        bool | float:
            Returns False if there was no collision. Returns the distance to
            the nearest triangle intersection found.
        """

        if not self.collides_aabb(bvh.aabb):
            return False

        if _exit_early:
            dist = self._recursive_check_bvh(bvh)
            return dist if dist < self.MAX_DISTANCE else False

        bmin = bvh.leaf_bmin_vectors
        bmax = bvh.leaf_bmax_vectors
        intersecting_leaves = bvh.leaves[
            self.collides_aabb(bmin=bmin, bmax=bmax)
        ]
        if not len(intersecting_leaves) > 0:
            return False
        triangles = np.concatenate(
            [node.triangles for node in intersecting_leaves]
        )
        intersection_distances = ray_triangle_intersections(
            triangles, self.origin, self.direction
        )
        indices = np.where(intersection_distances != -1)[0]
        if len(indices) > 0:
            return np.min(intersection_distances[indices])
        else:
            return False

    def intersects_triangles(self, triangles):
        """Given an array of triangles, returns an array containing the
        distances at which this ray intersects those triangles.

        Parameters
        ----------
        triangles : np.ndarray
            This array should have shape (n, 3, 3)

        Returns
        -------
        np.ndarray:
            Given an array of shape (n, 3, 3) returns an array of shape (n,).
            Values will either be the distance to the collision point, or -1
            for misses.
        """

        return ray_triangle_intersections(
            triangles, self.origin, self.direction
        )

    def _recursive_check_bvh(self, bvh_node, minimum=MAX_DISTANCE):
        def choose_order(node):
            """Choose which branches to prioritize."""
            # ignore dead end nodes
            if node.left is None:
                if node.right is None:
                    return ()
                return (node.right,)
            elif node.right is None:
                return (node.left,)

            # check leaf nodes first
            if node.left.indices is not None:
                if node.right.indices is None:
                    return node.left, node.right
            elif node.right.indices is not None:
                return node.right, node.left

            # else check closest first
            center = node.aabb.center
            dl = (center - node.left.aabb.center).magnitude
            dr = (center - node.left.aabb.center).magnitude
            if dl < dr:
                return node.left, node.right
            return node.right, node.left

        if bvh_node is None:
            return minimum
        if not self.collides_aabb(bvh_node.aabb):
            return minimum
        if bvh_node.vertices is not None:
            intersection_data = ray_triangle_intersections(
                bvh_node.triangles, self.origin, self.direction
            )
            detected = np.where(intersection_data != -1)[0]
            if len(detected) > 0:
                minimum = min(
                    minimum,
                    np.min(intersection_data[detected]),
                )
            return minimum

        order = choose_order(bvh_node)

        for bvh_node in order:
            minimum = self._recursive_check_bvh(bvh_node, minimum)
            if minimum < self.MAX_DISTANCE:
                return minimum

        return minimum


class AABB:
    """Axis Aligned Bounding Box

    Used for checking collisions without having to check against every single
    triangle.
    """

    nx = np.array((1, 0, 0), gl.vec3)
    ny = np.array((0, 1, 0), gl.vec3)
    nz = np.array((0, 0, 1), gl.vec3)

    __slots__ = ("_min", "_max")

    def __init__(self, bmin, bmax):
        """Create a bounding box from the minimum and maximum positions along
        each dimension.

        Parameters
        ----------
        bmin : Vec3 | Iterable[float]
        bmax : Vec3 | Iterable[float]
        """

        self.min = bmin
        self.max = bmax

    def __repr__(self):
        return f"<AABB(min={self._min}, max={self._max})>"

    def __eq__(self, other):
        """Compare equal only to other AABB's with the same boundaries."""

        if not isinstance(other, AABB):
            return False
        else:
            return self._min == other.min and self._max == other.max

    @property
    def min(self):
        """Get the box minimum.

        Returns
        -------
        Vec3
        """

        return self._min

    @min.setter
    def min(self, bmin):
        """Set the box minimum.

        Parameters
        ----------
        bmin : Vec3 | Iterable[float]
        """

        if not isinstance(bmin, Vec3):
            bmin = Vec3(*bmin)
        self._min = bmin

    @property
    def max(self):
        """Get the box maximum.

        Returns
        -------
        Vec3
        """

        return self._max

    @max.setter
    def max(self, bmax):
        """Set the box maximum.

        Parameters
        ----------
        bmax : Vec3 | Iterable[float]
        """

        if not isinstance(bmax, Vec3):
            bmax = Vec3(*bmax)
        self._max = bmax

    @property
    def shape(self):
        """Compute the shape of the box.

        Returns
        -------
        Vec3
        """

        return self._max - self._min

    @property
    def center(self):
        """Compute the center point of the box.

        Returns
        -------
        Vec3
        """

        return self._min + (self._max - self._min) / 2

    @center.setter
    def center(self, pos):
        """Compute new min and max values to move the center to the given
        position.

        Parameters
        ----------
        pos : Vec3 | Iterable[float]
        """

        diff = pos - self.center
        self.min += diff
        self.max += diff


class BVH:
    """Bounded Volume Hierarchy

    This is the node class of a tree type data structure used to narrow down
    the amount primitives (triangles) that need to be processed to check for
    collisions.
    """

    # only on the root
    ntris: int
    leaves: np.ndarray
    leaf_bmin_vectors: np.ndarray
    leaf_bmax_vectors: np.ndarray

    def __init__(self, aabb, vertices=None, indices=None):
        """Initialize a node. A node is considered a leaf node if it contains
        vertices and indices. If a node has no children and no vertices/indices
        then it's just a region of empty space.

        Parameters
        ----------
        aabb : AABB
            The space this node occupies.
        vertices : np.ndarray
            A pointer to the array of vertices this node is representing.
        indices : np.ndarray
            The indices into the models vertices that represents the triangles
            which are contained in this node.
        """

        self.aabb = aabb
        self.vertices = vertices
        self.indices = indices
        self.left = None
        self.right = None

    @property
    def triangles(self):
        """Get a view of the actual triangles by indexing into the vertices
        array.

        Returns
        -------
        np.ndarray | None:
            If this is a leaf node this will return the contained triangles,
            otherwise this will return None.
        """

        return None if self.vertices is None else self.vertices[self.indices]

    @classmethod
    def create_tree(cls, model, target_density=64):
        """Create a BVH tree for the given model, returning the root.

        Parameters
        ----------
        model : base.Model
        target_density : int
            The subdivision algorithm will create leaves when it reaches nodes
            containing this amount or fewer triangles.

        Returns
        -------
        BVH:
            The root of the tree.
        """

        aabb = AABB(model.v_min, model.v_max)

        # check if we already have created this bvh
        key = (id(model), tuple(aabb.min), tuple(aabb.max), target_density)
        ref = _bvh_cache.pop(key, None)
        if ref is not None:
            bvh = ref()
            if bvh is not None:
                _bvh_cache[key] = ref
                return bvh

        root = cls(aabb)
        BVH_Helper.divide(root, model.vertices, model.indices, target_density)
        bmin_vectors = []
        bmax_vectors = []
        leaves = []
        for node in root:
            if node.indices is not None:
                bmin_vectors.append(node.aabb.min)
                bmax_vectors.append(node.aabb.max)
                leaves.append(node)

        root.leaf_bmin_vectors = np.stack(bmin_vectors)
        root.leaf_bmax_vectors = np.stack(bmax_vectors)
        root.leaves = np.array(leaves, object)
        root.ntris = len(model.indices)

        # cache reference
        _bvh_cache[key] = weakref.ref(root)
        return root

    def __iter__(self):
        """Convenience for visiting each node.

        Returns
        -------
        Iterable[BVH]
        """

        nodes = []

        def collect_nodes(current):
            nodes.append(current)
            if current.left is not None:
                collect_nodes(current.left)
            if current.right is not None:
                collect_nodes(current.right)

        collect_nodes(self)
        for node in nodes:
            yield node

    def __repr__(self):
        ntris = 0 if self.triangles is None else len(self.triangles)
        return f"<BVH(aabb={self.aabb}, n_triangles={ntris})>"


class BVH_Helper:
    """A namespace for functions related to subdividing a model into a BVH."""

    @dataclasses.dataclass
    class Split:
        aabb: AABB
        vertices: np.ndarray
        indices: np.ndarray

    Splits = Tuple[Split, Split]

    @classmethod
    def divide(cls, node, vertices, indices, target_density):
        """Recursively divide a given node until it reaches leaves of target
        triangle density.

        Note that when a node splits it will clip vertices. This results in
        triangles that are shared. The more subdivisions you have the more
        shared triangles you will have.

        Parameters
        ----------
        node : BVH
            The node to subdivide.
        vertices : np.ndarray
            3-component mesh vertices.
        indices : np.ndarray
            Indices to construct the triangles array.
        target_density : int
            When to stop subdividing the tree.
        """

        # recursively divide the node
        ntris = len(indices)
        if ntris == 0:
            # dead end node
            node.vertices = None
            node.indices = None
            return
        if ntris <= target_density:
            # leaf node
            node.vertices = vertices
            node.indices = indices
            return

        this_split = cls.Split(node.aabb, vertices, indices)
        spl1, spl2 = cls._get_best_splits(this_split)
        if spl1 is None or spl2 is None:
            # If the call to get best splits bailed without getting any splits
            # then we should make this node a leaf node
            node.vertices = vertices
            node.indices = indices
            return

        node.left = BVH(spl1.aabb)
        node.right = BVH(spl2.aabb)

        cls.divide(node.left, vertices, spl1.indices, target_density)
        cls.divide(node.right, vertices, spl2.indices, target_density)

    @classmethod
    def _get_best_splits(cls, parent) -> Splits:
        """Generate splits and score them, returning the best one found. Might
        bail and return (None, None)."""

        best = 0
        best_splits = (None, None)
        for splits in cls._generate_possible_splits(parent):
            score = cls._score_splits(parent, splits)
            if score == 1_000:
                return splits
            if score >= best:
                best = score
                best_splits = splits
        return best_splits

    @classmethod
    def _generate_possible_splits(cls, parent) -> Iterable:
        """This function creates splits by bisecting the parent node along
        each axis."""

        for i in range(3):
            shape = tuple(parent.aabb.max - parent.aabb.min)
            bmin1 = tuple(parent.aabb.min)
            bmin2 = tuple(
                p_comp if j != i else p_comp + shape[i] / 2
                for j, p_comp in enumerate(bmin1)
            )
            bmax2 = tuple(parent.aabb.max)
            bmax1 = tuple(
                p_comp if j != i else bmin2[i]
                for j, p_comp in enumerate(bmax2)
            )

            aabb1 = AABB(bmin1, bmax1)
            aabb2 = AABB(bmin2, bmax2)
            spl1 = cls.Split(aabb1, parent.vertices, parent.indices)
            spl2 = cls.Split(aabb2, parent.vertices, parent.indices)
            cls._clamp_new_split(spl1)
            cls._clamp_new_split(spl2)
            yield spl1, spl2

    @classmethod
    def _clamp_new_split(cls, split) -> None:
        """Given a fresh split, clamp it to be as small as permitted."""

        indices = cls.clamp_aabb(split.aabb, split.vertices, split.indices)
        split.indices = indices

    @classmethod
    def _score_splits(cls, parent, splits) -> float:
        """Inspect newly created splits and decide how viable they are."""

        spl1, spl2 = splits
        length1, length2 = len(spl1.indices), len(spl2.indices)
        length_parent = len(parent.indices)
        parent_shape = parent.aabb.shape
        volume_parent = parent_shape.x * parent_shape.y * parent_shape.z

        # discard splits and opt to make parent a leaf node

        if (length1 + length2) >= (1.5 * length_parent):
            # if the splits contain mostly the same indices - stop splitting
            # some overlap is going to happen, since any triangle along the
            # split plane is bound to fall in both splits.
            return -1
        if (length1 == 0 and length2 == 0) or volume_parent == 0:
            # don't subdivide empty space
            return -1
        elif length1 == 0 or length2 == 0:
            # an empty leaf node will be chosen by default for now
            # might change in the future with benchmarks to actually
            # test differences
            return 1_000
        for spl in splits:
            # discard splits that go beyond a 5/1 shape ratio
            # try to stay more regular in shape
            if max(spl.aabb.shape) >= 8 * min(spl.aabb.shape):
                return -1

        score = 0
        volume_weight = 4
        dimensions_weight = 1.5
        distribution_weight = 1

        # reward score for minimizing contained volume
        expected = volume_parent / 2
        for spl in splits:
            shape = spl.aabb.max - spl.aabb.min
            v = shape.x * shape.y * shape.z
            t = (expected - v) / expected
            score += t * volume_weight / len(splits)

        # reward score for tending towards uniform dimensions
        for spl in splits:
            shape = spl.aabb.max - spl.aabb.min
            avg = sum(shape) / 3
            scaled = tuple(dim / avg for dim in shape)
            t = 1 + sum(abs(1 - dim) for dim in scaled)
            ratio = 1 / t
            score += dimensions_weight * ratio / len(splits)

        # reward score for even distribution of indices
        lengths = tuple(len(spl.indices) for spl in splits)
        avg = sum(lengths) / len(splits)
        scaled = tuple(length / avg for length in lengths)
        t = 1 + sum(abs(1 - dim) for dim in scaled)
        score += 1 / t * distribution_weight

        return score

    @classmethod
    def clamp_aabb(cls, aabb, vertices, triangle_indices):
        """Shrinks a bounding box along dimensions where it isn't clipping any
        triangles.

        Parameters
        ----------
        aabb : AABB
        vertices : np.ndarray
        triangle_indices : np.ndarray

        Returns
        -------
        np.ndarray:
            The triangle indices whose triangles actually intersect the box.
        """

        tris = vertices[triangle_indices]

        # Get all vertices who belong to triangles that intersect with this box
        mask = np.where(aabb_triangle_intersections(aabb, tris))
        clipped_indices = triangle_indices[mask]
        clipped_vertices = vertices[clipped_indices].reshape(-1, 3)
        if len(clipped_indices) == 0:
            return clipped_indices

        # Assume that this box doesn't need to stretch to accommodate vertices
        # that fall outside, but whose triangle still intersects. Those other
        # vertices are assumed to be under the jurisdiction of another box.
        # However, for a dimension of the box to be permitted to shrink, it IS
        # required to have the capacity for ALL clipped triangle vertices.
        bmin = tuple(
            max(aabb.min[i], np.min(clipped_vertices[:, i])) for i in range(3)
        )
        bmax = tuple(
            min(aabb.max[i], np.max(clipped_vertices[:, i])) for i in range(3)
        )
        aabb.min = bmin
        aabb.max = bmax
        return clipped_indices


def ray_triangle_intersections(triangles, origin, dir):
    """Batch triangle intersection calculations. This is highly unoptimized as
    it won't exit early, but should suffice for the time being. I would rather
    wait to implement hot paths like this in C later than to try and optimize
    the algorithm for numpy performance.

    Parameters
    ----------
    triangles : np.ndarray
        Array with shape (n, 3, 3) containing n triangles.
    origin : Iterable[float]
        3-component ray origin.
    dir : Iterable[float]
        3-component ray direction.

    Returns
    -------
    np.ndarray:
        Shape (n,) array containing the distances to intersected triangles,
        or -1 for triangles that didn't intersect.
    """

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
    epsilon = 0.0000001
    h = np.cross(dir, e2[:])
    a = np.sum(e1 * h, axis=1)
    f = 1 / a
    s = origin - v0
    u = f * np.sum(s * h, axis=1)
    q = np.cross(s, e1)
    v = f * np.sum(dir * q[:], axis=1)
    t = f * np.sum(e2 * q, axis=1)
    misses = (
        ((a < epsilon) & (a > -epsilon))
        | ((u < 0) | (u > 1))
        | ((v < 0) | (u + v > 1))
        | (t <= epsilon)
    )

    result = t * dir_mag
    result[misses] = -1
    return result


def aabb_triangle_intersections(aabb, triangles):
    """Check for AABB -> triangle intersection in batches using Separating Axis
    Theorem.

    Parameters
    ----------
    aabb : AABB
    triangles : np.ndarray
        Array with shape (n, 3, 3) containing n triangles.

    Returns
    -------
    np.ndarray:
        Shape (n,) array containing True for triangles that intersected and
        False otherwise.
    """

    bmin = np.array(tuple(aabb.min), gl.vec3)
    bmax = np.array(tuple(aabb.max), gl.vec3)

    # vertices
    v1 = triangles[:, 0]
    v2 = triangles[:, 1]
    v3 = triangles[:, 2]

    # transform aabb
    c = (bmin + bmax) / 2
    bmin -= c
    bmax -= c

    # transform triangles
    v1 -= c
    v2 -= c
    v3 -= c

    # compute triangle edge vectors
    e1 = v2 - v1
    e2 = v3 - v2
    e3 = v1 - v3

    # check axis for triangle edge -> aabb face
    axis = np.cross(AABB.nx, e1[:])
    intersects = _test_axis_for_separation(bmax, axis, v1, v2, v3)

    axis = np.cross(AABB.nx, e2[:])
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    axis = np.cross(AABB.nx, e3[:])
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    axis = np.cross(AABB.ny, e1[:])
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    axis = np.cross(AABB.ny, e2[:])
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    axis = np.cross(AABB.ny, e3[:])
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    axis = np.cross(AABB.nz, e1[:])
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    axis = np.cross(AABB.nz, e2[:])
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    axis = np.cross(AABB.nz, e3[:])
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    # check axis for aabb faces
    axis = AABB.nx.reshape(1, 3)
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    axis = AABB.ny.reshape(1, 3)
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    axis = AABB.nz.reshape(1, 3)
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    # check axis for triangle face normal
    axis = np.cross(e1, e2, axis=1)
    intersects = intersects & _test_axis_for_separation(bmax, axis, v1, v2, v3)

    return intersects


def _test_axis_for_separation(half_extents, test_axis, v1, v2, v3):
    """Test an axis to see if we can rule out this triangle for intersection."""

    proj1 = np.sum(test_axis * v1, axis=1)
    proj2 = np.sum(test_axis * v2, axis=1)
    proj3 = np.sum(test_axis * v3, axis=1)
    proj_aabb = (
        half_extents[0] * np.abs(np.sum(AABB.nx * test_axis[:], axis=1))
        + half_extents[1] * np.abs(np.sum(AABB.ny * test_axis[:], axis=1))
        + half_extents[2] * np.abs(np.sum(AABB.nz * test_axis[:], axis=1))
    )

    mx = np.maximum.reduce([proj1, proj2, proj3])
    mn = np.minimum.reduce([proj1, proj2, proj3])
    return np.maximum(-mx, mn) <= proj_aabb
