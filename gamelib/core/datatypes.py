import math

from typing import Iterable


class _Vector:
    """Base class for element-wise vector operations.

    Note that using this is probably going to come with a performance hit from
    all the getattr/setattr dynamic attribute access. This could be optimized
    in the future, but is not high on my list of things to do. When I later get
    to optimizations I'd rather focus on particularly hot code paths to be
    implemented in C.

    If the performance of this does present itself as an issue I imagine just
    trying to set/access attributes by x, y, z, w within a try except block
    and stopping when you hit an AttributeError could be a good place to start.
    """

    __slots__ = ()

    # make type checker happy
    # slots enabled on subclasses
    x: float
    y: float
    z: float
    w: float

    def __init__(self, *args, **kwargs):
        if kwargs:
            for key, value in kwargs:
                setattr(self, key, value)
        if len(args) == 1 and not isinstance(args[0], Iterable):
            value = args[0]
            for slot in self.__slots__:
                setattr(self, slot, value)
        else:
            for slot, value in zip(self.__slots__, args):
                setattr(self, slot, value)

    def __getitem__(self, index):
        return getattr(self, self.__slots__[index])

    def __setitem__(self, index, value):
        setattr(self, self.__slots__[index], value)

    def __iter__(self):
        return (getattr(self, slot) for slot in self.__slots__)

    def __eq__(self, other):
        return all(v1 == v2 for v1, v2 in zip(self, other))

    def __repr__(self):
        values_desc = ", ".join(
            f"{slot}={getattr(self, slot)}" for slot in self.__slots__
        )
        return f"<{self.__class__.__name__}({values_desc})>"

    def __neg__(self):
        for slot in self.__slots__:
            setattr(self, slot, getattr(self, slot) * -1)
        return self

    def __add__(self, other):
        if not isinstance(other, Iterable):
            return type(self)(*(val + other for val in self))
        return type(self)(*(v1 + v2 for v1, v2 in zip(self, other)))

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        if not isinstance(other, Iterable):
            for s in self.__slots__:
                setattr(self, s, getattr(self, s) + other)
            return self
        for s, v in zip(self.__slots__, other):
            setattr(self, s, getattr(self, s) + v)
        return self

    def __sub__(self, other):
        if not isinstance(other, Iterable):
            return type(self)(*(val - other for val in self))
        return type(self)(*(v1 - v2 for v1, v2 in zip(self, other)))

    def __rsub__(self, other):
        if isinstance(other, Iterable):
            return type(self)(*other) - self
        else:
            return type(self)(*(other for _ in self.__slots__)) - self

    def __isub__(self, other):
        if not isinstance(other, Iterable):
            for s in self.__slots__:
                setattr(self, s, getattr(self, s) - other)
            return self

        for s, v in zip(self.__slots__, other):
            setattr(self, s, getattr(self, s) - v)
        return self

    def __mul__(self, other):
        if not isinstance(other, Iterable):
            return type(self)(*(val * other for val in self))
        return type(self)(*(v1 * v2 for v1, v2 in zip(self, other)))

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        if not isinstance(other, Iterable):
            for s in self.__slots__:
                setattr(self, s, getattr(self, s) * other)
            return self

        for s, v in zip(self.__slots__, other):
            setattr(self, s, getattr(self, s) * v)
        return self

    def __truediv__(self, other):
        if not isinstance(other, Iterable):
            return type(self)(*(val / other for val in self))
        return type(self)(*(v1 / v2 for v1, v2 in zip(self, other)))

    def __rtruediv__(self, other):
        if isinstance(other, Iterable):
            return type(self)(*other) / self
        else:
            return type(self)(*(other for _ in self.__slots__)) / self

    def __itruediv__(self, other):
        if not isinstance(other, Iterable):
            for s in self.__slots__:
                setattr(self, s, getattr(self, s) / other)
            return self

        for s, v in zip(self.__slots__, other):
            setattr(self, s, getattr(self, s) / v)
        return self

    def __floordiv__(self, other):
        if not isinstance(other, Iterable):
            return type(self)(*(val // other for val in self))
        return type(self)(*(v1 // v2 for v1, v2 in zip(self, other)))

    def __rfloordiv__(self, other):
        if isinstance(other, Iterable):
            return type(self)(*other) // self
        else:
            return type(self)(*(other for _ in self.__slots__)) // self

    def __ifloordiv__(self, other):
        if not isinstance(other, Iterable):
            for s in self.__slots__:
                setattr(self, s, getattr(self, s) // other)
            return self

        for s, v in zip(self.__slots__, other):
            setattr(self, s, getattr(self, s) // v)
        return self

    @property
    def magnitude(self):
        """Get the length of the vector.

        Returns
        -------
        float
        """

        return math.sqrt(abs(self.dot(self)))

    def dot(self, other):
        """Compute a vector dot product.

        Parameters
        ----------
        other : Iterable
            Should be an iterable of scalars with the same length as this
            vector.

        Returns
        -------
        float
        """

        return sum(self * other)

    def normalize(self):
        """Normalize the vector to length 1.

        Returns
        -------
        _Vector:
            returns itself for convenience
        """

        magnitude = self.magnitude
        if magnitude == 0:
            return
        for slot in self.__slots__:
            setattr(self, slot, getattr(self, slot) / magnitude)
        return self

    def inverse(self):
        """Compute the inverse of this vector. Note that any 0 value component
        will have an inverse of math.inf.

        Returns
        -------
        _Vector
        """

        inv = []
        for v in self:
            try:
                component = 1 / v
            except ZeroDivisionError:
                component = math.inf
            inv.append(component)
        return type(self)(*inv)


class Vec2(_Vector):
    __slots__ = ("x", "y")


class Vec3(_Vector):
    __slots__ = ("x", "y", "z")

    def cross(self, other):
        """Compute the vector cross product.

        Parameters
        ----------
        other : Iterable
            Should be an iterable made up of 3 scalar values.

        Returns
        -------
        Vec3
        """

        if not isinstance(other, Vec3):
            other = Vec3(*other)
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


class Vec4(_Vector):
    __slots__ = ("x", "y", "z", "w")
