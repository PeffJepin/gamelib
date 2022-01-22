import math

from typing import Iterable


class _Vector:
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        if kwargs:
            for key, value in kwargs:
                setattr(self, key, value)
        else:
            for slot, value in zip(self.__slots__, args):
                setattr(self, slot, value)

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

    def __sub__(self, other):
        if not isinstance(other, Iterable):
            return type(self)(*(val - other for val in self))
        return type(self)(*(v1 - v2 for v1, v2 in zip(self, other)))

    def __rsub__(self, other):
        if isinstance(other, Iterable):
            return type(self)(*other) - self
        else:
            return type(self)(*(other for _ in self.__slots__)) - self

    def __mul__(self, other):
        if not isinstance(other, Iterable):
            return type(self)(*(val * other for val in self))
        return type(self)(*(v1 * v2 for v1, v2 in zip(self, other)))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, Iterable):
            return type(self)(*(val / other for val in self))
        return type(self)(*(v1 / v2 for v1, v2 in zip(self, other)))

    def __rtruediv__(self, other):
        if isinstance(other, Iterable):
            return type(self)(*other) / self
        else:
            return type(self)(*(other for _ in self.__slots__)) / self

    def __floordiv__(self, other):
        if not isinstance(other, Iterable):
            return type(self)(*(val // other for val in self))
        return type(self)(*(v1 // v2 for v1, v2 in zip(self, other)))

    def __rfloordiv__(self, other):
        if isinstance(other, Iterable):
            return type(self)(*other) // self
        else:
            return type(self)(*(other for _ in self.__slots__)) // self

    @property
    def magnitude(self):
        return math.sqrt(abs(self.dot(self)))

    def dot(self, other):
        return sum(self * other)

    def normalize(self):
        magnitude = self.magnitude
        if magnitude == 0:
            return
        for slot in self.__slots__:
            setattr(self, slot, getattr(self, slot) / magnitude)


class Vec2(_Vector):
    __slots__ = ("x", "y")


class Vec3(_Vector):
    __slots__ = ("x", "y", "z")

    def cross(self, other):
        if not isinstance(other, Vec3):
            other = Vec3(*other)
        return Vec3(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )


class Vec4(_Vector):
    __slots__ = ("x", "y", "z", "w")
