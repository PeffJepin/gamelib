from typing import Iterable

import numpy as np


class VectorComponent:
    """Descriptor to add a named field corresponding to an index in an array
    of data."""

    def __init__(self, index):
        self._index = index

    def __get__(self, obj, objtype=None):
        return obj[self._index]

    def __set__(self, obj, value):
        obj[self._index] = value

    def __set_name__(self, owner, name):
        self._owner = owner
        self._name = name

    def __repr__(self):
        return (
            f"<VectorComponent("
            f"owner={self._owner!r}, "
            f"name={self._name!r}, "
            f"index={self._index})>"
        )


class VectorType(np.ndarray):
    """Base vector type. Subclasses should specify _DTYPE if the default float
    is undesirable. VectorComponents should be added to subclasses."""

    _LENGTH: int
    _FIELDS: tuple

    _DTYPE = float

    def __init_subclass__(cls):
        """Checks what fields have been marked with VectorComponents."""

        fields = []
        for k, v in cls.__dict__.items():
            if isinstance(v, VectorComponent):
                fields.append(k)
        assert len(fields) > 1
        cls._LENGTH = len(fields)
        cls._FIELDS = tuple(fields)

    def __new__(cls, *args, **kwargs):
        """Create the underlying array."""

        return np.zeros(cls._LENGTH, cls._DTYPE).view(cls)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """Special treatment for certain numpy ufuncs."""

        args = []
        for arg in inputs:
            if isinstance(arg, VectorType):
                # view vectors as regular arrays for ufunc operations
                args.append(arg.view(np.ndarray))
            else:
                args.append(arg)

        result = getattr(ufunc, method)(*args, **kwargs)
        if ufunc == np.not_equal:
            # convert inequality checks to bool
            return np.any(result)
        elif ufunc == np.equal:
            # convert equality checks to bool
            return np.all(result)
        if result.shape == self.shape:
            # if an array involved in ufunc ops with this array conforms
            # to this vectors shape then view array as the vector type.
            return result.view(type(self))
        return result

    def __init__(self, *args, **kwargs):
        """Initialize a new vector. *args and **kwargs are mutually exclusive.

        Parameters
        ----------
        *args: Any
            Should be given in order described by VectorComponents.
        **kwargs : Any
            Keys should match VectorComponent names.
        """

        if kwargs:
            for name, value in kwargs.items():
                setattr(self, name, value)
        elif args:
            self._parse_args(args)

    def _parse_args(self, args):
        if len(args) == 0:
            # defaults to zeros
            return
        elif len(args) == 1:
            # args might be a single iterable
            if isinstance(args[0], Iterable):
                self._parse_iter(args[0])
        # else iterate over args
        else:
            self._parse_iter(args)

    def _parse_iter(self, iterable):
        # raises index error if iterable is too long
        for i, v in enumerate(iterable):
            self[i] = v

    @property
    def magnitude(self):
        """Get the length of the vector.

        Returns
        -------
        float
        """

        return np.sqrt(self.dot(self))

    def normalize(self):
        """Normalize the vector to length 1.

        Returns
        -------
        _Vector:
            returns itself for convenience
        """

        magnitude = self.magnitude
        if magnitude == 0:
            return self
        self[:] /= magnitude
        return self

    def inverse(self):
        """Compute the inverse of this vector. Note that any 0 value component
        will have an inverse of math.inf.

        Returns
        -------
        _Vector
        """

        with np.errstate(divide="ignore"):
            return 1 / self

    @classmethod
    def as_dtype(cls):
        """Express this vector type as a multidimensional numpy dtype.

        Returns
        -------
        np.dtype
        """

        return np.dtype((cls._DTYPE, cls._LENGTH))


class Vec2(VectorType):
    x = VectorComponent(0)
    y = VectorComponent(1)


class Vec3(VectorType):
    x = VectorComponent(0)
    y = VectorComponent(1)
    z = VectorComponent(2)

    def cross(self, other):
        return np.cross(self, other).view(Vec3)


class Vec4(VectorType):
    x = VectorComponent(0)
    y = VectorComponent(1)
    z = VectorComponent(2)
    w = VectorComponent(3)
