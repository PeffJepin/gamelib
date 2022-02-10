import numpy as np

from gamelib.core import gl


class ArrayStorage:
    def __init__(self, gltype, length=1):
        self._array = np.zeros(length, gltype)

    def __get__(self, obj, objtype=None):
        return self._array

    def __set__(self, obj, value):
        self._array[:] = value


class UniformBlock:
    def __init_subclass__(cls):
        cls._uniforms = [
            name
            for name, value in cls.__dict__.items()
            if isinstance(value, ArrayStorage)
        ]

    def __init__(self, *args, **kwargs):
        if args:
            for arg, name in zip(args, self._uniforms):
                setattr(self, name, arg)
        else:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def todict(self):
        return {name: getattr(self, name) for name in self._uniforms}


class AutoUniform:
    """Helper class for ShaderProgram to keep track of uniform sources."""

    def __init__(self, array, dtype, name):
        """
        Parameters
        ----------
        array : np.ndarray
        dtype : np.dtype | str
        name : str
        """
        self.array = array
        self.dtype = dtype
        self.name = name

    def update(self, prog):
        prog[self.name].write(self._data)

    @property
    def _data(self):
        return gl.coerce_array(self.array, self.dtype).tobytes()
