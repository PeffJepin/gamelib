from multiprocessing import shared_memory

import numpy as np


class SharedArray:
    def __init__(self, id_, shape, dtype):
        """Load into an already existing shared block of memory."""
        self._sm = shared_memory.SharedMemory(id_, create=False)
        self._arr = np.ndarray(shape, dtype, self._sm.buf)

    @classmethod
    def create(cls, id_, shape=None, dtype=None, array=None):
        """Copy an array into shared memory and return an array view into it."""
        if array is None:
            array = np.empty(shape, dtype)
        shm = shared_memory.SharedMemory(id_, create=True, size=array.nbytes)
        inst = cls(id_, array.shape, array.dtype)
        inst._arr[:] = array[:]
        shm.close()
        return inst

    def close(self):
        """
        Seems SharedMemory.unlink() has a bug in windows and doesn't
        behave as expected.

        My testing shows closing all open shm objects individually yields
        the desired result, where an new attempt to connect to the shm file
        raises FileNotFoundError.
        """
        self._arr = None
        self._sm.close()
        self._sm = None

    def __getitem__(self, key):
        return self._arr.__getitem__(key)

    def __setitem__(self, key, value):
        self._arr.__setitem__(key, value)

    def __eq__(self, other):
        return self._arr == other

    def __add__(self, other):
        return self._arr + other

    def __sub__(self, other):
        return self._arr - other

    def __mul__(self, other):
        return self._arr * other

    def __truediv__(self, other):
        return self._arr / other

    def __floordiv__(self, other):
        return self._arr // other

    def __iadd__(self, other):
        self._arr += other
        return self

    def __isub__(self, other):
        self._arr -= other
        return self

    def __imul__(self, other):
        self._arr *= other
        return self

    def __itruediv__(self, other):
        self._arr /= other
        return self

    def __ifloordiv__(self, other):
        self._arr //= other
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Documentation for np array class extensions found here.
        https://numpy.org/doc/stable/reference/arrays.classes.html

        This allows this class to behave like an np.ndarray for ufuncs by finding
        itself in the proposed function arguments and replacing itself with the current read array
        """
        corrected_inputs = (
            input_ if input_ is not self else self._arr for input_ in inputs
        )
        return getattr(ufunc, method)(*corrected_inputs, **kwargs)


class DoubleBufferedArray:
    def __init__(self, id_, shape, dtype):
        self._read_arr = SharedArray(id_ + '_r', shape, dtype)
        self._write_arr = SharedArray(id_ + '_w', shape, dtype)

    @classmethod
    def create(cls, id_, array=None, shape=None, dtype=None):
        if array is None:
            array = np.empty(shape, dtype)
        shm_r = shared_memory.SharedMemory(id_ + '_r', create=True, size=array.nbytes)
        shm_w = shared_memory.SharedMemory(id_ + '_w', create=True, size=array.nbytes)
        inst = cls(id_, array.shape, array.dtype)
        inst[:] = array[:]
        inst.swap()
        shm_r.close()
        shm_w.close()
        return inst

    def close(self):
        self._read_arr.close()
        self._write_arr.close()

    def swap(self):
        self._read_arr[:] = self._write_arr[:]

    def __eq__(self, other):
        return self._read_arr == other

    def __getitem__(self, idx):
        return self._read_arr[idx]

    def __setitem__(self, idx, value):
        self._write_arr[idx] = value

    def __add__(self, other):
        return self._read_arr + other

    def __sub__(self, other):
        return self._read_arr - other

    def __mul__(self, other):
        return self._read_arr * other

    def __truediv__(self, other):
        return self._read_arr / other

    def __iadd__(self, other):
        self._write_arr += other
        return self

    def __isub__(self, other):
        self._write_arr -= other
        return self

    def __imul__(self, other):
        self._write_arr *= other
        return self

    def __itruediv__(self, other):
        self._write_arr /= other
        return self

    def __ifloordiv__(self, other):
        self._write_arr //= other
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Documentation for np array class extensions found here.
        https://numpy.org/doc/stable/reference/arrays.classes.html

        This allows this class to behave like an np.ndarray for ufuncs by finding
        itself in the proposed function arguments and replacing itself with the current read array
        """
        corrected_inputs = (
            input_ if input_ is not self else self._read_arr for input_ in inputs
        )
        return getattr(ufunc, method)(*corrected_inputs, **kwargs)
