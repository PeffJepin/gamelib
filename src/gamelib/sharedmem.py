import os
from multiprocessing import shared_memory

import numpy as np

_POSIX = os.name == "posix"


class SharedArray:
    def __init__(self, id_, shape, dtype):
        """Load into an already existing shared block of memory."""
        self._shm = shared_memory.SharedMemory(id_)
        self._arr = np.ndarray(shape, dtype, self._shm.buf)

    @classmethod
    def create(cls, id_, shape=None, dtype=None, *, array=None):
        """
        Allocates the underlying shared memory and returns a view into it.

        Allocation can be specified by shape and dtype or just an existing numpy.ndarray
        If an array is used for specification shape and dtype params are ignored.

        Parameters
        ----------
        id_ : str
            Identifier for this shared block to be discovered from elsewhere.
        shape : tuple[int, ...]
            A valid shape for a numpy.ndarray
        dtype : numpy.dtype
        array : numpy.ndarray
            Optional initial data to initialize the memory block.

        Returns
        -------
        inst : SharedArray
            Creates and instance of the class.
            Note: SharedMemory could potentially be reclaimed if this were gc'd.
        """
        if array is None:
            array = np.empty(shape, dtype)
        shm = shared_memory.SharedMemory(id_, create=True, size=array.nbytes)
        inst = cls(id_, array.shape, array.dtype)
        inst._arr[:] = array[:]
        shm.close()
        return inst

    def unlink(self):
        """Should be called only once after the memory is no longer in use."""
        if not _POSIX:
            # SharedMemory.unlink() does nothing on windows but required on posix.
            # Close should still be called on this shm if `unlink` is the request.
            self.close()
            self._arr = None
            self._shm = None
            return

        self._shm.unlink()
        self._arr = None
        self._shm = None

    def close(self):
        """Should be called from every instance when that particular instance is no longer used."""
        self._shm.close()
        self._arr = None
        self._shm = None

    def __getattr__(self, item):
        return getattr(self._arr, item)

    def __getitem__(self, key):
        return self._arr.__getitem__(key)

    def __setitem__(self, key, value):
        self._arr.__setitem__(key, value)

    def __len__(self):
        return len(self._arr)

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
    """
    Maintains two SharedArray instances and delegates reads/writes to separate arrays.
    The flip method can be called to copy the write array into the read array.
    """

    def __init__(self, id_, shape, dtype):
        """Load into an already allocated block of shared memory."""
        self._read_arr = SharedArray(id_ + "_r", shape, dtype)
        self._write_arr = SharedArray(id_ + "_w", shape, dtype)

    @classmethod
    def create(cls, id_, shape=None, dtype=None, *, array=None):
        """
        Allocates the underlying shared memory and returns a view into it.

        Allocation can be specified by shape and dtype or just an existing numpy.ndarray
        If an array is used for specification shape and dtype params are ignored.

        Parameters
        ----------
        id_ : str
            Identifier for this shared block to be discovered from elsewhere.
        shape : tuple[int, ...]
            A valid shape for a numpy.ndarray
        dtype : numpy.dtype
        array : numpy.ndarray
            Optional initial data to initialize the memory block.

        Returns
        -------
        inst : DoubleBufferedArray
            Creates and instance of the class.
            Note: SharedMemory could potentially be reclaimed if this were gc'd.
        """
        if array is None:
            array = np.zeros(shape, dtype)
        shm_r = shared_memory.SharedMemory(id_ + "_r", create=True, size=array.nbytes)
        shm_w = shared_memory.SharedMemory(id_ + "_w", create=True, size=array.nbytes)
        inst = cls(id_, array.shape, array.dtype)
        inst[:] = array[:]
        inst.flip()
        shm_r.close()
        shm_w.close()
        return inst

    def unlink(self):
        """Should be called once when the shared memory can be reclaimed."""
        self._read_arr.unlink()
        self._write_arr.unlink()

    def close(self):
        """Should be called on every instance when no longer in use."""
        self._read_arr.close()
        self._write_arr.close()

    def flip(self):
        """Copy the write array into the read array."""
        self._read_arr[:] = self._write_arr[:]

    def __len__(self):
        return len(self._read_arr)

    def __eq__(self, other):
        return self._read_arr == other

    def __getitem__(self, idx):
        return self._read_arr[idx]

    def __setitem__(self, idx, value):
        self._write_arr[idx] = value

    def __getattr__(self, item):
        return getattr(self._read_arr, item)

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
