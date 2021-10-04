from __future__ import annotations

import os
from multiprocessing import shared_memory

import numpy as np

_POSIX = os.name == "posix"


class SharedBlock:
    def __init__(self, shared_arrays, *, name_extra=""):
        """
        Allocates passed in arrays and keeps references to the arrays and created shm.

        Parameters
        ----------
        shared_arrays : list[SharedArray]
            All of this SharedArray this block will manage.
        name_extra : str
            Optionally gets added to array.name to mutate shm names.
        """
        self._extra = str(name_extra)
        self._array_lookup = {
            array.name + self._extra: array for array in shared_arrays
        }
        self._shm_lookup = self._allocate_shm()

    def connect_arrays(self, *arrays):
        """
        Connects SharedArrays to appropriate shm.

        Parameters
        ----------
        arrays : SharedArray

        Raises
        ------
        KeyError:
            If this block hasn't been used to allocate the array.
        """
        for array in arrays:
            shm_name = array.name + self._extra
            shm = self._shm_lookup[shm_name]
            self._array_lookup[shm_name] = array
            array.open_view(shm)

    def close(self):
        """Closes this view into the block. Called on every instance for cleanup."""
        for array in self._array_lookup.values():
            array.close()
        for shm in self._shm_lookup.values():
            shm.close()
        self._array_lookup = None
        self._shm_lookup = None

    def unlink(self):
        """Un-links the shm. Called once by the main process for the lifetime of the shm."""
        for array in self._array_lookup.values():
            array.unlink()
        for shm in self._shm_lookup.values():
            shm.unlink()
        self._array_lookup = None
        self._shm_lookup = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array_lookup"] = dict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _allocate_shm(self):
        """Allocates SharedArrays based on their key values in the lookup."""
        shm_lookup = dict()
        for name, array in self._array_lookup.items():
            shm = array.allocate(name)
            shm_lookup[name] = shm
        return shm_lookup


class SharedArray:
    def __init__(self, name, shape, dtype, *, try_open=True):
        """
        Designate the general specification of the SharedArray.

        Does not automatically allocate the memory, though will load
        into existing memory with given name.

        Parameters
        ----------
        name : str
            Identifier to potentially find an existing shm.
        shape : tuple
            numpy.ndarray compatible shape
        dtype : numpy.dtype
        try_open : bool
            Should this array be opened on __init__ or not
        """
        self.name = str(name)
        self._shape = shape
        self._dtype = dtype
        self._arr = None
        self._shm = None
        if try_open:
            try:
                self.open_view()
            except FileNotFoundError:
                self._arr = None

    @property
    def is_open(self):
        return self._shm is not None

    @classmethod
    def create(cls, name, shape=None, dtype=None, *, array=None):
        """
        Allocates the underlying shared memory and returns a view into it.

        Allocation can be specified by shape and dtype or just an existing numpy.ndarray
        If an array is used for specification shape and dtype params are ignored.

        Parameters
        ----------
        name : str
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

        Raises
        ------
        FileExistsError:
            If shm is already allocated with this name.
        """
        if array is None:
            array = np.zeros(shape, dtype)
        inst = cls(name, array.shape, array.dtype)
        inst.allocate()
        inst._arr[:] = array[:]
        return inst

    def allocate(self, name=None):
        """
        Allocates the underlying shared memory and keeps a reference to it.

        Also initializes the arrays view into the memory.

        Parameters
        ----------
        name : str
            Optional override to shm name

        Returns
        -------
        shm : SharedMemory

        Raises
        ------
        FileExistsError:
            If memory is already allocated at this name.
        """
        blank = np.zeros(self._shape, self._dtype)
        self._shm = shared_memory.SharedMemory(
            name or self.name, create=True, size=blank.nbytes
        )
        self._arr = np.ndarray(self._shape, self._dtype, self._shm.buf)
        self._arr[:] = blank[:]
        return self._shm

    def open_view(self, shm=None):
        """
        Opens a view into an already initialized array given the appropriate shm.

        Parameters
        ----------
        shm : SharedMemory
            Should correspond to this arrays specification.
            Used in place of default lookup with specified name if given.

        Raises
        ------
        FileNotFoundError:
            If the memory has not been allocated.
        """
        self._arr = np.array(self._shape, self._dtype)
        self._shm = shm or shared_memory.SharedMemory(
            self.name, create=False, size=self._arr.nbytes
        )
        self._arr = np.ndarray(self._shape, self._dtype, self._shm.buf)

    def unlink(self):
        """Should be called only once after the memory is no longer in use."""
        if not _POSIX:
            # SharedMemory.unlink() does nothing on windows but is required on posix.
            # Close should still be called on this shm if `unlink` is the request.
            return self.close()

        if self._shm is None:
            try:
                self.open_view()
            except FileNotFoundError:
                self._arr = None
                return

        self._shm.unlink()
        self._arr = None
        self._shm = None

    def close(self):
        """Should be called from every instance when that particular instance is no longer in use."""
        if self._shm is None:
            self._arr = None
            return
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
        if self.is_open:
            return len(self._arr)
        return None

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

    def __init__(self, name, shape, dtype, *, try_open=True):
        """Initialize the specification of the arrays and attempts to open a view into them."""
        self._read_arr = SharedArray(name + "_r", shape, dtype, try_open=try_open)
        self._write_arr = SharedArray(name + "_w", shape, dtype, try_open=try_open)
        self._dtype = dtype
        self._shape = shape
        self.name = name

    @classmethod
    def create(cls, name, shape=None, dtype=None, *, array=None):
        """See SharedArray Documentation."""
        if array is None:
            array = np.zeros(shape, dtype)
        inst = cls(name, array.shape, array.dtype)
        inst.allocate()
        inst[:] = array[:]
        inst.flip()
        return inst

    def allocate(self, name=None):
        """See SharedArray documentation."""
        if name is not None:
            self._read_arr.allocate(name + "_r")
            self._write_arr.allocate(name + "_w")
            return
        self._read_arr.allocate()
        self._write_arr.allocate()

    def open_view(self):
        """See SharedArray Documentation."""
        self._read_arr.open_view()
        self._write_arr.open_view()

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

    @property
    def is_open(self):
        return all(array.is_open for array in self.arrays)

    @property
    def arrays(self):
        """Get the underlying SharedArray instances."""
        return self._read_arr, self._write_arr

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
