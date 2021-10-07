from __future__ import annotations

import os
from multiprocessing import shared_memory
from typing import Union, NamedTuple, Type, Iterable

import numpy as np

_POSIX = os.name == "posix"


class ArraySpec(NamedTuple):
    name: str
    dtype: Union[np.number, Type[int], Type[float]]


class SharedBlock:
    def __init__(self, array_specs, max_entities, *, name_extra=""):
        """
        Allocates passed in arrays and keeps references to the arrays and created shm.

        Parameters
        ----------
        array_specs : Iterable[ArraySpec]
            The datatype and identifier to classify a piece of shared memory
        max_entities : int
            Arrays are allocated at this length.
        name_extra : str
            Optionally gets added to array.name to mutate shm names.
        """
        self._extra = str(name_extra)
        self._max_entities = max_entities
        self._array_lookup = dict()
        self._shm_lookup = dict()

        for spec in array_specs:
            self._allocate_shm(spec)

    def __getitem__(self, spec):
        """
        Parameters
        ----------
        spec : ArraySpec

        Returns
        -------
        arr : np.ndarray
        """
        if spec.name not in self._array_lookup:
            self._connect_array(spec)
        return self._array_lookup[spec.name]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array_lookup"] = dict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def close_shm(self):
        """Closes this view into the block. Called on every instance for cleanup."""
        if self._shm_lookup is not None:
            for shm in self._shm_lookup.values():
                shm.close()
        self._array_lookup = None
        self._shm_lookup = None

    def unlink_shm(self):
        """Un-links the shm. Called once by the main process for the lifetime of the shm."""
        if self._shm_lookup is not None:
            for shm in self._shm_lookup.values():
                shm.unlink()
        self._array_lookup = None
        self._shm_lookup = None

    def _allocate_shm(self, spec: ArraySpec) -> None:
        init = np.zeros((self._max_entities,), spec.dtype)
        shm = shared_memory.SharedMemory(
            spec.name + self._extra, create=True, size=init.nbytes
        )
        arr = np.ndarray((self._max_entities,), spec.dtype, shm.buf)
        arr[:] = init[:]
        self._array_lookup[spec.name] = arr
        self._shm_lookup[spec.name] = shm

    def _connect_array(self, spec: ArraySpec) -> None:
        shm = self._shm_lookup[spec.name]
        array = np.ndarray((self._max_entities,), spec.dtype, shm.buf)
        self._array_lookup[spec.name] = array


class DoubleBufferedArray:
    """
    Maintains two SharedArray instances and delegates reads/writes to separate arrays.
    The flip method can be called to copy the write array into the read array.
    """

    def __init__(self, name, dtype):
        self._read_spec = ArraySpec(name + "_r", dtype)
        self._write_spec = ArraySpec(name + "_w", dtype)
        self._read_arr = None
        self._write_arr = None

    @property
    def is_open(self):
        return not any((arr is None for arr in (self._read_arr, self._write_arr)))

    @property
    def specs(self):
        """
        Get the specs describing the two internal arrays.

        Returns
        -------
        specs : tuple[ArraySpec]
        """
        return self._read_spec, self._write_spec

    def flip(self):
        """Copy the write array into the read array."""
        self._read_arr[:] = self._write_arr[:]

    def connect(self, shared_block):
        """
        Gets a reference to a view into shared memory.

        Parameters
        ----------
        shared_block : SharedBlock
            The shared block should have been initialized with reference to this array spec.
        """
        self._read_arr = shared_block[self._read_spec]
        self._write_arr = shared_block[self._write_spec]

    def disconnect(self):
        self._read_arr = None
        self._write_arr = None

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
