from __future__ import annotations

import os
from multiprocessing import shared_memory
from typing import Union, NamedTuple, Type, Iterable

import numpy as np

_POSIX = os.name == "posix"
_SHM_NAME = "__gamelib_shm__"
_shm_file = None
_arrays = []


def allocate(specs):
    """
    Allocates the shm file for an app. A call to allocate must
    come before any use of shared memory, and only shared memory
    described by parameter specs will be available.

    Parameters
    ----------
    specs : Iterable[ArraySpec]
        Specifications for allocating this apps memory.
    """
    global _shm_file
    if _shm_file is not None:
        unlink()
    _shm_file = _SharedMemoryFile(specs)


def connect(spec):
    """
    Connect to a shared block of memory described by spec.

    Parameters
    ----------
    spec : ArraySpec
        This spec should have been allocated by this point.

    Returns
    -------
    array : np.ndarray
        A view into the shared file.

    Raises:
    -------
    FileNotFoundError:
        If this spec hasn't been allocated prior to calling.
    """
    global _shm_file
    if _shm_file is None:
        _shm_file = _SharedMemoryFile()
    array = _shm_file.retrieve_from_spec(spec)
    _arrays.append(array)
    return array


def close():
    """
    Close this local view into the shared file.

    This function should be called from any child process accessing shared memory
    before exiting.

    Trying to use an array after tearing away its buffer may cause SegmentationFault
    """
    global _shm_file
    if _shm_file is None:
        return
    _shm_file.close()
    _shm_file = None


def unlink():
    """
    Unlink the underlying shared memory.

    This should be called once for the entire app using shared memory, once
    all other processes are finished with the memory.

    Unlink doesn't actually do anything on windows, instead windows connections
    should all be closed and the file will be unlinked. This function accounts for that.

    Trying to use an array after tearing away its buffer may cause SegmentationFault
    """
    global _shm_file

    if not _POSIX:
        return close()

    if _shm_file is not None:
        _shm_file.unlink()
        _shm_file = None
        return

    try:
        _attempted_connection = _SharedMemoryFile()
        _attempted_connection.unlink()
    except FileNotFoundError:
        pass


def _cleanup_arrays():
    for arr in _arrays:
        arr.buffer = None


class _SharedMemoryFile:
    """
    Internal manager of the shared memory file.

    Allocated specs are described in a header so no metadata
    needs to be passed across process boundaries for access.

    The first four bytes (int) describes the length of the header in bytes.
    """

    _HEADER_DTYPE = np.dtype([("name", "U100"), ("offset", "i4"), ("length", "i4")])
    _shm = shared_memory.SharedMemory

    def __init__(self, specs=None):
        if specs is None:
            self._load_header()
        else:
            self._create_new_shm(specs)

    def retrieve_from_spec(self, spec):
        for entry in self._header:
            if entry[0] == spec.name:
                return np.ndarray(
                    shape=(spec.length,),
                    dtype=spec.dtype,
                    buffer=self._shm.buf,
                    offset=entry[1],
                )

    def close(self):
        self._shm.close()
        self._shm = None

    def unlink(self):
        self._shm.unlink()
        self._shm = None

    def _load_header(self):
        self._shm = shared_memory.SharedMemory(_SHM_NAME)
        header_desc = np.ndarray(
            shape=(1,),
            dtype=int,
            buffer=self._shm.buf,
        )
        header_size = header_desc[0]
        self._header = np.ndarray(
            shape=(header_size//self._HEADER_DTYPE.itemsize,),
            dtype=self._HEADER_DTYPE,
            buffer=self._shm.buf,
            offset=header_desc.nbytes
        )

    def _create_new_shm(self, specs):
        # figure out how much memory is needed
        header_desc, header, body = self._calculate_allocations(specs)
        total_size = header_desc.nbytes + header.nbytes + body.nbytes
        self._shm = shared_memory.SharedMemory(_SHM_NAME, create=True, size=total_size)

        # copy initial data into shm
        offset = 0
        for initial_data in (header_desc, header, body):
            view = np.ndarray(
                shape=initial_data.shape,
                dtype=initial_data.dtype,
                buffer=self._shm.buf,
                offset=offset,
            )
            view[:] = initial_data[:]
            offset += initial_data.nbytes

        # cache a view into the header
        self._header = np.ndarray(
            shape=header.shape,
            dtype=header.dtype,
            buffer=self._shm.buf,
            offset=header_desc.nbytes,
        )

    def _calculate_allocations(self, specs):
        offset = 0
        header_data = []
        for spec in specs:
            blank = np.empty((spec.length,), spec.dtype)
            header_data.append((spec.name, offset, blank.nbytes))
            offset += blank.nbytes

        header = np.array(header_data, self._HEADER_DTYPE)
        header_desc = np.array([header.nbytes], int)
        header["offset"] += header.nbytes + header_desc.nbytes
        body = np.zeros((offset,), np.uint8)

        return header_desc, header, body


class ArraySpec(NamedTuple):
    name: str
    dtype: Union[np.number, Type[int], Type[float]]
    length: int


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

    def __init__(self, name, dtype, length):
        self._name = name
        self._dtype = dtype
        self._length = length
        self._read_arr = None
        self._write_arr = None

    @property
    def is_open(self):
        return not any((arr is None for arr in (self._read_arr, self._write_arr)))

    @property
    def _read_spec(self):
        return ArraySpec(self._name + '_r', self._dtype, self._length)

    @property
    def _write_spec(self):
        return ArraySpec(self._name + '_w', self._dtype, self._length)

    @property
    def specs(self):
        """
        Get the specs describing the two shared memory arrays.

        Returns
        -------
        specs : tuple[ArraySpec]
        """
        return self._read_spec, self._write_spec

    def flip(self):
        """Copy the write array into the read array."""
        self._read_arr[:] = self._write_arr[:]

    def connect(self):
        """Gets a reference to a view into shared memory."""
        self._read_arr = connect(self._read_spec)
        self._write_arr = connect(self._write_spec)

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
