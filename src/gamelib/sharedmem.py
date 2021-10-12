from __future__ import annotations

import os
from multiprocessing import shared_memory
from typing import NamedTuple, Iterable, Any

import numpy as np

_POSIX = os.name == "posix"
_SHM_NAME = "__gamelib_shm__"
_shm_file: _SharedMemoryFile | None = None


def allocate(specs):
    """
    Allocates the shm file for an app. A call to allocate must
    come before any use of shared memory, and only shared memory
    described by parameter specs will be available.

    Parameters
    ----------
    specs : Iterable[ArraySpec]
        Specifications for allocating this apps memory.

    Returns
    -------
    None

    Raises
    ------
    FileExistsError:
        If allocate has already been called and has not since been unlinked.
    """
    global _shm_file
    if _shm_file is not None:
        raise FileExistsError("Shared memory has already been allocated.")
    _shm_file = _SharedMemoryFile(specs)


def connect(spec, *, readonly=False):
    """
    Connect to a shared block of memory described by spec.

    Parameters
    ----------
    spec : ArraySpec
        This spec should have been allocated by this point.
    readonly : bool

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
    if readonly:
        array.flags.writeable = False
    return array


def close():
    """
    Close this local view into the shared file.

    This function should be called from any child process accessing shared memory
    before exiting.

    Trying to use an array after tearing away its buffer may cause SegmentationFault

    Returns
    -------
    None
    """
    global _shm_file
    if _shm_file is None:
        return
    _shm_file.close()
    _shm_file = None


def unlink():
    """
    Unlink the underlying shared memory.

    # WARNING
    Trying to use an array after tearing away its buffer may cause SegmentationFault

    This should be called once for the entire app using shared memory, once
    all other processes are finished with the memory.

    Unlink doesn't actually do anything on windows, instead windows connections
    should all be closed and the file will be unlinked. This function accounts for that.

    Returns
    -------
    None
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


class ArraySpec(NamedTuple):
    # See np.dtype documentation for supported data types.
    name: str
    dtype: Any
    length: int


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

    def retrieve_from_spec(self, spec) -> np.ndarray:
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
            shape=(header_size // self._HEADER_DTYPE.itemsize,),
            dtype=self._HEADER_DTYPE,
            buffer=self._shm.buf,
            offset=header_desc.nbytes,
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
            empty_spec = np.empty((spec.length,), spec.dtype)
            header_data.append((spec.name, offset, empty_spec.nbytes))
            offset += empty_spec.nbytes

        header = np.array(header_data, self._HEADER_DTYPE)
        header_desc = np.array([header.nbytes], int)
        header["offset"] += header.nbytes + header_desc.nbytes
        body = np.zeros((offset,), np.uint8)

        return header_desc, header, body
