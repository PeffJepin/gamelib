from multiprocessing import shared_memory as sm
from typing import Tuple, List

import numpy as np


class SharedArrays:
    ArraySpecification = Tuple[str, tuple, np.dtype]

    def __init__(self, blocks: List[ArraySpecification], create: bool = False):
        self._mem_lookup, self._arr_lookup = dict(), dict()
        for id_, shape, dtype in blocks:
            empty_array = np.empty(shape, dtype)
            shared_memory = sm.SharedMemory(id_, create, empty_array.nbytes)
            shared_array = np.ndarray(shape, dtype, shared_memory.buf)

            self._mem_lookup[id_] = shared_memory
            self._arr_lookup[id_] = shared_array

    def __getitem__(self, id_):
        return self._arr_lookup[id_]


class DoubleBufferedArrays(SharedArrays):
    """
    A shared array that implements itself with double buffers for concurrency.

    Note that all data is doubled this way, so shared data
    should be avoided as much to help reduce this overhead.

    Arrays can be retrieved through indexing just like in SharedArrays
    with read and write access getting managed internally.

    Example
    -------
    >>> array_specs = [('my_id', ...), ...]
    >>> dbl_buffered = DoubleBufferedArrays(array_specs, create=True)
    >>> shared_arr = dbl_buffered['my_id']
    >>> shared_arr[:] = 255

    >>> assert np.all(shared_arr == 255)
    False

    >>> dbl_buffered.swap()
    >>> assert np.all(shared_arr == 255)
    True
    """
    def __init__(self, blocks: List[SharedArrays.ArraySpecification], create: bool = False):
        all_blocks = []
        for block in blocks:
            id_, shape, dtype = block
            doubled_blocks = [(id_ + f'_{i}', shape, dtype) for i in range(2)]
            all_blocks.extend(doubled_blocks)
        indices = [('read', (1,), np.uint), ('write', (1,), np.uint)]
        all_blocks.extend(indices)
        super().__init__(all_blocks, create)
        self.read_index = 0
        self.write_index = 1

    def get_arr(self, id_):
        return self._arr_lookup[id_]

    @property
    def read_index(self):
        return self._arr_lookup['read'][0]

    @read_index.setter
    def read_index(self, value):
        self._arr_lookup['read'][0] = value

    @property
    def write_index(self):
        return self._arr_lookup['write'][0]

    @write_index.setter
    def write_index(self, value):
        self._arr_lookup['write'][0] = value

    def swap(self):
        self.read_index, self.write_index = self.write_index, self.read_index

    def __getitem__(self, id_):
        return _DoubleBufferedProxy(id_, self)


class _DoubleBufferedProxy:
    """A stand in for a double buffered shared array. """

    def __init__(self, arr_name: str, dbl_buffer_obj: DoubleBufferedArrays):
        self._arrays = tuple(dbl_buffer_obj.get_arr(f'{arr_name}_{i}') for i in range(2))
        self._dbl_buffer_obj = dbl_buffer_obj

    def __getitem__(self, key):
        return self._read_array.__getitem__(key)

    def __setitem__(self, key, value):
        self._write_array.__setitem__(key, value)

    def __getslice__(self, i, j):
        return self._read_array.__getslice__(i, j)

    def __setslice__(self, i, j, sequence):
        self._write_array.__setslice__(i, j, sequence)

    def __getattr__(self, item):
        """
        This tries to offer some of the underlying ndarray interface to those viewing this object.
        It wont expose __magic__ methods, this class will need to implement magic methods where
        it wants array-like behavior.
        """
        return self._read_array.__getattr__(item)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Documentation for np array class extensions found here.
        https://numpy.org/doc/stable/reference/arrays.classes.html

        This allows this class to behave like an np.ndarray for ufuncs by finding
        itself in the proposed function arguments and replacing itself with the current read buffer
        """
        corrected_inputs = (
            input_ if input_ is not self else self._read_array
            for input_ in inputs
        )
        return getattr(ufunc, method)(*corrected_inputs, **kwargs)

    def __eq__(self, other):
        return self._read_array == other

    @property
    def _read_array(self):
        """Looks up the appropriate array based off from the index stored in shared memory."""
        return self._arrays[self._dbl_buffer_obj.read_index]

    @property
    def _write_array(self):
        """Looks up the appropriate array based off from the index stored in shared memory."""
        return self._arrays[self._dbl_buffer_obj.write_index]
