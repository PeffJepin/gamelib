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
